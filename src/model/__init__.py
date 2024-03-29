#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import os
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid
from torch import nn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import BasicPointCloud
import math
from settings import Settings

class GaussianModel:

    def setup_functions(self):
                
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, settings, server_controller = None):
        self.settings = settings
        self.initialized = False

        self.server_controller = server_controller
        if(self.server_controller is not None):
            self.server_controller.subscribe_to_messages(
                'updateModelSettings', 
                self.handle_model_settings)
            
            self.server_controller.subscribe_to_messages(
                'saveModel', 
                self.handle_save_model)            
            self.server_controller.subscribe_to_messages(
                'loadModel', 
                self.handle_load_model)       
            self.server_controller.subscribe_to_messages(
                'requestAvailableModels', 
                self.request_available_models)
            
            self.server_controller.subscribe_to_messages(
                "initFromPCD", 
                self.init_from_pcd)
            

        self.active_sh_degree = 0
        if(self.settings.white_background):
            bg = torch.tensor([1.,1.,1.], device=self.settings.device)
        elif(not self.settings.random_background):
            bg = torch.tensor([0.,0.,0.], device=self.settings.device)
        else:
            bg = None
        self.background = bg
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.setup_functions()

    async def init_from_pcd(self, data, websocket):
        if self.server_controller.dataset.loaded:
            self.create_from_pcd(self.server_controller.dataset.scene_info.point_cloud)

    async def handle_model_settings(self, data, websocket):
        # Check if the data device exists
        try:
            a = torch.empty([32], dtype=torch.float32, device=data['device'])
            del a
        except Exception as e:
            await self.server_controller.send_error_message("Model settings error", 
                f"Device does not exist: {data['device']}")
            return
        
        if(data['sh_degree'] < 0 or data['sh_degree'] > 3):
            await self.server_controller.send_error_message("Model settings error", 
                f"SH degree is invalid: {data['device']}")
            return
        
        self.settings.params['sh_degree'] = data['sh_degree']
        self.settings.params['device'] = data['device']
        self.on_settings_update(self.settings)
        self.server_controller.trainer.on_settings_update(self.settings)

    async def handle_save_model(self, data, websocket):
        path = os.path.join(os.path.abspath(__file__), "..", "..", "..", "savedModels")
        name = data['modelPath']
        if(".ply" not in name[-4:]):
            name = name + ".ply"
        full_path = os.path.abspath(os.path.join(path, name))
        if not self.initialized:
            await self.server_controller.send_error_message("Model error", 
                f"Cannot save model before it is initialized")
            return
        self.save_ply(full_path)
        self.server_controller.broadcast({
            "type": "popup",
            "data": {
                "header": "Success",
                "body": f"Model successfully saved to {full_path}"
            }
        })
    
    async def handle_load_model(self, data, websocket):
        path = os.path.join(os.path.abspath(__file__), "..", "..", "..", "savedModels")
        name = data['modelPath']
        if not os.path.exists(os.path.join(path, name)):
            name2 = name + ".ply"
            if not os.path.exists(os.path.join(path, name2)):
                data = {"other" : {"error": f"Location doesn't exist: {path}"}}
                await self.server_controller.send_error_message("Model error", 
                    f"Cannot load model from {path}, does not exist.")
                return
            else:
                name = name2

        full_path = os.path.join(path, name)
        try:
            self.load_ply(full_path)
            self.server_controller.trainer.set_model(self)
            await self.server_controller.broadcast({
            "type": "popup",
            "data": {
                "header": "Success",
                "body": f"Model successfully loaded from {full_path}"
            }
        })
        except Exception as e:
            await self.server_controller.send_error_message("Model error", 
                f"Error loading the model.")
            return

    async def request_available_models(self, data, websocket):
        path = os.path.join(os.path.abspath(__file__), "..", "..", "..", "savedModels")
        availableModels = os.listdir(path)
        message = {
            "type": "availableModels",
            "data" :{
                "models": availableModels
            }
        }
        await self.server_controller.broadcast(message)

    def on_settings_update(self, new_settings):

        self.settings = new_settings
        
        if(self.settings.white_background):
            bg = torch.tensor([1.,1.,1.], device=self.settings.device)
        elif(not self.settings.random_background):
            bg = torch.tensor([0.,0.,0.], device=self.settings.device)
        else:
            bg = None
        self.background = bg

        if(self._xyz.device.type not in self.settings.device):
            with torch.no_grad():
                self._xyz = self._xyz.to(self.settings.device)
                self._features_dc = self._features_dc.to(self.settings.device)
                self._features_rest = self._features_rest.to(self.settings.device)
                self._scaling = self._scaling.to(self.settings.device)
                self._rotation = self._rotation.to(self.settings.device)
                self._opacity = self._opacity.to(self.settings.device)
                self.max_radii2D = self.max_radii2D.to(self.settings.device)
                self.background = self.background.to(self.settings.device)

    def create_from_random_pcd(self):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        self.create_from_pcd(pcd)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_num_gaussians(self):
        return self._xyz.shape[0]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.settings.sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.settings.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.settings.device))
        features = torch.zeros((fused_color.shape[0], 3, (self.settings.sh_degree + 1) ** 2)).float().to(self.settings.device)
        features[:, :3, 0 ] = fused_color
        features[:, :3, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(
                torch.from_numpy(np.asarray(pcd.points)).float().to(self.settings.device)), 0.0000001)

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.settings.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.settings.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.settings.device)
        self.initialized = True
        if self.server_controller is not None:
            self.server_controller.trainer.set_model(self)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.settings.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.settings.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.settings.device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.settings.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.settings.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.settings.device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.settings.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.settings.device).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.settings.device)

        self.active_sh_degree = self.settings.sh_degree
        self.initialized = True
        if self.server_controller is not None:
            self.server_controller.trainer.set_model(self)

    def render(self, viewpoint_camera, scaling_modifier = 1.0, alpha_modifier = 0.05,
               rgba_buffer = None, depth_buffer = None, selection_mask = None):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, 
                            requires_grad=True, device=self.settings.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background if self.background is not None \
                else torch.rand([3], device=self.settings.device, dtype=torch.float32),
            scale_modifier=scaling_modifier,
            alpha_modifier=alpha_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.get_xyz
        means2D = screenspace_points
        opacity = self.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = self.get_scaling
        rotations = self.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = self.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            scales = scales,
            rotations = rotations,
            rgba_buffer = rgba_buffer,
            depth_buffer = depth_buffer,
            selection_mask = selection_mask)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}
