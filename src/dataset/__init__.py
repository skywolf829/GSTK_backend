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
import random
import json
from dataset.dataset_readers import sceneLoadTypeCallbacks
from settings import Settings
import numpy as np
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from utils.system_utils import mkdir_p
import subprocess

class Dataset:
    def __init__(self, settings : Settings, 
                 resolution_scales=[1.0], server_controller=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = settings.save_path
        self.loaded_iter = None
        self.resolution_scales = resolution_scales
        self.white_background = settings.white_background
        self.train_cameras = {}
        self.test_cameras = {}
        self.settings = settings
        self.server_controller = server_controller
        self.loading = False
        self.loaded = False
        self.colmap_path = ""
        self.imagemagick_path = ""
        self.available_datasets = self.get_available_datasets()

        if(self.server_controller is not None):            
            self.server_controller.subscribe_to_messages(
                'datasetInitialize', 
                self.initialize_dataset)
              
    def load_dataset(self):
        self.loaded = False
        if os.path.exists(os.path.join(self.settings.dataset_path, "sparse")):
            print("Found sparse folder, assuming COLMAP dataset.")
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                self.settings.dataset_path, "images", False)
        elif os.path.exists(os.path.join(self.settings.dataset_path, 
                                         "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender dataset.")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.settings.dataset_path, 
                    self.settings.white_background, False)
        else:
            print("Could not recognize scene type!")
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            #with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            #with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            #    json.dump(json_cams, file)

        random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, self.settings)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, self.settings)
        self.scene_info = scene_info

        min_pos = self.scene_info.point_cloud.points.min(axis=0)
        max_pos = self.scene_info.point_cloud.points.max(axis=0)
        max_diff = np.max(max_pos - min_pos)
        self.settings.spatial_lr_scale = max_diff.flatten()[0]

        self.loaded = True
        print("Dataset loaded")

    def get_available_datasets(self):
        datasets_path = os.path.abspath(os.path.join(os.path.abspath(__file__), 
                        "..", "..", "..", "data"))
        return os.listdir(datasets_path)

    def __getitem__(self, idx, scale=1.0):
        return self.train_cameras[scale][idx % len(self.train_cameras[scale])]

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    async def send_state(self):
        self.available_datasets = self.get_available_datasets()
        if self.server_controller is not None:
            message = {
                "type": "datasetState",
                "data": {
                    "datasetPath": self.settings.dataset_path,
                    "datasetDevice": self.settings.data_device,
                    "colmapPath": self.colmap_path,
                    "imagemagickPath": self.imagemagick_path,
                    "loading": self.loading,
                    "loaded": self.loaded,
                    "availableDatasets": self.available_datasets
                }
            }
            await self.server_controller.broadcast(message)

    async def initialize_dataset(self, data, websocket):
        if self.loading:
            return
        self.loading = True

        # relative dataset path
        
        self.colmap_path = data['colmap_path']
        self.imagemagick_path = data['imagemagick_path']

        datasets_path = os.path.abspath(os.path.join(os.path.abspath(__file__), 
                        "..", "..", "..", "data"))
        data['dataset_path'] = os.path.join(datasets_path, data['dataset_path'])

        # Just load all key data to settings
        for k in data.keys():
            # Check if the keys line up
            if k in self.settings.keys():
                self.settings.params[k] = data[k]

        # Check to make sure dataset path exists
        if not os.path.exists(data["dataset_path"]):
            await self.server_controller.broadcast({
                    "type": "datasetError", 
                    "data": {
                        "header": "Dataset error",
                        "body": f"No data found at {data['dataset_path']}"
                    }                
                })
            self.loading = False
            return
        
        # Check if the data device exists
        try:
            a = torch.empty([32], dtype=torch.float32, device=data['data_device'])
            del a
        except Exception as e:
            await self.server_controller.broadcast({
                    "type": "datasetError", 
                    "data": {
                        "header": "Dataset error",
                        "body": f"Device does not exist: data_device={data['data_device']}"
                    }                
                })
            self.loading = False
            return
        
        # Create dataset
        try:
            await self.server_controller.broadcast(
                {
                    "type": "datasetLoading", 
                    "data": {    
                        "loaded": False,
                        "header": "Dataset loading...",
                        "message": "Loading dataset from storage...",
                        "percent": 0,
                        "totalPercent": 0
                    }
                }
            )
            self.load_dataset()
            self.server_controller.trainer.set_dataset(self)
            self.server_controller.trainer.on_settings_update(self.settings)
        except Exception as e:
            # Doesn't recognize dataset, use COLMAP to turn it into a dataset from images
            await self.server_controller.broadcast(
                {
                    "type": "datasetLoading", 
                    "data": {    
                        "loaded": False,
                        "header": "Dataset loading...",
                        "message": "Attempting to create dataset from COLMAP...",
                        "percent": 0,
                        "totalPercent": 0
                    }
                }
            )

            # move images in directory to an <input> folder
            if not os.path.exists(os.path.join(data['dataset_path'], "input")):
                mkdir_p(os.path.join(data['dataset_path'], "input"))
            for item in os.listdir(data['dataset_path']):
                if '.png' in item.lower() or ".jpg" in item.lower() or ".jpeg" in item.lower():
                    os.rename(os.path.join(data['dataset_path'], item), 
                              os.path.join(data['dataset_path'], "input", item))
            
            convert_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "convert.py"))
            cmd = ["python", convert_path, "-s", data['dataset_path']]
            if data['colmap_path'] != "":
                cmd.append("--colmap_executable")
                cmd.append(os.path.abspath(data['colmap_path']))
            if data['imagemagick_path'] != "":    
                cmd.append("--imagemagick_path")
                cmd.append(os.path.abspath(data['imagemagick_path']))      

            try:
                s = subprocess.Popen(cmd)
                s.wait()
                try:
                    await self.server_controller.broadcast(
                        {
                            "type": "datasetLoading", 
                            "data": {    
                                "loaded": False,
                                "header": "Dataset loading...",
                                "message": "COLMAP complete, loading dataset...",
                                "percent": 75,
                                "totalPercent": 75
                            }
                        }
                    )
                    self.load_dataset()
                    #self.trainer.set_dataset(self.dataset)
                    #self.trainer.on_settings_update(self.settings)
                except Exception as e:
                    # Doesn't recognize dataset still
                    await self.server_controller.broadcast({
                        "type": "datasetError", 
                        "data": {
                            "header": "Dataset error",
                            "body": f"Error loading dataset."
                        }                
                    })
                    self.loading = False
                    return
            except Exception as e:
                await self.server_controller.broadcast({
                        "type": "datasetError", 
                        "data": {
                            "header": "Dataset error",
                            "body": "Error running colmap on new dataset."
                        }                
                    })
                self.loading = False
                return
                
        await self.server_controller.broadcast(
            {
                "type": "datasetLoading", 
                "data": {    
                    "loaded": True
                }
            }
        )
        self.loading = False