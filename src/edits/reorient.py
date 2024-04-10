import torch
import numpy as np
from dataset import Dataset
from model import GaussianModel
from renderer import Renderer
from trainer import Trainer
from settings import Settings
from edits import EditCommand
import pygfx as gfx
import kornia as K

def quaternion_mult(q1, q2):
    out = torch.empty_like(q1)
    out[:,0] = q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] - q1[:,3]*q2[:,3]
    out[:,1] = q1[:,0]*q2[:,1] - q1[:,1]*q2[:,0] - q1[:,2]*q2[:,3] - q1[:,3]*q2[:,2]
    out[:,2] = q1[:,0]*q2[:,2] - q1[:,1]*q2[:,3] - q1[:,2]*q2[:,0] - q1[:,3]*q2[:,1]
    out[:,3] = q1[:,0]*q2[:,3] - q1[:,1]*q2[:,2] - q1[:,2]*q2[:,1] - q1[:,3]*q2[:,0]
    return out

class Reorient_Edit(EditCommand):
    key = "reorient"
    use_selection_mask = False

    def __init__(self, model: GaussianModel, renderer: Renderer, dataset: Dataset, trainer: Trainer, settings : Settings):
        super().__init__(model, renderer, dataset, trainer, settings)
        
    def undo(self):
        if(self.completed):
            pass

    def execute(self, payload):
        t = torch.linalg.inv(torch.tensor(self.renderer.selector.mesh.world.matrix, dtype=torch.float32, device=self.settings.device).T)
        with torch.no_grad():
            xyz_hom = torch.cat([self.model._xyz, 
                                 torch.ones([self.model.get_num_gaussians, 1],
                                 device=self.settings.device)], dim=1)
            xyz_rot = xyz_hom @ t
            self.model._xyz[:,:] = xyz_rot[:,0:3]

            #rots = self.model.get_rotation
            #new_rot = K.geometry.conversions.rotation_matrix_to_quaternion(t[0:3, 0:3].T)
            #new_rot = K.geometry.conversions.normalize_quaternion(new_rot)
            #rots = quaternion_mult(new_rot.expand([rots.shape[0], -1]), rots)
            #self.model._rotation[:,:] = rots[:,:]

            scales = self.model.get_scaling
            scales @= t[0:3, 0:3]
            scales = self.model.scaling_inverse_activation(torch.abs(scales))
            self.model._scaling[:,:] = scales[:,:]
        
        self.renderer.selector.reset_mesh()
        self.settings.orientation @= t.cpu().numpy()
        #self.dataset.update_orientation()
        self.trainer.set_model(self.model)
        self.completed = True