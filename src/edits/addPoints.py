import torch
import numpy as np
from dataset import Dataset
from model import GaussianModel
from renderer import Renderer
from trainer import Trainer
from settings import Settings
from utils.sh_utils import RGB2SH
from edits import EditCommand

class Add_Edit(EditCommand):
    def __init__(self, model: GaussianModel, renderer: Renderer, dataset: Dataset, trainer: Trainer, settings : Settings):
        super().__init__(model, renderer, dataset, trainer, settings)

        self.num_points = 0
        self.dist_type = ""
        
        self.key = "addPoints"

    def undo(self):
        if(self.completed):
            mask = torch.zeros([self.model.get_num_gaussians], dtype=torch.bool, device=self.settings.device)
            start = self.model.get_num_gaussians - self.num_points
            mask[start:] = True
            self.trainer.prune_points(mask)

    def execute(self, payload):
        self.num_points = payload['num_points']
        self.dist_type = payload['distribution']

        if(self.dist_type == "uniform"):
            samples = torch.rand([self.num_points, 3], device=self.settings.device, dtype=torch.float32)-0.5
        elif(self.dist_type == "normal"):
            samples = (torch.randn([self.num_points, 3], device=self.settings.device, dtype=torch.float32)/5).clamp(-0.5, 0.5)
        elif(self.dist_type == "inverse_normal"):
            samples = (torch.randn([self.num_points, 3], device=self.settings.device, dtype=torch.float32)/5).clamp(-0.5, 0.5)
            samples[samples>0] = 0.5 - samples[samples>0]
            samples[samples<0] = -0.5 - samples[samples<0]

        samples = self.renderer.selector.transform_to_selector_world(samples)

        max_scale = (samples.amax(dim=0)-samples.amin(dim=0)) / 50
        scales = torch.ones_like(samples) * max_scale
        scales = self.model.scaling_inverse_activation(scales)
        rots = torch.zeros([self.num_points, 4], device=self.settings.device, dtype=torch.float32)
        rots[:,0] = 1.
        opacities = self.model.inverse_opacity_activation(0.1 * torch.ones((self.num_points, 1), dtype=torch.float32, device=self.settings.device))

        rgb = RGB2SH(0.5 * torch.ones([self.num_points, 3], dtype=torch.float32, device=self.settings.device))
        features = torch.zeros((self.num_points, 3, (self.settings.sh_degree + 1) ** 2)).float().to(self.settings.device)
        features[:, :3, 0 ] = rgb
        features[:, :3, 1:] = 0.0
        self.trainer.densification_postfix(samples, features[:,0:3,0:1].mT, features[:,:,1:].mT, opacities, scales, rots)

        self.completed = True