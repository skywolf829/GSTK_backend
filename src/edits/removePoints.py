import torch
from edits import EditCommand
import numpy as np

class Remove_Edit(EditCommand):
    key = "removePoints"
    use_selection_mask = True
    
    def __init__(self, model, renderer, dataset, trainer, settings):
        super().__init__(model, renderer, dataset, trainer, settings)

        self.remove_pct = 100
        self.invert_selection = False
        
    def undo(self):
        if(self.completed):
            mask = torch.zeros([self.model.get_num_gaussians], dtype=torch.bool, device=self.settings.device)
            start = self.model.get_num_gaussians - self.num_points
            mask[start:] = True
            self.trainer.prune_points(mask)

    def execute(self, payload):
        self.remove_pct = payload['removePercent']
        self.invert_selection = payload['invertSelection']

        mask = self.renderer.get_selection_mask(self.model.get_xyz).type(torch.bool)
        if(self.remove_pct == 100):            
            if(self.invert_selection):
                mask = ~mask
            self.trainer.prune_points(mask)
        else:
            new_xyz, new_scales, new_rots, new_feats, new_opacities = self.decimate_model(mask)
            self.trainer.prune_points(mask)
            self.trainer.densification_postfix(new_xyz, new_feats[:,0:1,:], new_feats[:,1:,:], new_opacities, new_scales, new_rots)

        self.completed = True
    
    def decimate_model(self, mask):
        
        points_xyz = self.model.get_xyz
        if(mask is not None):         
            mask_indices = torch.argwhere(mask)[:,0]
        else:
            mask_indices = torch.arange(self.model.get_num_gaussians, dtype=torch.long, device=self.model.get_xyz.device)
        num_points_final = int(mask.sum() * (self.remove_pct/100))
        
        tree_nodes = self.decimate_KDTree(points_xyz, num_points_final, mask_indices)
        new_xyz = []
        new_scales = []
        new_rots = []
        new_feats = []
        new_opacities = []
        for node in tree_nodes:
            avg_xyz = self.model.get_xyz[node].mean(dim=0)[None,:]
            avg_scale = self.model.scaling_inverse_activation(self.model.get_scaling[node].mean(dim=0) * (100/self.remove_pct))[None,:]
            Q = self.model.get_rotation[node]
            # https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
            _, vecs = torch.linalg.eig(Q.T @ Q)
            avg_rot = torch.nn.functional.normalize(vecs[:,0:1].real).T
            avg_feat = self.model.get_features[node].mean(dim=0)[None,:,:]
            avg_opacity = self.model.inverse_opacity_activation(self.model.get_opacity[node].mean(dim=0))[None,:]

            new_xyz.append(avg_xyz)
            new_scales.append(avg_scale)
            new_rots.append(avg_rot)
            new_feats.append(avg_feat)
            new_opacities.append(avg_opacity)

        new_xyz = torch.cat(new_xyz, dim=0)    
        new_scales = torch.cat(new_scales, dim=0)
        new_rots = torch.cat(new_rots, dim=0)    
        new_feats = torch.cat(new_feats, dim=0)    
        new_opacities = torch.cat(new_opacities, dim=0)
        
        return new_xyz, new_scales, new_rots, new_feats, new_opacities

    def decimate_KDTree(self, points, num_points_final, starting_mask):
        node_indices = [starting_mask]
        finished_nodes = []

        while len(node_indices)+len(finished_nodes) < num_points_final:
            node_idx = node_indices.pop(0)
            these_points = points[node_idx]
            dim_ranges = (these_points.amax(dim=0)-these_points.amin(dim=0))
            max_dim = torch.argmax(dim_ranges)

            center = torch.median(these_points[:,max_dim])
            mask = these_points[:,max_dim] < center
            left = node_idx[mask]
            right = node_idx[~mask]

            if(left.shape[0] > 1):
                node_indices.append(left)
            elif(left.shape[0] == 1):
                finished_nodes.append(left)
            if(right.shape[0] > 1):
                node_indices.append(right)
            elif(right.shape[0] == 1):
                finished_nodes.append(right)

        node_indices.extend(finished_nodes)

        return node_indices   
  
    def decimate_FPS(self, points, num_points_final, starting_mask):
        """
        Subsamples a point cloud using the Farthest Point Sampling (FPS) algorithm.
        
        Parameters:
        - points: A numpy array of shape (N, D) where N is the number of points and D is the dimension.
        - num_samples: The number of points to sample in the subsampled point cloud.
        
        Returns:
        - A numpy array of shape (num_samples, D) containing the subsampled points.
        """
        p = points[starting_mask]
        N, D = p.shape
        # Start with a random point from the dataset
        farthest_pts = torch.zeros((num_points_final, D), dtype=torch.float32, device=points.device)
        farthest_pts[0] = p[np.random.randint(N)]
        distances = torch.full([N], torch.inf, device=points.device)
        indices = []
        
        for i in range(1, num_points_final):
            dist = torch.sum((p - farthest_pts[i-1])**2, dim=1)
            distances = torch.minimum(dist, distances)
            ind = torch.argmax(distances)
            farthest_pts[i] = p[ind]
        
        return farthest_pts

    def decimate_octree(self, points, num_points_final, starting_mask):
        pass

    def decimate_random(self, points, num_points_final, starting_mask):
        pass

if __name__ == "__main__":
    edit = Remove_Edit(None, None, None, None, None)
    N = 1000000
    points = torch.rand([N, 3], dtype=torch.float32, device='cuda')
    mask = torch.ones([N], dtype=torch.bool, device='cuda')
    final_points = edit.decimate_octree(points, 100000, mask)