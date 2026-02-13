import numpy as np
import torch
from torch import nn
from model_utils.resnet import ResNet, BottleneckBlock
import torch.nn.functional as F


def rand_indices(pop_size, num_samples, device):
    """Use torch.randperm to generate indices on a 32-bit GPU tensor."""
    return torch.randperm(pop_size, dtype=torch.int32, device=device)[:num_samples]

def masked_farthest_point_sampling(pts, masks, n_samples=1024):
    """
    Farthest-point sampling over masked points (batched).

    Args:
    pts (torch.Tensor): Float tensor of shape (B, N, 3) containing point coordinates.
    masks (torch.Tensor or torch.BoolTensor): Mask of shape (B, N) where True
    indicates a valid point. May be boolean or numeric (0/1).
    n_samples (int, optional): Number of points to sample per batch. Default: 1024.

    Returns:
    f_pts (torch.Tensor): Sampled point coordinates of shape (B, n_samples, 3).
    f_inds (torch.LongTensor): Sampled indices of shape (B, n_samples), dtype torch.long.
    """
    B, N, _ = pts.shape
    masks = masks.float().view(B,N)
    is_empty = ~masks.any(dim=-1)
    ### Dirty hack to prevent crash on empty masks
    masks[:,0] = masks[:,0] + is_empty.int()

    initial_inds = torch.multinomial(masks, 1).repeat(1,3).unsqueeze(1) ### Bx1x3
    initial_pts = torch.gather(pts, dim=1, index=initial_inds)
    dists = torch.norm(pts - initial_pts, dim=-1)#+ (1 - masks) * float('-inf')
    dists = torch.where(masks.bool(), dists, float('-inf'))
    selected_inds = [initial_inds[:,:,0]]### Bx1
    selected_pts = [initial_pts]

    for i in range(1, n_samples):
        farthest_point_inds = dists.argmax(dim=1, keepdim=True).repeat(1,3).unsqueeze(1) ### Bx1x3
        farthest_point = torch.gather(pts, dim=1, index=farthest_point_inds)
        new_dists = torch.norm(pts - farthest_point, dim=-1)
        dists = torch.min(dists, new_dists)
        selected_inds.append(farthest_point_inds[:,:,0]) ### Bx1
        selected_pts.append(farthest_point)

    f_inds = torch.cat(selected_inds, dim=1)
    f_pts = torch.cat(selected_pts, dim=1)
    return f_pts, f_inds




class CategoryPrototype(nn.Module):
    """ 
    Main category prototype module. Contains the canonical representation for each category in the form of keypoints (self.kps)
    and their associated features (self.kps_feats). Also handle logging points for visualization.
    """


    def __init__(self, feature_dims=768, n_cats=18, num_keypoints=30, n_logged_points=2**13, n_sample_points=2**10):

        super(CategoryPrototype, self).__init__()

        self.feat_dim = feature_dims

        self.n_cats = n_cats
        self.num_keypoints = num_keypoints
        self.register_buffer('num_keypoints_per_cat' , torch.tensor([25,12,17,14,10,30,30,15,14,21,16,20,13,20,9,21,18,16])) # number of keypoints per SPair category

        self.kps = nn.Parameter(torch.randn(self.n_cats,self.num_keypoints,3) * 1e-5) #randomly initialize close to zero
        self.kps_feats = nn.Parameter(torch.randn(self.n_cats,self.num_keypoints,self.feat_dim) / np.sqrt(self.feat_dim))

        self.proto_idx_start = [0] * self.n_cats
        self.n_logged_points = n_logged_points
        self.n_sample_points = n_sample_points
        self.proto_points = nn.Parameter(torch.zeros(self.n_cats,self.n_logged_points,3))
        self.proto_adapted_feats = nn.Parameter(torch.randn(self.n_cats,self.n_logged_points,self.feat_dim) / np.sqrt(self.feat_dim))

        self.geom_temp = nn.Parameter(torch.zeros(1))

        self.cat_dict = {
            'aeroplane':    0,
            'bicycle':      1,
            'bird':         2,
            'boat':         3,
            'bottle':       4,
            'bus':          5,
            'car':          6,
            'cat':          7,
            'chair':        8,
            'cow':          9,
            'dog':         10,
            'horse':       11,
            'motorbike':   12,
            'person':      13,
            'pottedplant': 14,
            'sheep':       15,
            'train':       16,
            'tvmonitor':   17,
            }


    def get_cat_proto(self, cat):
        """
        Retrieve prototype tensors for a single category.
        """
        cat_id = self.cat_dict[cat]
        return self.kps[cat_id], self.kps_feats[cat_id], self.proto_points[cat_id], self.proto_adapted_feats[cat_id]

    def get_cat_mask(self, cat):
        """
        Return a boolean mask indicating which keypoints are valid for a given category.
        """
        cat_id = self.cat_dict[cat]
        mask = torch.zeros(self.num_keypoints)
        end_index = self.num_keypoints_per_cat[cat_id]
        mask[:end_index] = 1

        return mask


    def postprocess_mask_with_depth(self, mask, depth, n_mad=5):
        """
        Refine segmentation masks using depth outlier rejection (median absolute deviation) to prevent oversegmentation issues.

        Args:
        mask (torch.BoolTensor): Bool tensor of shape (B, H*W) or (B, N) indicating valid pixels/points.
        depth (torch.Tensor): Float tensor of matching shape (B, H*W) or (B, N) containing depth values.
        n_mad (float, optional): Threshold in units of MAD for outlier rejection. Default: 5.

        Returns:
        torch.BoolTensor: Refined mask of shape (B, N).
        """
        new_masks = []
        for (m, d) in zip(mask, depth):
            valid_depths = torch.masked_select(d,m)
            median = valid_depths.nanmedian()
            MAD = (valid_depths - median).abs().nanmedian()
            new_mask = (d - median).abs() < n_mad * MAD
            new_masks.append(new_mask)
        new_masks = torch.stack(new_masks, dim=0)

        ret = mask & new_masks
        valid_mask = ret.any(dim=-1, keepdim=True).any(dim=-2, keepdim=True)
        ret = torch.where(valid_mask, ret, mask)

        return ret




    def sample_object_points(self, point_map, depth, mask, feats):
        """
        Sample object points and associated features from per-pixel point map using masks.
        Args:
        point_map (torch.Tensor): Tensor of shape (B, N, 3) camera-space points.
        depth (torch.Tensor): Tensor of shape (B, N) with depth values (used for mask postprocessing).
        mask (torch.Tensor): Tensor of shape (B, N) with values in [0,1] or booleans indicating object occupancy.
        feats (torch.Tensor): Per-point features of shape (B, N, D).

        Returns:
        tuple:
        object_points (torch.Tensor): Sampled points of shape (B, n_sample_points, 3).
        object_feats (torch.Tensor): Corresponding features of shape (B, n_sample_points, D).
        """

        B = point_map.size(0)
        mask = mask.ge(.99)
        mask = self.postprocess_mask_with_depth(mask, depth, n_mad=5)


        object_points, sampled_inds = masked_farthest_point_sampling(point_map, mask, n_samples=self.n_sample_points)

        object_feats = torch.gather(feats,
                                     dim=1,
                                     index=sampled_inds.reshape(B, self.n_sample_points, 1).expand(-1,-1,self.feat_dim))
        return object_points, object_feats



    def update_protos_from_pts(self, cats, canonical_pts, canonical_adapted_feats, masks=None, n_samples=128):
        # This function randomly logs object points in the canonical space. It is used for visualization purposes only
        with torch.no_grad():
            for idx, cat in enumerate(cats):
                if masks is not None and ~masks[0].any():
                    continue
                cat_id = self.cat_dict[cat]
                idx_start = self.proto_idx_start[cat_id]
                idx_end = min(idx_start + n_samples, self.n_logged_points)
                real_n_samples = idx_end - idx_start
                selected_inds_idx = rand_indices(canonical_pts.size(1), real_n_samples, device=canonical_pts.device)
                self.proto_points[cat_id, idx_start:idx_end] = canonical_pts[idx,selected_inds_idx]
                self.proto_adapted_feats[cat_id, idx_start:idx_end] = canonical_adapted_feats[idx,selected_inds_idx]
                self.proto_idx_start[cat_id] = idx_end % self.n_logged_points

