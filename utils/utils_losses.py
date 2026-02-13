import torch
import numpy as np
import torch.nn.functional as F
from utils.utils_geoware import permute_indices, flip_keypoints, kabsch, apply_affine, closest_geodesic_neighbors
from .utils_distributed import get_model_attribute
import os

def get_corr_map_loss(img1_desc, img2_desc, corr_map_net, img1_patch_idx, gt_flow, num_patches=60, img2_patch_idx = None):
    # img1_desc shape [1,3600,768]
    corr_map = torch.matmul(img1_desc, img2_desc.transpose(1,2)) # [1,3600,3600]
    corr_map = corr_map.reshape(1, num_patches, num_patches, num_patches, num_patches)
    # feed into network
    corr_map = corr_map_net(corr_map) # [1,60,60,2] 2 for x and y
    corr_map = corr_map.reshape(1, num_patches*num_patches, 2)
    # get the predicted flow
    predict_flow = corr_map[0, img1_patch_idx, :]
    EPE_loss = torch.norm(predict_flow - gt_flow, dim=-1).mean()

    return EPE_loss

def self_contrastive_loss(feat_map, instance_mask=None):
    """
    input: feat_map (B, C, H, W) mask (B, H', W')
    """
    B, C, H, W = feat_map.size()
    if instance_mask is not None:
        # interpolate the mask to the size of the feature map
        instance_mask = F.interpolate(instance_mask.cuda().unsqueeze(1).float(), size=(H, W), mode='bilinear')>0.5
        # mask out the feature map
        feat_map = feat_map * instance_mask
        # make where all zeros to be 1
        feat_map = feat_map + (~instance_mask)
    # Define neighborhood for local loss (8-neighborhood)
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
    local_loss = 0.0
    for i, j in offsets:
        # Shift feature map
        shifted_map = torch.roll(feat_map, shifts=(i, j), dims=(2, 3))
        # Compute the dot product
        dot_product = (feat_map * shifted_map).sum(dim=1)  # Sum along channel dimension
        # Only consider valid region (to avoid wrapping around difference)
        if i > 0:
            dot_product[:, :i, :] = 0
        if j > 0:
            dot_product[:, :, :j] = 0
        if i < 0:
            dot_product[:, i:, :] = 0
        if j < 0:
            dot_product[:, :, j:] = 0
        local_loss -= dot_product.mean()  # negative because we want to maximize the dot product for neighbors

    # For global loss, random sample non-neighbor pixels
    num_samples = H * W  # you can adjust this number based on your requirement
    idx_i = torch.randint(0, H, (num_samples,)).cuda()
    idx_j = torch.randint(0, W, (num_samples,)).cuda()
    idx_k = torch.randint(0, H, (num_samples,)).cuda()
    idx_l = torch.randint(0, W, (num_samples,)).cuda()

    # Ensure they are not neighbors
    mask = ((idx_k-idx_i).abs() > 1) | ((idx_l-idx_j).abs() > 1)
    if instance_mask is not None:
        mask = mask & instance_mask[0, 0, idx_i, idx_j] & instance_mask[0, 0, idx_k, idx_l]
    idx_i, idx_j, idx_k, idx_l = idx_i[mask], idx_j[mask], idx_k[mask], idx_l[mask]
    global_loss = 0.0
    for i, j, k, l in zip(idx_i, idx_j, idx_k, idx_l):
        dot_product = (feat_map[:, :, i, j] * feat_map[:, :, k, l]).sum(dim=1)
        global_loss += dot_product.mean()  # positive because we want to minimize the dot product for non-neighbors
    # Combine local and global losses
    lambda_factor = 0.1  # this can be adjusted based on cross-validation
    loss = local_loss + lambda_factor * global_loss
    return loss 

def get_logits(image_features, text_features, logit_scale):
    # Compute base logits
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T

    return logits_per_image, logits_per_text

def cal_clip_loss(image_features, text_features, logit_scale, self_logit_scale = None):
    total_loss = 0

    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss += (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2

    return total_loss

def calculate_patch_indices_and_loss(args, kps_1, kps_2, desc_1, desc_2, scale_factor, num_patches, aggre_net, threshold, corr_map_net=None, device='cuda'):
    """
    Calculate patch indices and corresponding loss.
    
    Args:
    - kps_1, kps_2: Keypoints for the two images.
    - desc_1, desc_2: Descriptors for the two images.
    - scale_factor, num_patches: Parameters for calculating patch indices.
    - aggre_net: Aggregation network for calculating loss.
    - DENSE_OBJ: Boolean indicating if to use dense objective.
    - GAUSSIAN_AUGMENT: Gaussian augmentation factor.
    - threshold: Threshold for Gaussian augmentation.
    - corr_map_net: Correlation map network, required if CORR_MAP is True.
    - device: Device to use for tensor operations.
    
    Returns:
    - Loss calculated based on the provided parameters.
    """
    def get_patch_idx(scale_factor, num_patches, img1_y, img1_x):
        scaled_img1_y = scale_factor * img1_y
        scaled_img1_x = scale_factor * img1_x
        img1_y_patch = scaled_img1_y.astype(np.int32)
        img1_x_patch = scaled_img1_x.astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch
        if args.DENSE_OBJ:
            return scaled_img1_y, scaled_img1_x, img1_patch_idx
        else:
            return img1_y_patch, img1_x_patch, img1_patch_idx 
        
    # Calculate patch indices for both keypoints
    y1, x1 = kps_1[:, 1].numpy(), kps_1[:, 0].numpy()
    y2, x2 = kps_2[:, 1].numpy(), kps_2[:, 0].numpy()
    y_patch_1, x_patch_1, patch_idx_1 = get_patch_idx(scale_factor, num_patches, y1, x1)
    y_patch_2, x_patch_2, patch_idx_2 = get_patch_idx(scale_factor, num_patches, y2, x2)
    
    # Calculate loss based on whether correlation map is used
    if not args.DENSE_OBJ:
        desc_patch_1 = desc_1[0, patch_idx_1, :]
        desc_patch_2 = desc_2[0, patch_idx_2, :]
        loss = cal_clip_loss(desc_patch_1, desc_patch_2, get_model_attribute(aggre_net, 'logit_scale').exp(), self_logit_scale=get_model_attribute(aggre_net, 'self_logit_scale').exp())
    else:
        gt_flow = torch.stack([torch.tensor(x_patch_2) - torch.tensor(x_patch_1), torch.tensor(y_patch_2) - torch.tensor(y_patch_1)], dim=-1).to(device)
        if args.GAUSSIAN_AUGMENT > 0:
            std = args.GAUSSIAN_AUGMENT * threshold / 2
            noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
            gt_flow += noise
        loss = get_corr_map_loss(desc_1, desc_2, corr_map_net, patch_idx_1, gt_flow, num_patches, img2_patch_idx=patch_idx_2)
    
    return loss

def calculate_loss(args, aggre_net, img1_kps, img2_kps, img1_desc, img2_desc, img1_threshold, img2_threshold, mask1, mask2, num_patches, device, raw_permute_list=None, img1_desc_flip=None, img2_desc_flip=None, corr_map_net=None, category_prototype=None):

    def get_patch_idx(args, scale_factor, num_patches, img1_y, img1_x):
        scaled_img1_y = scale_factor * img1_y
        scaled_img1_x = scale_factor * img1_x
        img1_y_patch = scaled_img1_y.astype(np.int32)
        img1_x_patch = scaled_img1_x.astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch
        if args.DENSE_OBJ:
            return scaled_img1_y, scaled_img1_x, img1_patch_idx
        else:
            return img1_y_patch, img1_x_patch, img1_patch_idx

    vis = (img1_kps[:, 2] * img2_kps[:, 2]).bool()
    scale_factor = num_patches / args.ANNO_SIZE
    img1_y, img1_x = img1_kps[vis, 1].numpy(), img1_kps[vis, 0].numpy()
    img1_y_patch, img1_x_patch, img1_patch_idx = get_patch_idx(args, scale_factor, num_patches, img1_y, img1_x)
    img2_y, img2_x = img2_kps[vis, 1].numpy(), img2_kps[vis, 0].numpy()
    img2_y_patch, img2_x_patch, img2_patch_idx = get_patch_idx(args, scale_factor, num_patches, img2_y, img2_x)

    loss = cal_clip_loss(img1_desc[0, img1_patch_idx,:], img2_desc[0, img2_patch_idx,:], get_model_attribute(aggre_net, 'logit_scale').exp(), self_logit_scale=get_model_attribute(aggre_net, 'self_logit_scale').exp())

    if args.DENSE_OBJ > 0: # dense training objective loss
        flow_idx = img1_patch_idx
        flow_idx2 = img2_patch_idx
        gt_flow = torch.stack([torch.tensor(img2_x_patch) - torch.tensor(img1_x_patch), torch.tensor(img2_y_patch) - torch.tensor(img1_y_patch)], dim=-1).to(device)
        if args.GAUSSIAN_AUGMENT>0:
            std = args.GAUSSIAN_AUGMENT * img2_threshold / 2     # 2 sigma within the threshold
            noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
            gt_flow = gt_flow + noise
        EPE_loss = get_corr_map_loss(img1_desc, img2_desc, corr_map_net, flow_idx, gt_flow, num_patches, img2_patch_idx=flow_idx2)
        loss += EPE_loss

    if args.ADAPT_FLIP > 0 or args.AUGMENT_SELF_FLIP > 0 or args.AUGMENT_DOUBLE_FLIP > 0:  # augment with flip
        loss = [loss]
        loss_weight = [1]
        
        permute_list = permute_indices(raw_permute_list)
        img1_kps_flip = flip_keypoints(img1_kps, args.ANNO_SIZE, permute_list)
        img2_kps_flip = flip_keypoints(img2_kps, args.ANNO_SIZE, permute_list)
        img1_kps = img1_kps[:len(permute_list), :]
        img2_kps = img2_kps[:len(permute_list), :]
        
        # Calculate losses for each augmentation type
        if args.ADAPT_FLIP > 0:
            vis_flip = img1_kps_flip[:, 2] * img2_kps[:, 2] > 0  # mutual visibility after flip
            if vis_flip.sum() > 0:
                loss_flip = calculate_patch_indices_and_loss(args, img1_kps_flip[vis_flip], img2_kps[vis_flip], img1_desc_flip, img2_desc, scale_factor, num_patches, aggre_net, img2_threshold, corr_map_net, device)
                loss.append(loss_flip)
                loss_weight.append(args.ADAPT_FLIP)

        if args.AUGMENT_DOUBLE_FLIP > 0:
            vis_double_flip = img1_kps_flip[:, 2] * img2_kps_flip[:, 2] > 0  # mutual visibility after flip
            if vis_double_flip.sum() > 0:
                loss_double_flip = calculate_patch_indices_and_loss(args, img1_kps_flip[vis_double_flip], img2_kps_flip[vis_double_flip], img1_desc_flip, img2_desc_flip, scale_factor, num_patches, aggre_net, img2_threshold, corr_map_net, device)
                loss.append(loss_double_flip)
                loss_weight.append(args.AUGMENT_DOUBLE_FLIP)
        
        if args.AUGMENT_SELF_FLIP > 0:
            vis_self_flip = img1_kps_flip[:, 2] * img1_kps[:, 2] > 0
            if vis_self_flip.sum() > 0:
                loss_self_flip = calculate_patch_indices_and_loss(args, img1_kps_flip[vis_self_flip], img1_kps[vis_self_flip], img1_desc_flip, img1_desc, scale_factor, num_patches, aggre_net, img1_threshold, corr_map_net, device)
                loss.append(loss_self_flip)
                loss_weight.append(args.AUGMENT_SELF_FLIP)
        
        # Aggregate losses
        loss = sum([l * w for l, w in zip(loss, loss_weight)]) / sum(loss_weight)

    if args.SELF_CONTRAST_WEIGHT>0:
        contrast_loss1 = self_contrastive_loss(img1_desc.permute(0,2,1).reshape(-1,args.PROJ_DIM,num_patches,num_patches), mask1.unsqueeze(0)) * args.SELF_CONTRAST_WEIGHT
        contrast_loss2 = self_contrastive_loss(img2_desc.permute(0,2,1).reshape(-1,args.PROJ_DIM,num_patches,num_patches), mask2.unsqueeze(0)) * args.SELF_CONTRAST_WEIGHT
        contrast_loss = (contrast_loss1 + contrast_loss2) / 2 * args.SELF_CONTRAST_WEIGHT
        loss += contrast_loss

    return loss






############################################################
#################### 3D CATEGORY PROTOTYPE #################
############################################################




def get_joint_pca_mat(l, in_dim, n_dim=3, normalize=True, batched=False):
    if batched:
        x = torch.cat([t for t in l], dim=0)
    else:
        x = torch.cat([t.view(-1, in_dim) for t in l], dim=0)
    if normalize:
        x = F.normalize(x, dim=-1)
    U, S, V = torch.pca_lowrank(x, q=n_dim)
    return V

def viz_with_pca(x, mat):
    red_x = (x @ mat)
    red_x = (red_x - red_x.min()) / (red_x.max() - red_x.min())
    return red_x



def smooth_dist(x, y, dim=-1, beta=1.0, **kwargs):

    euclidean_distances = torch.norm(x - y, dim=dim)
    l = F.smooth_l1_loss(euclidean_distances, torch.zeros_like(euclidean_distances), beta=beta, **kwargs)

    return l


def cosine_loss(x, y, thresh=1.0, reduction=True, dim=-1):
    assert thresh > -1.0 and thresh <= 1.0
    cos_dist = thresh - F.cosine_similarity(x, y, dim=dim).clip(max=thresh)
    if reduction:
        return cos_dist.mean()
    return cos_dist

def kps_alignment_loss(src_kps, trg_kps, kps_mask=None, subsample_points=None, eps=1e-5):
    B, N, k_dim = src_kps.shape

    if kps_mask is None:
        kps_mask = torch.ones_like(trg_kps).bool()

    if subsample_points is not None:
        kps_mask = kps_mask.clone()

        for b in range(B):
            per_point_mask = kps_mask[b, :, 0]          # (N,)
            num_valid = int(per_point_mask.sum())

            if num_valid > subsample_points:
                # too many points →  randomly keep exactly n of them
                valid_idx = torch.nonzero(per_point_mask, as_tuple=False).flatten()
                keep_idx  = valid_idx[torch.randperm(len(valid_idx),
                                                     device=valid_idx.device)[:subsample_points]]
                # new per-point mask with exactly n ‘True’
                new_point_mask = torch.zeros(N, dtype=torch.bool,
                                                   device=kps_mask.device)
                new_point_mask[keep_idx] = True
                # broadcast to (N,3)
                kps_mask[b] = new_point_mask[:, None].expand(-1, k_dim)


    if not kps_mask.any():
        return torch.tensor(0., device=src_kps.device, dtype=src_kps.dtype), None

    with torch.no_grad():
        align_RT = kabsch(src_kps, trg_kps, mask=kps_mask, estimate_scale=True, min_scale=1e0)
    aligned_src_kps = apply_affine(align_RT, src_kps)

    trg_scale = (trg_kps*kps_mask).norm(dim=-1, keepdim=True).max(dim=-2, keepdim=True)[0]
    src_scale = (aligned_src_kps*kps_mask).norm(dim=-1, keepdim=True).max(dim=-2, keepdim=True)[0]
    scale = torch.maximum(trg_scale.detach(), src_scale.detach()) + eps

    valid_src_kps = torch.masked_select(aligned_src_kps / scale, kps_mask).view(-1,3)
    valid_trg_kps = torch.masked_select(trg_kps / scale, kps_mask).view(-1,3)

    loss = smooth_dist(valid_src_kps, valid_trg_kps, beta=1e-3, reduction='none')

    return loss.mean(), align_RT


def nn_geom_loss(posed_coords, canonical_coords, n_neighbors=32, eps=1e-5):
    B, n_points, _3 = posed_coords.shape
    canon_nn_dist, canon_nn_dist_inds = torch.cdist(canonical_coords, canonical_coords).topk(k=n_neighbors, largest=False, dim=-1)

    posed_nn = torch.gather(posed_coords.unsqueeze(-3).expand(-1, n_points, -1, -1), dim=-2, index=canon_nn_dist_inds.unsqueeze(-1).expand(-1, -1, -1, 3))
    posed_nn_diff = posed_nn - posed_coords.unsqueeze(-2)
    canon_nn = torch.gather(canonical_coords.unsqueeze(-3).expand(-1, n_points, -1, -1), dim=-2, index=canon_nn_dist_inds.unsqueeze(-1).expand(-1, -1, -1, 3))
    canon_nn_diff = canon_nn - canonical_coords.unsqueeze(-2)

    flat_posed_nn = posed_nn_diff.view(-1, n_neighbors, 3)
    flat_canon_nn = canon_nn_diff.view(-1, n_neighbors, 3)
    with torch.no_grad():
        align_RT = kabsch(flat_canon_nn, flat_posed_nn, estimate_scale=True, min_scale=1e0)
    aligned_flat_canon_nn = apply_affine(align_RT, flat_canon_nn)

    posed_nn_scale = posed_nn_diff.norm(dim=-1).max(dim=-1, keepdim=True)[0].view(B * n_points, 1, 1).expand(-1,n_neighbors,3).detach()
    canon_nn_scale = aligned_flat_canon_nn.norm(dim=-1).max(dim=-1, keepdim=True)[0].view(B * n_points, 1, 1).expand(-1,n_neighbors,3).detach()
    scale = torch.maximum(posed_nn_scale, canon_nn_scale) + eps

    loss = smooth_dist(flat_posed_nn/scale, aligned_flat_canon_nn/scale, beta=1e-3)#, reduction='none').mean(dim=-1).max()
    return loss


def geodesic_geom_loss(source_coords, target_coords, n_neighbors=32, eps=1e-5):
    B, n_points, _3 = source_coords.shape
    target_nn, target_nn_inds = closest_geodesic_neighbors(target_coords, K=n_neighbors)

    source_nn = torch.gather(source_coords.unsqueeze(-3).expand(-1, n_points, -1, -1), dim=-2, index=target_nn_inds.unsqueeze(-1).expand(-1, -1, -1, 3))
    source_nn_diff = source_nn - source_coords.unsqueeze(-2)
    target_nn = torch.gather(target_coords.unsqueeze(-3).expand(-1, n_points, -1, -1), dim=-2, index=target_nn_inds.unsqueeze(-1).expand(-1, -1, -1, 3))
    target_nn_diff = target_nn - target_coords.unsqueeze(-2)

    flat_source_nn = source_nn_diff.view(-1, n_neighbors, 3)
    flat_target_nn = target_nn_diff.view(-1, n_neighbors, 3)
    with torch.no_grad():
        align_RT = kabsch(flat_target_nn, flat_source_nn, estimate_scale=True, min_scale=1e0)
    aligned_flat_target_nn = apply_affine(align_RT, flat_target_nn)

    source_nn_scale = source_nn_diff.norm(dim=-1).max(dim=-1, keepdim=True)[0].view(B * n_points, 1, 1).expand(-1,n_neighbors,3).detach()
    target_nn_scale = aligned_flat_target_nn.norm(dim=-1).max(dim=-1, keepdim=True)[0].view(B * n_points, 1, 1).expand(-1,n_neighbors,3).detach()
    scale = torch.maximum(source_nn_scale, target_nn_scale) + eps

    loss = smooth_dist(flat_source_nn/scale, aligned_flat_target_nn/scale, beta=1e-3)
    return loss


def gather_by_indices(feats, kps, valid_mask=None):
    B, n_kps, _ = kps.shape
    _, feat_dim, H, W = feats.shape

    # Create masks for valid indices
    if valid_mask is None:
        valid_mask = (kps[..., 0] >= 0) & (kps[..., 0] < H) & (kps[..., 1] >= 0) & (kps[..., 1] < W)

    # Normalize keypoint positions to the range [-1, 1]
    kps_normalized = torch.empty_like(kps.float())
    kps_normalized[:, :, 0] = 2 * kps[:, :, 0] / (H - 1.0) - 1  # Normalize y coordinates
    kps_normalized[:, :, 1] = 2 * kps[:, :, 1] / (W - 1.0) - 1  # Normalize x coordinates

    # Ensure the keypoints are within the bounds after floating-point inaccuracies
    kps_normalized = kps_normalized.clamp(min=-1, max=1)

    # Reshape for grid_sample (needs BCHW, where C=2)
    kps_grid = kps_normalized.view(B, 1, n_kps, 2)

    # Use grid_sample for bilinear interpolation
    # align_corners=False treats the coordinates in the corner centers
    gathered_feats = F.grid_sample(feats, kps_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Reshape back to the original dimensions, but with the feature dimension last
    gathered_feats = gathered_feats.view(B, feat_dim, n_kps).permute(0, 2, 1)

    # Apply the mask to replace invalid entries with default_value
    gathered_feats[~valid_mask.view(B, n_kps, 1).expand(-1, -1, feat_dim)] = 0

    return gathered_feats

def pad_to_square(x):
    H, W = x.shape[-2:]

    # Calculate padding to make the image square and center the original data
    if H > W:
        diff = H - W
        padding = (diff // 2, diff - diff // 2, 0, 0)  # (left, right, top, bottom)
    else:
        diff = W - H
        padding = (0, 0, diff // 2, diff - diff // 2)  # (left, right, top, bottom)

    # Pad the input tensor to make it square with centered data
    padded_x = F.pad(x, padding, mode='constant', value=0)

    return padded_x, max(H,W)

def calculate_3d_loss(args, aggre_net, category_prototype, full_img1_kps, full_img2_kps, img1_desc, img2_desc, img1_points, img2_points, img1_depths, img2_depths, segmask_1, segmask_2,
                    mask1, mask2, cat, num_patches, device, raw_permute_list=None, img1_desc_flip=None, img2_desc_flip=None, save=False, n_neighbors=64
                    ):

    proto_kps_points, proto_kps_feats, proto_points, proto_feats = category_prototype.get_cat_proto(cat)
    n_proto_points = proto_points.size(-2)
    n_proto_kps = proto_kps_points.size(-2)
    proto_kps_feats = F.normalize(proto_kps_feats, dim=-1)

    #3D Prototype losses
    img1_kps_mask = (full_img1_kps[...,-1:]).bool().reshape(1,-1).to(device)
    img1_points, padded_size = pad_to_square(img1_points)
    img1_depths, _ = pad_to_square(img1_depths)
    segmask_1, _ = pad_to_square(segmask_1)
    img1_kps = full_img1_kps[...,:-1].unsqueeze(0).to(device)
    scale_factor = padded_size / args.ANNO_SIZE
    rescaled_kps = img1_kps * scale_factor
    kps1_3d = gather_by_indices(img1_points, rescaled_kps.long(), valid_mask=img1_kps_mask)

    visible_kps1_3d = kps1_3d[img1_kps_mask].reshape(1,-1,3)
    visible_proto_kps_points = proto_kps_points.reshape(1,-1,3)[img1_kps_mask].reshape(1,-1,3)
    kps_loc_loss, align_RT = kps_alignment_loss(visible_kps1_3d, visible_proto_kps_points)
    aligned_visible_kps1_3d = apply_affine(align_RT, visible_kps1_3d)



    ## Clip_loss on kps feats
    cat_mask = category_prototype.get_cat_mask(cat).view(1,-1).to(proto_kps_points.device).bool()
    feat_dim = category_prototype.feat_dim
    scale_factor = num_patches / args.ANNO_SIZE
    rescaled_kps = img1_kps * scale_factor
    kps1_feats = gather_by_indices(img1_desc.permute(0,2,1).reshape(1,-1,num_patches,num_patches), rescaled_kps.long(), valid_mask=img1_kps_mask)

    visible_kps1_feats = kps1_feats[img1_kps_mask]

    valid_proto_kps_feats = proto_kps_feats.reshape(1,-1, feat_dim)[cat_mask].view(1,-1,feat_dim)
    im_kps_to_proto_kps_sim = F.normalize(visible_kps1_feats, dim=-1) @ F.normalize(valid_proto_kps_feats.unsqueeze(0), dim=-1).transpose(-2,-1)
    kps_feats_logits = im_kps_to_proto_kps_sim.mul(get_model_attribute(category_prototype, 'geom_temp').exp())[0,0]
    labels = torch.nonzero(img1_kps_mask[0])
    kps_feat_loss = F.cross_entropy(kps_feats_logits, labels[:,0])


    

    # 3D Structure Loss
    flat_points = img1_points.permute(0,2,3,1).reshape(1,-1,3)
    flat_depths = img1_depths.permute(0,2,3,1).reshape(1,-1,1)
    flat_mask = segmask_1.permute(0,2,3,1).reshape(1,-1,1)
    flat_feats = F.interpolate(img1_desc.permute(0,2,1).reshape(1,feat_dim, num_patches, num_patches), size=padded_size, mode='bilinear', align_corners=False).permute(0,2,3,1).reshape(1,-1,feat_dim)
    sampled_points, sampled_object_feats = category_prototype.sample_object_points(flat_points, flat_depths, flat_mask, flat_feats)

    valid_proto_kps_points = proto_kps_points.reshape(1,-1,3)[cat_mask].view(1,-1,3)
    feats_to_kps_sim = F.normalize(sampled_object_feats, dim=-1) @ F.normalize(valid_proto_kps_feats.unsqueeze(0), dim=-1).transpose(-2,-1)
    feats_weight = feats_to_kps_sim.mul(get_model_attribute(category_prototype, 'geom_temp').exp()).softmax(dim=-1)[0]
    sampled_canonical_coords = feats_weight @ valid_proto_kps_points.detach()

    object_to_proto_geom_loss = geodesic_geom_loss(sampled_canonical_coords, sampled_points, n_neighbors=n_neighbors)
    proto_to_object_geom_loss = nn_geom_loss(sampled_points, sampled_canonical_coords, n_neighbors=n_neighbors)

    kps_feats_weight = im_kps_to_proto_kps_sim.mul(get_model_attribute(category_prototype, 'geom_temp').exp()).softmax(dim=-1)[0]
    kps_canonical_coords = kps_feats_weight @ valid_proto_kps_points.detach()


    if save:
        os.makedirs(os.path.dirname(f'logs/{args.NOTE}/{cat}/'), exist_ok=True)
        viz_mat = get_joint_pca_mat([F.normalize(proto_feats, dim=-1)], in_dim=feat_dim, batched=True)
        full_im_kps = torch.cat([apply_affine(align_RT, kps1_3d), img1_kps_mask.float().unsqueeze(-1)], dim=-1)
        np.save(f'logs/{args.NOTE}/{cat}/sampled_feats.npy',  viz_with_pca(F.normalize(sampled_object_feats, dim=-1), viz_mat).numpy(force=True))
        np.save(f'logs/{args.NOTE}/{cat}/canon_coords.npy', sampled_canonical_coords.numpy(force=True))
        np.save(f'logs/{args.NOTE}/{cat}/posed_coords.npy', apply_affine(align_RT, sampled_points).numpy(force=True))
        np.save(f'logs/{args.NOTE}/{cat}/object_kps_coords.npy', full_im_kps.numpy(force=True))
        np.save(f'logs/{args.NOTE}/{cat}/canon_kps_coords.npy', kps_canonical_coords.numpy(force=True))
        np.save(f'logs/{args.NOTE}/{cat}/proto.npy', proto_points.numpy(force=True))
        np.save(f'logs/{args.NOTE}/{cat}/proto_kps.npy', proto_kps_points.numpy(force=True))
        np.save(f'logs/{args.NOTE}/{cat}/proto_feats.npy', viz_with_pca(F.normalize(proto_feats, dim=-1), viz_mat).numpy(force=True))
        
    category_prototype.update_protos_from_pts([cat], sampled_canonical_coords, sampled_object_feats)

    kps_loc_weight = 1
    kps_feat_weight = .3
    coarse_geom_weight = 0
    f_geom_weight = 1
    b_geom_weight = 1
    dense_proto_weight = 0
    canonical_kps_loc_weight = 0

    total_loss = kps_loc_weight * kps_loc_loss +\
                 kps_feat_weight * kps_feat_loss +\
                 f_geom_weight * object_to_proto_geom_loss +\
                 b_geom_weight * proto_to_object_geom_loss

    return total_loss, kps_loc_loss.item(), kps_feat_loss.item(), (f_geom_weight * object_to_proto_geom_loss + b_geom_weight * proto_to_object_geom_loss).item()


