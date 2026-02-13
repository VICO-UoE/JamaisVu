from PIL import Image
import math
import numpy as np
import torch

SPAIR_GEO_AWARE = {
    'aeroplane': [0, 1, 2, 3, [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], 22, 23, 24],
    # 45 landing_gear 67 engine_front 89 wing_end 1011 engine_back 1213 wing_foot_front 1415 wing_foot_back 1617 tailplane_end 1819 tailplane_foot_front 2021 tailplane_foot_back
    'bicycle': [0, 1, [2, 3], 4, 5, [6, 7], 8, [9, 10], 11+2], #01 front and back wheel
    # 23 handle 67 seat_back_end 910 pedal
    'bird': [0, [1, 2], 3, 4, 5, 6, 7, 8, 9, [10, 11], [12, 13], [14, 15], 16], #45 mouth #78 eyes
    # 12 wing_end 1011 foot 1213 knee 1415 hip
    'boat': [0, [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], 13],
    # 12 upper_front 34 upper_side 56 upper_back # 78 lower_front 910 lower_side 1112 lower_back
    'bottle': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    # 01 cap 23 neck 45 shoulder 67 body 89 base
    'bus': [[0, 1], [2, 3, 5, 6], [4, 7], [8+2, 11+2, 18+2, 21+2], [9+2, 12+2, 19+2, 22+2], [10+2, 13+2, 20+2, 23+2], [14+2, 15+2, 24+2, 25+2], [16+2, 17+2, 26+2, 27+2]],
    # 01 rearview_mirror 2356 light 47 licence_plate 10132023 front_fender 11142124 wheel 12152225 rear_fender 16172627 window_top_corner 18192829 window_bottom_corner
    'car': [[0, 1], [2, 3, 6, 7], [4, 8], [5, 9], [10, 13, 20, 23], [11, 14, 21, 24], [12, 15, 22, 25], [16, 17, 26, 27], [18, 19, 28, 29]],
    # 01 rearview_mirror 2367 light 48 licence_plate 59 brand_logo 10132023 rear_fender 11142124 wheel 12152225 front_fender 16172627 window_bottom_corner 18192829 window_top_corner
    'cat': [0, 1, [2, 3], 4, 5, 6, 7, 8, [9, 10, 11, 12], 13, 14], #01 below ear 45 eyes 67 nose
    # 23 ear 9101112 paw
    'chair': [[0, 1], [2, 3], [4, 5, 6, 7], [8, 9], [10, 11], [12, 13]],
    # 01 cushion_front 23 cushion_back 4567 legs 89 backrest_top 1011 armrest_front 1213 armrest_back
    'cow': [0, 1, [2, 3], 4, 5, 6, 7, 8, [9, 10, 11, 12], 13, 14, [15, 16, 17, 18], [19, 20]], # 01 below ear 45 eyes 67 nose
    # 23 ear 9101112 hoof 15161718 knee 1920 horn
    'dog': [0, 1, [2, 3], 4, 5, 6, 7, 8, [9, 10, 11, 12], 13, 14, 15], #01 below ear 45 eyes
    # 23 ear 9101112 paw
    'horse': [0, 1, [2, 3], 4, 5, 6, 7, 8, 9, [10, 11, 12, 13], 14, 15, [16, 17, 18, 19]], # 01 below ear 45 eyes 67 nose
    # 23 ear 10111213 hoof 16171819 knee
    'motorbike': [[0, 1], [2, 3], 4, 5, 6, 7, 8, 9, 10, 11, 12], #67 front back wheel
    # 01 rearview_mirror 23 handle
    'person': [0, 1, 2, 3, 4, 5, 6, 7, [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], # 01 eyes 23 ears
    # 89 shoulder 1011 elbow 1213 wrist 1415 knee 1617 ankle 1819 foot
    'pottedplant': [[0, 1, 2, 3], [4, 5], [6, 8], 7],
    # 0123 top 45 side_wall 68 bottom
    'sheep': [0, 1, [2, 3], 4, 5, 6, 7, 8, [9, 10, 11, 12], 13, 14, [15, 16, 17, 18], [19, 20]], # 01 below ear 45 eyes 67 nose
    # 23 ear 9101112 hoof 15161718 knee 1920 horn
    'train': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17]],
    # 01 front_top 23 front_bottom 45 back_top 67 back_bottom 89 window_corner_top_outer 1011 window_corner_bottom_outer 1213 window_corner_top_inner 1415 window_corner_bottom_inner 1617 front light
    'tvmonitor': [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]
    # 0246 outer_corner # 1357 outer_side # 8101214 inner_corner # 9111315 inner_side
}

AP10K_GEO_AWARE = [0,1, # eye
                    2, # nose
                    3, # neck
                    4, # root of tail
                    [5,8], # shoulder
                    [6,9,12,15], # elbow # knee
                    [7,10,13,16], # front paw # back paw
                    [11,14], # hip
                    ]

SPAIR_FLIP = {
    'aeroplane': [0,1,2,3,[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21],22,23,24],
    # 'bicycle': [0,1,[2,3],4,5,[6,7],8,[9,10],11+2], #11,12 are dummy
    'bicycle': [0,1,[2,3],4,5,[6,7],8,[9,10],11], 
    'bird': [0,[1,2],3,[4,5],6,[7,8],9,[10,11],[12,13],[14,15],16],
    'boat': [0,[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],13],
    'bottle': [[0,1],[2,3],[4,5],[6,7],[8,9]],
    # 'bus': [[0,1],[2,3],[5,6],4,7,[8+2,18+2],[11+2,21+2],[9+2,19+2],[12+2,22+2],[10+2,20+2],[13+2,23+2],[14+2,15+2],[24+2,25+2],[16+2,17+2],[26+2,27+2]],  #8,9 are dummy
    'bus': [[0,1],[2,3],[5,6],4,7,[8,18],[11,21],[9,19],[12,22],[10,20],[13,23],[14,15],[24,25],[16,17],[26,27]], 
    'car':[[0,1],[2,3],4,5,[6,7],8,9,[10,20],[13,23],[11,21],[14,24],[12,22],[15,25],[16,17],[26,27],[18,19],[28,29]],
    'cat':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14],
    'chair':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]],
    'cow':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'dog':[[0,1],[2,3],[4,5],6,7,8,[9,10],[11,12],13,14,15],
    'horse':[[0,1],[2,3],[4,5],[6,7],8,9,[10,11],[12,13],14,15,[16,17],[18,19]],
    'motorbike':[[0,1],[2,3],4,5,6,7,8,9,10,11,12],
    'person':[[0,1],[2,3],4,5,6,7,[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]],
    'pottedplant':[[0,2],1,3,[4,5],[6,8],7],
    'sheep':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'train':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]],
    'tvmonitor':[[0,2],[4,6],1,5,[3,7],[8,10],[12,14],9,13,[11,15]]
}

SPAIR_FLIP_TRN = SPAIR_FLIP
SPAIR_FLIP_TRN = {
    'aeroplane': [0,1,2,3,[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21],22,23,24],
    'bicycle': [0,1,[2,3],4,5,[6,7],8,[9,10],11+2], #11,12 are dummy
    # 'bicycle': [0,1,[2,3],4,5,[6,7],8,[9,10],11], 
    'bird': [0,[1,2],3,[4,5],6,[7,8],9,[10,11],[12,13],[14,15],16],
    'boat': [0,[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],13],
    'bottle': [[0,1],[2,3],[4,5],[6,7],[8,9]],
    'bus': [[0,1],[2,3],[5,6],4,7,[8+2,18+2],[11+2,21+2],[9+2,19+2],[12+2,22+2],[10+2,20+2],[13+2,23+2],[14+2,15+2],[24+2,25+2],[16+2,17+2],[26+2,27+2]],  #8,9 are dummy
    # 'bus': [[0,1],[2,3],[5,6],4,7,[8,18],[11,21],[9,19],[12,22],[10,20],[13,23],[14,15],[24,25],[16,17],[26,27]], 
    'car':[[0,1],[2,3],4,5,[6,7],8,9,[10,20],[13,23],[11,21],[14,24],[12,22],[15,25],[16,17],[26,27],[18,19],[28,29]],
    'cat':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14],
    'chair':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]],
    'cow':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'dog':[[0,1],[2,3],[4,5],6,7,8,[9,10],[11,12],13,14,15],
    'horse':[[0,1],[2,3],[4,5],[6,7],8,9,[10,11],[12,13],14,15,[16,17],[18,19]],
    'motorbike':[[0,1],[2,3],4,5,6,7,8,9,10,11,12],
    'person':[[0,1],[2,3],4,5,6,7,[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]],
    'pottedplant':[[0,2],1,3,[4,5],[6,8],7],
    'sheep':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'train':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]],
    'tvmonitor':[[0,2],[4,6],1,5,[3,7],[8,10],[12,14],9,13,[11,15]]
}

AP10K_FLIP = [
                [0,1], # eye
                2, # nose
                3, # neck
                4, # root of tail
                [5,8], # shoulder
                [6,9], # elbow # knee
                [12, 15], # knee
                [7,10], # front paw 
                [13,16], # back paw
                [11,14], # hip
                              ]

SPAIR_LR = { # 0 neutral, 1 left, 2 right
        'aeroplane': [0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 0],
        'bicycle': [0, 0, 2, 1, 0, 0, 2, 1, 0, 2, 1, 0, 0, 0],
        'bird': [0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2, 1, 2, 1, 0],
        'boat': [0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0],
        'bottle': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'bus': [2, 1, 2, 1, 0, 2, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1],
        'car': [2, 1, 2, 1, 0, 0, 2, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1],
        'cat': [2, 1, 2, 1, 2, 1, 2, 1, 0, 2, 1, 2, 1, 0, 0],
        'chair': [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        'cow': [2, 1, 2, 1, 2, 1, 2, 1, 0, 2, 1, 2, 1, 0, 0, 2, 1, 2, 1, 2, 1],
        'dog': [2, 1, 2, 1, 2, 1, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0],
        'horse': [2, 1, 2, 1, 2, 1, 2, 1, 0, 0, 2, 1, 2, 1, 0, 0, 2, 1, 2, 1],
        'motorbike': [2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'person': [2, 1, 2, 1, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        'pottedplant': [2, 0, 1, 0, 2, 1, 2, 0, 1],
        'sheep': [2, 1, 2, 1, 2, 1, 2, 1, 0, 2, 1, 2, 1, 0, 0, 2, 1, 2, 1, 2, 1],
        'train': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'tvmonitor': [2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 1]
}


def renumber_indices(lst, counter=[0]):
    """
    input: a nested list
    output: a nested list with all the numbers replaced by the next number in the counter
    """
    new_lst = []
    for item in lst:
        if isinstance(item, list):
            new_lst.append(renumber_indices(item, counter))
        else:
            new_lst.append(counter[0])
            counter[0] += 1
    return new_lst

def permute_indices(flip_list, vis=None):
    """
    Permutes indices based on the rules provided in the flip_list.
    
    Args:
    - flip_list (list): List containing integers or lists of integers.
    - vis (list, optional): A list of boolean values used to determine whether a particular permutation should be applied.
    
    Returns:
    - List[int]: Permuted indices list.

    Sample Input 1:
    flip_list = [1, 2, [0, 3]]
    vis = [True, True, True, False]
    Output:
    [0, 1, 2, 3]

    Sample Input 1:
    flip_list = [1, 2, [0, 3]]
    vis = None
    Output:
    [3, 1, 2, 0]
    """
    # Flatten the list to find the max index
    flat_list = [item for sublist in flip_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    max_idx = max(flat_list)

    # Create a list of indices from 0 to max index
    indices = list(range(max_idx + 1))

    # Permute the indices where necessary
    for item in flip_list:
        if isinstance(item, list):
            # if all elem in item is True in vis, then flip
            if vis is None or all(vis[i] for i in item):
                for i in range(len(item)):
                    indices[item[i]] = item[(i + 1) % len(item)]

    return indices

def flip_image_keypoints(image, keypoints, img_size, permute_list=None):
    img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    keypoints_flip = keypoints.detach().clone()
    keypoints_flip[:, 0] = img_size - keypoints_flip[:, 0]
    if permute_list is not None:
        keypoints_flip = keypoints_flip[permute_list]
    return img_flip, keypoints_flip

def flip_keypoints(keypoints, img_size, permute_list=None):
    keypoints_flip = keypoints.detach().clone()
    keypoints_flip[:, 0] = img_size - keypoints_flip[:, 0]
    if permute_list is not None:
        keypoints_flip = keypoints_flip[permute_list]
    return keypoints_flip

def edge_pad_rotate_and_crop(img: Image.Image, angle: float, output_size = None) -> Image.Image:
    """
    Applies edge padding, rotation, and cropping to an input image.

    Args:
        img (PIL.Image.Image): The input image to be transformed.
        angle (float): The angle (in degrees) by which to rotate the image.
        output_size (int, optional): The size of the output image. Defaults to None.

    Returns:
        PIL.Image.Image: The transformed image.
    """
    # Convert the input image to a NumPy array
    img_np = np.array(img)
    height, width = img_np.shape[:2]
    max_edge = max(height, width)

    # Calculate the length of the diagonal
    diagonal = int(math.ceil(math.sqrt(max_edge ** 2 + max_edge ** 2)))

    # Calculate the amount of padding needed
    left_pad = (diagonal - width) // 2
    right_pad = diagonal - width - left_pad
    top_pad = (diagonal - height) // 2
    bottom_pad = diagonal - height - top_pad

    # Apply edge padding using NumPy
    img_padded_np = np.pad(
        img_np,
        ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
        'edge'
    )

    # Convert the padded NumPy array back to a PIL image
    img_padded = Image.fromarray(np.uint8(img_padded_np))

    # Rotate the image
    img_rotated = img_padded.rotate(angle)

    # Crop the rotated image to the original size
    center_x, center_y = diagonal // 2, diagonal // 2
    img_cropped = img_rotated.crop(
        (
            center_x - max_edge // 2,
            center_y - max_edge // 2,
            center_x + (max_edge + 1) // 2,
            center_y + (max_edge + 1) // 2
        )
    )

    # Resize the cropped image if output_size is specified
    if output_size is not None:
        img_cropped = img_cropped.resize((output_size, output_size), Image.BILINEAR)

    # Return the transformed image
    return img_cropped

def renumber_used_points(kpts, idx):
    N, C = kpts.shape
    out = torch.zeros(30, C)
    out[idx] = kpts
    return out

def optimized_kps_1_to_2(args, kps_1_to_2, kps_1_to_2_flip, img1_kps, img2_kps, flip_dist, original_dist, vis, permute_list):
    img1_kps_mutual_visible = img1_kps * vis.unsqueeze(-1).float()
    img1_kps_flip_visible = flip_keypoints(img1_kps_mutual_visible, args.ANNO_SIZE, permute_indices(permute_list, None))
    vis_flip = (img1_kps_flip_visible[:, 2] * img2_kps[:, 2] * img1_kps[:, 2] > 0).cuda() # mutual visible after flip
    
    if flip_dist < original_dist:
        kps_1_to_2[vis_flip] = kps_1_to_2_flip[vis_flip]
    else:
        kps_1_to_2 = kps_1_to_2

    return kps_1_to_2
















############################################################
#################### 3D utils functions ####################
############################################################



def weighted_mean(A, M, **kwargs):

    # Ensure A and M can be broadcasted to the same shape
    if torch.broadcast_shapes(A.shape, M.shape) != A.shape:
        raise ValueError("Shapes of A and M are not broadcastable to a common shape")

    # Calculate the weighted sum and the sum of weights using kwargs
    weighted_sum = torch.sum(A * M, **kwargs)
    total_weight = torch.sum(M, **kwargs)

    # Check if the total weight is zero
    if torch.all(total_weight == 0):
        return torch.mean(A, **kwargs)

    # Compute and return the weighted mean
    return weighted_sum / total_weight + 1e-6

def apply_affine(RT, pts):
    homogenous = torch.cat((pts, torch.ones_like(pts[...,0:1])), dim=-1)
    res = homogenous @ RT
    return res[...,:3]


def kabsch(src, trg, mask=None, estimate_scale=False, min_scale=None, max_scale=None):
    assert src.shape == trg.shape
    assert len(src.shape) == 2 or len(src.shape) == 3
    batched = len(src.shape) == 3

    if mask is None:
        mean_src = src.mean(dim=-2, keepdim=True)
        mean_trg = trg.mean(dim=-2, keepdim=True)
        src_c = src - mean_src
        trg_c = trg - mean_trg
    else:
        mean_src = weighted_mean(src, mask, dim=-2, keepdim=True)
        mean_trg = weighted_mean(trg, mask, dim=-2, keepdim=True)
        src_c = src - mean_src
        trg_c = trg - mean_trg
        src_c = src_c * mask
        trg_c = trg_c * mask

    cov = src_c.transpose(-2,-1) @ trg_c
    U, S, Vt = torch.linalg.svd(cov)

    d = torch.linalg.det(U@Vt)
    corrector = torch.eye(3, device=src.device)
    if batched:
        B = src.size(0)
        corrector = corrector.repeat(B,1,1)
    corrector[...,-1,-1] = d
    R = U @ corrector @ Vt

    s = 1
    if estimate_scale:
        trace = (torch.diagonal(corrector, dim1=-2, dim2=-1) * S).sum(-1)
        src_cov = (src_c * src_c).sum((-1, -2))

        # the scaling component
        s = trace / torch.clamp(src_cov, 1e-5)

        s = s.view(B,1,1)

        if min_scale is not None:
            s = s.clip(min=min_scale)

        if max_scale is not None:
            s = s.clip(max=max_scale)

    T = (mean_trg - s * mean_src @ R) 

    RT = torch.cat((s*R,T), dim=-2)
    RT = torch.cat([RT,torch.zeros_like(RT[...,0:1])], dim=-1)

    RT[...,3,3] = 1


    nans = RT.isnan().any(dim=-2, keepdim=True).any(dim=-1, keepdim=True)
    if nans.any():
        if batched:
            expanded = nans.repeat(1,4,4)
            ident = torch.eye(4, device=RT.device).view(1,4,4).repeat(B,1,1)
            RT = torch.where(expanded, ident, RT)
        else:
            RT = torch.eye(4, device=RT.device)

    return RT

    
def closest_geodesic_neighbors(pts, K=256):
    B, N, _ = pts.shape
    # Initialize lists to store the indices and coordinates of the K closest geodesic neighbors
    selected_inds = [torch.arange(N, device=pts.device).view(1, N, 1).expand(B, N, 1)]  # Initial indices
    selected_pts = [pts.unsqueeze(2)]  # Initial points, shape: (B, N, 1, 3)

    # Calculate pairwise distances between all points using torch.cdist
    dists = torch.cdist(pts, pts)  # Shape: (B, N, N)

    # Initialize a mask to track selected points
    selected_mask = torch.eye(N, device=pts.device).unsqueeze(0).expand(B, -1, -1).bool()

    for _ in range(1, K):
        # Apply the mask to the distances to set the distance of selected points to infinity
        masked_dists = dists.where(~selected_mask, float('inf'))

        # Find the index of the closest point to the selected set of points for each point
        closest_point_inds = masked_dists.argmin(dim=2, keepdim=True)

        # Update the mask to include the newly selected points
        selected_mask.scatter_(-1, closest_point_inds, True)

        # Gather the closest points
        closest_point = torch.gather(pts, 1, closest_point_inds.expand(-1, -1, 3))

        # Update distances to reflect the minimum distance to any of the selected points
        new_dists = torch.cdist(pts, closest_point)
        dists = torch.min(dists, new_dists)

        # Append the new closest point to the selected set
        selected_inds.append(closest_point_inds)
        selected_pts.append(closest_point.unsqueeze(2))

    # Concatenate the selected indices and points
    closest_neighbors_indices = torch.cat(selected_inds, dim=2)
    closest_neighbors_pts = torch.cat(selected_pts, dim=2)

    return closest_neighbors_pts, closest_neighbors_indices