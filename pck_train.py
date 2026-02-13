import os
import torch
import pickle
import wandb
import argparse
from PIL import Image
from tqdm import tqdm
from loguru import logger
from itertools import chain
torch.set_num_threads(16)
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from preprocess_map import set_seed
from model_utils.projection_network import AggregationNetwork, DummyAggregationNetwork
from model_utils.category_prototype import CategoryPrototype
from model_utils.corr_map_model import Correlation2Displacement
import utils.utils_losses as utils_losses
import utils.utils_visualization as utils_visualization
from utils.logger import get_logger, log_geo_stats, update_stats, update_geo_stats, log_weighted_pcks, load_config
from utils.utils_geoware import AP10K_GEO_AWARE, AP10K_FLIP, SPAIR_GEO_AWARE, SPAIR_FLIP, SPAIR_FLIP_TRN, permute_indices, renumber_indices, flip_keypoints, renumber_used_points, optimized_kps_1_to_2
from utils.utils_correspondence import kpts_to_patch_idx, load_img_and_kps, convert_to_binary_mask, calculate_keypoint_transformation, get_distance, get_distance_mutual_nn, extract_kap_vals
from utils.utils_dataset import load_eval_data, load_and_prepare_data, get_dataset_info
from utils.utils_io import read_exr, get_pascal_mask, get_owl_mask

import timm

from sklearn.metrics import average_precision_score

device  = 'cuda' if torch.cuda.is_available() else 'cpu'

def normalize_feats(args, feats, epsilon=1e-10):
    if args.DUMMY_NET: # seperate norm
        feat_sd = feats[..., :640+1280+1280] #sd feature
        feat_dino = feats[..., 640+1280+1280:] #dino feature
        norms_sd = torch.linalg.norm(feat_sd, dim=-1)[:, :, None]
        norm_feats_sd = feat_sd / (norms_sd + epsilon)
        norms_dino = torch.linalg.norm(feat_dino, dim=-1)[:, :, None]
        norm_feats_dino = feat_dino / (norms_dino + epsilon)
        if args.ONLY_DINO:
            feats = norm_feats_dino
        elif args.ONLY_SD:
            feats = norm_feats_sd
        else:
            feats = torch.cat([norm_feats_sd, norm_feats_dino], dim=-1)
    # (b, w*h, c)
    norms = torch.linalg.norm(feats, dim=-1)[:, :, None]
    norm_feats = feats / (norms + epsilon)
    # norm_feats = feats / norms

    return norm_feats

def prepare_feature_paths_and_load(aggre_net, img_path, flip, ensemble, num_patches, device):
    # Construct feature paths
    feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
    suffix_flip = '_flip' if flip else ''
    ensemble_folder = f'features_ensemble{ensemble}' if ensemble > 1 else 'features'
    mask_path = f"{feature_base}_mask{suffix_flip}.png"
    sd_path = f"{feature_base}_sd{suffix_flip}.pt".replace('features', ensemble_folder)
    dino_path = f"{feature_base}_dino{suffix_flip}.pt".replace('features', ensemble_folder)
    # Load SD and DINO features
    features_sd = torch.load(sd_path, weights_only=True, map_location=device)
    desc_dino = torch.load(dino_path, weights_only=True, map_location=device)
    # Prepare descriptors
    desc_gathered = torch.cat([
        features_sd['s3'],
        F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
        F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
        desc_dino
    ], dim=1)
    desc = aggre_net(desc_gathered).reshape(1, 1, -1, num_patches**2).permute(0, 1, 3, 2)
    # Load mask if it exists
    mask = None
    if os.path.exists(mask_path):
        mask = convert_to_binary_mask(mask_path)

    return desc, mask

def load_points_and_seg(files, img_idx, device, use_sam_masks=True):
    img_path = files[img_idx]

    # Construct feature paths
    pointmap_path = img_path.replace('JPEGImages', 'MoGeMaps').replace('.jpg', '/points.exr')
    points, _ = read_exr(pointmap_path)
    points = torch.tensor(points)
    points = points.clip(min=-1e3, max=1e3)
    points = points.view(1,points.size(0),points.size(1),3).permute(0,3,1,2).to(device)


    depthmap_path = img_path.replace('JPEGImages', 'MoGeMaps').replace('.jpg', '/depth.exr')
    depths, _ = read_exr(depthmap_path)
    depths = torch.tensor(depths)
    depths = depths.clip(max=1e3)
    depths = depths.view(1,depths.size(0),depths.size(1),1).permute(0,3,1,2).to(device)

    if use_sam_masks:
        mask_path = img_path.replace('JPEGImages', 'owlv2_sam_masks').replace('.jpg', '.npy')
        mask = get_owl_mask(mask_path)
    else:
        mask_path = img_path.replace('JPEGImages', 'Segmentation').replace('.jpg', '.png')
        mask = get_pascal_mask(mask_path)
    mask = mask.to(device)

    return points, depths, mask

def get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=False, flip2=False, img1=None, img2=None, device='cuda', sph_mapper=None):
    img_path_1 = files[pair_idx * 2]
    img_path_2 = files[pair_idx * 2 + 1]
    # save the imgs for cases if the feature doesn't exist
    img1_desc, mask1 = prepare_feature_paths_and_load(aggre_net, img_path_1, flip, args.ENSEMBLE, num_patches, device)
    img2_desc, mask2 = prepare_feature_paths_and_load(aggre_net, img_path_2, flip2, args.ENSEMBLE, num_patches, device)
    #run the sphere mapper if present
    if args.SPH:
        sph_map1 = F.normalize(sph_mapper(img1_desc[0,..., 640+1280+1280:]), dim=-1)
        sph_map2 = F.normalize(sph_mapper(img2_desc[0,..., 640+1280+1280:]), dim=-1)
    # normalize the desc
    img1_desc = normalize_feats(args, img1_desc[0])
    img2_desc = normalize_feats(args, img2_desc[0])
    #concatenate the spherical maps
    if args.SPH:
        img1_desc = torch.cat([img1_desc, sph_map1], dim=-1)
        img2_desc = torch.cat([img2_desc, sph_map2], dim=-1)
    return img1_desc, img2_desc, mask1, mask2

def compute_pck(args, save_path, aggre_net, files, kps, category=None, used_points=None, thresholds=None, verbose=True, sph_mapper=None):
    out_results = []
    num_patches = args.NUM_PATCHES
    current_save_results = 0
    gt_correspondences, pred_correspondences, img_acc_001, img_acc_005, img_acc_01, len_kpts = ([] for _ in range(6))
    pckd_valid_matches = []
    if thresholds is not None:
        thresholds = torch.tensor(thresholds).to(device)
        bbox_size=[]
    N = len(files) // 2
    desc_list = []
    mask_list = []
    img_list = []
    idx_list = []
    pbar = tqdm(total=N, dynamic_ncols=True, disable=not verbose)

    if args.COMPUTE_GEOAWARE_METRICS:   # get the geo-aware idx list
        geo_aware_count = geo_aware_total_count = 0
        geo_idx_all, influ_list_geo_filtered = [], []
        if args.EVAL_DATASET == 'ap10k':
            influ_list_geo = AP10K_GEO_AWARE
        else:
            influ_list_geo = SPAIR_GEO_AWARE[category] if category in SPAIR_GEO_AWARE else None
        for item in influ_list_geo:
            item = [item] if isinstance(item, int) else item
            temp_list = [idx for idx in item if idx in used_points]
            if len(temp_list) >= 1:
                influ_list_geo_filtered.append(temp_list)
        raw_geo_aware = renumber_indices(influ_list_geo_filtered, counter=[0])

    if args.ADAPT_FLIP: # get the permute list for flipping
        FLIP_ANNO = AP10K_FLIP if args.EVAL_DATASET == 'ap10k' else SPAIR_FLIP[category]
        if sum(len(i) if isinstance(i, list) else 1 for i in FLIP_ANNO) == kps[0].shape[0]:
            permute_list = FLIP_ANNO
        else:
            influ_list_filtered = []
            influ_list = FLIP_ANNO
            for item in influ_list:
                item = [item] if isinstance(item, int) else item
                temp_list = [idx for idx in item if idx in used_points]
                if len(temp_list) >= 1:
                    influ_list_filtered.append(temp_list)
            permute_list = renumber_indices(influ_list_filtered, counter=[0])

    for pair_idx in range(N):
        # Load images and keypoints
        img1, img1_kps = load_img_and_kps(idx=2*pair_idx, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        img2, img2_kps = load_img_and_kps(idx=2*pair_idx+1, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        # Get mutual visibility
        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        vis2 = img2_kps[:, 2]
        # Get patch descriptors
        with torch.no_grad():
            img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, img1=img1, img2=img2, sph_mapper=sph_mapper)
        # Get patch index for the keypoints
        img1_patch_idx = kpts_to_patch_idx(args, img1_kps, num_patches)
        # Get similarity matrix
        kps_1_to_2 = calculate_keypoint_transformation(args, img1_desc, img2_desc, img1_patch_idx, num_patches)

        if args.ADAPT_FLIP:
            img1_flip = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img1_desc_flip, _, mask1_flip, _ = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=True, img1=img1.transpose(Image.FLIP_LEFT_RIGHT), img2=img2, sph_mapper=sph_mapper)
            img1_kps_flip = flip_keypoints(img1_kps, args.ANNO_SIZE, permute_indices(permute_list, vis))
            img1_patch_idx_flip = kpts_to_patch_idx(args, img1_kps_flip, num_patches)
            kps_1_to_2_flip = calculate_keypoint_transformation(args, img1_desc_flip, img2_desc, img1_patch_idx_flip, num_patches)

            # get the distance for the flip and original img
            if args.MUTUAL_NN:
                original_dist = get_distance_mutual_nn(img1_desc, img2_desc)
                flip_dist = get_distance_mutual_nn(img1_desc_flip, img2_desc)
            else:
                original_dist = get_distance(img1_desc, img2_desc, mask1, mask2)
                flip_dist = get_distance(img1_desc_flip, img2_desc, mask1_flip, mask2)

            kps_1_to_2 = optimized_kps_1_to_2(args, kps_1_to_2, kps_1_to_2_flip, img1_kps, img2_kps, flip_dist, original_dist, vis, permute_list)
        # collect the result for more complicated eval
        single_result = {
            "src_fn": files[2*pair_idx],  # must
            "trg_fn": files[2*pair_idx+1],  # must
            # "category": category,
            # "used_points": used_points.cpu().numpy(),
            # "src_kpts": renumber_used_points(img1_kps, used_points).cpu().numpy(),
            # "trg_kpts": renumber_used_points(img2_kps, used_points).cpu().numpy(),
            "src_kpts_pred": renumber_used_points(kps_1_to_2.cpu(), used_points).cpu().detach().numpy(),  # must
            # "threshold": thresholds[pair_idx].item() if thresholds is not None else 0,
            "resize_resolution": args.ANNO_SIZE,  # must
        }
        out_results.append(single_result)

        #Compute wether the gt kp is the closest to the pred
        if args.PCKD:
            preds_to_gt_dists = torch.cdist(kps_1_to_2, img2_kps[...,:2].to(device))
            preds_to_gt_dists += preds_to_gt_dists.max() * (1 - vis.to(device).float().reshape(1,-1))
            closest_kp_index = preds_to_gt_dists.argmin(dim=-1)
            is_valid = closest_kp_index == torch.arange(closest_kp_index.size(-1), device=device)
            is_valid = is_valid[vis].cpu()
            pckd_valid_matches.append(is_valid)


        gt_kps = img2_kps[vis][:, [1,0]]
        prd_kps = kps_1_to_2[vis][:, [1,0]]
        gt_correspondences.append(gt_kps)
        pred_correspondences.append(prd_kps)
        len_kpts.append(vis.sum().item())

        # compute per image acc
        if not args.KPT_RESULT: # per img result
            single_gt_correspondences = img2_kps[vis][:, [1,0]]
            single_pred_correspondences = kps_1_to_2[vis][:, [1,0]]
            alpha = torch.tensor([0.1, 0.05, 0.01]) if args.EVAL_DATASET != 'pascal' else torch.tensor([0.1, 0.05, 0.15])
            # correct = torch.zeros(3)
            err = (single_gt_correspondences - single_pred_correspondences.cpu()).norm(dim=-1)
            err = err.unsqueeze(0).repeat(3, 1)
            if thresholds is not None:
                single_bbox_size = thresholds[pair_idx].repeat(vis.sum()).cpu()
                correct = (err < alpha.unsqueeze(-1) * single_bbox_size.unsqueeze(0)).float()
            else:
                correct = (err < alpha.unsqueeze(-1) * args.ANNO_SIZE).float()
            if args.PCKD:
                correct = correct * is_valid.unsqueeze(0).repeat(3, 1)
            correct = correct.mean(-1)
            img_acc_01.append(correct[0].item())
            img_acc_005.append(correct[1].item())
            img_acc_001.append(correct[2].item())

        if thresholds is not None:
            pckthres = thresholds[pair_idx].repeat(vis.sum())
            bbox_size.append(pckthres)

        if args.COMPUTE_GEOAWARE_METRICS:
            geo_aware_list, geo_aware_full_list = ([] for _ in range(2))
            for item in raw_geo_aware:
                # convert to list
                item = [item] if isinstance(item, int) else item
                # check if all items are visible
                temp_list = [idx for idx in item if vis[idx]]
                temp_list2 = [idx for idx in item if vis2[idx]]
                # if more than 2 items are visible, add to geo_aware_list
                if len(temp_list2) >= 2 and len(temp_list) >= 1:
                    for temp_idx in temp_list:
                        geo_aware_list.append([temp_idx])
                    geo_aware_full_list.append(temp_list)

            geo_aware_idx = [item for sublist in geo_aware_list for item in sublist]
            geo_idx_mask = torch.zeros(len(vis)).bool()
            geo_idx_mask[geo_aware_idx] = True
            geo_idx_mask = geo_idx_mask[vis]
            geo_idx_all.append(torch.tensor(geo_idx_mask))

            # count the number of geo-aware pairs
            if len(geo_aware_full_list) > 0:
                geo_aware_total_count += len(geo_aware_idx)     # per keypoint
                geo_aware_count += 1                            # per img

        if current_save_results!=args.TOTAL_SAVE_RESULT:
            if args.ADAPT_FLIP and (flip_dist < original_dist): # save the flip result
                utils_visualization.save_visualization(thresholds, pair_idx, vis, save_path, category,
                       img1_kps_flip, img1_flip, img2, kps_1_to_2, img2_kps, args.ANNO_SIZE, args.ADAPT_FLIP)
            else:
                utils_visualization.save_visualization(thresholds, pair_idx, vis, save_path, category,
                       img1_kps, img1, img2, kps_1_to_2, img2_kps, args.ANNO_SIZE, args.ADAPT_FLIP)
            if current_save_results!=args.TOTAL_SAVE_RESULT-1:
                _, _, desc_mask = load_points_and_seg(files=files, img_idx=2*pair_idx+1, device=device)
                desc_list.append(img2_desc[...,-3:].reshape(60,60, -1).permute(2,0,1).detach())
                img_list.append(img2)
                mask_list.append(desc_mask[0].detach())
                idx_list.append(pair_idx)
            else:
                utils_visualization.save_desc_pca(desc_list=desc_list, save_path=save_path, category=category, pair_idx_list=idx_list, mask_list=mask_list, img_list=img_list)
            current_save_results += 1

        pbar.update(1)
    pbar.close()
    if not args.KPT_RESULT:
        img_correct = torch.tensor([img_acc_01, img_acc_005, img_acc_001])
        img_correct = img_correct.mean(dim=-1).tolist()
        img_correct.append(N)
    else:
        img_correct = None
    gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
    pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
    alpha = torch.tensor([0.1, 0.05, 0.01]) if args.EVAL_DATASET != 'pascal' else torch.tensor([0.1, 0.05, 0.15])
    correct = torch.zeros(len(alpha))
    err = (pred_correspondences - gt_correspondences).norm(dim=-1)
    err = err.unsqueeze(0).repeat(len(alpha), 1)
    if thresholds is not None:
        bbox_size = torch.cat(bbox_size, dim=0).cpu()
        threshold = alpha.unsqueeze(-1) * bbox_size.unsqueeze(0)
        correct_all = err < threshold
    else:
        threshold = alpha * args.ANNO_SIZE
        correct_all = err < threshold.unsqueeze(-1)
    if args.PCKD:
        pckd_valid_matches = torch.cat(pckd_valid_matches, dim=0).cpu()
        correct_all = correct_all * pckd_valid_matches.unsqueeze(0).repeat(len(alpha), 1)
        
    log_func = logger.info if verbose else logger.debug

    correct = correct_all.sum(dim=-1) / len(gt_correspondences)
    correct = correct.tolist()
    correct.append(len(gt_correspondences))
    alpha2pck = zip(alpha.tolist(), correct[:3]) if args.KPT_RESULT else zip(alpha.tolist(), img_correct[:3])
    log_func(f'{category}...'+' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
        for alpha, pck_alpha in alpha2pck]))

    geo_score = []
    if args.COMPUTE_GEOAWARE_METRICS:
        geo_idx_all = torch.cat(geo_idx_all, dim=0).cpu()
        correct_geo = correct_all[:,geo_idx_all].sum(dim=-1) / geo_idx_all.sum().item()
        correct_geo = correct_geo.tolist()
        geo_score.append(geo_aware_count / N)
        geo_score.append(geo_aware_total_count / len(gt_correspondences))
        geo_score.extend(correct_geo)
        geo_score.append(geo_idx_all.sum().item())
        alpha2pck_geo = zip(alpha.tolist(), correct_geo[:3])
        log_func(' | '.join([f'PCK-Transfer_geo-aware@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                        for alpha, pck_alpha in alpha2pck_geo]))
        log_func(f'Geo-aware occurence count: {geo_aware_count}, with ratio {geo_aware_count / N * 100:.2f}%; total count ratio {geo_aware_total_count / len(gt_correspondences) * 100:.2f}%')

    return correct, geo_score, out_results, img_correct


def mAP(positive_preds, negative_preds):
    # Concatenate predictions and create labels
    all_preds = torch.cat((positive_preds, negative_preds))
    labels = torch.cat((torch.ones_like(positive_preds), torch.zeros_like(negative_preds)))

    # Sort predictions in descending order
    sorted_indices = torch.argsort(all_preds, descending=True)
    sorted_labels = labels[sorted_indices]

    # Calculate true positives and false positives
    true_positives = torch.cumsum(sorted_labels, dim=0)
    false_positives = torch.cumsum(1 - sorted_labels, dim=0)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / true_positives[-1]

    # Calculate Average Precision (AP) using the rectangle method matching sklearn implementation
    # Sum precision values where recall changes
    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] != recall[i - 1]:
            ap += precision[i] * (recall[i] - recall[i - 1])

    return ap.item()


def compute_kap(args, save_path, aggre_net, files, kps, category=None, used_points=None, thresholds=None, verbose=True, sph_mapper=None):
    out_results = []
    num_patches = args.NUM_PATCHES
    current_save_results = 0
    pos_samples, neg_samples, img_acc_001, img_acc_005, img_acc_01, len_kpts = ([] for _ in range(6))
    if thresholds is not None:
        thresholds = torch.tensor(thresholds).to(device)
        bbox_size=[]
    N = len(files) // 2

    alpha = torch.tensor([0.1, 0.05, 0.01]) if args.EVAL_DATASET != 'pascal' else torch.tensor([0.1, 0.05, 0.15])
    alpha = alpha.to(device)

    pbar = tqdm(total=N, dynamic_ncols=True, disable=not verbose)

    for pair_idx in range(N):
        # Load images and keypoints
        img1, img1_kps = load_img_and_kps(idx=2*pair_idx, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        img2, img2_kps = load_img_and_kps(idx=2*pair_idx+1, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        # Get mutual visibility
        vis1 = img1_kps[:, 2]
        vis2 = img2_kps[:, 2]
        vis = vis1 * vis2 > 0
        # Get patch descriptors
        with torch.no_grad():
            img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, img1=img1, img2=img2, sph_mapper=sph_mapper)
        # Get patch index for the keypoints
        img1_patch_idx = kpts_to_patch_idx(args, img1_kps, num_patches)
        # Get similarity matrix
        thresh = thresholds[pair_idx] if thresholds is not None else args.ANNO_SIZE
        im_threshs = thresh * alpha
        pos_samples_im, neg_samples_im = extract_kap_vals(args, img1_desc, img2_desc, img1_patch_idx, vis1, img2_kps.to(img1_desc.device), num_patches, im_threshs)

        pos_samples.append(pos_samples_im)
        neg_samples.append(neg_samples_im)
        len_kpts.append(vis.sum().item())

        # compute per image acc
        if not args.KPT_RESULT: # per img result
            preds_im = torch.cat([pos_samples_im, neg_samples_im], dim=-1).numpy(force=True)
            labels_im = torch.cat([torch.ones_like(pos_samples_im), torch.zeros_like(neg_samples_im)], dim=-1).numpy(force=True)
            img_acc_01.append(average_precision_score(labels_im[0], preds_im[0]))
            img_acc_005.append(average_precision_score(labels_im[1], preds_im[1]))
            img_acc_001.append(average_precision_score(labels_im[2], preds_im[2]))


        pbar.update(1)
    pbar.close()
    if not args.KPT_RESULT:
        img_correct = torch.tensor([img_acc_01, img_acc_005, img_acc_001])
        img_correct = img_correct.mean(dim=-1).tolist()
        img_correct.append(N)
    else:
        img_correct = None

    pos_samples = torch.cat(pos_samples, dim=-1).cpu()
    neg_samples = torch.cat(neg_samples, dim=-1).cpu()
    preds = torch.cat([pos_samples, neg_samples], dim=-1).numpy(force=True)
    labels = torch.cat([torch.ones_like(pos_samples), torch.zeros_like(neg_samples)], dim=-1).numpy(force=True)

    kap_01 = average_precision_score(labels[0], preds[0])
    kap_005 = average_precision_score(labels[1], preds[1])
    kap_001 = average_precision_score(labels[2], preds[2])

    correct =[kap_01, kap_005, kap_001, len(neg_samples[0])]
    alpha2pck = zip(alpha.tolist(), correct[:3]) if args.KPT_RESULT else zip(alpha.tolist(), img_correct[:3])
    logger.info(f'{category}...'+' | '.join([f'KAP-Transfer@{alpha:.2f}: {kap_alpha * 100:.2f}%'
        for alpha, kap_alpha in alpha2pck]))

    return correct, [], out_results, img_correct

def train(args, aggre_net, corr_map_net, cat_proto, optimizer, scheduler, logger, save_path):
    # gather training data
    files, kps, _, _, all_thresholds = load_and_prepare_data(args)
    # train
    num_patches = args.NUM_PATCHES
    N = len(files) // 2
    pbar = tqdm(total=N, dynamic_ncols=True)
    max_pck_010 = max_pck_005 = max_pck_001 = max_iter = im_loss_count = kpl_loss_count = kpf_loss_count = geom_loss_count = count = 0
    seen_samples = 0
    for epoch in range(args.EPOCH):
        pbar.reset()
        for j in range(0, N, args.BZ):
            optimizer.zero_grad()
            batch_loss = 0  # collect the loss for each batch
            for pair_idx in range(j, min(j+args.BZ, N)):
                category = files[2*pair_idx].split('/')[-2]
                # Load images and keypoints
                img1, img1_kps = load_img_and_kps(idx=2*pair_idx, files=files, kps=kps, edge=False)
                img2, img2_kps = load_img_and_kps(idx=2*pair_idx+1, files=files, kps=kps, edge=False)
                # Get patch descriptors/feature maps
                img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, img1=img1, img2=img2)
                # Load pointmaps and segmentation
                img1_points, img1_depths, img1_mask = load_points_and_seg(img_idx=2*pair_idx, files=files, device=img1_desc.device)
                img2_points, img2_depths, img2_mask = load_points_and_seg(img_idx=2*pair_idx+1, files=files, device=img1_desc.device)
                if args.ADAPT_FLIP > 0 or args.AUGMENT_SELF_FLIP > 0 or args.AUGMENT_DOUBLE_FLIP > 0:  # augment with flip
                    img1_desc_flip, img2_desc_flip, _, _ = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=True, flip2=True, img1=img1.transpose(Image.FLIP_LEFT_RIGHT), img2=img2.transpose(Image.FLIP_LEFT_RIGHT))
                    raw_permute_list = AP10K_FLIP if args.TRAIN_DATASET == 'ap10k' else SPAIR_FLIP_TRN[files[pair_idx * 2].split('/')[-2]]
                else:
                    img1_desc_flip = img2_desc_flip = raw_permute_list = None
                # Get the threshold for each patch
                scale_factor = num_patches / args.ANNO_SIZE
                if args.BBOX_THRE:
                    img1_threshold = all_thresholds[2*pair_idx] * scale_factor
                    img2_threshold = all_thresholds[2*pair_idx+1] * scale_factor
                else: # image threshold
                    img1_threshold = img2_threshold = args.ANNO_SIZE

                # Compute loss
                im_loss = utils_losses.calculate_loss(args, aggre_net, img1_kps, img2_kps, img1_desc, img2_desc, img1_threshold, img2_threshold, mask1, mask2,
                                                   num_patches, device, raw_permute_list, img1_desc_flip, img2_desc_flip, corr_map_net, )

                # Compute 3d loss
                if args.CAT_PROTO:
                    proto_loss, loc_loss, feat_loss, geom_loss = utils_losses.calculate_3d_loss(args, aggre_net, cat_proto, img1_kps, img2_kps, img1_desc, img2_desc, img1_points, img2_points, img1_depths, img2_depths,
                                                                 img1_mask, img2_mask, mask1, mask2, category, num_patches, device, raw_permute_list, img1_desc_flip, img2_desc_flip,
                                                                 save=(pair_idx + epoch * N) % 100 == 0)
                else:
                    proto_loss, loc_loss, feat_loss, geom_loss = torch.tensor(0), 0, 0, 0


                loss = im_loss + proto_loss 
                # Accumulate loss over iterations
                im_loss_count += im_loss.item()
                kpl_loss_count += loc_loss
                kpf_loss_count += feat_loss
                geom_loss_count += geom_loss
                count += args.BZ
                batch_loss += loss
                pbar.update(1)
                seen_samples += 1

                with torch.no_grad():
                    # Log loss periodically or at the end of the dataset
                    if ((pair_idx + epoch * N) % 100 == 0 and pair_idx > 0) or pair_idx == N-1: # Log every 100 iterations and at the end of the dataset
                        # print(f'KPS LOC:{loc_loss}, KPS FEAT:{feat_loss}, GEOM: {geom_loss}')
                        pbar.set_description(f'Step {pair_idx + epoch * N} | Loss: {im_loss_count / count:.4f} | kp loc loss: {kpl_loss_count / count:.4f} | kp feat loss: {kpf_loss_count / count:.4f} | geom loss: {geom_loss_count / count:.4f}')
                        logger.debug(f'Step {pair_idx + epoch * N} | Loss: {im_loss_count / count:.4f}')
                        wandb_dict = {'loss': im_loss_count / count}
                        im_loss_count = kpl_loss_count = kpf_loss_count = geom_loss_count = count = 0 # reset loss count
                        if not args.NOT_WANDB: wandb.log(wandb_dict, step=pair_idx + epoch * N)
                    # Evaluate model periodically, at the end of the dataset, or under specific conditions
                    if ((pair_idx + epoch * N) % args.EVAL_EPOCH == 0 and pair_idx > 0) or pair_idx == N-1:  # Evaluate every args.EVAL_EPOCH iterations and at the end of the dataset
                        pck_010, pck_005, pck_001, total_result = eval(args, aggre_net, save_path, verbose=False)  # Perform evaluation
                        wandb_dict = {'pck_010': pck_010, 'pck_005': pck_005, 'pck_001': pck_001}
                        # Update best model based on PCK scores and dataset type
                        if (pck_010 > max_pck_010 and args.EVAL_DATASET != 'pascal') or (pck_005 > max_pck_005 and args.EVAL_DATASET == 'pascal'): # different criteria for PASCAL_EVAL
                            max_pck_010, max_pck_005, max_pck_001 = pck_010, pck_005, pck_001
                            max_iter = pair_idx + epoch * N
                            torch.save(aggre_net.state_dict(), f'{save_path}/best.pth') # Save the best model
                            torch.save(cat_proto.state_dict(), f'{save_path}/best_cp.pth') # Save the best model
                        else:
                            torch.save(aggre_net.state_dict(), f'{save_path}/last.pth') # Save the last model if it's not the best
                            torch.save(cat_proto.state_dict(), f'{save_path}/last_cp.pth') # Save the last model if it's not the best
                        # Log the best PCK scores
                        logger.info(f'Best PCK0.10: {max_pck_010 * 100:.2f}% at step {max_iter}, with PCK0.05: {max_pck_005 * 100:.2f}%, PCK0.01: {max_pck_001 * 100:.2f}%')
                        if not args.NOT_WANDB: wandb.log(wandb_dict, step=pair_idx + epoch * N)

            batch_loss /= args.BZ
            batch_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    pbar.close()


def eval(args, aggre_net, save_path, sph_mapper=None, split='val', verbose=True):
    aggre_net.eval()  # Set the network to evaluation mode
    # Configure data directory and categories based on the dataset type
    data_dir, categories, split = get_dataset_info(args, split)

    # Initialize lists for results and statistics
    total_out_results, pcks, pcks_05, pcks_01, weights, kpt_weights = ([] for _ in range(6))
    if args.COMPUTE_GEOAWARE_METRICS: geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo = ([] for _ in range(6))

    # Process each category
    for category in categories:
        # Load data based on the dataset
        files, kps, thresholds, used_points = load_eval_data(args, data_dir, category, split, unseen_kps=args.UNSEEN_KPS)
        # Compute PCK with or without bbox threshold
        compute_args = (save_path, aggre_net, files, kps, category, used_points)
        if args.KAP:
            pck, correct_geo, out_results, img_correct = compute_kap(args, *compute_args, sph_mapper=sph_mapper, thresholds=thresholds, verbose=verbose) if args.BBOX_THRE else compute_pck(args, *compute_args, sph_mapper=sph_mapper, verbose=verbose)
        else:
            pck, correct_geo, out_results, img_correct = compute_pck(args, *compute_args, sph_mapper=sph_mapper, thresholds=thresholds, verbose=verbose) if args.BBOX_THRE else compute_pck(args, *compute_args, sph_mapper=sph_mapper, verbose=verbose)
        total_out_results.extend(out_results)
        update_stats(args, pcks, pcks_05, pcks_01, weights, kpt_weights, pck, img_correct)
        if args.COMPUTE_GEOAWARE_METRICS: update_geo_stats(geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, correct_geo)

    # Calculate and log weighted PCKs
    pck_010, pck_005, pck_001 = log_weighted_pcks(args, logger, pcks, pcks_05, pcks_01, weights)
    if args.COMPUTE_GEOAWARE_METRICS: log_geo_stats(args, geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, kpt_weights, total_out_results)

    aggre_net.train()  # Set the network back to training mode
    return pck_010, pck_005, pck_001, total_out_results

def main(args):
    set_seed(args.SEED)
    args.NUM_PATCHES = 60
    args.BBOX_THRE = not (args.IMG_THRESHOLD or args.EVAL_DATASET == 'pascal')
    args.AUGMENT_FLIP, args.AUGMENT_DOUBLE_FLIP, args.AUGMENT_SELF_FLIP = (1.0, 1.0, 0.25) if args.PAIR_AUGMENT else (0, 0, 0) # set different weight for different augmentation
    if args.SAMPLE == 0: args.SAMPLE = None # use all the data
    feature_dims = [640,1280,1280,768] # dimensions for three layers of SD and one layer of DINOv2 features

    # Determine the evaluation type and project name based on args
    save_path = f'./results_{args.EVAL_DATASET}/pck_train_{args.NOTE}_sample_{args.EPOCH}_{args.SAMPLE}_lr_{args.LR}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not args.NOT_WANDB:
        wandb.init(project=args.EVAL_DATASET, name=f'{args.NOTE}_sample_{args.EPOCH}_{args.SAMPLE}_lr_{args.LR}', config=args)

    logger = get_logger(save_path+'/result.log')
    logger.info(args)
    if args.DUMMY_NET:
        aggre_net = DummyAggregationNetwork()
    else:
        aggre_net = AggregationNetwork(feature_dims=feature_dims, projection_dim=args.PROJ_DIM, device=device, feat_map_dropout=args.FEAT_MAP_DROPOUT)

    category_prototype = CategoryPrototype(feature_dims=args.PROJ_DIM)

    if args.LOAD is not None:
        pretrained_dict = torch.load(args.LOAD, weights_only=True)
        aggre_net.load_pretrained_weights(pretrained_dict)
        logger.info(f'Load model from {args.LOAD}')
        if args.CAT_PROTO:
            cat_proto_dict_path = os.path.splitext(args.LOAD)[0] + '_cp.pth'
            if os.path.exists(cat_proto_dict_path):
                cat_proto_dict = torch.load(cat_proto_dict_path, weights_only=True)
                category_prototype.load_state_dict(cat_proto_dict)
                logger.info(f'Load cat proto from {cat_proto_dict_path}')
            else:
                logger.info(f'No cat proto checkpoint found!')
    if args.SPH:
        sph_mapper = torch.nn.Sequential(
            torch.nn.Linear(768, 384),
            torch.nn.GELU(),
            timm.models.vision_transformer.Block(dim=384, num_heads=6),
            torch.nn.Linear(384, 3),
            )
        sph_ckpt = torch.load("sph_mapper.pth", map_location='cpu', weights_only=True)
        sph_ckpt = {k[14:]: v for k, v in sph_ckpt.items() if 'sphere_mapper' in k}
        sph_mapper.load_state_dict(sph_ckpt)
        sph_mapper.to(device).eval()
    else:
        sph_mapper = None
    aggre_net.to(device)
    category_prototype = category_prototype.to(device)
    total_args = chain(aggre_net.parameters(), category_prototype.parameters())
    if args.DENSE_OBJ>0:
        corr_map_net = Correlation2Displacement(setting=args.DENSE_OBJ, window_size=args.SOFT_TRAIN_WINDOW).to(device)
        total_args = chain(total_args, corr_map_net.parameters())
    else:
        corr_map_net = None


    optimizer = torch.optim.AdamW(total_args, lr=args.LR, weight_decay=args.WD)
    if args.SCHEDULER is not None:
        if args.SCHEDULER == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=(53339+args.BZ)//args.BZ, eta_min=1e-6) #53339 is the number of training pairs for SPair-71k
        if args.SCHEDULER == 'one_cycle':
            scheduler = OneCycleLR(optimizer, max_lr=args.LR, steps_per_epoch=(53339+args.BZ)//args.BZ, epochs=args.EPOCH, pct_start=args.SCHEDULER_P1)
    else:
        scheduler = None

    if args.DO_EVAL: # eval on test set
        with torch.no_grad():
            _,_,_,result = eval(args, aggre_net, save_path, sph_mapper=sph_mapper, split='test')
            with open(save_path+'/result.pkl', 'wb') as f:
                pickle.dump(result, f)
    else:
        train(args, aggre_net, corr_map_net, category_prototype, optimizer, scheduler, logger, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # load config
    parser.add_argument('--config', type=str, default=None)                         # path to the config file

    # basic training setting
    parser.add_argument('--SEED', type=int, default=42)                             # random seed
    parser.add_argument('--NOTE', type=str, default='')                             # note for the experiment
    parser.add_argument('--SAMPLE', type=int, default=0)                            # sample 100 pairs for each category for training, set to 0 to use all pairs
    parser.add_argument('--TEST_SAMPLE', type=int, default=20)                      # sample 20 pairs for each category for testing, set to 0 to use all pairs
    parser.add_argument('--TOTAL_SAVE_RESULT', type=int, default=0)                 # save the qualitative results for the first 5 pairs
    parser.add_argument('--IMG_THRESHOLD', action='store_true', default=False)      # set the pck threshold to the image size rather than the bbox size
    parser.add_argument('--ANNO_SIZE', type=int, default=840)                       # image size for the annotation input
    parser.add_argument('--LR', type=float, default=1.25e-3)                        # learning rate
    parser.add_argument('--WD', type=float, default=1e-3)                           # weight decay
    parser.add_argument('--BZ', type=int, default=1)                                # batch size
    parser.add_argument('--SCHEDULER', type=str, default=None)                      # set to use lr scheduler, one_cycle, cosine, plateau
    parser.add_argument('--SCHEDULER_P1', type=float, default=0.3)                  # set the first parameter for the scheduler
    parser.add_argument('--EPOCH', type=int, default=1)                             # number of epochs
    parser.add_argument('--EVAL_EPOCH', type=int, default=5000)                     # number of steps for evaluation
    parser.add_argument('--NOT_WANDB', action='store_true', default=False)          # set true to not use wandb
    parser.add_argument('--TRAIN_DATASET', type=str, default='spair')               # set the training dataset, 'spair' for SPair-71k, 'pascal' for PF-Pascal, 'ap10k' for AP10k

    # training model setup
    parser.add_argument('--LOAD', type=str, default=None)                           # path to load the pretrained model
    parser.add_argument('--DENSE_OBJ', type=int, default=1)                         # set true to use the dense training objective, 1: enable; 0: disable
    parser.add_argument('--GAUSSIAN_AUGMENT', type=float, default=0.1)              # set float to use the gaussian augment, float for std
    parser.add_argument('--FEAT_MAP_DROPOUT', type=float, default=0.2)              # set true to use the dropout for the feat map
    parser.add_argument('--ENSEMBLE', type=int, default=1)                          # set true to use the ensembles of sd feature maps
    parser.add_argument('--PROJ_DIM', type=int, default=768)                        # projection dimension of the post-processor
    parser.add_argument('--PAIR_AUGMENT', action='store_true', default=False)       # set true to enable pose-aware pair augmentation
    parser.add_argument('--SELF_CONTRAST_WEIGHT', type=float, default=0)            # set true to use the self supervised loss
    parser.add_argument('--SOFT_TRAIN_WINDOW', type=int, default=0)                 # set true to use the window soft argmax during training, default is using standard soft argmax

    # evaluation setup
    parser.add_argument('--DO_EVAL', action='store_true', default=False)            # set true to do the evaluation on test set
    parser.add_argument('--DUMMY_NET', action='store_true', default=False)          # set true to use the dummy net, used for zero-shot setting
    parser.add_argument('--EVAL_DATASET', type=str, default='spair')                # set the evaluation dataset, 'spair' for SPair-71k, 'pascal' for PF-Pascal, 'ap10k' for AP10k
    parser.add_argument('--AP10K_EVAL_SUBSET', type=str, default='intra-species')          # set the test setting for ap10k dataset, `intra-species`, `cross-species`, `cross-family`
    parser.add_argument('--COMPUTE_GEOAWARE_METRICS', action='store_true', default=False)   # set true to use the geo-aware count
    parser.add_argument('--KPT_RESULT', action='store_true', default=False)         # set true to evaluate per kpt result, in the paper, this is used for comparing unsupervised methods, following ASIC
    parser.add_argument('--ADAPT_FLIP', action='store_true', default=False)         # set true to use the flipped images, adaptive flip
    parser.add_argument('--MUTUAL_NN', action='store_true', default=False)          # set true to use the flipped images, adaptive flip, mutual nn as metric
    parser.add_argument('--SOFT_EVAL', action='store_true', default=False)          # set true to use the soft argmax eval
    parser.add_argument('--SOFT_EVAL_WINDOW', type=int, default=7)                  # set true to use the window soft argmax eval, window size is 2*SOFT_EVAL_WINDOW+1, 0 to be standard soft argmax

    # add Jamais Vu options
    parser.add_argument('--CAT_PROTO', action='store_true', default=False)          # set true to train 3d prototype on top
    parser.add_argument('--UNSEEN_KPS', action='store_true', default=False)         # set true to evaluate on SPair-U keypoints
    parser.add_argument('--ONLY_DINO', action='store_true', default=False)          # set true to evaluate on DINO features only
    parser.add_argument('--ONLY_SD', action='store_true', default=False)            # set true to evaluate on SD features only
    parser.add_argument('--SPH', action='store_true', default=False)                # set true to evaluate on DINO+SD+spherical maps
    parser.add_argument('--KAP', action='store_true', default=False)                # set true to evaluate using KAP
    parser.add_argument('--PCKD', action='store_true', default=False)               # set true to evaluate unsing PCK^\dagger

    args = parser.parse_args()
    if args.config is not None: # load config file and update the args
        args_dict = vars(args)
        args_dict.update(load_config(args.config))
        args = argparse.Namespace(**args_dict)
    main(args)
