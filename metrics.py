import os
import glob
import numpy as np
import sys
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from scipy.io import loadmat
from scipy.spatial.distance import cdist, directed_hausdorff
from tqdm import tqdm
import math
from typing import Optional, Tuple, Dict, List
import config

BASE_DIR = r"<ADD_YOUR_BASE_DIR>"
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "test12K")
PRED_FOLDER = os.path.join(RESULTS_DIR, "predictions")
LABEL_FOLDER = os.path.join(RESULTS_DIR, "labels")

IMAGE_SIZE = 256
NLSR_DICE_THRESHOLD = 0.7

pred_files = sorted(glob.glob(os.path.join(PRED_FOLDER, '*.mat')))
label_files = sorted(glob.glob(os.path.join(LABEL_FOLDER, '*.mat')))

if len(pred_files) != len(label_files):
    print(f"Error: Number of prediction files ({len(pred_files)}) does not match label files ({len(label_files)}).")
    exit()

if not pred_files:
    print(f"Error: No prediction files found in {PRED_FOLDER}. Run Test.py first.")
    exit()

print(f"Found {len(pred_files)} prediction/label pairs.")

dsc_scores, iou_scores = [], []
mhd_scores = []
te_scores, nlr_scores = [], []
success_flags = []
image_center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)


def calculate_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Prediction and ground truth masks must have the same shape.")
    if not ((pred_mask == 0) | (pred_mask == 1)).all():
        raise ValueError("Prediction mask must be binary (0 or 1).")
    if not ((gt_mask == 0) | (gt_mask == 1)).all():
        raise ValueError("Ground truth mask must be binary (0 or 1).")

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    dsc = (2. * tp + eps) / (2. * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        'dsc': dsc,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }


def calculate_mhd(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_boundary = get_boundary_points(pred_mask)
    gt_boundary = get_boundary_points(gt_mask)

    if pred_boundary is None or gt_boundary is None or pred_boundary.shape[0] == 0 or gt_boundary.shape[0] == 0:
        return np.nan

    dist_pred_to_gt = cdist(pred_boundary, gt_boundary)
    dist_gt_to_pred = cdist(gt_boundary, pred_boundary)

    mhd1 = np.mean(np.min(dist_pred_to_gt, axis=1))
    mhd2 = np.mean(np.min(dist_gt_to_pred, axis=1))

    return max(mhd1, mhd2)


def find_tip_coordinates(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if not np.any(mask):
        return None

    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8)
    else:
        mask_u8 = mask

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.squeeze(axis=1)

    if points.ndim == 0:
        return None
    if points.ndim == 1:
        points = points.reshape(1, -1)

    if points.shape[0] == 0:
        return None

    tip_index = np.argmax(points[:, 1])
    tip = tuple(points[tip_index].astype(int))
    return tip


def get_boundary_points(mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.sum() == 0:
        return None
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8)
    else:
        mask_u8 = mask

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    boundary_points = np.concatenate(contours, axis=0).squeeze(axis=1)
    return boundary_points


def calculate_fragmentation_score(pred_mask: np.ndarray, area_threshold: int = 10) -> int:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)

    fragmentation_score = 0
    if num_labels > 1:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > area_threshold:
                fragmentation_score += 1

    if fragmentation_score == 0 and np.any(pred_mask):
        if num_labels > 1:
            if any(stats[i, cv2.CC_STAT_AREA] > 0 for i in range(1, num_labels)):
                return 1
    elif fragmentation_score == 0 and not np.any(pred_mask):
        return 0

    return fragmentation_score


def get_principal_axis_and_length(mask: np.ndarray, eps: float = 1e-6) -> Optional[
    Tuple[np.ndarray, float, np.ndarray]]:
    if not np.any(mask):
        return None

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2)

    if points.shape[0] < 5:
        return None

    points = points.astype(np.float32)

    try:
        line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_params.flatten()

        axis_vector_tuple = (vx, vy)
        center_tuple = (x0, y0)

        center_np = np.array([x0, y0], dtype=np.float32)
        axis_vector_flat = np.array([vx, vy], dtype=np.float32)

        diff = points - center_np
        projections = np.dot(diff, axis_vector_flat)

        min_proj = np.min(projections)
        max_proj = np.max(projections)
        length = max_proj - min_proj

        return axis_vector_flat, length, center_np
    except cv2.error as e:
        print(f"cv2.fitLine error: {e}")
        return None
    except Exception as e:
        print(f"Error in get_principal_axis_and_length: {e}")
        return None


def is_localization_successful(fs_score: int, dice_score: float) -> bool:
    if np.isnan(dice_score) or fs_score is None:
        return False
    return fs_score == 1 and dice_score > NLSR_DICE_THRESHOLD


def calculate_distance_line_to_point(axis_info: Optional[Tuple[np.ndarray, float, np.ndarray]],
                                     point: Tuple[int, int]) -> float:
    if axis_info is None:
        return np.nan

    axis_vector, _, centroid = axis_info
    line_point = np.array(centroid)
    target_point = np.array(point)

    norm = np.linalg.norm(axis_vector)
    if norm < 1e-6:
        return np.linalg.norm(target_point - line_point)

    unit_axis_vector = axis_vector / norm
    vx, vy = unit_axis_vector

    vec_to_point = target_point - line_point

    distance = abs(vy * vec_to_point[0] - vx * vec_to_point[1])
    return distance


def calculate_targeting_error(pred_axis_info: Optional[Tuple[np.ndarray, float, np.ndarray]],
                                    gt_axis_info: Optional[Tuple[np.ndarray, float, np.ndarray]],
                                    image_center: Tuple[int, int]) -> float:
    d_true = calculate_distance_line_to_point(gt_axis_info, image_center)
    d_segmentation = calculate_distance_line_to_point(pred_axis_info, image_center)

    if np.isnan(d_true) or np.isnan(d_segmentation):
        return np.nan

    te = abs(d_true - d_segmentation)
    return te


def calculate_needle_length_ratio(pred_length: Optional[float], gt_length: Optional[float], eps: float = 1e-6) -> float:
    if pred_length is None or gt_length is None or gt_length < eps:
        return np.nan

    nlr = pred_length / (gt_length + eps)
    return nlr


for i, (pred_file, label_file) in tqdm(enumerate(zip(pred_files, label_files)), total=len(pred_files)):
    try:
        pred_basename = os.path.basename(pred_file)
        label_basename = os.path.basename(label_file)
        if pred_basename != label_basename:
            print(f"Warning: Skipping pair - Mismatched filenames: {pred_basename} vs {label_basename}")
            continue

        pred_data = loadmat(pred_file)
        label_data = loadmat(label_file)

        pred_mask = pred_data['data'].astype(np.uint8)
        label_mask = label_data['data'].astype(np.uint8)

        pred_mask = (pred_mask > 0).astype(np.uint8).squeeze()
        label_mask = (label_mask > 0).astype(np.uint8).squeeze()

        if pred_mask.shape != label_mask.shape:
            print(
                f"Warning: Skipping sample {pred_basename} due to shape mismatch: Pred {pred_mask.shape}, GT {label_mask.shape}")
            continue

        if not np.any(pred_mask) or not np.any(label_mask):
            print(
                f"Warning: Skipping sample {pred_basename} because prediction or label mask is empty after processing.")
            dsc_scores.append(np.nan)
            iou_scores.append(np.nan)
            mhd_scores.append(np.nan)
            te_scores.append(np.nan)
            nlr_scores.append(np.nan)
            success_flags.append(0)
            continue

        seg_metrics = calculate_segmentation_metrics(pred_mask, label_mask)
        current_dice = seg_metrics['dsc']
        fs = calculate_fragmentation_score(pred_mask)

        is_success = is_localization_successful(fs, current_dice)
        success_flags.append(1 if is_success else 0)

        dsc_scores.append(current_dice)
        iou_scores.append(seg_metrics['iou'])

        if is_success:
            pred_axis_info = get_principal_axis_and_length(pred_mask)
            gt_axis_info = get_principal_axis_and_length(label_mask)

            mhd_val = calculate_mhd(pred_mask, label_mask)
            mhd_scores.append(mhd_val)

            te_val = calculate_targeting_error(pred_axis_info, gt_axis_info, image_center)
            te_scores.append(te_val)

            if pred_axis_info is not None and gt_axis_info is not None:
                _, pred_len, _ = pred_axis_info
                _, gt_len, _ = gt_axis_info
                nlr_val = calculate_needle_length_ratio(pred_len, gt_len)
                nlr_scores.append(nlr_val)
        else:
            mhd_scores.append(np.nan)
            te_scores.append(np.nan)
            nlr_scores.append(np.nan)

    except Exception as e:
        print(f"Error processing file {pred_file}: {e}")
        dsc_scores.append(np.nan)
        iou_scores.append(np.nan)
        mhd_scores.append(np.nan)
        te_scores.append(np.nan)
        nlr_scores.append(np.nan)
        success_flags.append(np.nan)

metric_results = {
    "DSC": dsc_scores,
    "IoU": iou_scores,
    "MHD": mhd_scores,
    "TE": te_scores,
    "NLR": nlr_scores,
}

valid_samples_count = 0
for name, scores in metric_results.items():
    scores_np = np.array(scores)
    valid_count = np.sum(~np.isnan(scores_np))
    if valid_samples_count == 0 and name == 'DSC':
        valid_samples_count = valid_count

    if valid_count > 0:
        mean_val = np.nanmean(scores_np)
        std_val = np.nanstd(scores_np)
        print(f"{name}: Mean={mean_val:.4f}, SD={std_val:.4f} (Valid Calc: {valid_count})")
    else:
        print(f"{name}: Mean=NaN, SD=NaN (Valid Calc: 0)")