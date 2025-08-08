from typing import Tuple
import numpy as np
import cv2
import os
from scipy.io import savemat
import matplotlib.pyplot as plt


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0),
            alpha: float = 0.5) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if len(mask.shape) == 3:
        mask = mask.squeeze()

    if image.shape[:2] != mask.shape[:2]:
        print(f"Warning: Image shape {image.shape[:2]} doesn't match mask shape {mask.shape[:2]}")
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)

    mask_bool = mask.astype(bool)

    if not np.any(mask_bool):
        print("Warning: Mask is empty (no positive values)")
        return image.copy()

    colored_overlay = image.copy()
    colored_overlay[mask_bool] = color

    combined = cv2.addWeighted(image, 1.0 - alpha, colored_overlay, alpha, 0)

    return combined


def overlay_contour(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0),
                    thickness: int = 2) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if len(mask.shape) == 3:
        mask = mask.squeeze()

    if image.shape[:2] != mask.shape[:2]:
        print(f"Warning: Image shape {image.shape[:2]} doesn't match mask shape {mask.shape[:2]}")
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)

    return result


def overlay_filled(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0),
                   alpha: float = 0.3) -> np.ndarray:
    filled = overlay(image, mask, color, alpha)

    result = overlay_contour(filled, mask, color, thickness=2)

    return result


def save_mat(file, i, dir, folder_name):
    folder = os.path.join(dir, folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    savemat(os.path.join(folder, f'{i}.mat'), {'data': file})
    return None
