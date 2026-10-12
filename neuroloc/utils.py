"""
Utility functions and synthetic brain MRI generation for NeuroLoc-XAI.
Provides synthetic MRI generator for unit testing, offline demos, and figure plotting helpers.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_synthetic_mri(
    tumor_type: str = "glioma",
    image_size: Tuple[int, int] = (224, 224),
    noise_level: float = 0.02,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate a realistic synthetic 2D brain MRI slice with realistic cranial contour,
    brain parenchyma, ventricles, and a class-specific lesion.

    Args:
        tumor_type: 'glioma', 'meningioma', 'pituitary', or 'notumor'.
        image_size: (H, W) in pixels.
        noise_level: Gaussian noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        image: uint8 numpy array of shape (H, W, 3) simulating a T1-Gd MRI scan.
        metadata: Ground truth bounding box, centroid, and lesion information.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = image_size
    canvas = np.zeros((h, w), dtype=np.float32)

    # 1. Skull ring (outer cranial bone)
    center_x, center_y = w // 2, h // 2
    skull_rx, skull_ry = int(w * 0.40), int(h * 0.45)
    cv2.ellipse(canvas, (center_x, center_y), (skull_rx, skull_ry), 0, 0, 360, 0.45, -1)

    # 2. CSF space (subdural hypointense rim)
    csf_rx, csf_ry = int(skull_rx * 0.96), int(skull_ry * 0.96)
    cv2.ellipse(canvas, (center_x, center_y), (csf_rx, csf_ry), 0, 0, 360, 0.15, -1)

    # 3. Brain parenchyma (gray/white matter background)
    brain_rx, brain_ry = int(skull_rx * 0.92), int(skull_ry * 0.92)
    cv2.ellipse(canvas, (center_x, center_y), (brain_rx, brain_ry), 0, 0, 360, 0.50, -1)

    # 4. Ventricles (bilateral anterior and posterior horns)
    left_vent_x, right_vent_x = center_x - int(w * 0.08), center_x + int(w * 0.08)
    vent_y = center_y - int(h * 0.03)
    cv2.ellipse(canvas, (left_vent_x, vent_y), (int(w * 0.03), int(h * 0.10)), 15, 0, 360, 0.12, -1)
    cv2.ellipse(canvas, (right_vent_x, vent_y), (int(w * 0.03), int(h * 0.10)), -15, 0, 360, 0.12, -1)

    # Interhemispheric fissure (subtle central dark midline)
    cv2.line(canvas, (center_x, center_y - brain_ry), (center_x, center_y + brain_ry), 0.25, 1)

    # 5. Insert pathology based on tumor type
    gt_bbox = None
    gt_centroid = None
    lesion_mask = np.zeros((h, w), dtype=np.uint8)

    if tumor_type == "pituitary":
        # Sellar / suprasellar lesion (central midline, base of brain)
        lx, ly = center_x, center_y + int(h * 0.06)
        lrx, lry = int(w * 0.08), int(h * 0.08)
        # Hyperintense contrast-enhancing adenoma
        cv2.ellipse(canvas, (lx, ly), (lrx, lry), 0, 0, 360, 0.92, -1)
        # Necrotic/cystic hypointense center
        cv2.circle(canvas, (lx, ly), int(lrx * 0.35), 0.35, -1)
        cv2.ellipse(lesion_mask, (lx, ly), (lrx, lry), 0, 0, 360, 255, -1)
        gt_bbox = (lx - lrx, ly - lry, lx + lrx, ly + lry)
        gt_centroid = (lx, ly)

    elif tumor_type == "meningioma":
        # Extra-axial dural-based mass (e.g. right frontal-parietal convexity)
        lx = center_x - int(w * 0.26)
        ly = center_y - int(h * 0.18)
        lrx, lry = int(w * 0.10), int(h * 0.09)
        # Strongly enhancing uniform mass with dural tail
        cv2.ellipse(canvas, (lx, ly), (lrx, lry), 30, 0, 360, 0.95, -1)
        # Subtle dural tail attachment
        cv2.line(canvas, (lx - lrx, ly), (lx - lrx - 8, ly - 12), 0.85, 2)
        cv2.ellipse(lesion_mask, (lx, ly), (lrx, lry), 30, 0, 360, 255, -1)
        gt_bbox = (lx - lrx, ly - lry, lx + lrx, ly + lry)
        gt_centroid = (lx, ly)

    elif tumor_type == "glioma":
        # Intra-axial infiltrative mass with peritumoral vasogenic edema (e.g. left temporal/parietal)
        lx = center_x + int(w * 0.20)
        ly = center_y + int(h * 0.10)
        lrx, lry = int(w * 0.12), int(h * 0.11)
        # Peritumoral edema (dark halo)
        cv2.ellipse(canvas, (lx, ly), (int(lrx * 1.35), int(lry * 1.35)), -20, 0, 360, 0.28, -1)
        # Ring-enhancing core
        cv2.ellipse(canvas, (lx, ly), (lrx, lry), -20, 0, 360, 0.88, -1)
        # Central necrotic core
        cv2.circle(canvas, (lx, ly), int(lrx * 0.45), 0.20, -1)
        cv2.ellipse(lesion_mask, (lx, ly), (int(lrx * 1.35), int(lry * 1.35)), -20, 0, 360, 255, -1)
        gt_bbox = (lx - int(lrx * 1.35), ly - int(lry * 1.35), lx + int(lrx * 1.35), ly + int(lry * 1.35))
        gt_centroid = (lx, ly)

    elif tumor_type == "notumor":
        # Normal scan - no mass, just normal anatomic variation
        pass
    else:
        raise ValueError(f"Unknown tumor_type: {tumor_type}. Choose from 'glioma', 'meningioma', 'pituitary', 'notumor'.")

    # Add Gaussian noise and scanner texture
    noise = np.random.normal(0, noise_level, canvas.shape)
    canvas = np.clip(canvas + noise, 0.0, 1.0)

    # Convert to 3-channel uint8
    image_uint8 = (canvas * 255.0).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

    metadata = {
        "tumor_type": tumor_type,
        "has_lesion": tumor_type != "notumor",
        "gt_bbox": gt_bbox,
        "gt_centroid": gt_centroid,
        "gt_mask": lesion_mask,
        "image_size": image_size,
    }

    return image_bgr, metadata
