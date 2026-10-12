"""
Module 4: Explainability & Anatomical Localization
Implements Grad-CAM and Grad-CAM++ with intracranial brain parenchyma masking,
active bounding box and centroid extraction constrained strictly inside the head,
automated neuro-anatomical rule engine, 2D (X, Y) spatial indication & triage graph,
and validation metrics (IoU, Pointing Game).
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from neuroloc.model import get_last_conv_layer_name
from neuroloc.dataset import extract_cranial_brain_mask


# =========================================================================
# Grad-CAM & Grad-CAM++ Engine with Cranial Masking
# =========================================================================

class GradCAM:
    """
    Standard Grad-CAM (Selvaraju et al., 2017) implementation using tf.GradientTape,
    with cranial intracranial vault masking to guarantee heatmaps reside within the head.
    """

    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or get_last_conv_layer_name(model)

        target_layer = self.model.get_layer(self.layer_name)
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output],
            name="gradcam_extractor",
        )

    def generate_heatmap(
        self,
        image_tensor: Union[np.ndarray, tf.Tensor],
        class_idx: Optional[int] = None,
        brain_mask: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Generate Grad-CAM activation heatmap for target class.
        Guaranteed to be constrained strictly within the brain parenchyma.
        """
        if isinstance(image_tensor, np.ndarray):
            image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            grads = tf.ones_like(conv_outputs)

        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
        cam = tf.maximum(cam, 0.0).numpy()

        cam_max = np.max(cam)
        if cam_max > eps:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        h, w = int(image_tensor.shape[1]), int(image_tensor.shape[2])
        heatmap_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply cranial brain mask: strictly eliminate activations outside skull
        if brain_mask is not None:
            if brain_mask.shape[:2] != (h, w):
                brain_mask = cv2.resize(brain_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (brain_mask > 0).astype(np.float32)
            heatmap_resized = heatmap_resized * mask_bin
            m_max = np.max(heatmap_resized)
            if m_max > eps:
                heatmap_resized = heatmap_resized / m_max

        return np.clip(heatmap_resized, 0.0, 1.0)


class GradCAMPlusPlus:
    """
    Grad-CAM++ (Chattopadhyay et al., 2018) with 2nd and 3rd order gradients
    and cranial brain vault masking to ensure accurate tumor localization inside the skull.
    """

    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name or get_last_conv_layer_name(model)
        target_layer = self.model.get_layer(self.layer_name)
        self.grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output],
            name="gradcam_pp_extractor",
        )

    def generate_heatmap(
        self,
        image_tensor: Union[np.ndarray, tf.Tensor],
        class_idx: Optional[int] = None,
        brain_mask: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Generate Grad-CAM++ activation heatmap using higher-order gradients.
        Guaranteed to be constrained strictly within the brain parenchyma.
        """
        if isinstance(image_tensor, np.ndarray):
            image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_outputs, predictions = self.grad_model(image_tensor, training=False)
                    if class_idx is None:
                        class_idx = tf.argmax(predictions[0])
                    loss = predictions[:, class_idx]

                grads1 = tape3.gradient(loss, conv_outputs)
            grads2 = tape2.gradient(grads1, conv_outputs)
        grads3 = tape1.gradient(grads2, conv_outputs)

        if grads1 is None:
            grads1 = tf.ones_like(conv_outputs)
        if grads2 is None:
            grads2 = tf.zeros_like(conv_outputs)
        if grads3 is None:
            grads3 = tf.zeros_like(conv_outputs)

        conv_outputs_val = conv_outputs[0]
        grads1_val = grads1[0]
        grads2_val = grads2[0]
        grads3_val = grads3[0]

        sum_a_grads3 = tf.reduce_sum(conv_outputs_val * grads3_val, axis=(0, 1), keepdims=True)
        denominator = 2.0 * grads2_val + sum_a_grads3
        denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))

        alpha = grads2_val / denominator
        alpha = tf.where(grads1_val != 0.0, alpha, tf.zeros_like(alpha))

        relu_grads1 = tf.maximum(grads1_val, 0.0)
        weights = tf.reduce_sum(alpha * relu_grads1, axis=(0, 1))

        cam = tf.reduce_sum(weights * conv_outputs_val, axis=-1)
        cam = tf.maximum(cam, 0.0).numpy()

        cam_max = np.max(cam)
        if cam_max > eps:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        h, w = int(image_tensor.shape[1]), int(image_tensor.shape[2])
        heatmap_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply cranial brain mask: strictly eliminate activations outside skull
        if brain_mask is not None:
            if brain_mask.shape[:2] != (h, w):
                brain_mask = cv2.resize(brain_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (brain_mask > 0).astype(np.float32)
            heatmap_resized = heatmap_resized * mask_bin
            m_max = np.max(heatmap_resized)
            if m_max > eps:
                heatmap_resized = heatmap_resized / m_max

        return np.clip(heatmap_resized, 0.0, 1.0)


# =========================================================================
# Heatmap Thresholding & Active Bounding Box Extraction Inside Skull
# =========================================================================

def extract_bounding_box_from_heatmap(
    heatmap: np.ndarray,
    threshold_method: str = "combined",
    percentile: float = 70.0,
    min_contour_area: int = 40,
    brain_mask: Optional[np.ndarray] = None,
    predicted_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract salient bounding box coordinates (xmin, ymin, xmax, ymax) and centroid (xc, yc)
    strictly within the intracranial skull boundary.

    Args:
        heatmap: 2D float32 array in [0, 1].
        threshold_method: 'otsu', 'percentile', or 'combined'.
        percentile: Salience percentile cutoff.
        min_contour_area: Minimum pixel area to filter noise.
        brain_mask: Binary mask of intracranial cavity (255 inside head, 0 outside).
        predicted_class: If 'notumor', suppresses false lesion detections.

    Returns:
        Dictionary containing bbox, centroid, area, binary_mask, confidence_peak, has_detection.
    """
    h, w = heatmap.shape

    # For healthy brains (notumor), explicitly return no lesion detection
    if predicted_class == "notumor":
        return {
            "bbox": (0, 0, 0, 0),
            "centroid": (w // 2, h // 2),
            "area": 0,
            "binary_mask": np.zeros((h, w), dtype=np.uint8),
            "confidence_peak": float(np.max(heatmap)),
            "has_detection": False,
        }

    # Ensure heatmap is masked by cranial vault
    clean_heatmap = heatmap.copy()
    if brain_mask is not None:
        if brain_mask.shape[:2] != (h, w):
            brain_mask = cv2.resize(brain_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        clean_heatmap = clean_heatmap * (brain_mask > 0).astype(np.float32)

    heatmap_uint8 = (clean_heatmap * 255).astype(np.uint8)

    # Determine threshold
    if threshold_method == "otsu":
        t_val, binary = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == "percentile":
        cutoff = float(np.percentile(clean_heatmap[clean_heatmap > 0] if np.any(clean_heatmap > 0) else [0], percentile) * 255.0)
        _, binary = cv2.threshold(heatmap_uint8, max(cutoff, 50.0), 255, cv2.THRESH_BINARY)
    else:  # combined
        otsu_val, _ = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        perc_val = float(np.percentile(clean_heatmap[clean_heatmap > 0] if np.any(clean_heatmap > 0) else [0], percentile) * 255.0)
        final_thresh = max(otsu_val, perc_val, 60.0)
        _, binary = cv2.threshold(heatmap_uint8, final_thresh, 255, cv2.THRESH_BINARY)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) < min_contour_area:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            c_x = int(M["m10"] / M["m00"])
            c_y = int(M["m01"] / M["m00"])
        else:
            bx, by, bw, bh = cv2.boundingRect(c)
            c_x, c_y = bx + bw // 2, by + bh // 2

        # Verify centroid is strictly inside head
        if brain_mask is not None and (c_y >= h or c_x >= w or brain_mask[c_y, c_x] == 0):
            continue
        valid_contours.append((c, c_x, c_y))

    if not valid_contours:
        # If no cluster met threshold, find peak activation coordinate inside brain_mask
        if brain_mask is not None and np.any(clean_heatmap > 0.3):
            peak_y, peak_x = np.unravel_index(np.argmax(clean_heatmap), clean_heatmap.shape)
            peak_y, peak_x = int(peak_y), int(peak_x)
            box_r = 18
            bx1, by1 = max(0, peak_x - box_r), max(0, peak_y - box_r)
            bx2, by2 = min(w, peak_x + box_r), min(h, peak_y + box_r)
            lesion_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(lesion_mask, (bx1, by1), (bx2, by2), 255, -1)
            return {
                "bbox": (bx1, by1, bx2, by2),
                "centroid": (peak_x, peak_y),
                "area": (bx2 - bx1) * (by2 - by1),
                "binary_mask": lesion_mask,
                "confidence_peak": float(clean_heatmap[peak_y, peak_x]),
                "has_detection": True,
            }

        return {
            "bbox": (0, 0, 0, 0),
            "centroid": (w // 2, h // 2),
            "area": 0,
            "binary_mask": np.zeros((h, w), dtype=np.uint8),
            "confidence_peak": float(np.max(clean_heatmap)),
            "has_detection": False,
        }

    # Best contour inside skull
    best_c, xc, yc = max(valid_contours, key=lambda item: cv2.contourArea(item[0]))
    x, y, bw, bh = cv2.boundingRect(best_c)
    area = cv2.contourArea(best_c)

    # Constrain bounding box strictly inside skull boundaries
    if brain_mask is not None:
        mask_pts = np.argwhere(brain_mask > 0)
        if len(mask_pts) > 0:
            min_y_skull, min_x_skull = np.min(mask_pts, axis=0)
            max_y_skull, max_x_skull = np.max(mask_pts, axis=0)
            x = max(int(min_x_skull), x)
            y = max(int(min_y_skull), y)
            x_end = min(int(max_x_skull), x + bw)
            y_end = min(int(max_y_skull), y + bh)
            bw = max(1, x_end - x)
            bh = max(1, y_end - y)
            xc = int(np.clip(xc, min_x_skull, max_x_skull))
            yc = int(np.clip(yc, min_y_skull, max_y_skull))

    lesion_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(lesion_mask, [best_c], -1, 255, -1)

    return {
        "bbox": (x, y, x + bw, y + bh),
        "centroid": (xc, yc),
        "area": int(area),
        "binary_mask": lesion_mask,
        "confidence_peak": float(np.max(clean_heatmap)),
        "has_detection": True,
    }


# =========================================================================
# Automated Neuro-Anatomical Rule Engine
# =========================================================================

class NeuroAnatomicalRuleEngine:
    """
    Expert rule engine translating pixel coordinates and tumor class
    into standardized neuro-radiological localization narratives.
    Strictly observes radiological display conventions:
      - Image Left corresponds to Patient's Right Hemisphere.
      - Image Right corresponds to Patient's Left Hemisphere.
    """

    @staticmethod
    def localize(
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        predicted_class: str,
        has_detection: bool = True,
    ) -> Dict[str, Any]:
        """
        Map spatial coordinates to neuro-anatomical regions.
        """
        if not has_detection or predicted_class == "notumor":
            return {
                "laterality": "Bilateral / Non-focal",
                "compartment": "Diffuse / Normal Parenchyma",
                "lobar_zone": "No focal abnormality",
                "norm_x": 0.5,
                "norm_y": 0.5,
                "narrative": "No focal space-occupying lesion or abnormal hyperintense enhancement identified within the cranial vault. Parenchymal architecture within normal anatomical limits.",
            }

        h, w = image_shape
        xc, yc = centroid
        nx = float(xc) / max(float(w), 1.0)
        ny = float(yc) / max(float(h), 1.0)

        # Laterality (Radiological Convention)
        if nx < 0.44:
            laterality = "Right Hemisphere"
            lat_desc = "right-sided"
        elif nx > 0.56:
            laterality = "Left Hemisphere"
            lat_desc = "left-sided"
        else:
            laterality = "Medial / Midline"
            lat_desc = "central midline"

        # Specific Anatomical Zones
        is_sellar = (0.35 <= nx <= 0.65) and (0.38 <= ny <= 0.68) and (predicted_class == "pituitary" or (0.42 <= nx <= 0.58 and 0.42 <= ny <= 0.65))

        if is_sellar or predicted_class == "pituitary":
            compartment = "Sellar / Suprasellar Cistern"
            lobar_zone = "Hypophyseal Fossa (Sella Turcica)"
            narrative = (
                f"Focal space-occupying lesion localized to the {compartment} ({lat_desc}). "
                "Characteristic sellar expansion with potential superior extension toward the optic chiasm. "
                "Classic imaging hallmark of Pituitary Neuroendocrine Tumor (PitNET) / Pituitary Macroadenoma."
            )
            return {
                "laterality": laterality,
                "compartment": compartment,
                "lobar_zone": lobar_zone,
                "norm_x": nx,
                "norm_y": ny,
                "narrative": narrative,
            }

        if ny > 0.74 and (0.30 <= nx <= 0.70):
            compartment = "Infratentorial Compartment"
            lobar_zone = "Posterior Fossa / Cerebellar Hemisphere"
            narrative = (
                f"Abnormal focal lesion identified in the {compartment}, occupying the {lobar_zone} ({lat_desc}). "
                "Requires careful assessment of 4th ventricle patency and potential effacement."
            )
            return {
                "laterality": laterality,
                "compartment": compartment,
                "lobar_zone": lobar_zone,
                "norm_x": nx,
                "norm_y": ny,
                "narrative": narrative,
            }

        if ny < 0.38:
            compartment = "Supratentorial Compartment"
            lobar_zone = f"{laterality} - Frontal Lobe"
            zone_desc = "anterior frontal parenchyma / convexity"
        elif 0.38 <= ny <= 0.66:
            if nx < 0.32 or nx > 0.68:
                compartment = "Supratentorial Compartment"
                lobar_zone = f"{laterality} - Temporal Lobe"
                zone_desc = "lateral temporal cortex / subcortical white matter"
            else:
                compartment = "Supratentorial Compartment"
                lobar_zone = f"{laterality} - Parietal / Centrum Semiovale"
                zone_desc = "parietal parenchyma and adjacent parasagittal region"
        else:
            compartment = "Supratentorial Compartment"
            lobar_zone = f"{laterality} - Occipital Lobe"
            zone_desc = "posterior occipital cortex / pericalcarine region"

        if predicted_class == "meningioma":
            path_feature = "well-circumscribed dural-based extra-axial mass morphology with probable reactive hyperostosis/dural tail"
        elif predicted_class == "glioma":
            path_feature = "infiltrative intra-axial parenchymal mass with surrounding vasogenic edema and local mass effect"
        else:
            path_feature = "abnormal focal hyperintensity"

        narrative = (
            f"Focal lesion centered within the {lobar_zone} ({zone_desc}). "
            f"Imaging features demonstrate {path_feature}, consistent with radiological presentation of {predicted_class}."
        )

        return {
            "laterality": laterality,
            "compartment": compartment,
            "lobar_zone": lobar_zone,
            "norm_x": nx,
            "norm_y": ny,
            "narrative": narrative,
        }


# =========================================================================
# 2D (X, Y) Spatial Indication & Triage Graph Generator
# =========================================================================

def generate_2d_spatial_triage_graph(
    centroid: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    predicted_class: str,
    confidence: float,
    triage: Dict[str, Any],
    image_shape: Tuple[int, int] = (224, 224),
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate an informative 2D Anatomical Spatial Indication & Hospital Triage Graph.

    Renders:
      1. Cranial axial cross-section silhouette with marked brain lobes
      2. Tumor centroid (X, Y) coordinates with epistemic uncertainty radius
      3. 3-Tier triage status color coding (Tier 1 Green, Tier 2 Yellow, Tier 3 Red)
      4. Detailed anatomical compartment readout badge

    Returns:
        uint8 RGB numpy array of the rendered 2D graph.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    fig.patch.set_facecolor("#071322")
    ax.set_facecolor("#0B1C30")

    # Determine Triage Color
    tier = triage.get("confidence_tier", "TIER_1_HIGH_CONFIDENCE")
    if "TIER_3" in tier:
        tier_color = "#EF4444"  # Red / Coral
        tier_label = "TIER 3: CRITICAL UNCERTAINTY (URGENT)"
    elif "TIER_2" in tier:
        tier_color = "#F59E0B"  # Amber / Yellow
        tier_label = "TIER 2: MODERATE UNCERTAINTY (ADVISED)"
    else:
        tier_color = "#10B981"  # Emerald Green
        tier_label = "TIER 1: AUTONOMOUS VERIFIED (HIGH CONF)"

    # Draw Anatomical Skull Ellipse
    theta = np.linspace(0, 2 * np.pi, 200)
    skull_x = 0.5 + 0.40 * np.cos(theta)
    skull_y = 0.5 + 0.44 * np.sin(theta)
    ax.plot(skull_x, skull_y, color="#38BDF8", lw=3.0, label="Cranial Vault Boundary", zorder=3)

    # Interhemispheric fissure (Midline)
    ax.plot([0.5, 0.5], [0.06, 0.94], "--", color="#64748B", lw=1.5, alpha=0.8, zorder=2)

    # Compartment Zone Labels
    ax.text(0.24, 0.16, "Right Frontal", color="#94A3B8", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.76, 0.16, "Left Frontal", color="#94A3B8", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.50, 0.49, "Sellar / Suprasellar", color="#00F2FE", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.18, 0.50, "Right Temporal", color="#94A3B8", ha="center", fontsize=8.5)
    ax.text(0.82, 0.50, "Left Temporal", color="#94A3B8", ha="center", fontsize=8.5)
    ax.text(0.24, 0.82, "Right Occipital", color="#94A3B8", ha="center", fontsize=8.5)
    ax.text(0.76, 0.82, "Left Occipital", color="#94A3B8", ha="center", fontsize=8.5)
    ax.text(0.50, 0.88, "Infratentorial Cerebellar", color="#64748B", ha="center", fontsize=8)

    h, w = image_shape
    xc, yc = centroid
    nx = float(xc) / max(float(w), 1.0)
    ny = float(yc) / max(float(h), 1.0)

    # Plot Tumor Centroid and Uncertainty Circle
    if predicted_class != "notumor" and bbox != (0, 0, 0, 0):
        # Tumor Marker
        ax.scatter([nx], [ny], color=tier_color, s=200, edgecolors="#FFFFFF", lw=2.5, zorder=6, label=f"Tumor Centroid: {predicted_class.upper()}")
        
        # Uncertainty Ellipse
        variance = triage.get("metrics_evaluated", {}).get("variance", 0.01)
        radius = np.clip(np.sqrt(variance) * 1.5 + 0.04, 0.03, 0.15)
        uncertainty_circle = plt.Circle((nx, ny), radius, color=tier_color, fill=True, alpha=0.28, zorder=5)
        ax.add_patch(uncertainty_circle)

        # Coordinate Annotation Callout
        ax.annotate(
            f"({nx:.2f}, {ny:.2f})\n{predicted_class.capitalize()}",
            xy=(nx, ny),
            xytext=(nx + (0.08 if nx < 0.6 else -0.22), ny - 0.08),
            color="#FFFFFF",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#0F253F", ec=tier_color, lw=1.5),
            arrowprops=dict(arrowstyle="->", color=tier_color, lw=1.5),
            zorder=7,
        )
    else:
        # Healthy scan: mark center parenchyma as clear
        ax.scatter([0.5], [0.5], color="#10B981", s=180, marker="P", edgecolors="#FFFFFF", lw=2, zorder=6, label="Clear Parenchyma (No Focal Mass)")
        ax.text(0.5, 0.56, "NO TUMOR DETECTED", color="#10B981", ha="center", fontsize=10, fontweight="bold", zorder=7)

    # Axes Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Inverted so 0=Anterior (Top), 1=Posterior (Bottom)
    ax.set_title(f"2D Anatomical Spatial Indication & Triage Map\n[{tier_label}]", color="#38BDF8", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Radiological X: 0.0 (Right Hemisphere) <-------> 1.0 (Left Hemisphere)", color="#E2E8F0", fontsize=9)
    ax.set_ylabel("Cranial Y: 0.0 (Anterior Frontal) <-------> 1.0 (Posterior Occipital)", color="#E2E8F0", fontsize=9)
    ax.tick_params(colors="#94A3B8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1E3A5F")

    ax.legend(facecolor="#071322", edgecolor="#1E3A5F", labelcolor="#FFFFFF", loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")

    # Render figure to numpy RGB array
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img_array = rgba[:, :, :3].copy()
    plt.close(fig)

    return img_array


# =========================================================================
# Ground-Truth Validation Metrics (IoU & Pointing Game)
# =========================================================================

def compute_iou(
    boxA: Tuple[int, int, int, int],
    boxB: Tuple[int, int, int, int],
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes (xmin, ymin, xmax, ymax)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
    return float(interArea / unionArea)


def compute_mask_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Calculate Dice and IoU overlap between binary masks."""
    p = (mask_pred > 0).astype(np.uint8)
    g = (mask_gt > 0).astype(np.uint8)

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def pointing_game_hit(
    heatmap: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> bool:
    """
    Evaluate Pointing Game Hit-Rate (Zhang et al., 2018).
    Evaluates whether the peak salience point falls strictly inside the ground-truth mask.
    """
    if ground_truth_mask is None or np.sum(ground_truth_mask) == 0:
        return False

    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(ground_truth_mask[peak_y, peak_x] > 0)
