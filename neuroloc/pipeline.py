"""
End-to-End Diagnostic Pipeline & Visualization Generator
Integrates data preprocessing, Monte Carlo Dropout inference, epistemic uncertainty
quantification, cranial-constrained Grad-CAM++ explainability, neuro-anatomical localization,
2D (X, Y) spatial indication & triage graph, and multi-panel clinical diagnostic reporting.
"""

from typing import Dict, Any, Optional, Union, Tuple
import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from neuroloc.dataset import (
    CLASS_NAMES,
    preprocess_mri_image,
    ben_graham_skull_crop,
    apply_clahe,
    extract_cranial_brain_mask,
)
from neuroloc.model import build_neuroloc_model, get_last_conv_layer_name, MCDropout
from neuroloc.uncertainty import compute_epistemic_uncertainty, triage_clinical_case
from neuroloc.explainability import (
    GradCAM,
    GradCAMPlusPlus,
    extract_bounding_box_from_heatmap,
    NeuroAnatomicalRuleEngine,
    generate_2d_spatial_triage_graph,
)
from neuroloc.train_eval import format_clinical_report

logger = logging.getLogger(__name__)


class NeuroLocPipeline:
    """
    Unified end-to-end diagnostic pipeline for explainable brain tumor diagnosis.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights: Optional[str] = "imagenet",
        backbone: str = "efficientnetb0",
        target_layer_name: Optional[str] = None,
    ):
        """
        Initialize NeuroLoc pipeline.
        """
        if model_path and os.path.exists(model_path):
            logger.info("Loading trained NeuroLoc model from '%s'...", model_path)
            custom_objects = {"MCDropout": MCDropout}
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            logger.info("Instantiating NeuroLoc model with %s backbone...", backbone)
            self.model = build_neuroloc_model(
                input_shape=(224, 224, 3),
                num_classes=len(CLASS_NAMES),
                backbone_name=backbone,
                dropout_rate=0.3,
                weights=weights,
            )

        self.target_layer_name = target_layer_name or get_last_conv_layer_name(self.model)
        logger.info("Target convolutional layer for XAI: '%s'", self.target_layer_name)

        # Initialize XAI engines
        self.gradcam = GradCAM(self.model, layer_name=self.target_layer_name)
        self.gradcam_pp = GradCAMPlusPlus(self.model, layer_name=self.target_layer_name)
        self.rule_engine = NeuroAnatomicalRuleEngine()

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        num_mc_passes: int = 30,
        xai_method: str = "gradcam++",
        patient_id: str = "SCAN-ANON-7842",
        variance_threshold: float = 0.015,
        entropy_threshold: float = 0.60,
        confidence_threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Execute full clinical diagnostic forward pass:
          1. Load & preprocess scan (skull crop + CLAHE + resize in range [0, 255]).
          2. Extract cranial brain vault mask to strictly constrain localization inside head.
          3. Compute Bayesian Epistemic Uncertainty via T stochastic forward passes.
          4. Evaluate automated clinical triage logic.
          5. Compute skull-constrained Grad-CAM / Grad-CAM++ activation heatmap.
          6. Extract lesion bounding box coordinates and centroid strictly inside head.
          7. Map coordinates to anatomical regions and generate 2D spatial triage graph.
          8. Synthesize standardized clinical radiology report.

        Returns:
            Comprehensive diagnosis dictionary.
        """
        # 1. Load original image
        if isinstance(image_input, str):
            orig_bgr = cv2.imread(image_input)
            if orig_bgr is None:
                raise FileNotFoundError(f"Cannot load image: {image_input}")
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = image_input.copy()
            if len(orig_rgb.shape) == 2:
                orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_GRAY2RGB)

        # 2. Preprocess scan: maintain float in [0, 255] for EfficientNet internal rescaling
        preprocessed = preprocess_mri_image(
            orig_rgb,
            target_size=(224, 224),
            apply_crop=True,
            apply_contrast=True,
            normalize=False,
        )
        preprocessed_uint8 = np.clip(preprocessed, 0, 255).astype(np.uint8)

        # Extract cranial brain mask: 255 inside head, 0 outside
        brain_mask = extract_cranial_brain_mask(preprocessed_uint8)

        # 3. Epistemic Uncertainty via Monte Carlo Dropout
        uncertainty = compute_epistemic_uncertainty(
            self.model,
            preprocessed,
            num_passes=num_mc_passes,
        )

        pred_idx = uncertainty["predicted_class_idx"]
        pred_class = uncertainty["predicted_class_name"]
        pred_conf = uncertainty["predicted_confidence"]
        pred_var = uncertainty["predicted_class_variance"]

        # 4. Clinical Triage Evaluation
        triage = triage_clinical_case(
            uncertainty,
            variance_threshold=variance_threshold,
            entropy_threshold=entropy_threshold,
            confidence_threshold=confidence_threshold,
        )

        # 5. Skull-Constrained Explainability (Grad-CAM / Grad-CAM++)
        input_tensor = np.expand_dims(preprocessed, axis=0)
        if xai_method.lower() == "gradcam++":
            heatmap = self.gradcam_pp.generate_heatmap(input_tensor, class_idx=pred_idx, brain_mask=brain_mask)
        else:
            heatmap = self.gradcam.generate_heatmap(input_tensor, class_idx=pred_idx, brain_mask=brain_mask)

        # 6. Salient Bounding Box & Centroid Extraction strictly inside skull
        bbox_data = extract_bounding_box_from_heatmap(
            heatmap,
            threshold_method="combined",
            percentile=70.0,
            brain_mask=brain_mask,
            predicted_class=pred_class,
        )
        bbox = bbox_data["bbox"]
        centroid = bbox_data["centroid"]
        has_detection = bbox_data["has_detection"]

        # 7. Neuro-Anatomical Rule Engine Localization
        anat_loc = self.rule_engine.localize(
            centroid=centroid,
            bbox=bbox,
            image_shape=(224, 224),
            predicted_class=pred_class,
            has_detection=has_detection,
        )

        # 8. Create Diagnostic Overlay Image
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(preprocessed_uint8, 0.65, heatmap_colored_rgb, 0.35, 0)

        # Draw bounding box and centroid if lesion detected
        if has_detection and pred_class != "notumor":
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.circle(overlay, centroid, 4, (255, 0, 0), -1)
            label_txt = f"{pred_class.upper()} ({pred_conf:.1%})"
            cv2.putText(overlay, label_txt, (xmin, max(ymin - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

        # 9. Generate 2D Anatomical Spatial Indication & Triage Graph
        spatial_2d_graph = generate_2d_spatial_triage_graph(
            centroid=centroid,
            bbox=bbox,
            predicted_class=pred_class,
            confidence=pred_conf,
            triage=triage,
            image_shape=(224, 224),
        )

        # 10. Assemble results
        result = {
            "patient_id": patient_id,
            "predicted_class": pred_class,
            "predicted_class_idx": pred_idx,
            "confidence": pred_conf,
            "epistemic_variance": pred_var,
            "mean_probabilities": uncertainty["mean_probabilities"],
            "predictive_entropy": uncertainty["predictive_entropy"],
            "expected_entropy": uncertainty["expected_entropy"],
            "mutual_information": uncertainty["mutual_information"],
            "credible_interval_95": uncertainty["credible_interval_95"],
            "mc_samples": uncertainty["mc_samples"],
            "triage": triage,
            "heatmap": heatmap,
            "brain_mask": brain_mask,
            "bbox": bbox,
            "centroid": centroid,
            "has_detection": has_detection,
            "anatomical_localization": anat_loc,
            "original_image": orig_rgb,
            "preprocessed_image": preprocessed_uint8,
            "overlay_image": overlay,
            "spatial_2d_graph": spatial_2d_graph,
            "xai_method": xai_method,
        }

        # 11. Generate clinical text report
        report_str = format_clinical_report(result)
        result["clinical_report"] = report_str

        return result

    def visualize(
        self,
        result: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Generate publication-ready 6-panel clinical neuro-radiology figure:
          Panel 1: Original MRI Scan
          Panel 2: Skull-Cropped MRI + Cranial Vault Mask
          Panel 3: Skull-Constrained Grad-CAM++ Salience Heatmap
          Panel 4: Anatomical Localization Overlay (BBox + Centroid strictly inside head)
          Panel 5: 2D Anatomical Spatial Indication & Hospital Triage Graph
          Panel 6: Bayesian Monte Carlo Posterior Distribution (T=30 Samples)
        """
        fig = plt.figure(figsize=(24, 4.5), dpi=150)
        fig.patch.set_facecolor("#121212")
        text_color = "#E0E0E0"

        # Panel 1: Original MRI
        ax1 = fig.add_subplot(1, 6, 1)
        ax1.imshow(result["original_image"])
        ax1.set_title("1. Original MRI Scan", color=text_color, fontsize=10, fontweight="bold", pad=8)
        ax1.axis("off")

        # Panel 2: Preprocessed MRI + Brain Mask
        ax2 = fig.add_subplot(1, 6, 2)
        ax2.imshow(result["preprocessed_image"])
        # Overlay cranial contour
        contours, _ = cv2.findContours(result["brain_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c.squeeze()
            if pts.ndim == 2:
                ax2.plot(pts[:, 0], pts[:, 1], color="#38BDF8", lw=1.2, linestyle="--")
        ax2.set_title("2. Skull-Crop & Vault", color=text_color, fontsize=10, fontweight="bold", pad=8)
        ax2.axis("off")

        # Panel 3: Skull-Constrained Heatmap
        ax3 = fig.add_subplot(1, 6, 3)
        im3 = ax3.imshow(result["heatmap"], cmap="jet")
        method_name = result.get("xai_method", "Grad-CAM++").upper()
        ax3.set_title(f"3. {method_name} (In-Head)", color=text_color, fontsize=10, fontweight="bold", pad=8)
        ax3.axis("off")
        cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_color, fontsize=7)

        # Panel 4: Diagnostic Overlay
        ax4 = fig.add_subplot(1, 6, 4)
        ax4.imshow(result["overlay_image"])
        pred_cls = result["predicted_class"].upper()
        conf = result["confidence"]
        ax4.set_title(f"4. BBox Localization\n[{pred_cls}: {conf:.1%}]", color=text_color, fontsize=9.5, fontweight="bold", pad=6)
        ax4.axis("off")

        # Panel 5: 2D Spatial Indication & Triage Graph
        ax5 = fig.add_subplot(1, 6, 5)
        ax5.imshow(result["spatial_2d_graph"])
        ax5.set_title("5. 2D Spatial & Triage Map", color=text_color, fontsize=10, fontweight="bold", pad=8)
        ax5.axis("off")

        # Panel 6: Monte Carlo Posterior Distributions
        ax6 = fig.add_subplot(1, 6, 6)
        ax6.set_facecolor("#1E1E1E")
        mc_samples = result["mc_samples"]
        means = result["mean_probabilities"]
        colors = ["#38BDF8", "#0EA5E9", "#10B981", "#00F2FE"]

        x_pos = np.arange(len(CLASS_NAMES))
        bars = ax6.bar(x_pos, means * 100, color=colors, alpha=0.85, edgecolor="#E0E0E0", width=0.55)
        bars[result["predicted_class_idx"]].set_edgecolor("#00F2FE")
        bars[result["predicted_class_idx"]].set_linewidth(2.0)

        # Scatter individual MC iterations
        for i in range(len(CLASS_NAMES)):
            y_scatter = mc_samples[:, i] * 100
            x_scatter = np.random.normal(i, 0.05, size=len(y_scatter))
            ax6.scatter(x_scatter, y_scatter, color="#FFFFFF", alpha=0.35, s=9, zorder=3)

        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([c.capitalize() for c in CLASS_NAMES], rotation=30, ha="right", color=text_color, fontsize=8)
        ax6.set_ylabel("Posterior Probability (%)", color=text_color, fontsize=8)
        ax6.set_ylim(0, 105)
        ax6.set_title(f"6. Bayesian MC (T=30)\n[sigma^2 = {result['epistemic_variance']:.4f}]", color=text_color, fontsize=9.5, fontweight="bold", pad=6)
        ax6.grid(axis="y", linestyle="--", alpha=0.25, color="#666666")
        ax6.tick_params(axis="y", colors=text_color, labelsize=7)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=150)
            logger.info("Saved diagnostic figure to: %s", save_path)

        if show:
            plt.show()

        return fig
