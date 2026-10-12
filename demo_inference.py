"""
NeuroLoc-XAI: Example Standalone Inference & Clinical Reporting Script
Demonstrates an end-to-end forward pass:
  1. Input MRI ingestion (real or synthetic)
  2. Skull contour cropping & CLAHE enhancement
  3. Bayesian Monte Carlo Dropout uncertainty estimation (T=30 passes)
  4. Clinical triage decision
  5. Grad-CAM++ anatomical localization & bounding box extraction
  6. Clinical Neuro-Radiology Summary Report generation
  7. Multi-panel diagnostic visualization
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NeuroLoc-Demo")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from neuroloc.pipeline import NeuroLocPipeline
from neuroloc.utils import create_synthetic_mri
from neuroloc.dataset import CLASS_NAMES


def find_sample_scan(target_class: str = "pituitary") -> str:
    """Find a real sample image from Testing or Training directory."""
    for root in ["Testing", "Training"]:
        cls_dir = os.path.join(root, target_class)
        if os.path.isdir(cls_dir):
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                matches = glob.glob(os.path.join(cls_dir, ext))
                if matches:
                    return matches[0]
    return None


def run_inference_demo(
    image_path: str = None,
    tumor_type: str = "pituitary",
    num_mc_passes: int = 30,
    xai_method: str = "gradcam++",
    save_plot_path: str = "reports/Result_1/Result_1.png",
    model_checkpoint: str = "checkpoints/neuroloc_best.keras",
):
    """
    Execute full inference pipeline and render visualization.
    """
    logger.info("Initializing NeuroLoc-XAI Diagnostic Pipeline...")
    pipeline = NeuroLocPipeline(
        model_path=model_checkpoint,
        weights="imagenet" if model_checkpoint is None else None,
        backbone="efficientnetb0",
    )

    # Resolve input image
    sample_gt = None
    if image_path is None:
        # Check if real scan exists in local dataset
        real_scan_path = find_sample_scan(tumor_type)
        if real_scan_path and os.path.exists(real_scan_path):
            logger.info("Using real MRI scan from local dataset: %s", real_scan_path)
            input_data = real_scan_path
            patient_id = f"PATIENT-{os.path.basename(real_scan_path).split('.')[0].upper()}"
        else:
            logger.info("Generating synthetic brain MRI scan (%s) with realistic lesion...", tumor_type)
            synthetic_bgr, meta = create_synthetic_mri(tumor_type=tumor_type, image_size=(224, 224))
            input_data = synthetic_bgr
            sample_gt = meta
            patient_id = f"SYNTH-CASE-{tumor_type.upper()}-001"
    else:
        logger.info("Using provided image path: %s", image_path)
        input_data = image_path
        patient_id = f"PATIENT-{os.path.basename(image_path).split('.')[0].upper()}"

    # Execute end-to-end diagnostic forward pass
    logger.info("Executing Monte Carlo Dropout Bayesian forward passes (T=%d)...", num_mc_passes)
    result = pipeline.predict(
        image_input=input_data,
        num_mc_passes=num_mc_passes,
        xai_method=xai_method,
        patient_id=patient_id,
        variance_threshold=0.015,
        entropy_threshold=0.60,
        confidence_threshold=0.70,
    )

    # Print clinical report to terminal
    print(result["clinical_report"])

    # Output directory
    out_dir = os.path.dirname(os.path.abspath(save_plot_path))
    os.makedirs(out_dir, exist_ok=True)

    # Save multi-panel clinical figure
    logger.info("Generating multi-panel diagnostic figure...")
    pipeline.visualize(result, save_path=save_plot_path, show=False)
    logger.info("Report visualization successfully saved to: %s", save_plot_path)

    # Save accompanying report text and spatial graph
    base_stem = os.path.splitext(os.path.basename(save_plot_path))[0]
    txt_path = os.path.join(out_dir, f"{base_stem}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["clinical_report"])

    # Also save standalone spatial graph and heatmap overlay if available
    import cv2
    if result.get("spatial_2d_graph") is not None and result["spatial_2d_graph"].size > 0:
        cv2.imwrite(os.path.join(out_dir, "spatial_2d_graph.png"), cv2.cvtColor(result["spatial_2d_graph"], cv2.COLOR_RGB2BGR))
    if result.get("overlay") is not None and result["overlay"].size > 0:
        cv2.imwrite(os.path.join(out_dir, "heatmap_overlay.png"), cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))

    return result


def main():
    parser = argparse.ArgumentParser(description="NeuroLoc-XAI Standalone Inference Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input MRI scan (optional)")
    parser.add_argument("--class-name", type=str, default="pituitary", choices=CLASS_NAMES, help="Tumor class for demo scan")
    parser.add_argument("--mc-passes", type=int, default=30, help="Number of Monte Carlo forward passes")
    parser.add_argument("--xai", type=str, default="gradcam++", choices=["gradcam++", "gradcam"], help="Explainability algorithm")
    parser.add_argument("--output", type=str, default="reports/Result_1/Result_1.png", help="Output figure path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/neuroloc_best.keras", help="Path to trained model checkpoint")

    args = parser.parse_args()
    run_inference_demo(
        image_path=args.image,
        tumor_type=args.class_name,
        num_mc_passes=args.mc_passes,
        xai_method=args.xai,
        save_plot_path=args.output,
        model_checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
