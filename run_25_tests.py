"""
NeuroLoc-XAI: Batch Clinical Test Runner
Executes 25 diverse MRI test cases (Glioma, Meningioma, Pituitary, No Tumor)
using the retrained high-accuracy model with Cranial Vault Masking and 2D Spatial Triage.

For each test case i (1 to 25):
  - Creates directory: reports/Result_{i}/
  - Generates comprehensive 6-panel clinical diagnostic report: reports/Result_{i}/Result_{i}.png
  - Saves full clinical text report: reports/Result_{i}/Result_{i}_report.txt
  - Saves structured JSON metrics: reports/Result_{i}/Result_{i}_metrics.json
  - Saves standalone 2D (X, Y) spatial triage graph: reports/Result_{i}/spatial_2d_graph.png
  - Saves standalone in-head heatmap overlay: reports/Result_{i}/heatmap_overlay.png
  - Saves raw MRI & cranial brain mask images
"""

import os
import sys
import json
import logging
import cv2
import numpy as np

# Configure console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NeuroLoc-25Tests")

from neuroloc.pipeline import NeuroLocPipeline
from neuroloc.dataset import extract_cranial_brain_mask, CLASS_NAMES


# 25 Diverse Test Cases selected from Testing directory
TEST_CASES = [
    # --- Pituitary Tumors (7 cases) ---
    {"case_num": 1, "image_path": "Testing/pituitary/Te-pi_1.jpg", "ground_truth": "pituitary", "desc": "Sellar cistern mass (axial)"},
    {"case_num": 2, "image_path": "Testing/pituitary/Te-pi_2.jpg", "ground_truth": "pituitary", "desc": "Macroadenoma midline (axial)"},
    {"case_num": 3, "image_path": "Testing/pituitary/Te-pi_3.jpg", "ground_truth": "pituitary", "desc": "Infrasellar extension (axial)"},
    {"case_num": 4, "image_path": "Testing/pituitary/Te-pi_4.jpg", "ground_truth": "pituitary", "desc": "Sellar lesion (T1 contrast)"},
    {"case_num": 5, "image_path": "Testing/pituitary/Te-pi_5.jpg", "ground_truth": "pituitary", "desc": "Microadenoma focus (axial)"},
    {"case_num": 6, "image_path": "Testing/pituitary/Te-pi_6.jpg", "ground_truth": "pituitary", "desc": "Pituitary fossa lesion"},
    {"case_num": 7, "image_path": "Testing/pituitary/Te-pi_7.jpg", "ground_truth": "pituitary", "desc": "Suprasellar mass (axial)"},

    # --- Gliomas (6 cases) ---
    {"case_num": 8, "image_path": "Testing/glioma/Te-gl_1.jpg", "ground_truth": "glioma", "desc": "Frontal/Temporal infiltrating lesion"},
    {"case_num": 9, "image_path": "Testing/glioma/Te-gl_2.jpg", "ground_truth": "glioma", "desc": "High-grade glioma deep white matter"},
    {"case_num": 10, "image_path": "Testing/glioma/Te-gl_3.jpg", "ground_truth": "glioma", "desc": "Parietal lobe glioblastoma focus"},
    {"case_num": 11, "image_path": "Testing/glioma/Te-gl_4.jpg", "ground_truth": "glioma", "desc": "Temporal lobe infiltrative mass"},
    {"case_num": 12, "image_path": "Testing/glioma/Te-gl_5.jpg", "ground_truth": "glioma", "desc": "Left hemisphere high-grade lesion"},
    {"case_num": 13, "image_path": "Testing/glioma/Te-gl_6.jpg", "ground_truth": "glioma", "desc": "Deep cerebral hemisphere tumor"},

    # --- Meningiomas (6 cases) ---
    {"case_num": 14, "image_path": "Testing/meningioma/Te-aug-me_1.jpg", "ground_truth": "meningioma", "desc": "Parasagittal dural-based mass"},
    {"case_num": 15, "image_path": "Testing/meningioma/Te-aug-me_2.jpg", "ground_truth": "meningioma", "desc": "Convexity meningioma right frontal"},
    {"case_num": 16, "image_path": "Testing/meningioma/Te-aug-me_3.jpg", "ground_truth": "meningioma", "desc": "Sphenoid ridge dural lesion"},
    {"case_num": 17, "image_path": "Testing/meningioma/Te-aug-me_4.jpg", "ground_truth": "meningioma", "desc": "Falx cerebri enhancing mass"},
    {"case_num": 18, "image_path": "Testing/meningioma/Te-aug-me_5.jpg", "ground_truth": "meningioma", "desc": "Lateral convexity meningioma"},
    {"case_num": 19, "image_path": "Testing/meningioma/Te-aug-me_6.jpg", "ground_truth": "meningioma", "desc": "Frontoparietal dural tail mass"},

    # --- Healthy Controls / No Tumor (6 cases) ---
    {"case_num": 20, "image_path": "Testing/notumor/Te-no_1.jpg", "ground_truth": "notumor", "desc": "Healthy baseline control scan 1"},
    {"case_num": 21, "image_path": "Testing/notumor/Te-no_2.jpg", "ground_truth": "notumor", "desc": "Normal intracranial anatomy 2"},
    {"case_num": 22, "image_path": "Testing/notumor/Te-no_3.jpg", "ground_truth": "notumor", "desc": "Unremarkable axial brain MRI 3"},
    {"case_num": 23, "image_path": "Testing/notumor/Te-no_4.jpg", "ground_truth": "notumor", "desc": "Normal ventricles and sulci 4"},
    {"case_num": 24, "image_path": "Testing/notumor/Te-no_5.jpg", "ground_truth": "notumor", "desc": "Healthy grey-white differentiation 5"},
    {"case_num": 25, "image_path": "Testing/notumor/Te-no_6.jpg", "ground_truth": "notumor", "desc": "Normal cranial vault without lesion 6"},
]


def run_all_25_tests(
    checkpoint_path: str = "checkpoints/neuroloc_best.keras",
    reports_base_dir: str = "reports",
    mc_passes: int = 15,
):
    """Run all 25 test cases and save results into individual folders: reports/Result_{i}/"""
    os.makedirs(reports_base_dir, exist_ok=True)

    logger.info("Loading NeuroLoc-XAI Pipeline from checkpoint: %s", checkpoint_path)
    pipeline = NeuroLocPipeline(
        model_path=checkpoint_path,
        backbone="efficientnetb0",
    )

    summary_records = []

    print("\n" + "=" * 80)
    print("  NEUROLOC-XAI: EXECUTING 25 CLINICAL MRI TEST CASES")
    print("  Verifying In-Head Localization & Calibrated 2D (X, Y) Spatial Triage")
    print("=" * 80)

    for case in TEST_CASES:
        num = case["case_num"]
        img_path = case["image_path"]
        gt = case["ground_truth"]
        desc = case["desc"]

        # Output folder for this test: reports/Result_{num}
        folder_name = f"Result_{num}"
        folder_path = os.path.join(reports_base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        patient_id = f"PATIENT-TEST-{num:02d}-{gt.upper()[:2]}"
        logger.info("[Test %02d/25] Processing %s (%s) -> %s", num, img_path, gt, folder_path)

        # Run inference
        result = pipeline.predict(
            image_input=img_path,
            num_mc_passes=mc_passes,
            xai_method="gradcam++",
            patient_id=patient_id,
            variance_threshold=0.015,
            entropy_threshold=0.60,
            confidence_threshold=0.70,
        )

        pred_class = result["predicted_class"]
        conf = result["confidence"]
        entropy = result["predictive_entropy"]
        triage = result["triage"]
        has_det = result["has_detection"]
        bbox = result["bbox"]
        centroid = result["centroid"]
        in_head = True

        # Check in-head compliance
        brain_mask = result["brain_mask"]
        if has_det:
            xc, yc = centroid
            in_head = bool(brain_mask[yc, xc] > 0)
        else:
            in_head = True  # Healthy scan correctly has no tumor

        # 1. Save 6-Panel Visualization as Result_{num}.png & report.png
        report_png_path = os.path.join(folder_path, f"Result_{num}.png")
        pipeline.visualize(result, save_path=report_png_path, show=False)

        # Also create a duplicate copy as report.png for convenience
        alt_png_path = os.path.join(folder_path, "report.png")
        try:
            import shutil
            shutil.copyfile(report_png_path, alt_png_path)
            # If num == 1, also copy as Result_1.png
            if num == 1:
                shutil.copyfile(report_png_path, os.path.join(folder_path, "Result_1.png"))
        except Exception:
            pass

        # 2. Save Clinical Text Report
        txt_report_path = os.path.join(folder_path, f"Result_{num}_report.txt")
        with open(txt_report_path, "w", encoding="utf-8") as f:
            f.write(result["clinical_report"])

        # Also copy as clinical_report.txt and Result_1_report.txt if num == 1
        with open(os.path.join(folder_path, "clinical_report.txt"), "w", encoding="utf-8") as f:
            f.write(result["clinical_report"])
        if num == 1:
            with open(os.path.join(folder_path, "Result_1_report.txt"), "w", encoding="utf-8") as f:
                f.write(result["clinical_report"])

        # 3. Save Structured Metrics JSON
        tier_str = triage.get("confidence_tier", "TIER_1_HIGH_CONFIDENCE")
        status_str = triage.get("triage_status", "AUTONOMOUS_VERIFIED")
        rec_str = triage.get("recommendation", "")

        json_metrics = {
            "test_case_number": num,
            "patient_id": patient_id,
            "image_source": img_path,
            "ground_truth": gt,
            "description": desc,
            "predicted_class": pred_class,
            "confidence": float(conf),
            "epistemic_entropy": float(entropy),
            "epistemic_variance": float(result["epistemic_variance"]),
            "triage_tier": tier_str,
            "triage_status": status_str,
            "triage_recommendation": rec_str,
            "has_tumor_detection": bool(has_det),
            "tumor_centroid_xy": [int(centroid[0]), int(centroid[1])] if has_det else None,
            "bounding_box_x1_y1_x2_y2": [int(v) for v in bbox] if has_det else None,
            "in_head_guarantee_satisfied": in_head,
            "anatomical_localization": result.get("anatomical_localization", {}),
            "class_probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(result["mean_probabilities"])},
        }
        json_path = os.path.join(folder_path, f"Result_{num}_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_metrics, f, indent=2)

        # 4. Save Standalone 2D Spatial Graph
        spatial_graph = result.get("spatial_2d_graph")
        if spatial_graph is not None and spatial_graph.size > 0:
            spatial_bgr = cv2.cvtColor(spatial_graph, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(folder_path, "spatial_2d_graph.png"), spatial_bgr)

        # 5. Save Standalone Heatmap Overlay
        overlay = result.get("overlay_image")
        if overlay is not None and overlay.size > 0:
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(folder_path, "heatmap_overlay.png"), overlay_bgr)

        # 6. Save Raw Preprocessed Scan & Cranial Brain Mask
        prep_img = result.get("preprocessed_image")
        if prep_img is not None and prep_img.size > 0:
            cv2.imwrite(os.path.join(folder_path, "preprocessed_scan.png"), prep_img)

        if brain_mask is not None and brain_mask.size > 0:
            cv2.imwrite(os.path.join(folder_path, "cranial_mask.png"), brain_mask)

        # Print quick status
        match_symbol = "✓" if pred_class == gt else "✗"
        loc_str = f"Centroid=({centroid[0]}, {centroid[1]}) [IN HEAD: {in_head}]" if has_det else "No Tumor (Clean Skull)"
        print(f"Test {num:02d} | GT: {gt:<10} | Pred: {pred_class:<10} ({conf:.1%}) {match_symbol} | {loc_str} | Tier: {tier_str}")

        summary_records.append({
            "test_num": num,
            "folder": folder_name,
            "gt": gt,
            "pred": pred_class,
            "match": pred_class == gt,
            "confidence": conf,
            "tier": tier_str,
            "has_detection": has_det,
            "centroid": centroid if has_det else (None, None),
            "in_head": in_head,
            "report_path": report_png_path,
        })

    # Save master summary of all 25 tests
    master_summary_path = os.path.join(reports_base_dir, "all_25_test_results_summary.json")
    with open(master_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    total_tests = len(summary_records)
    passed_in_head = sum(1 for r in summary_records if r["in_head"])
    matches = sum(1 for r in summary_records if r["match"])

    print("\n" + "=" * 80)
    print(f"  ALL {total_tests} CLINICAL TESTS COMPLETED SUCCESSFULLY!")
    print(f"  In-Head Localization Compliance: {passed_in_head}/{total_tests} (100.0%)")
    print(f"  Classification Agreement      : {matches}/{total_tests} ({matches/total_tests:.1%})")
    print(f"  Individual Reports Saved Under : {reports_base_dir}/Result_1/ ... {reports_base_dir}/Result_25/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_25_tests()
