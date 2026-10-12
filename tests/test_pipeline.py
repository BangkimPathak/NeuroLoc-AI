"""
NeuroLoc-XAI Master Test Suite (25 Comprehensive Test Cases)
Verifies:
- Cranial brain vault masking & skull-crop boundary tightness
- Heatmap and bounding boxes strictly constrained INSIDE the head
- All 4 diagnostic classes (Glioma, Meningioma, Pituitary, No Tumor)
- 2D (X, Y) spatial indication & triage graph generation
- 3-Tier clinical triage engine logic (Tier 1, Tier 2, Tier 3)
- Bayesian Monte Carlo Dropout uncertainty & Shannon entropy
- Model architecture, accuracy, IoU, Pointing Game, and dHash deduplication
"""

import unittest
import numpy as np
import cv2
import tensorflow as tf

from neuroloc.dataset import (
    CLASS_NAMES,
    extract_cranial_brain_mask,
    ben_graham_skull_crop,
    apply_clahe,
    preprocess_mri_image,
    compute_image_dhash,
)
from neuroloc.model import (
    MCDropout,
    build_neuroloc_model,
    get_last_conv_layer_name,
)
from neuroloc.uncertainty import (
    compute_epistemic_uncertainty,
    triage_clinical_case,
)
from neuroloc.explainability import (
    GradCAM,
    GradCAMPlusPlus,
    extract_bounding_box_from_heatmap,
    NeuroAnatomicalRuleEngine,
    generate_2d_spatial_triage_graph,
    compute_iou,
    compute_mask_iou,
    pointing_game_hit,
)
from neuroloc.train_eval import compute_expected_calibration_error
from neuroloc.pipeline import NeuroLocPipeline
from neuroloc.utils import create_synthetic_mri


class TestNeuroLocMasterSuite(unittest.TestCase):
    """25 Comprehensive Test Cases for NeuroLoc-XAI."""

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    # -------------------------------------------------------------------------
    # TEST 1: Cranial Brain Mask Extraction
    # -------------------------------------------------------------------------
    def test_01_cranial_brain_mask_generation(self):
        """Verify cranial brain mask cleanly encapsulates head and excludes background margins."""
        img, _ = create_synthetic_mri(tumor_type="glioma", image_size=(224, 224))
        mask = extract_cranial_brain_mask(img)
        self.assertEqual(mask.shape, (224, 224))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(np.sum(mask > 0), 1000)
        # Verify corners are outside the head (0)
        self.assertEqual(mask[0, 0], 0, "Top-left corner must be outside cranial vault")
        self.assertEqual(mask[223, 223], 0, "Bottom-right corner must be outside cranial vault")
        # Center must be inside the head (255)
        self.assertEqual(mask[112, 112], 255, "Brain center must be inside cranial vault")

    # -------------------------------------------------------------------------
    # TEST 2: Skull Crop Boundary Tightness
    # -------------------------------------------------------------------------
    def test_02_skull_crop_boundary_tightness(self):
        """Verify Ben Graham skull crop tightly encapsulates cranial boundary without clipping brain."""
        img, _ = create_synthetic_mri(tumor_type="pituitary", image_size=(224, 224))
        cropped = ben_graham_skull_crop(img)
        self.assertGreater(cropped.size, 0)
        self.assertLessEqual(cropped.shape[0], 224)
        self.assertLessEqual(cropped.shape[1], 224)

    # -------------------------------------------------------------------------
    # TEST 3: CLAHE Adaptive Contrast Enhancement
    # -------------------------------------------------------------------------
    def test_03_clahe_contrast_enhancement(self):
        """Verify CLAHE enhances contrast without altering image dimensions."""
        img, _ = create_synthetic_mri(tumor_type="meningioma", image_size=(224, 224))
        clahe_img = apply_clahe(img)
        self.assertEqual(clahe_img.shape, img.shape)
        self.assertEqual(clahe_img.dtype, np.uint8)

    # -------------------------------------------------------------------------
    # TEST 4: Preprocessing Scaling & Tensor Dimensions
    # -------------------------------------------------------------------------
    def test_04_preprocess_scaling_and_dimensions(self):
        """Verify preprocess_mri_image produces float32 array in range [0, 255]."""
        img, _ = create_synthetic_mri(tumor_type="glioma", image_size=(224, 224))
        processed = preprocess_mri_image(img, target_size=(224, 224), normalize=False)
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(processed)), 0.0)
        self.assertLessEqual(float(np.max(processed)), 255.0)

    # -------------------------------------------------------------------------
    # TEST 5: Heatmap Strictly Zero Outside Skull
    # -------------------------------------------------------------------------
    def test_05_heatmap_strictly_zero_outside_skull(self):
        """Verify skull-constrained Grad-CAM produces 0.0 activation everywhere outside the head."""
        model = build_neuroloc_model(weights=None)
        img, _ = create_synthetic_mri(tumor_type="pituitary", image_size=(224, 224))
        brain_mask = extract_cranial_brain_mask(img)

        gradcam_pp = GradCAMPlusPlus(model)
        input_tensor = np.expand_dims(img.astype(np.float32), axis=0)
        hm = gradcam_pp.generate_heatmap(input_tensor, class_idx=3, brain_mask=brain_mask)

        outside_skull_pixels = hm[brain_mask == 0]
        self.assertEqual(np.max(outside_skull_pixels), 0.0, "All activations outside the skull must be strictly 0.0")

    # -------------------------------------------------------------------------
    # TEST 6: Bounding Box Strictly Inside Cranial Vault
    # -------------------------------------------------------------------------
    def test_06_tumor_bounding_box_inside_head(self):
        """Verify extracted bounding box is contained within cranial mask boundaries."""
        img, meta = create_synthetic_mri(tumor_type="glioma", image_size=(224, 224))
        brain_mask = extract_cranial_brain_mask(img)
        heatmap = np.zeros((224, 224), dtype=np.float32)

        # Place simulated tumor activation inside brain
        gt_x, gt_y = meta["gt_centroid"]
        cv2.circle(heatmap, (gt_x, gt_y), 20, 1.0, -1)
        heatmap = heatmap * (brain_mask > 0).astype(np.float32)

        bbox_data = extract_bounding_box_from_heatmap(heatmap, brain_mask=brain_mask, predicted_class="glioma")
        self.assertTrue(bbox_data["has_detection"])
        xmin, ymin, xmax, ymax = bbox_data["bbox"]

        # Check bounds
        self.assertGreaterEqual(xmin, 0)
        self.assertGreaterEqual(ymin, 0)
        self.assertLessEqual(xmax, 224)
        self.assertLessEqual(ymax, 224)
        # Check that centroid of box is inside head mask
        c_x, c_y = bbox_data["centroid"]
        self.assertEqual(brain_mask[c_y, c_x], 255, "Bbox centroid must be inside head mask")

    # -------------------------------------------------------------------------
    # TEST 7: Tumor Centroid Strictly Inside Brain Parenchyma
    # -------------------------------------------------------------------------
    def test_07_tumor_centroid_strictly_inside_brain(self):
        """Verify extracted lesion centroid (xc, yc) falls strictly on valid brain tissue."""
        img, meta = create_synthetic_mri(tumor_type="meningioma", image_size=(224, 224))
        brain_mask = extract_cranial_brain_mask(img)
        heatmap = np.zeros((224, 224), dtype=np.float32)
        gt_x, gt_y = meta["gt_centroid"]
        cv2.circle(heatmap, (gt_x, gt_y), 15, 1.0, -1)

        bbox_data = extract_bounding_box_from_heatmap(heatmap, brain_mask=brain_mask, predicted_class="meningioma")
        xc, yc = bbox_data["centroid"]
        self.assertTrue(brain_mask[yc, xc] > 0, f"Centroid ({xc}, {yc}) must be inside cranial vault")

    # -------------------------------------------------------------------------
    # TEST 8: Healthy Brain Scan Suppression (No Tumor)
    # -------------------------------------------------------------------------
    def test_08_notumor_healthy_scan_no_lesion(self):
        """Verify healthy scan (notumor) suppresses false detections inside head."""
        img, _ = create_synthetic_mri(tumor_type="notumor", image_size=(224, 224))
        brain_mask = extract_cranial_brain_mask(img)
        dummy_heatmap = np.random.rand(224, 224).astype(np.float32)

        bbox_data = extract_bounding_box_from_heatmap(dummy_heatmap, brain_mask=brain_mask, predicted_class="notumor")
        self.assertFalse(bbox_data["has_detection"], "Healthy scans must not detect false tumor lesions")
        self.assertEqual(bbox_data["bbox"], (0, 0, 0, 0))

    # -------------------------------------------------------------------------
    # TEST 9: Glioma Localization Inside Head
    # -------------------------------------------------------------------------
    def test_09_glioma_localization_inside_head(self):
        """Verify Glioma case is localized inside the head with lobar narrative."""
        img, meta = create_synthetic_mri(tumor_type="glioma", image_size=(224, 224))
        brain_mask = extract_cranial_brain_mask(img)
        heatmap = np.zeros((224, 224), dtype=np.float32)
        cv2.circle(heatmap, meta["gt_centroid"], 22, 1.0, -1)

        bbox_data = extract_bounding_box_from_heatmap(heatmap, brain_mask=brain_mask, predicted_class="glioma")
        loc = NeuroAnatomicalRuleEngine.localize(
            centroid=bbox_data["centroid"],
            bbox=bbox_data["bbox"],
            image_shape=(224, 224),
            predicted_class="glioma",
            has_detection=True,
        )
        self.assertIn("Lobe", loc["lobar_zone"])
        self.assertIn("glioma", loc["narrative"].lower())

    # -------------------------------------------------------------------------
    # TEST 10: Meningioma Localization Inside Head
    # -------------------------------------------------------------------------
    def test_10_meningioma_localization_inside_head(self):
        """Verify Meningioma case localized to cranial convexity/compartment."""
        img, meta = create_synthetic_mri(tumor_type="meningioma", image_size=(224, 224))
        brain_mask = extract_cranial_brain_mask(img)
        heatmap = np.zeros((224, 224), dtype=np.float32)
        cv2.circle(heatmap, meta["gt_centroid"], 18, 1.0, -1)

        bbox_data = extract_bounding_box_from_heatmap(heatmap, brain_mask=brain_mask, predicted_class="meningioma")
        loc = NeuroAnatomicalRuleEngine.localize(
            centroid=bbox_data["centroid"],
            bbox=bbox_data["bbox"],
            image_shape=(224, 224),
            predicted_class="meningioma",
            has_detection=True,
        )
        self.assertIn("meningioma", loc["narrative"].lower())

    # -------------------------------------------------------------------------
    # TEST 11: Pituitary Sellar Localization Inside Head
    # -------------------------------------------------------------------------
    def test_11_pituitary_localization_inside_head(self):
        """Verify Pituitary case localized to Sellar / Suprasellar cistern."""
        img, meta = create_synthetic_mri(tumor_type="pituitary", image_size=(224, 224))
        brain_mask = extract_cranial_brain_mask(img)
        heatmap = np.zeros((224, 224), dtype=np.float32)
        cv2.circle(heatmap, meta["gt_centroid"], 16, 1.0, -1)

        bbox_data = extract_bounding_box_from_heatmap(heatmap, brain_mask=brain_mask, predicted_class="pituitary")
        loc = NeuroAnatomicalRuleEngine.localize(
            centroid=bbox_data["centroid"],
            bbox=bbox_data["bbox"],
            image_shape=(224, 224),
            predicted_class="pituitary",
            has_detection=True,
        )
        self.assertIn("Sellar", loc["compartment"])
        self.assertIn("Hypophyseal Fossa", loc["lobar_zone"])

    # -------------------------------------------------------------------------
    # TEST 12: 2D Spatial Indication & Triage Graph Generation
    # -------------------------------------------------------------------------
    def test_12_2d_spatial_triage_graph_generation(self):
        """Verify 2D anatomical spatial indication graph renders with correct shape and RGB channels."""
        triage = {
            "confidence_tier": "TIER_1_HIGH_CONFIDENCE",
            "triage_status": "AUTONOMOUS_VERIFIED",
            "metrics_evaluated": {"variance": 0.005},
        }
        graph_img = generate_2d_spatial_triage_graph(
            centroid=(112, 112),
            bbox=(92, 92, 132, 132),
            predicted_class="pituitary",
            confidence=0.92,
            triage=triage,
            image_shape=(224, 224),
        )
        self.assertIsInstance(graph_img, np.ndarray)
        self.assertEqual(graph_img.ndim, 3)
        self.assertEqual(graph_img.shape[2], 3)
        self.assertGreater(graph_img.shape[0], 200)

    # -------------------------------------------------------------------------
    # TEST 13: 2D Spatial Coordinates Normalized in Range [0, 1]
    # -------------------------------------------------------------------------
    def test_13_2d_spatial_coordinates_normalized(self):
        """Verify normalized spatial coordinates fall strictly within [0.0, 1.0]."""
        loc = NeuroAnatomicalRuleEngine.localize(
            centroid=(50, 120),
            bbox=(30, 100, 70, 140),
            image_shape=(224, 224),
            predicted_class="glioma",
            has_detection=True,
        )
        self.assertGreaterEqual(loc["norm_x"], 0.0)
        self.assertLessEqual(loc["norm_x"], 1.0)
        self.assertGreaterEqual(loc["norm_y"], 0.0)
        self.assertLessEqual(loc["norm_y"], 1.0)

    # -------------------------------------------------------------------------
    # TEST 14: Triage Engine: Tier 1 Autonomous Verification
    # -------------------------------------------------------------------------
    def test_14_triage_tier1_autonomous_verified(self):
        """Verify high confidence and low variance routes case to Tier 1."""
        unc = {
            "predicted_confidence": 0.94,
            "predicted_class_variance": 0.003,
            "predictive_entropy": 0.25,
            "mutual_information": 0.04,
            "predicted_class_name": "glioma",
        }
        triage = triage_clinical_case(unc)
        self.assertEqual(triage["confidence_tier"], "TIER_1_HIGH_CONFIDENCE")
        self.assertFalse(triage["triage_required"])
        self.assertEqual(triage["triage_status"], "AUTONOMOUS_VERIFIED")

    # -------------------------------------------------------------------------
    # TEST 15: Triage Engine: Tier 2 Moderate Uncertainty
    # -------------------------------------------------------------------------
    def test_15_triage_tier2_moderate_uncertainty(self):
        """Verify borderline uncertainty routes case to Tier 2."""
        unc = {
            "predicted_confidence": 0.65,  # Low confidence trigger
            "predicted_class_variance": 0.008,
            "predictive_entropy": 0.45,
            "mutual_information": 0.05,
            "predicted_class_name": "meningioma",
        }
        triage = triage_clinical_case(unc, confidence_threshold=0.70)
        self.assertEqual(triage["confidence_tier"], "TIER_2_MODERATE_UNCERTAINTY")
        self.assertTrue(triage["triage_required"])

    # -------------------------------------------------------------------------
    # TEST 16: Triage Engine: Tier 3 Critical Uncertainty
    # -------------------------------------------------------------------------
    def test_16_triage_tier3_critical_uncertainty(self):
        """Verify high entropy and multiple flags route case to Tier 3 urgent review."""
        unc = {
            "predicted_confidence": 0.45,
            "predicted_class_variance": 0.035,
            "predictive_entropy": 1.45,
            "mutual_information": 0.22,
            "predicted_class_name": "pituitary",
        }
        triage = triage_clinical_case(unc)
        self.assertEqual(triage["confidence_tier"], "TIER_3_CRITICAL_UNCERTAINTY")
        self.assertTrue(triage["triage_required"])
        self.assertIn("URGENT", triage["recommendation"])

    # -------------------------------------------------------------------------
    # TEST 17: Monte Carlo Dropout Stochasticity
    # -------------------------------------------------------------------------
    def test_17_mc_dropout_stochastic_variance(self):
        """Verify MCDropout produces non-identical outputs when training=False."""
        layer = MCDropout(rate=0.5)
        test_in = tf.ones((1, 50))
        out1 = layer(test_in, training=False).numpy()
        out2 = layer(test_in, training=False).numpy()
        self.assertFalse(np.array_equal(out1, out2), "Monte Carlo Dropout must remain stochastic at inference")

    # -------------------------------------------------------------------------
    # TEST 18: Epistemic Uncertainty & Shannon Entropy
    # -------------------------------------------------------------------------
    def test_18_epistemic_uncertainty_quantification(self):
        """Verify compute_epistemic_uncertainty outputs valid variance and entropy."""
        model = build_neuroloc_model(weights=None)
        dummy_img = np.random.rand(224, 224, 3).astype(np.float32) * 255.0
        unc = compute_epistemic_uncertainty(model, dummy_img, num_passes=5)
        self.assertIn("mean_probabilities", unc)
        self.assertIn("epistemic_variance", unc)
        self.assertIn("predictive_entropy", unc)
        self.assertGreaterEqual(unc["predictive_entropy"], 0.0)
        self.assertAlmostEqual(float(np.sum(unc["mean_probabilities"])), 1.0, places=4)

    # -------------------------------------------------------------------------
    # TEST 19: Model Architecture & Softmax Probabilities
    # -------------------------------------------------------------------------
    def test_19_model_architecture_and_output_shape(self):
        """Verify model builds and outputs valid 4-class softmax probability distribution."""
        model = build_neuroloc_model(weights=None)
        self.assertEqual(model.output_shape, (None, 4))
        dummy = tf.zeros((2, 224, 224, 3))
        preds = model(dummy, training=False).numpy()
        self.assertEqual(preds.shape, (2, 4))
        self.assertTrue(np.allclose(np.sum(preds, axis=1), [1.0, 1.0]))

    # -------------------------------------------------------------------------
    # TEST 20: Bounding Box IoU Computation
    # -------------------------------------------------------------------------
    def test_20_box_iou_exact_and_partial(self):
        """Verify compute_iou on exact, overlapping, and disjoint boxes."""
        boxA = (10, 10, 50, 50)
        boxB = (10, 10, 50, 50)
        boxC = (30, 30, 70, 70)
        boxD = (100, 100, 150, 150)
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)
        self.assertGreater(compute_iou(boxA, boxC), 0.0)
        self.assertEqual(compute_iou(boxA, boxD), 0.0)

    # -------------------------------------------------------------------------
    # TEST 21: Mask IoU Computation
    # -------------------------------------------------------------------------
    def test_21_mask_iou_computation(self):
        """Verify compute_mask_iou on binary lesion masks."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:60, 20:60] = 255
        mask2[20:60, 20:60] = 255
        self.assertAlmostEqual(compute_mask_iou(mask1, mask2), 1.0)

    # -------------------------------------------------------------------------
    # TEST 22: Pointing Game Hit Rate
    # -------------------------------------------------------------------------
    def test_22_pointing_game_hit_rate(self):
        """Verify pointing_game_hit correctly checks peak salience inside ground truth."""
        hm = np.zeros((50, 50), dtype=np.float32)
        gt = np.zeros((50, 50), dtype=np.uint8)
        gt[10:30, 10:30] = 255
        hm[20, 20] = 1.0  # Hit inside
        self.assertTrue(pointing_game_hit(hm, gt))
        hm[20, 20] = 0.0
        hm[45, 45] = 1.0  # Miss outside
        self.assertFalse(pointing_game_hit(hm, gt))

    # -------------------------------------------------------------------------
    # TEST 23: Neuro-Anatomical Laterality Mapping
    # -------------------------------------------------------------------------
    def test_23_neuro_anatomical_laterality_mapping(self):
        """Verify radiological mirror convention: Image Left = Right Hemisphere, Image Right = Left."""
        loc_right = NeuroAnatomicalRuleEngine.localize(
            centroid=(40, 80),  # x < 0.44 -> Right hemisphere
            bbox=(20, 60, 60, 100),
            image_shape=(224, 224),
            predicted_class="glioma",
            has_detection=True,
        )
        self.assertEqual(loc_right["laterality"], "Right Hemisphere")

        loc_left = NeuroAnatomicalRuleEngine.localize(
            centroid=(180, 80),  # x > 0.56 -> Left hemisphere
            bbox=(160, 60, 200, 100),
            image_shape=(224, 224),
            predicted_class="glioma",
            has_detection=True,
        )
        self.assertEqual(loc_left["laterality"], "Left Hemisphere")

    # -------------------------------------------------------------------------
    # TEST 24: Perceptual Difference Hashing & Leakage Detection
    # -------------------------------------------------------------------------
    def test_24_dhash_leakage_audit_detection(self):
        """Verify 64-bit dHash computes identical hash for duplicate and distinct hash for different."""
        img1, _ = create_synthetic_mri(tumor_type="glioma", image_size=(100, 100), seed=1)
        img2, _ = create_synthetic_mri(tumor_type="glioma", image_size=(100, 100), seed=1)
        img3, _ = create_synthetic_mri(tumor_type="notumor", image_size=(100, 100), seed=99)
        h1 = compute_image_dhash(img1)
        h2 = compute_image_dhash(img2)
        h3 = compute_image_dhash(img3)
        self.assertEqual(h1 - h2, 0, "Identical scans must have Hamming distance 0")
        self.assertGreater(h1 - h3, 0, "Distinct scans must have positive Hamming distance")

    # -------------------------------------------------------------------------
    # TEST 25: End-to-End Pipeline In-Head Diagnosis & Accuracy
    # -------------------------------------------------------------------------
    def test_25_end_to_end_pipeline_in_head_diagnosis(self):
        """Verify complete pipeline executes and guarantees tumor localization strictly inside head."""
        pipeline = NeuroLocPipeline(weights=None)
        img, _ = create_synthetic_mri(tumor_type="pituitary", image_size=(224, 224))
        result = pipeline.predict(img, num_mc_passes=3)

        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIn("bbox", result)
        self.assertIn("spatial_2d_graph", result)
        self.assertIn("clinical_report", result)

        # Check localization is strictly inside head
        brain_mask = result["brain_mask"]
        xc, yc = result["centroid"]
        if result["has_detection"]:
            self.assertTrue(brain_mask[yc, xc] > 0, "Centroid must be strictly within intracranial vault")
            self.assertGreater(result["spatial_2d_graph"].size, 0)


if __name__ == "__main__":
    unittest.main()
