"""
NeuroLoc-XAI: Explainable Multi-Class Brain Tumor Diagnosis with
Automated Anatomical Localization and Epistemic Uncertainty Estimation in TensorFlow 2.x
"""

from neuroloc.model import MCDropout, build_neuroloc_model
from neuroloc.uncertainty import compute_epistemic_uncertainty, triage_clinical_case
from neuroloc.explainability import (
    GradCAM,
    GradCAMPlusPlus,
    NeuroAnatomicalRuleEngine,
    extract_bounding_box_from_heatmap,
    generate_2d_spatial_triage_graph,
    compute_iou,
    compute_mask_iou,
    pointing_game_hit,
)
from neuroloc.dataset import (
    CLASS_NAMES,
    LABEL_MAP,
    preprocess_mri_image,
    ben_graham_skull_crop,
    apply_clahe,
    extract_cranial_brain_mask,
    audit_and_deduplicate_dataset,
    create_dataset,
)
from neuroloc.pipeline import NeuroLocPipeline

__version__ = "1.0.0"
__all__ = [
    "MCDropout",
    "build_neuroloc_model",
    "compute_epistemic_uncertainty",
    "triage_clinical_case",
    "GradCAM",
    "GradCAMPlusPlus",
    "NeuroAnatomicalRuleEngine",
    "extract_bounding_box_from_heatmap",
    "generate_2d_spatial_triage_graph",
    "compute_iou",
    "compute_mask_iou",
    "pointing_game_hit",
    "CLASS_NAMES",
    "LABEL_MAP",
    "preprocess_mri_image",
    "ben_graham_skull_crop",
    "apply_clahe",
    "extract_cranial_brain_mask",
    "audit_and_deduplicate_dataset",
    "create_dataset",
    "NeuroLocPipeline",
]
