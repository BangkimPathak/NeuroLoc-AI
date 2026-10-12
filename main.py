"""
NeuroLoc-XAI: Master Command-Line Interface
Provides commands for Training, Evaluation, Deduplication Auditing, and Clinical Inference.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import tensorflow as tf

from neuroloc.dataset import (
    CLASS_NAMES,
    create_dataset,
    audit_and_deduplicate_dataset,
    download_kaggle_dataset,
)
from neuroloc.model import build_neuroloc_model, MCDropout
from neuroloc.train_eval import (
    train_neuroloc_model,
    evaluate_model_comprehensive,
)
from neuroloc.pipeline import NeuroLocPipeline
from neuroloc.utils import create_synthetic_mri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NeuroLoc-Main")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def handle_deduplicate(args):
    """Audit dataset splits for perceptual duplicates and benchmark leakage."""
    logger.info("Starting Perceptual Hashing (dHash) Audit on '%s' and '%s'...", args.train_dir, args.test_dir)
    report = audit_and_deduplicate_dataset(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        hamming_threshold=args.hamming_threshold,
        quarantine_duplicates=args.quarantine,
        quarantine_dir=args.quarantine_dir,
    )

    print("\n" + "=" * 60)
    print("  NEUROLOC-XAI: DATASET DEDUPLICATION & LEAKAGE AUDIT")
    print("=" * 60)
    print(f"Training Images Analyzed    : {report['train_images_analyzed']}")
    print(f"Testing Images Analyzed     : {report['test_images_analyzed']}")
    print(f"Exact Duplicate Leaks (d=0) : {report['exact_leakage_count']}")
    print(f"Near-Duplicate Leaks (d<={args.hamming_threshold}) : {report['near_leakage_count']}")
    print(f"Total Benchmark Leakages    : {report['total_leakage_count']}")
    print(f"Test Split Leakage Rate     : {report['leakage_rate']:.2%}")
    print("=" * 60)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        # Convert non-serializable fields
        serializable_report = {k: v for k, v in report.items() if k != "leakage_records"}
        def _json_serial(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            return str(obj)

        with open(args.output_json, "w") as f:
            json.dump(serializable_report, f, indent=2, default=_json_serial)
        logger.info("Audit report written to '%s'", args.output_json)


def handle_train(args):
    """Execute two-stage training with Bayesian Monte Carlo Dropout."""
    logger.info("Preparing training and validation datasets...")
    train_ds = create_dataset(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        image_size=(224, 224),
        is_training=True,
        sample_limit=args.sample_limit,
    )
    val_ds = create_dataset(
        data_dir=args.test_dir,
        batch_size=args.batch_size,
        image_size=(224, 224),
        is_training=False,
        sample_limit=args.sample_limit,
    )

    logger.info("Constructing NeuroLoc model with %s backbone...", args.backbone)
    model = build_neuroloc_model(
        input_shape=(224, 224, 3),
        num_classes=len(CLASS_NAMES),
        backbone_name=args.backbone,
        dropout_rate=args.dropout_rate,
        weights="imagenet" if not args.from_scratch else None,
        freeze_backbone=True,
    )

    logger.info("Beginning two-stage training pipeline...")
    history = train_neuroloc_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        learning_rate_stage1=args.lr_stage1,
        learning_rate_stage2=args.lr_stage2,
        checkpoint_dir=args.checkpoint_dir,
    )
    logger.info("Training complete. Best checkpoint saved under: %s", args.checkpoint_dir)


def handle_evaluate(args):
    """Execute comprehensive clinical evaluation suite."""
    logger.info("Loading model for evaluation from '%s'...", args.model_path)
    custom_objects = {"MCDropout": MCDropout}
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)

    test_ds = create_dataset(
        data_dir=args.test_dir,
        batch_size=args.batch_size,
        image_size=(224, 224),
        is_training=False,
        sample_limit=args.sample_limit,
    )

    logger.info("Running evaluation across all diagnostic metrics...")
    metrics = evaluate_model_comprehensive(model, test_ds, class_names=CLASS_NAMES)

    print("\n" + "=" * 65)
    print("  NEUROLOC-XAI: COMPREHENSIVE CLINICAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Overall Accuracy          : {metrics['accuracy']:.2%}")
    print(f"Macro F1-Score            : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score         : {metrics['weighted_f1']:.4f}")
    print(f"Multi-Class ROC-AUC (OvR) : {metrics['roc_auc_ovr_macro']:.4f}")
    print(f"Expected Calibration Error: {metrics['expected_calibration_error']:.4f} (ECE)")
    print(f"Maximum Calibration Error : {metrics['maximum_calibration_error']:.4f} (MCE)")
    print(f"Brier Multi-Class Score   : {metrics['brier_score']:.4f}")
    print("-" * 65)
    print("Per-Class Sensitivity (Recall) & Specificity:")
    for c_name, c_dict in metrics["per_class_metrics"].items():
        print(f"  * {c_name.capitalize():<12}: Sensitivity={c_dict['sensitivity']:.2%} | Specificity={c_dict['specificity']:.2%} | F1={c_dict['f1_score']:.4f}")
    print("=" * 65)

    if args.output_json:
        def _json_serial(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            return str(obj)

        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2, default=_json_serial)
        logger.info("Saved evaluation metrics JSON to: %s", args.output_json)


def handle_infer(args):
    """Run inference on an input scan."""
    pipeline = NeuroLocPipeline(
        model_path=args.model_path,
        backbone=args.backbone,
        weights="imagenet" if args.model_path is None else None,
    )

    result = pipeline.predict(
        image_input=args.image,
        num_mc_passes=args.mc_passes,
        xai_method=args.xai,
        patient_id=args.patient_id,
    )

    print(result["clinical_report"])

    if args.output_plot:
        pipeline.visualize(result, save_path=args.output_plot)


def main():
    parser = argparse.ArgumentParser(
        description="NeuroLoc-XAI: Explainable Multi-Class Brain Tumor Diagnosis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operating Mode")

    # Mode: Demo
    demo_parser = subparsers.add_parser("demo", help="Run end-to-end clinical inference demonstration")
    demo_parser.add_argument("--image", type=str, default=None, help="Path to input MRI scan")
    demo_parser.add_argument("--class-name", type=str, default="pituitary", choices=CLASS_NAMES)
    demo_parser.add_argument("--mc-passes", type=int, default=30)
    demo_parser.add_argument("--xai", type=str, default="gradcam++", choices=["gradcam++", "gradcam"])
    demo_parser.add_argument("--output", type=str, default="reports/Result_1/Result_1.png")
    demo_parser.add_argument("--checkpoint", type=str, default="checkpoints/neuroloc_best.keras")

    # Mode: Test 25
    test25_parser = subparsers.add_parser("test25", help="Run all 25 clinical test cases into reports/Result_1 ... reports/Result_25")
    test25_parser.add_argument("--checkpoint", type=str, default="checkpoints/neuroloc_best.keras")
    test25_parser.add_argument("--output-dir", type=str, default="reports")
    test25_parser.add_argument("--mc-passes", type=int, default=15)

    # Mode: Deduplicate
    dedup_parser = subparsers.add_parser("deduplicate", help="Audit dataset splits for perceptual leakage")
    dedup_parser.add_argument("--train-dir", type=str, default="Training")
    dedup_parser.add_argument("--test-dir", type=str, default="Testing")
    dedup_parser.add_argument("--hamming-threshold", type=int, default=2)
    dedup_parser.add_argument("--quarantine", action="store_true")
    dedup_parser.add_argument("--quarantine-dir", type=str, default=None)
    dedup_parser.add_argument("--output-json", type=str, default="reports/deduplication_audit.json")

    # Mode: Train
    train_parser = subparsers.add_parser("train", help="Train NeuroLoc model with Monte Carlo Dropout")
    train_parser.add_argument("--train-dir", type=str, default="Training")
    train_parser.add_argument("--test-dir", type=str, default="Testing")
    train_parser.add_argument("--backbone", type=str, default="efficientnetb0", choices=["efficientnetb0", "densenet121"])
    train_parser.add_argument("--epochs-stage1", type=int, default=5)
    train_parser.add_argument("--epochs-stage2", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--dropout-rate", type=float, default=0.3)
    train_parser.add_argument("--lr-stage1", type=float, default=1e-3)
    train_parser.add_argument("--lr-stage2", type=float, default=1e-4)
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    train_parser.add_argument("--sample-limit", type=int, default=None)
    train_parser.add_argument("--from-scratch", action="store_true")

    # Mode: Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model calibration and clinical metrics")
    eval_parser.add_argument("--model-path", type=str, default="checkpoints/neuroloc_best.keras")
    eval_parser.add_argument("--test-dir", type=str, default="Testing")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--sample-limit", type=int, default=None)
    eval_parser.add_argument("--output-json", type=str, default="reports/evaluation_metrics.json")

    # Mode: Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference on a single image")
    infer_parser.add_argument("--image", type=str, required=True, help="Path to input MRI image")
    infer_parser.add_argument("--model-path", type=str, default="checkpoints/neuroloc_best.keras")
    infer_parser.add_argument("--backbone", type=str, default="efficientnetb0")
    infer_parser.add_argument("--mc-passes", type=int, default=30)
    infer_parser.add_argument("--xai", type=str, default="gradcam++")
    infer_parser.add_argument("--patient-id", type=str, default="PATIENT-EXAM-101")
    infer_parser.add_argument("--output-plot", type=str, default="reports/Result_1/Result_1.png")

    args = parser.parse_args()

    if args.mode == "demo":
        from demo_inference import run_inference_demo
        run_inference_demo(
            image_path=args.image,
            tumor_type=args.class_name,
            num_mc_passes=args.mc_passes,
            xai_method=args.xai,
            save_plot_path=args.output,
            model_checkpoint=args.checkpoint,
        )
    elif args.mode == "test25":
        from run_25_tests import run_all_25_tests
        run_all_25_tests(
            checkpoint_path=args.checkpoint,
            reports_base_dir=args.output_dir,
            mc_passes=args.mc_passes,
        )
    elif args.mode == "deduplicate":
        handle_deduplicate(args)
    elif args.mode == "train":
        handle_train(args)
    elif args.mode == "evaluate":
        handle_evaluate(args)
    elif args.mode == "infer":
        handle_infer(args)


if __name__ == "__main__":
    main()
