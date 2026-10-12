"""
Module 5: Training, Evaluation & Calibration Metrics
Full training loop with callbacks, multi-class evaluation metrics
(F1, Per-Class Sensitivity & Specificity, ROC-AUC),
Expected Calibration Error (ECE) analysis, and Clinical Neuro-Radiology Report generation.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import logging
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)

from neuroloc.dataset import CLASS_NAMES, NUM_CLASSES
from neuroloc.model import unfreeze_backbone_layers

logger = logging.getLogger(__name__)


# =========================================================================
# Training Pipeline with Two-Stage Transfer Learning
# =========================================================================

def train_neuroloc_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs_stage1: int = 5,
    epochs_stage2: int = 5,
    learning_rate_stage1: float = 1e-3,
    learning_rate_stage2: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    class_weights: Optional[Dict[int, float]] = None,
) -> tf.keras.callbacks.History:
    """
    Two-stage training procedure:
      Stage 1: Train classification head with frozen backbone.
      Stage 2: Unfreeze top 20 backbone layers for fine-tuning at lower learning rate.

    Args:
        model: NeuroLoc model instance.
        train_ds: tf.data.Dataset for training.
        val_ds: tf.data.Dataset for validation.
        epochs_stage1: Number of warm-up epochs for classification head.
        epochs_stage2: Number of fine-tuning epochs.
        learning_rate_stage1: Stage 1 learning rate.
        learning_rate_stage2: Stage 2 learning rate.
        checkpoint_dir: Directory to save best model checkpoints.
        class_weights: Optional dictionary of class weights.

    Returns:
        Keras training history.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "neuroloc_best.keras")

    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # STAGE 1: Train Head Only
    logger.info("=== STAGE 1: Warm-up Classification Head (%d epochs) ===", epochs_stage1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_stage1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=callbacks_stage1,
        class_weight=class_weights,
    )

    # STAGE 2: Fine-Tuning Top Backbone Layers
    if epochs_stage2 > 0:
        logger.info("=== STAGE 2: Fine-Tuning Backbone (%d epochs) ===", epochs_stage2)
        unfreeze_backbone_layers(model, num_layers_to_unfreeze=25)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_stage2),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        callbacks_stage2 = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage1 + epochs_stage2,
            initial_epoch=epochs_stage1,
            callbacks=callbacks_stage2,
            class_weight=class_weights,
        )
        return history2

    return history1


# =========================================================================
# Expected Calibration Error (ECE) & Reliability Metrics
# =========================================================================

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    to assess probability reliability for clinical deployment (Guo et al., 2017).

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Ground truth integer labels shape (N,).
        y_probs: Predicted probability distributions shape (N, K).
        num_bins: Number of confidence bins (typically 10).

    Returns:
        Dictionary containing ECE, MCE, bin confidences, bin accuracies, and counts.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0

    bin_data = []
    total_samples = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_accuracy = float(np.mean(accuracies[in_bin]))
            bin_confidence = float(np.mean(confidences[in_bin]))
            abs_diff = abs(bin_accuracy - bin_confidence)

            ece += (bin_size / total_samples) * abs_diff
            mce = max(mce, abs_diff)

            bin_data.append({
                "bin_idx": i,
                "range": (bin_lower, bin_upper),
                "count": int(bin_size),
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
                "calibration_gap": abs_diff,
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "range": (bin_lower, bin_upper),
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
                "calibration_gap": 0.0,
            })

    # Multi-class Brier Score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    y_true_onehot = np.zeros_like(y_probs)
    for idx, true_c in enumerate(y_true):
        y_true_onehot[idx, true_c] = 1.0
    brier_score = float(np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1)))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "num_bins": num_bins,
        "bin_details": bin_data,
    }


# =========================================================================
# Comprehensive Evaluation Suite
# =========================================================================

def evaluate_model_comprehensive(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Run complete diagnostic evaluation suite:
      - Multi-class confusion matrix
      - Macro & Weighted F1-scores, Precision, Recall
      - Per-class Sensitivity (True Positive Rate) & Specificity (True Negative Rate)
      - Multi-class ROC-AUC (One-vs-Rest, macro and weighted)
      - Expected Calibration Error (ECE)

    Args:
        model: Trained tf.keras.Model.
        dataset: tf.data.Dataset yielding (images, labels).
        class_names: List of class labels.

    Returns:
        Structured evaluation metrics dictionary.
    """
    all_preds = []
    all_labels = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_probs = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.argmax(y_probs, axis=1)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # 2. Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 3. Per-Class Sensitivity and Specificity
    # For class i:
    # TP = CM[i, i]
    # FN = sum(CM[i, :]) - TP
    # FP = sum(CM[:, i]) - TP
    # TN = total - (TP + FN + FP)
    total_samples = len(y_true)
    per_class_metrics = {}

    for i, c_name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - (tp + fn + fp))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class_metrics[c_name] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
        }

    # 4. Multi-class ROC-AUC
    y_true_onehot = np.eye(len(class_names))[y_true]
    try:
        roc_auc_ovr_macro = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="macro"))
        roc_auc_ovr_weighted = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="weighted"))
    except Exception as e:
        logger.warning("ROC-AUC computation notice: %s", e)
        roc_auc_ovr_macro = 0.0
        roc_auc_ovr_weighted = 0.0

    # 5. Expected Calibration Error
    calibration = compute_expected_calibration_error(y_true, y_probs, num_bins=10)

    accuracy = float(np.mean(y_pred == y_true))

    return {
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "weighted_f1": float(weighted_f1),
        "roc_auc_ovr_macro": roc_auc_ovr_macro,
        "roc_auc_ovr_weighted": roc_auc_ovr_weighted,
        "expected_calibration_error": calibration["ece"],
        "maximum_calibration_error": calibration["mce"],
        "brier_score": calibration["brier_score"],
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "calibration_details": calibration,
    }


# =========================================================================
# Formatted Clinical Neuro-Radiology Summary Report
# =========================================================================

def format_clinical_report(report_data: Dict[str, Any]) -> str:
    """
    Generate a standardized, clinical-grade Neuro-Radiology Diagnostic Report.

    Args:
        report_data: Unified pipeline prediction output dictionary.

    Returns:
        Formatted ASCII clinical diagnostic report string.
    """
    patient_id = report_data.get("patient_id", "SCAN-ANON-7842")
    diag = report_data.get("predicted_class", "UNKNOWN").upper()
    prob = report_data.get("confidence", 0.0)
    var = report_data.get("epistemic_variance", 0.0)
    entropy = report_data.get("predictive_entropy", 0.0)
    mi = report_data.get("mutual_information", 0.0)
    ci = report_data.get("credible_interval_95", (0.0, 0.0))

    triage = report_data.get("triage", {})
    triage_status = triage.get("triage_status", "UNKNOWN")
    tier = triage.get("confidence_tier", "UNKNOWN")
    recommendation = triage.get("recommendation", "Review required.")
    flags = triage.get("flags", [])

    anat = report_data.get("anatomical_localization", {})
    laterality = anat.get("laterality", "N/A")
    compartment = anat.get("compartment", "N/A")
    lobar_zone = anat.get("lobar_zone", "N/A")
    narrative = anat.get("narrative", "N/A")

    bbox = report_data.get("bbox", (0, 0, 0, 0))
    centroid = report_data.get("centroid", (0, 0))

    flags_str = "\n".join([f"    [!] {f}" for f in flags]) if flags else "    None (Parameters within nominal threshold)"

    border = "=" * 80
    sub_border = "-" * 80

    report = f"""
{border}
  NEUROLOC-XAI: CLINICAL NEURO-RADIOLOGY DIAGNOSTIC REPORT
{border}
ACCESSION ID        : {patient_id}
IMAGE PROTOCOL      : Brain MRI (T1-Gd / Contrast-Enhanced Multi-Slice)
FRAMEWORK           : TensorFlow 2.x Bayesian Deep Inference Engine
STATUS              : {triage_status} [{tier}]
{sub_border}
I. PRIMARY DIAGNOSTIC IMPRESSION
{sub_border}
  * Predicted Pathology   : {diag}
  * Bayesian Posterior p  : {prob:.2%}  (95% CI: [{ci[0]:.2%}, {ci[1]:.2%}])
  * Epistemic Uncertainty : sigma^2 = {var:.5f}  (Monte Carlo T=30 forward passes)
  * Total Entropy H(y|x)  : {entropy:.4f} nats
  * Knowledge Info Gain I : {mi:.4f} nats (Mutual Information)

{sub_border}
II. AUTOMATED CLINICAL TRIAGE DECISION
{sub_border}
  * Triage Status         : {triage_status}
  * Confidence Level      : {tier}
  * Uncertainty Flags     :
{flags_str}
  * Clinical Advisory     :
    {recommendation}

{sub_border}
III. AUTOMATED NEURO-ANATOMICAL LOCALIZATION (Grad-CAM++ Salience)
{sub_border}
  * Laterality            : {laterality}
  * Cranial Compartment   : {compartment}
  * Regional Lobar Zone   : {lobar_zone}
  * Bounding Coordinates  : [xmin={bbox[0]}, ymin={bbox[1]}, xmax={bbox[2]}, ymax={bbox[3]}]
  * Centroid Coordinates  : (x={centroid[0]}, y={centroid[1]})

  * Radiological Narrative:
    "{narrative}"
{border}
  DISCLAIMER: NeuroLoc-XAI is an investigational clinical decision-support tool.
  Final diagnosis must be validated by a certified attending Neuroradiologist.
{border}
"""
    return report
