"""
Module 3: Epistemic Uncertainty Quantification & Triage Engine
Performs Bayesian Monte Carlo Dropout sampling to calculate predictive mean,
epistemic variance, Shannon entropy, and mutual information.
Implements clinical triage rules for autonomous vs. radiologist-review routing.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import tensorflow as tf
from neuroloc.dataset import CLASS_NAMES


def compute_epistemic_uncertainty(
    model: tf.keras.Model,
    image: Union[np.ndarray, tf.Tensor],
    num_passes: int = 30,
    batch_size: int = 32,
    epsilon: float = 1e-12,
) -> Dict[str, Any]:
    """
    Perform Bayesian Monte Carlo Dropout sampling to quantify epistemic (model) uncertainty.

    Args:
        model: NeuroLoc model containing active MCDropout layers.
        image: Single image array of shape (H, W, C) or (1, H, W, C), scaled to [0, 1].
        num_passes: Number of stochastic forward passes T (default 30).
        batch_size: Batch execution size for forward passes.
        epsilon: Small numerical stability constant for log calculations.

    Returns:
        Dictionary containing:
            - mean_probabilities: np.ndarray shape (K,)
            - epistemic_variance: np.ndarray shape (K,)
            - total_epistemic_variance: float
            - predicted_class_idx: int
            - predicted_class_name: str
            - predicted_confidence: float
            - predicted_class_variance: float
            - predictive_entropy: float (total uncertainty)
            - expected_entropy: float (aleatoric uncertainty estimate)
            - mutual_information: float (pure epistemic / knowledge uncertainty)
            - credible_interval_95: Tuple[float, float]
            - mc_samples: np.ndarray shape (T, K)
    """
    # Standardize input dimensions
    if isinstance(image, tf.Tensor):
        img_np = image.numpy()
    else:
        img_np = np.asarray(image, dtype=np.float32)

    if img_np.ndim == 3:
        img_batch = np.expand_dims(img_np, axis=0)
    elif img_np.ndim == 4:
        img_batch = img_np
    else:
        raise ValueError(f"Expected 3D or 4D image input, got shape: {img_np.shape}")

    # Replicate single image across T forward passes for fast parallel execution
    tiled_inputs = np.repeat(img_batch, repeats=num_passes, axis=0)

    # Perform stochastic forward passes
    # Model uses MCDropout with training=True, so each slice in tiled_inputs gets different dropout masks
    predictions = model.predict(tiled_inputs, batch_size=batch_size, verbose=0)
    # predictions shape: (num_passes, num_classes)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

    # 1. Predictive Mean Probability Vector \mu
    mean_probs = np.mean(predictions, axis=0)
    # Re-normalize to ensure strictly valid probability distribution
    mean_probs = mean_probs / np.sum(mean_probs)

    # 2. Epistemic Variance \sigma^2 per class
    # \sigma_k^2 = (1/T) \sum_{t=1}^T (P_{t,k} - \mu_k)^2
    variances = np.var(predictions, axis=0)
    total_variance = float(np.mean(variances))

    # Argmax prediction
    predicted_idx = int(np.argmax(mean_probs))
    predicted_name = CLASS_NAMES[predicted_idx]
    predicted_conf = float(mean_probs[predicted_idx])
    predicted_var = float(variances[predicted_idx])

    # 3. Predictive Entropy H(y | x) = - \sum_k \mu_k \log(\mu_k)
    # Represents total predictive uncertainty
    predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + epsilon)))

    # 4. Expected Entropy E_{p(W|D)} [ H(y | x, W) ]
    # Represents data / aleatoric uncertainty
    sample_entropies = -np.sum(predictions * np.log(predictions + epsilon), axis=1)
    expected_entropy = float(np.mean(sample_entropies))

    # 5. Mutual Information I(y, W | x) = H(y | x) - E[H(y | x, W)]
    # Directly captures Epistemic (model ignorance) uncertainty
    mutual_info = float(max(0.0, predictive_entropy - expected_entropy))

    # 6. 95% Bayesian Credible Interval for winning class
    winning_samples = predictions[:, predicted_idx]
    ci_lower = float(np.percentile(winning_samples, 2.5))
    ci_upper = float(np.percentile(winning_samples, 97.5))

    return {
        "mean_probabilities": mean_probs,
        "epistemic_variance": variances,
        "total_epistemic_variance": total_variance,
        "predicted_class_idx": predicted_idx,
        "predicted_class_name": predicted_name,
        "predicted_confidence": predicted_conf,
        "predicted_class_variance": predicted_var,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_info,
        "credible_interval_95": (ci_lower, ci_upper),
        "mc_samples": predictions,
    }


def triage_clinical_case(
    uncertainty_dict: Dict[str, Any],
    variance_threshold: float = 0.015,
    entropy_threshold: float = 0.60,
    confidence_threshold: float = 0.70,
    mutual_info_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Automated Clinical Triage Engine.
    Examines epistemic variance, predictive entropy, and posterior confidence
    to route cases to either autonomous reporting or urgent human neuroradiologist review.

    Args:
        uncertainty_dict: Output of compute_epistemic_uncertainty.
        variance_threshold: Threshold tau_var on predicted class epistemic variance.
        entropy_threshold: Threshold tau_ent on total Shannon entropy.
        confidence_threshold: Minimum acceptable winning confidence tau_conf.
        mutual_info_threshold: Threshold on epistemic mutual information.

    Returns:
        Triage assessment dictionary containing:
            - triage_required: bool
            - triage_status: str
            - confidence_tier: str (Tier 1 / Tier 2 / Tier 3)
            - flags: List[str]
            - recommendation: str
    """
    conf = uncertainty_dict["predicted_confidence"]
    var = uncertainty_dict["predicted_class_variance"]
    entropy = uncertainty_dict["predictive_entropy"]
    mi = uncertainty_dict["mutual_information"]

    flags = []

    if var > variance_threshold:
        flags.append(f"HIGH_EPISTEMIC_VARIANCE (sigma^2 = {var:.4f} > {variance_threshold:.4f})")
    if entropy > entropy_threshold:
        flags.append(f"HIGH_PREDICTIVE_ENTROPY (H = {entropy:.4f} > {entropy_threshold:.4f})")
    if conf < confidence_threshold:
        flags.append(f"LOW_PREDICTIVE_CONFIDENCE (p = {conf:.2%} < {confidence_threshold:.2%})")
    if mi > mutual_info_threshold:
        flags.append(f"HIGH_MODEL_IGNORANCE_MUTUAL_INFO (MI = {mi:.4f} > {mutual_info_threshold:.4f})")

    if len(flags) >= 2 or var > (1.5 * variance_threshold):
        confidence_tier = "TIER_3_CRITICAL_UNCERTAINTY"
        triage_status = "TRIAGE_REQUIRED"
        recommendation = (
            "URGENT REVIEW: High model disagreement across Monte Carlo forward passes. "
            "Scan features deviate from core distribution. Mandatory evaluation by attending neuroradiologist."
        )
        triage_required = True
    elif len(flags) == 1:
        confidence_tier = "TIER_2_MODERATE_UNCERTAINTY"
        triage_status = "TRIAGE_ADVISED"
        recommendation = (
            "SECONDARY REVIEW: Borderline decision boundaries detected. "
            "Recommend clinical correlation and secondary radiologist verification."
        )
        triage_required = True
    else:
        confidence_tier = "TIER_1_HIGH_CONFIDENCE"
        triage_status = "AUTONOMOUS_VERIFIED"
        recommendation = (
            "CONFIDENT DIAGNOSIS: Consistent Bayesian convergence across all 30 stochastic passes. "
            "Autonomous diagnostic report approved for radiology review."
        )
        triage_required = False

    return {
        "triage_required": triage_required,
        "triage_status": triage_status,
        "confidence_tier": confidence_tier,
        "flags": flags,
        "recommendation": recommendation,
        "metrics_evaluated": {
            "predicted_class": uncertainty_dict["predicted_class_name"],
            "confidence": conf,
            "variance": var,
            "entropy": entropy,
            "mutual_information": mi,
        },
    }
