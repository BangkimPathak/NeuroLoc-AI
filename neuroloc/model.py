"""
Module 2: Model Architecture with Monte Carlo Dropout
Custom MCDropout layer maintaining stochastic activation during inference
for Bayesian approximation, paired with transfer learning backbones
(EfficientNetB0 / DenseNet121).
"""

from typing import Tuple, Optional, Union
import logging
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package="NeuroLoc")
class MCDropout(layers.Dropout):
    """
    Monte Carlo Dropout layer for Bayesian deep learning approximation.
    Subclasses tf.keras.layers.Dropout and forces training=True during forward
    passes, even at inference time, yielding stochastic samples from the model's
    posterior parameter distribution (Gal & Ghahramani, 2016).
    """

    def __init__(self, rate: float, **kwargs):
        super().__init__(rate=rate, **kwargs)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Enforce dropout sampling regardless of the training flag
        return super().call(inputs, training=True)

    def get_config(self):
        config = super().get_config()
        return config


def build_neuroloc_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    backbone_name: str = "efficientnetb0",
    dropout_rate: float = 0.3,
    weights: Optional[str] = "imagenet",
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """
    Construct the NeuroLoc-XAI deep learning architecture.

    Architecture:
        Input (224x224x3)
        -> Pretrained Backbone (EfficientNetB0 / DenseNet121)
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> MCDropout(dropout_rate)
        -> Dense(128, activation='relu')
        -> BatchNormalization
        -> MCDropout(dropout_rate)
        -> Dense(num_classes, activation='softmax')

    Args:
        input_shape: Input tensor shape (H, W, C).
        num_classes: Number of diagnostic categories (default 4).
        backbone_name: 'efficientnetb0' or 'densenet121'.
        dropout_rate: Dropout rate for stochastic MC sampling.
        weights: Pretrained weights ('imagenet' or None).
        freeze_backbone: Whether to freeze backbone weights initially for transfer learning.

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_scan")

    # Instantiate backbone
    backbone_lower = backbone_name.lower()
    if "efficientnet" in backbone_lower:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
        )
    elif "densenet" in backbone_lower:
        base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
        )
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. Choose 'efficientnetb0' or 'densenet121'."
        )

    # Configure backbone trainability
    base_model.trainable = not freeze_backbone
    if freeze_backbone:
        logger.info("Freezing base model backbone (%s) for Stage 1 warm-up.", backbone_name)
    else:
        logger.info("Base model backbone (%s) is fully trainable.", backbone_name)

    # Feature extraction & classification head
    features = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    x = layers.BatchNormalization(name="head_batch_norm_1")(x)
    x = MCDropout(rate=dropout_rate, name="mc_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_1")(x)
    x = layers.BatchNormalization(name="head_batch_norm_2")(x)
    x = MCDropout(rate=dropout_rate, name="mc_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"NeuroLoc_{backbone_name}")
    return model


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Automatically detect the name of the final 2D convolutional layer
    in the model hierarchy for Grad-CAM / Grad-CAM++ targeting.

    Args:
        model: tf.keras.Model instance.

    Returns:
        Layer name string (e.g. 'top_conv' for EfficientNetB0).
    """
    # Common known final conv layer names for popular backbones
    known_targets = ["top_conv", "conv5_block16_2_conv", "relu"]
    for layer in reversed(model.layers):
        if layer.name in known_targets:
            return layer.name

    # Fallback: search backwards for any 4D tensor producing conv layer
    for layer in reversed(model.layers):
        # Inspect inner backbone if model wraps a functional/sequential model
        if hasattr(layer, "layers"):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, (layers.Conv2D, layers.DepthwiseConv2D)):
                    return inner_layer.name
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            return layer.name

    # If top layer is inside functional graph
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name

    raise ValueError("Could not automatically locate a convolutional layer in the model.")


def unfreeze_backbone_layers(
    model: tf.keras.Model,
    num_layers_to_unfreeze: int = 20,
) -> None:
    """
    Unfreeze the top N layers of the backbone for Stage 2 fine-tuning,
    keeping lower feature extractors (edges/textures) stable.

    Args:
        model: NeuroLoc tf.keras.Model.
        num_layers_to_unfreeze: Number of top backbone layers to unfreeze.
    """
    # Find base model layer or iterate top-level layers
    target_layers = model.layers
    unfrozen_count = 0

    for layer in reversed(target_layers):
        # Don't touch head layers
        if layer.name in ("classification_head", "mc_dropout_2", "head_batch_norm_2",
                          "head_dense_1", "mc_dropout_1", "head_batch_norm_1", "global_average_pooling"):
            continue

        # Keep BatchNormalization frozen during fine-tuning for stability
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue

        layer.trainable = True
        unfrozen_count += 1
        if unfrozen_count >= num_layers_to_unfreeze:
            break

    logger.info("Unfroze top %d backbone layers for fine-tuning.", unfrozen_count)
