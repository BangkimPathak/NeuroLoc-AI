import os
import tensorflow as tf
from neuroloc.dataset import create_dataset, CLASS_NAMES
from neuroloc.model import build_neuroloc_model, MCDropout

print("Loading balanced training and testing datasets (range 0..255)...")
train_ds = create_dataset('Training', batch_size=32, is_training=True, sample_limit=1600)
val_ds = create_dataset('Testing', batch_size=32, is_training=False, sample_limit=400)

print("Building EfficientNetB0 NeuroLoc model...")
model = build_neuroloc_model(
    input_shape=(224, 224, 3),
    num_classes=len(CLASS_NAMES),
    backbone_name="efficientnetb0",
    dropout_rate=0.3,
    weights="imagenet",
    freeze_backbone=True,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/neuroloc_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    ),
]

print("Training high-accuracy clinical model (5 epochs)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
    verbose=1,
)

print("Final Validation Accuracy:", max(history.history['val_accuracy']))
print(f"Model saved to {checkpoint_path}")
