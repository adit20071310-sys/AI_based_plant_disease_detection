# ============================================
# PLANT DISEASE DETECTOR — TRANSFER LEARNING
# Uses MobileNetV2 pretrained on ImageNet
# Much better real-world accuracy than CNN
# from scratch
# ============================================

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers, models
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "dataset"
IMG_SIZE     = 224          # MobileNetV2 expects 224x224
BATCH_SIZE   = 32
MODEL_PATH   = "model_cnn.h5"

SELECTED_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
]

NUM_CLASSES = len(SELECTED_CLASSES)

print(f"\n📦 Classes: {NUM_CLASSES}")
print(f"📁 Dataset: {DATASET_PATH}")
print(f"🖼  Image size: {IMG_SIZE}x{IMG_SIZE}\n")

# ============================================
# PHASE 1 DATA GENERATORS
# Heavy augmentation so the model learns
# to handle real-world phone photos —
# different angles, lighting, backgrounds
# ============================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,

    # Geometry
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,

    # Color/lighting — key for real-world photos
    brightness_range=[0.6, 1.4],
    channel_shift_range=30.0,

    fill_mode="nearest",
    validation_split=0.2
)

# Validation: only rescale, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=SELECTED_CLASSES,
    subset="training",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=SELECTED_CLASSES,
    subset="validation",
    shuffle=False,
)

print(f"✅ Training samples  : {train_gen.samples}")
print(f"✅ Validation samples: {val_gen.samples}")
print(f"✅ Class map: {train_gen.class_indices}\n")

# ============================================
# SAVE CLASS NAMES
# ============================================

class_names = list(train_gen.class_indices.keys())

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print("✅ class_names.json saved\n")

# ============================================
# BUILD MODEL — TRANSFER LEARNING
#
# MobileNetV2 pretrained on ImageNet is used
# as a feature extractor. We freeze its
# weights in Phase 1 and only train the new
# classification head. In Phase 2 we unfreeze
# the top layers for fine-tuning.
# ============================================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,          # remove ImageNet classifier
    weights="imagenet"          # use pretrained weights
)

# Freeze base — don't touch pretrained weights yet
base_model.trainable = False

# Build full model
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# ============================================
# PHASE 1 — Train only the new head
# Base model stays frozen
# ============================================

print("\n🚀 PHASE 1: Training classification head...\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase1 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    callbacks=callbacks_phase1
)

# ============================================
# PHASE 2 — Fine-tune top layers of base
#
# Unfreeze the last 30 layers of MobileNetV2
# and retrain with a very low learning rate.
# This lets the base adapt slightly to leaf
# images without forgetting ImageNet features.
# ============================================

print("\n🔧 PHASE 2: Fine-tuning top layers...\n")

base_model.trainable = True

# Freeze all except last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with much lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks_phase2
)

# ============================================
# FINAL EVALUATION
# ============================================

print("\n📊 Evaluating on validation set...")

loss, acc = model.evaluate(val_gen, verbose=1)
print(f"\n✅ Final Validation Accuracy: {acc*100:.2f}%")
print(f"✅ Final Validation Loss    : {loss:.4f}")
print(f"✅ Best model saved to      : {MODEL_PATH}")

# ============================================
# PLOT TRAINING CURVES
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History (Phase 1 + Phase 2)", fontsize=13)

# Combine histories
acc_all     = history1.history["accuracy"]     + history2.history["accuracy"]
val_acc_all = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss_all    = history1.history["loss"]         + history2.history["loss"]
val_loss_all= history1.history["val_loss"]     + history2.history["val_loss"]
phase_split = len(history1.history["accuracy"])

epochs_range = range(1, len(acc_all) + 1)

axes[0].plot(epochs_range, acc_all,     label="Train Acc",  color="#4ade80", linewidth=2)
axes[0].plot(epochs_range, val_acc_all, label="Val Acc",    color="#22d3ee", linewidth=2)
axes[0].axvline(phase_split, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(epochs_range, loss_all,     label="Train Loss", color="#f97316", linewidth=2)
axes[1].plot(epochs_range, val_loss_all, label="Val Loss",   color="#ef4444", linewidth=2)
axes[1].axvline(phase_split, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("📊 training_curves.png saved")
