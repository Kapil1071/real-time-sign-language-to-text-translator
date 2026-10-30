"""
train_model.py - Optimized, Robust Sign Language CNN with Dropout & Data Augmentation
Designed to prevent overfitting and ensure strong generalization on unseen data and varying lighting.
"""

import os, json, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET_DIR = Path(r"C:\Users\ASUS\Downloads\SIGN_DATA\asl_alphabet_train\asl_alphabet_train")
UNSEEN_TEST_DIR = Path(r"C:\Users\ASUS\Downloads\SIGN_DATA\asl_alphabet_test\asl_alphabet_test")
IMG_SIZE = 64
MAX_PER_CLASS = 400   # 400 * 29 = 11,600 images: fast training, balanced representation
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def prepare_rgb(pil_image):
    return np.asarray(ImageOps.fit(pil_image.convert("RGB"), (IMG_SIZE, IMG_SIZE),
                                   method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
print(f"Discovered {len(classes)} classes: {classes}")

X, y = [], []
for idx, label in enumerate(classes):
    folder = DATASET_DIR / label
    files = [f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    random.shuffle(files)
    if MAX_PER_CLASS is not None:
        files = files[:MAX_PER_CLASS]
    for f in files:
        try:
            with Image.open(f) as im:
                X.append(prepare_rgb(im))
            y.append(idx)
        except Exception:
            continue
    print(f"  [{label}] loaded: {len(files)} images")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)
print(f"\nTotal Dataset: {len(X)} images")

# 70% Train, 15% Validation, 15% Held-out Test
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
del X, y, X_tmp, y_tmp

# ── ARCHITECTURE WITH SPATIAL & DENSE DROPOUT ─────────────────────────────────
# 1. Spatial data augmentation handles dark lighting, translation, scale shifts
# 2. Dropout layers after Conv blocks (SpatialDropout) and Dense layers prevent co-adaptation/overfitting
model = models.Sequential([
    layers.Input((IMG_SIZE, IMG_SIZE, 3)),

    # Data Augmentation: Contrast/Brightness, Rotation, Zoom, Translation
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.20),

    # Block 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.20),  # Dropout 1

    # Block 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Dropout 2

    # Block 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.30),  # Dropout 3

    # Dense Classifier with Dropout
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.40),  # Dropout 4 (Dense)
    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── TRAINING ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Also save model.keras and labels.json
model.save("model.keras")
with open("labels.json", "w") as f:
    json.dump(classes, f)
print("\nSaved model.h5, model.keras, and labels.json")

# ── HELD-OUT TEST EVALUATION ──────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n==========================================")
print(f"  HELD-OUT TEST ACCURACY: {test_acc * 100:.2f}% | LOSS: {test_loss:.4f}")
print(f"==========================================")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification Report (Held-Out Split):")
print(classification_report(y_test, y_pred, target_names=classes, digits=4))

# ── UNSEEN REAL TEST DATASET EVALUATION ───────────────────────────────────────
if UNSEEN_TEST_DIR.exists():
    unseen_files = sorted([f for f in UNSEEN_TEST_DIR.iterdir() if f.suffix.lower() in {".jpg", ".png"}])
    correct = 0
    total = len(unseen_files)
    print("\n==========================================")
    print("  EVALUATING UNSEEN TEST SAMPLES")
    print("==========================================")
    for f in unseen_files:
        true_label = f.stem.split("_")[0].upper()
        try:
            with Image.open(f) as im:
                arr = prepare_rgb(im)
            preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_label = classes[pred_idx].upper()
            conf = preds[pred_idx]
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            print(f"  {f.name:18s} -> True: {true_label:7s} | Pred: {pred_label:7s} ({conf*100:5.1f}%) | {'PASS' if is_correct else 'FAIL'}")
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    if total > 0:
        print(f"\nUnseen Test Dataset Accuracy: {correct}/{total} = {correct*100.0/total:.2f}%")

# ── SAVE PERFORMANCE PLOTS ────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy with Regularization")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss (Train vs Val)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix (Regularized Model)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved training_curves.png and confusion_matrix.png plots.")
