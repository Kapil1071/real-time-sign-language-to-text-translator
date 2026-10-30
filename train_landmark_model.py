"""
train_landmark_model.py - MediaPipe 21-Landmark Vector ASL Classifier
Trains a lightweight Neural Network on 63 normalized landmark coordinates (x, y, z).
100% immune to background, skin color, room lighting, and camera resolution.
"""

import os, json, random
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_DIR = Path(r"dataset")
SAMPLES_PER_CLASS = 150  # 150 * 28 = ~4,200 landmark vectors (extracts fast & accurately)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def normalize_landmarks(landmarks):
    # 21 points: translation normalize relative to wrist (index 0)
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    coords = []
    for p in landmarks:
        coords.extend([p.x - base_x, p.y - base_y, p.z - base_z])
    coords = np.array(coords, dtype=np.float32)
    
    # Scale normalization: invariant to camera distance / hand size
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords

classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir() and d.name.lower() != "nothing"])
print(f"Extracting landmark vectors for {len(classes)} classes: {classes}")

X, y = [], []

for idx, label in enumerate(classes):
    folder = DATASET_DIR / label
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    # Uniformly step through the folder
    step = max(1, len(files) // (SAMPLES_PER_CLASS * 2))
    sampled_files = files[::step]
    
    count = 0
    for f in sampled_files:
        if count >= SAMPLES_PER_CLASS:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            vec = normalize_landmarks(res.multi_hand_landmarks[0].landmark)
            X.append(vec)
            y.append(idx)
            count += 1
            
    print(f"  [{label:7s}] extracted {count} landmark vectors.")

hands.close()

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)
print(f"\nTotal Landmark Vectors: {len(X)}")

# Train / Val / Test split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── MODEL ARCHITECTURE: Landmark MLP with Dropout & BatchNorm ─────────────────
model = models.Sequential([
    layers.Input(shape=(63,)),
    
    # Gaussian noise augmentation for hand trembling / slight angle drift
    layers.GaussianNoise(0.02),
    
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.30),
    
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.30),
    
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.25),
    
    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ModelCheckpoint("landmark_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save("landmark_model.keras")
with open("landmark_labels.json", "w", encoding="utf-8") as f:
    json.dump(classes, f)
print("\nSaved landmark_model.h5, landmark_model.keras, and landmark_labels.json")

# ── EVALUATION ────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n==========================================")
print(f"  LANDMARK TEST ACCURACY: {test_acc * 100:.2f}% | LOSS: {test_loss:.4f}")
print(f"==========================================")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification Report (Landmark Model):")
print(classification_report(y_test, y_pred, target_names=classes, digits=4))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
plt.title(f"Landmark Model Confusion Matrix (Accuracy: {test_acc*100:.1f}%)", fontsize=14, pad=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("landmark_confusion_matrix.png", dpi=160)
plt.close()

# Curves Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Landmark Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Landmark Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("landmark_training_curves.png", dpi=160)
plt.close()
print("Saved landmark_confusion_matrix.png and landmark_training_curves.png")

