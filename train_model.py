"""
train_model.py - Train a Convolutional Neural Network (CNN) on Sign Language Dataset
Author: Viva Project
Description:
    1. Loads images from 'dataset/' (29 classes: A-Z, del, nothing, space)
    2. Resizes to 64x64 and normalizes pixel values (0 to 1)
    3. Builds a simple 3-layer CNN classifier
    4. Uses EarlyStopping to prevent overfitting
    5. Saves the best model as 'model.h5' and labels as 'labels.json'
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 1. HYPERPARAMETERS & SETTINGS

DATASET_DIR = "dataset"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 13            # Maximum epochs (EarlyStopping stops automatically)
MAX_PER_CLASS = 150          # 150 images per class for fast training

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("=" * 60)
print("  SIGN LANGUAGE CNN TRAINING STARTED")
print(f"  Dataset path: {DATASET_DIR}")
print("=" * 60)


# 2. DISCOVER CLASSES (Folder names)

classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

print(f"Found {len(classes)} classes: {classes}")


# 3. LOAD & PREPROCESS IMAGES

X, y = [], []

for label_idx, label_name in enumerate(classes):

    folder = os.path.join(DATASET_DIR, label_name)
    files = os.listdir(folder)

    # Randomly select up to MAX_PER_CLASS images
    if len(files) > MAX_PER_CLASS:
        files = random.sample(files, MAX_PER_CLASS)

    for fname in files:

        fpath = os.path.join(folder, fname)

        try:
            # Open image, convert to RGB, resize and normalize
            img = Image.open(fpath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

            img_arr = np.array(img, dtype="float32") / 255.0

            X.append(img_arr)
            y.append(label_idx)

        except Exception:
            continue


X = np.array(X, dtype="float32")
y = np.array(y, dtype="int32")

print(f"Loaded {len(X)} total images across {len(classes)} classes.")


# 4. SPLIT DATA (70% Train, 15% Validation, 15% Test)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")


# 5. BUILD SIMPLE CNN ARCHITECTURE

model = models.Sequential([

    # Layer 1: Feature Extraction (Edges & basic contours)
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Layer 2: Complex Feature Extraction (Fingers & hand shapes)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Layer 3: High-level classification
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten 2D matrices into a 1D vector
    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    # Dropout prevents overfitting
    layers.Dropout(0.3),

    # Output Layer: Probabilities for each sign class
    layers.Dense(len(classes), activation='softmax')

])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 6. EARLY STOPPING & CALLBACKS

# Stop training if validation loss does not improve for 5 epochs
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Automatically save the best-performing model
checkpoint = ModelCheckpoint(
    filepath="model.h5",
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Reduce learning rate if validation loss stops improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)


# 7. TRAIN THE MODEL

print("\nTraining CNN with Early Stopping...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ],

    verbose=1
)


# Display actual training epochs
actual_epochs = len(history.history['loss'])

# Find best epoch based on minimum validation loss
best_epoch = np.argmin(history.history['val_loss']) + 1

print("\n" + "=" * 50)
print("  TRAINING SUMMARY")
print("=" * 50)

print(f"Maximum Epochs:       {EPOCHS}")
print(f"Actual Epochs:        {actual_epochs}")
print(f"Best Epoch:           {best_epoch}")

print(f"Best Validation Loss: {min(history.history['val_loss']):.4f}")

print(f"Best Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")

print("=" * 50)


# 8. EVALUATE ON TEST SET

# Restore the best checkpoint before final evaluation
model = models.load_model("model.h5")

test_loss, test_acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 50)
print(f"  TEST ACCURACY: {test_acc * 100:.2f}%")
print(f"  TEST LOSS:     {test_loss:.4f}")
print("=" * 50)


# Generate predictions for classification report

y_pred = np.argmax(
    model.predict(X_test),
    axis=1
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )
)


# 9. SAVE MODEL AND LABELS

# Save the restored best model
model.save("model.h5")

with open("labels.json", "w") as f:
    json.dump(classes, f)

print("Saved 'model.h5' and 'labels.json' successfully.")


# 10. SAVE TRAINING PLOTS

plt.figure(figsize=(12, 5))


# Accuracy Graph

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Val Accuracy'
)

plt.axvline(
    x=best_epoch - 1,
    color='red',
    linestyle='--',
    label=f'Best Epoch: {best_epoch}'
)

plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# Loss Graph

plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Val Loss'
)

plt.axvline(
    x=best_epoch - 1,
    color='red',
    linestyle='--',
    label=f'Best Epoch: {best_epoch}'
)

plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()

plt.savefig("training_curves.png", dpi=300)

plt.show()

print("Saved 'training_curves.png' plot.")

print("\nAll training tasks complete!")
print("Ready to run 'python app.py'.")