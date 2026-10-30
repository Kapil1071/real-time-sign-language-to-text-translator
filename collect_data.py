"""
collect_data.py - Capture hand sign images from webcam
Author: Viva Project
Description:
    Detects hand using MediaPipe, draws 21 landmark dots,
    and saves cropped 64x64 images when you press a key.
"""

import os
import time
import cv2
import mediapipe as mp

# 1. SETUP & CONFIGURATION
SAVE_DIR = "dataset"
IMG_SIZE = 64
PADDING = 20

# Initialize MediaPipe Hands detector
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Open default webcam (0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Please plug in or enable camera.")
    exit()

print("=" * 50)
print("  DATA COLLECTION RUNNING")
print("  - Press any letter (A-Z) to save image")
print("  - Press 1 for 'del', 2 for 'space'")
print("  - Press Q to Quit")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for natural mirror feel
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_box = None

    # If a hand is found in the frame
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw 21 landmark dots and connecting skeleton lines
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate bounding box around the hand
        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x1 = max(0, min(xs) - PADDING)
        y1 = max(0, min(ys) - PADDING)
        x2 = min(w, max(xs) + PADDING)
        y2 = min(h, max(ys) + PADDING)
        hand_box = (x1, y1, x2, y2)

        # Draw green box around the hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Hand Detected", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show instructions on screen
    cv2.putText(frame, "Press A-Z to save | Q to Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Sign Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Map pressed key to label name
    label = None
    if ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
        label = chr(key).upper()
    elif key == ord('1'):
        label = "del"
    elif key == ord('2'):
        label = "space"

    # If valid key pressed and hand is visible, save the cropped image
    if label and hand_box:
        x1, y1, x2, y2 = hand_box
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            target_folder = os.path.join(SAVE_DIR, label)
            os.makedirs(target_folder, exist_ok=True)

            # Resize to 64x64 square
            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            file_path = os.path.join(target_folder, f"{int(time.time() * 1000)}.jpg")
            cv2.imwrite(file_path, resized)

            total_saved = len(os.listdir(target_folder))
            print(f"[Saved] Label '{label}' -> Total images in folder: {total_saved}")

cap.release()
cv2.destroyAllWindows()
