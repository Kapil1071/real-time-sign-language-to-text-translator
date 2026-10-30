"""
app.py - Production-Ready Real-Time Sign Language AI Web Platform
Description:
    - Hosts responsive full-scale landing page and live interactive demo.
    - MediaPipe 21 landmark hand tracking with OpenCV.
    - Real-time ASL CNN inference with temporal stability filter.
    - Camera Start/Stop and AI Detection Toggle controls.
    - Offline Text-to-Speech (pyttsx3) and web API integration.
"""

import os
import time
import json
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import keras
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# ── CONFIGURATION ────────────────────────────────────────────────────────────
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.70      # 70% threshold
STABLE_FRAMES = 12               # Frames to confirm sign
COOLDOWN_SECONDS = 1.0           # Cooldown between accepted signs
PADDING = 25

# ── LOAD MODEL & LABELS ──────────────────────────────────────────────────────
if not os.path.exists("model.h5") or not os.path.exists("labels.json"):
    print("ERROR: 'model.h5' or 'labels.json' missing. Run 'python train_model.py' first.")
    exit()

print("Loading CNN model...")
model = keras.models.load_model("model.h5")

with open("labels.json", "r") as f:
    LABELS = json.load(f)

# ── MEDIAPIPE & TTS ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

tts_engine = pyttsx3.init()

# ── GLOBAL SYSTEM STATE ──────────────────────────────────────────────────────
camera_active = True
detection_active = True

state = {
    "sign": "",
    "confidence": 0.0,
    "sentence": "",
    "status": "Ready - Position hand in view",
    "fps": 30.0,
    "camera_active": True,
    "detection_active": True
}

stable_count = 0
last_predicted_sign = None
last_accepted_time = 0
prev_time = time.time()

# ── STANDBY FRAME (Shown when camera is paused) ──────────────────────────────
def create_standby_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (15, 23, 42)  # Dark slate blue background (#0f172a)
    cv2.putText(frame, "Camera Paused", (200, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (148, 163, 184), 2)
    cv2.putText(frame, "Click 'Start Camera' below to resume live detection", (110, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 116, 139), 1)
    ret, buf = cv2.imencode('.jpg', frame)
    return buf.tobytes()

# ── WEBCAM STREAM GENERATOR (MJPEG) ──────────────────────────────────────────
def generate_frames():
    global stable_count, last_predicted_sign, last_accepted_time, prev_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    standby_bytes = create_standby_frame()

    while True:
        if not camera_active:
            # Yield standby image when paused
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + standby_bytes + b'\r\n')
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # Mirror mode
        h, w, _ = frame.shape

        # Calculate FPS
        curr_time = time.time()
        state["fps"] = round(1.0 / max(curr_time - prev_time, 1e-6), 1)
        prev_time = curr_time

        predicted_label = ""
        confidence = 0.0

        if detection_active:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw 21 landmark dots (green) and skeleton lines (cyan)
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(16, 185, 129), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(6, 182, 212), thickness=2)
                )

                # Get bounding box
                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1 = max(0, min(xs) - PADDING)
                y1 = max(0, min(ys) - PADDING)
                x2 = min(w, max(xs) + PADDING)
                y2 = min(h, max(ys) + PADDING)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)

                # Crop & CNN classification
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
                    input_tensor = np.expand_dims(crop_resized, axis=0)

                    preds = model.predict(input_tensor, verbose=0)[0]
                    best_idx = int(np.argmax(preds))
                    confidence = float(preds[best_idx])
                    predicted_label = LABELS[best_idx]

                    state["sign"] = predicted_label
                    state["confidence"] = confidence

                    # Stability logic
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if predicted_label == last_predicted_sign:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_predicted_sign = predicted_label

                        if stable_count >= STABLE_FRAMES and (curr_time - last_accepted_time) > COOLDOWN_SECONDS:
                            if predicted_label == "space":
                                state["sentence"] += " "
                            elif predicted_label == "del":
                                state["sentence"] = state["sentence"][:-1]
                            elif predicted_label != "nothing":
                                state["sentence"] += predicted_label

                            last_accepted_time = curr_time
                            stable_count = 0
                            state["status"] = f"✓ Confirmed: {predicted_label}"
                        else:
                            state["status"] = f"Tracking: {predicted_label} ({stable_count}/{STABLE_FRAMES})"
                    else:
                        stable_count = 0
                        state["status"] = f"Detecting: {predicted_label} ({int(confidence*100)}%)"
            else:
                state["sign"] = ""
                state["confidence"] = 0.0
                stable_count = 0
                state["status"] = "No hand detected"
        else:
            state["status"] = "AI Detection Paused"
            state["sign"] = ""
            state["confidence"] = 0.0

        # Encode and stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    state["camera_active"] = camera_active
    state["detection_active"] = detection_active
    return jsonify(state)

@app.route('/camera/toggle', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    state["camera_active"] = camera_active
    state["status"] = "Camera Started" if camera_active else "Camera Stopped"
    return jsonify({"camera_active": camera_active})

@app.route('/detection/toggle', methods=['POST'])
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    state["detection_active"] = detection_active
    state["status"] = "AI Detection Enabled" if detection_active else "AI Detection Paused"
    return jsonify({"detection_active": detection_active})

@app.route('/action/<action_name>', methods=['POST'])
def handle_action(action_name):
    if action_name == 'clear':
        state['sentence'] = ""
        state['status'] = "Sentence Cleared"
    elif action_name == 'space':
        state['sentence'] += " "
    elif action_name == 'backspace':
        state['sentence'] = state['sentence'][:-1]
    elif action_name == 'speak':
        text_to_say = state['sentence'].strip()
        if text_to_say:
            try:
                tts_engine.say(text_to_say)
                tts_engine.runAndWait()
            except Exception:
                pass

    return jsonify({"success": True, "sentence": state["sentence"]})

@app.route('/set_sentence', methods=['POST'])
def set_sentence():
    data = request.get_json() or {}
    new_text = data.get("sentence", "")
    state["sentence"] = new_text
    return jsonify({"success": True, "sentence": state["sentence"]})

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  SIGN2TEXT WEB PLATFORM READY")
    print("  Access the full site at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
