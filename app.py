"""Flask webcam sign recognizer. Single capture worker; stream latest JPEG, never queue old frames."""
import os, json, time, threading
from collections import deque, Counter
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import keras
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = .80
PREDICT_EVERY = 3
STABLE_PREDICTIONS = 5
COOLDOWN_SECONDS = 1.0
PADDING = 25
JPEG_QUALITY = 70
CAMERA_INDEX = 0
# If your training images are NOT tight hand crops, collect/retrain with matching crops.
model_path = "model.keras" if os.path.exists("model.keras") else "model.h5"
if not os.path.exists(model_path) or not os.path.exists("labels.json"):
    raise FileNotFoundError("Run train_model_optimized.py first (model and labels missing).")
model = keras.models.load_model(model_path, compile=False)
with open("labels.json", encoding="utf-8") as f:
    LABELS = json.load(f)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
lock = threading.RLock()
frame_condition = threading.Condition()
latest_jpeg = None
frame_sequence = 0
camera_active = True
detection_active = True
state = dict(sign="", confidence=0.0, sentence="", status="Ready - Position hand in view",
             fps=0.0, camera_active=True, detection_active=True)
tts_lock = threading.Lock()

def standby():
    image = np.full((480, 640, 3), (42, 23, 15), dtype=np.uint8)
    cv2.putText(image, "Camera Paused", (170, 230), cv2.FONT_HERSHEY_SIMPLEX,
                1, (220, 220, 220), 2)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()

STANDBY = standby()

def publish(data):
    global latest_jpeg, frame_sequence
    with frame_condition:
        latest_jpeg = data
        frame_sequence += 1
        frame_condition.notify_all()

def crop_for_model(clean_frame, box):
    x1, y1, x2, y2 = box
    crop = clean_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Same center-crop / resize policy as ImageOps.fit in training.
    h, w = crop.shape[:2]
    side = min(h, w)
    x0, y0 = (w-side)//2, (h-side)//2
    crop = crop[y0:y0+side, x0:x0+side]
    
    # Adaptive Histogram Equalization (CLAHE) on luminance for robust low-light & dark room detection
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_crop = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    
    rgb = cv2.cvtColor(cv2.resize(enhanced_crop, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.astype(np.float32) / 255., axis=0)

def capture_worker():
    global camera_active, detection_active
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        with lock:
            state["status"] = "Cannot open webcam. Check camera permissions/index."
        publish(STANDBY)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # driver-dependent
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_no = 0
    last_time = time.monotonic()
    candidate = None
    stable = 0
    last_accepted = 0.0
    # Create one MediaPipe tracker for this single capture thread.
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=.7,
                        min_tracking_confidence=.5) as hands:
        while True:
            with lock:
                enabled = camera_active
                detecting = detection_active
            if not enabled:
                with lock:
                    state.update(sign="", confidence=0., fps=0., status="Camera Paused")
                publish(STANDBY)
                time.sleep(.15)
                # Drain a frame on resume instead of replaying a backlog.
                continue
            ok, frame = cap.read()
            if not ok:
                with lock:
                    state["status"] = "Camera read failed"
                time.sleep(.05)
                continue
            frame_no += 1
            now = time.monotonic()
            fps = 1. / max(now-last_time, 1e-6)
            last_time = now
            frame = cv2.flip(frame, 1)
            clean = frame.copy()  # NEVER classify landmark drawings.
            if detecting:
                rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    landmarks = result.multi_hand_landmarks[0]
                    h, w = frame.shape[:2]
                    xs = [int(l.x*w) for l in landmarks.landmark]
                    ys = [int(l.y*h) for l in landmarks.landmark]
                    box = (max(0, min(xs)-PADDING), max(0, min(ys)-PADDING),
                           min(w, max(xs)+PADDING), min(h, max(ys)+PADDING))
                    # Draw only on display frame, after preserving clean frame.
                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(frame, box[:2], box[2:], (16, 185, 129), 2)
                    if frame_no % PREDICT_EVERY == 0:
                        tensor = crop_for_model(clean, box)
                        if tensor is not None:
                            scores = model(tensor, training=False).numpy()[0]
                            idx = int(np.argmax(scores))
                            label, conf = LABELS[idx], float(scores[idx])
                            with lock:
                                state.update(sign=label, confidence=conf)
                                if conf >= CONFIDENCE_THRESHOLD:
                                    stable = stable+1 if candidate == label else 1
                                    candidate = label
                                    if stable >= STABLE_PREDICTIONS and now-last_accepted >= COOLDOWN_SECONDS:
                                        if label == "space":
                                            state["sentence"] += " "
                                        elif label == "del":
                                            state["sentence"] = state["sentence"][:-1]
                                        elif label != "nothing":
                                            state["sentence"] += label
                                        last_accepted = now
                                        stable = 0
                                        state["status"] = f"Confirmed: {label}"
                                    else:
                                        state["status"] = f"Tracking: {label} ({stable}/{STABLE_PREDICTIONS})"
                                else:
                                    stable, candidate = 0, None
                                    state["status"] = f"Low confidence: {label} ({conf:.0%})"
                else:
                    stable, candidate = 0, None
                    with lock:
                        state.update(sign="", confidence=0., status="No hand detected")
            else:
                stable, candidate = 0, None
                with lock:
                    state.update(sign="", confidence=0., status="AI Detection Paused")
            with lock:
                state["fps"] = round(fps, 1)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                publish(encoded.tobytes())
    cap.release()

def generate_frames():
    seen = -1
    while True:
        with frame_condition:
            frame_condition.wait_for(lambda: frame_sequence != seen, timeout=1.0)
            seen = frame_sequence
            data = latest_jpeg or STANDBY
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})

@app.route("/get_status")
def get_status():
    with lock:
        return jsonify(dict(state))

@app.route("/camera/toggle", methods=["POST"])
def toggle_camera():
    global camera_active
    with lock:
        camera_active = not camera_active
        state["camera_active"] = camera_active
        return jsonify(camera_active=camera_active)

@app.route("/detection/toggle", methods=["POST"])
def toggle_detection():
    global detection_active
    with lock:
        detection_active = not detection_active
        state["detection_active"] = detection_active
        return jsonify(detection_active=detection_active)

def speak_async(message):
    if not tts_lock.acquire(blocking=False):
        return
    try:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
        engine.stop()
    finally:
        tts_lock.release()

@app.route("/action/<action_name>", methods=["POST"])
def handle_action(action_name):
    if action_name not in {"clear", "space", "backspace", "speak"}:
        return jsonify(error="Unknown action"), 400
    with lock:
        if action_name == "clear":
            state["sentence"] = ""
            state["status"] = "Sentence Cleared"
        elif action_name == "space":
            state["sentence"] += " "
        elif action_name == "backspace":
            state["sentence"] = state["sentence"][:-1]
        sentence = state["sentence"]
    if action_name == "speak" and sentence.strip():
        threading.Thread(target=speak_async, args=(sentence.strip(),), daemon=True).start()
    return jsonify(success=True, sentence=sentence)

@app.route("/set_sentence", methods=["POST"])
def set_sentence():
    data = request.get_json(silent=True) or {}
    sentence = data.get("sentence", "")
    if not isinstance(sentence, str):
        return jsonify(error="sentence must be a string"), 400
    with lock:
        state["sentence"] = sentence[:10000]
        return jsonify(success=True, sentence=state["sentence"])

if __name__ == "__main__":
    threading.Thread(target=capture_worker, daemon=True).start()
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
