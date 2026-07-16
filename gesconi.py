"""
Gesconi — gesture-controlled mouse
------------------------------------
Uses your webcam + MediaPipe hand tracking to move the system cursor
and trigger clicks, drags, scrolling, screenshots, and letter typing
using hand gestures.

Supports dual modes:
  1. Web Mode (default): Starts a Flask server serving a beautiful dashboard
     website with an interactive "False Cursor Pad" sandboxed live demo.
  2. Desktop Mode (run with --desktop): The original desktop-only python command.
"""

import os
import sys
import time
import math
import argparse
import threading
import queue
import json
from datetime import datetime

import cv2
import mediapipe as mp
import pyautogui
from flask import Flask, Response, jsonify, send_from_directory, request

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

CAM_INDEX = 0                # change if you have multiple webcams
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Region of the webcam frame that maps to the screen
FRAME_MARGIN = 100

SMOOTHING = 5                # higher = smoother but laggier cursor movement
PINCH_TRIGGER_DIST = 0.045   # normalized distance (0-1) that counts as a pinch
PINCH_HOLD_TIME = 0.35       # seconds a pinch must be held before it becomes a drag
CLICK_COOLDOWN = 0.4         # seconds between repeat clicks of the same type
FINGER_EXTENDED_RATIO = 1.15 # finger extended threshold ratio

SCREENSHOT_PINCH_DIST = 0.08
SCREENSHOT_COOLDOWN = 1.5
SCREENSHOT_DIR = "screenshots"

LETTER_COOLDOWN = 0.6

SCROLL_ZONE_DEADZONE = 0.06
SCROLL_STEP = 40

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

# ---------------------------------------------------------------------------
# Math & Tracking Helpers
# ---------------------------------------------------------------------------

def landmark_dist(a, b):
    """Euclidean distance between two landmarks (normalized coordinates)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(landmarks, extended_ratio=FINGER_EXTENDED_RATIO):
    """Returns a dict of which fingers are extended."""
    wrist = landmarks[0]
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pips = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

    up = {}
    for name in tips:
        tip_dist = landmark_dist(wrist, landmarks[tips[name]])
        pip_dist = landmark_dist(wrist, landmarks[pips[name]])
        up[name] = tip_dist > pip_dist * extended_ratio
    return up


def classify_gesture(landmarks, pinch_start_time, prev_gesture, 
                     pinch_dist=PINCH_TRIGGER_DIST,
                     extended_ratio=FINGER_EXTENDED_RATIO):
    """Classifies the hand landmark arrangement into a gesture string."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    pinky_tip = landmarks[20]

    pinch_index_dist = landmark_dist(thumb_tip, index_tip)
    pinch_middle_dist = landmark_dist(thumb_tip, middle_tip)
    pinch_pinky_dist = landmark_dist(thumb_tip, pinky_tip)
    up = fingers_up(landmarks, extended_ratio)

    is_fist = not any(up.values())
    if is_fist:
        return "pause", up

    # Thumb + pinky pinch -> screenshot
    if (
        pinch_pinky_dist < SCREENSHOT_PINCH_DIST
        and pinch_index_dist >= pinch_dist
        and pinch_middle_dist >= pinch_dist
    ):
        return "screenshot", up

    # Thumb + middle pinch -> right click
    if pinch_middle_dist < pinch_dist and pinch_index_dist >= pinch_dist:
        return "right_click", up

    # Thumb + index pinch -> left click or drag
    if pinch_index_dist < pinch_dist:
        now = time.time()
        if pinch_start_time["t"] is None:
            pinch_start_time["t"] = now
        held_for = now - pinch_start_time["t"]
        if held_for >= PINCH_HOLD_TIME:
            return "drag", up
        return "click_pending", up
    else:
        pinch_start_time["t"] = None

    # Type letter check (edge triggered from pause/fist)
    if prev_gesture == "pause":
        if up["index"] and up["middle"] and up["ring"] and not up["pinky"]:
            return "type_c", up
        if up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
            return "type_b", up
        if up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
            return "type_a", up

    # Scroll
    if up["index"] and up["middle"] and up["ring"] and up["pinky"]:
        return "scroll", up

    # Move cursor
    if up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
        return "move", up

    return "idle", up


# ---------------------------------------------------------------------------
# Gesture Tracker Engine
# ---------------------------------------------------------------------------

class GestureTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.video_subscribers = []
        self.data_subscribers = []

        # Runtime states
        self.control_real_mouse = False
        self.current_gesture = "idle"
        self.hand_detected = False
        self.pointer_coords = (0.5, 0.5)
        self.fps = 0

        # Customizable parameters
        self.smoothing = SMOOTHING
        self.pinch_trigger_dist = PINCH_TRIGGER_DIST
        self.finger_extended_ratio = FINGER_EXTENDED_RATIO

    def subscribe_video(self):
        with self.lock:
            q = queue.Queue(maxsize=3)
            self.video_subscribers.append(q)
            self._check_start()
            return q

    def unsubscribe_video(self, q):
        with self.lock:
            if q in self.video_subscribers:
                self.video_subscribers.remove(q)
            self._check_stop()

    def subscribe_data(self):
        with self.lock:
            q = queue.Queue(maxsize=5)
            self.data_subscribers.append(q)
            self._check_start()
            return q

    def unsubscribe_data(self, q):
        with self.lock:
            if q in self.data_subscribers:
                self.data_subscribers.remove(q)
            self._check_stop()

    def _check_start(self):
        if not self.running and (len(self.video_subscribers) > 0 or len(self.data_subscribers) > 0 or self.control_real_mouse):
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("Background tracking thread started.")

    def _check_stop(self):
        if self.running and len(self.video_subscribers) == 0 and len(self.data_subscribers) == 0 and not self.control_real_mouse:
            self.running = False
            print("Background tracking thread stopping...")

    def _run_loop(self):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            self.running = False
            return

        prev_screen_x, prev_screen_y = pyautogui.position()
        # Initial normalize coordinates starts at center
        prev_norm_x, prev_norm_y = 0.5, 0.5
        pinch_start_time = {"t": None}
        is_dragging = False
        last_action_time = {"click": 0.0, "right_click": 0.0, "screenshot": 0.0, "letter": 0.0}
        prev_gesture = "idle"

        last_frame_time = time.time()
        fps_smoothed = 30.0

        with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:

            while self.running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    time.sleep(0.01)
                    continue

                # Calculate smoothed FPS
                now = time.time()
                diff = now - last_frame_time
                last_frame_time = now
                if diff > 0:
                    fps_smoothed = fps_smoothed * 0.9 + (1.0 / diff) * 0.1
                self.fps = int(fps_smoothed)

                frame = cv2.flip(frame, 1)  # mirror frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                gesture = "idle"
                hand_detected = False
                coords = (0.5, 0.5)

                if results.multi_hand_landmarks:
                    hand_detected = True
                    landmarks = results.multi_hand_landmarks[0].landmark
                    
                    # Draw visual skeleton
                    mp_drawing.draw_landmarks(
                        frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(220, 100, 70), thickness=2, circle_radius=2), # Cyan-ish/Blue lines
                        mp_drawing.DrawingSpec(color=(60, 240, 100), thickness=2, circle_radius=3)   # Green/Orange points
                    )

                    gesture, up = classify_gesture(landmarks, pinch_start_time, prev_gesture,
                                                   self.pinch_trigger_dist, self.finger_extended_ratio)
                    
                    index_tip = landmarks[8]

                    # Map coordinates within margins
                    x = min(max(index_tip.x * FRAME_WIDTH, FRAME_MARGIN), FRAME_WIDTH - FRAME_MARGIN)
                    y = min(max(index_tip.y * FRAME_HEIGHT, FRAME_MARGIN), FRAME_HEIGHT - FRAME_MARGIN)

                    norm_x = (x - FRAME_MARGIN) / (FRAME_WIDTH - 2 * FRAME_MARGIN)
                    norm_y = (y - FRAME_MARGIN) / (FRAME_HEIGHT - 2 * FRAME_MARGIN)

                    # Smooth normalized coords
                    smooth_norm_x = prev_norm_x + (norm_x - prev_norm_x) / self.smoothing
                    smooth_norm_y = prev_norm_y + (norm_y - prev_norm_y) / self.smoothing
                    prev_norm_x, prev_norm_y = smooth_norm_x, smooth_norm_y
                    coords = (smooth_norm_x, smooth_norm_y)

                    # Execute system-level mouse control via PyAutoGUI if enabled
                    if self.control_real_mouse:
                        screen_x = int(smooth_norm_x * SCREEN_W)
                        screen_y = int(smooth_norm_y * SCREEN_H)

                        if gesture == "pause":
                            if is_dragging:
                                pyautogui.mouseUp()
                                is_dragging = False

                        elif gesture == "move":
                            if is_dragging:
                                pyautogui.mouseUp()
                                is_dragging = False
                            pyautogui.moveTo(screen_x, screen_y)

                        elif gesture == "drag":
                            if not is_dragging:
                                pyautogui.mouseDown()
                                is_dragging = True
                            pyautogui.moveTo(screen_x, screen_y)

                        elif gesture == "right_click":
                            curr_time = time.time()
                            if curr_time - last_action_time["right_click"] > CLICK_COOLDOWN:
                                pyautogui.click(button="right")
                                last_action_time["right_click"] = curr_time

                        elif gesture == "scroll":
                            palm_y = landmarks[9].y
                            center = 0.5
                            if palm_y < center - SCROLL_ZONE_DEADZONE:
                                pyautogui.scroll(SCROLL_STEP)
                            elif palm_y > center + SCROLL_ZONE_DEADZONE:
                                pyautogui.scroll(-SCROLL_STEP)

                        elif gesture == "screenshot":
                            curr_time = time.time()
                            if curr_time - last_action_time["screenshot"] > SCREENSHOT_COOLDOWN:
                                shot = pyautogui.screenshot()
                                filename = os.path.join(
                                    SCREENSHOT_DIR,
                                    f"gesconi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                )
                                shot.save(filename)
                                last_action_time["screenshot"] = curr_time

                        elif gesture in ("type_a", "type_b", "type_c"):
                            letter = gesture.split("_")[1]
                            curr_time = time.time()
                            if curr_time - last_action_time["letter"] > LETTER_COOLDOWN:
                                pyautogui.press(letter)
                                last_action_time["letter"] = curr_time

                        if prev_gesture == "click_pending" and gesture not in ("click_pending", "drag"):
                            curr_time = time.time()
                            if curr_time - last_action_time["click"] > CLICK_COOLDOWN:
                                pyautogui.click(button="left")
                                last_action_time["click"] = curr_time

                    prev_gesture = gesture
                else:
                    if self.control_real_mouse and is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                    pinch_start_time["t"] = None
                    prev_gesture = "idle"
                    gesture = "idle"

                self.current_gesture = gesture
                self.hand_detected = hand_detected
                self.pointer_coords = coords

                # Draw local visual overlay for Flask stream
                mode_str = "SYSTEM MOUSE" if self.control_real_mouse else "WEB SANDBOX"
                cv2.putText(
                    frame, f"MODE: {mode_str} | GESTURE: {self.current_gesture.upper()}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 100) if self.control_real_mouse else (50, 150, 250), 2
                )
                cv2.putText(
                    frame, f"FPS: {self.fps}", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
                )

                # Send image frame to subscribers
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    jpeg_bytes = jpeg.tobytes()
                    with self.lock:
                        for q in list(self.video_subscribers):
                            try:
                                if q.full():
                                    q.get_nowait()
                                q.put_nowait(jpeg_bytes)
                            except Exception:
                                pass

                # Send data dictionary to subscribers
                data_packet = {
                    "hand_detected": self.hand_detected,
                    "gesture": self.current_gesture,
                    "x": coords[0],
                    "y": coords[1],
                    "fps": self.fps,
                    "control_real_mouse": self.control_real_mouse
                }
                with self.lock:
                    for q in list(self.data_subscribers):
                        try:
                            if q.full():
                                q.get_nowait()
                            q.put_nowait(data_packet)
                        except Exception:
                            pass

        cap.release()
        print("Webcam released, background tracking stopped.")


# Instantiate global tracker
tracker = GestureTracker()

# ---------------------------------------------------------------------------
# Flask Web App Server
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)

@app.route('/')
def home_route():
    return send_from_directory(DOCS_DIR, 'index.html')

@app.route('/gestures')
@app.route('/gestures.html')
def gestures_route():
    return send_from_directory(DOCS_DIR, 'gestures.html')

@app.route('/demo')
@app.route('/demo.html')
def demo_route():
    return send_from_directory(DOCS_DIR, 'demo.html')

@app.route('/install')
@app.route('/install.html')
def install_route():
    return send_from_directory(DOCS_DIR, 'install.html')

@app.route('/video-feed')
def video_feed():
    def video_stream():
        q = tracker.subscribe_video()
        try:
            while True:
                # Blocks until next frame is available
                frame = q.get()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            pass
        finally:
            tracker.unsubscribe_video(q)

    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream-data')
def stream_data():
    def data_stream():
        q = tracker.subscribe_data()
        try:
            while True:
                data = q.get()
                yield f"data: {json.dumps(data)}\n\n"
        except GeneratorExit:
            pass
        finally:
            tracker.unsubscribe_data(q)

    return Response(data_stream(), mimetype='text/event-stream')


@app.route('/api/toggle-control', methods=['POST'])
def toggle_control():
    data = request.json or {}
    enable_real = data.get('control_real_mouse', False)
    tracker.control_real_mouse = enable_real
    if enable_real:
        tracker._check_start()
    else:
        tracker._check_stop()
    return jsonify({"control_real_mouse": tracker.control_real_mouse})


@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        data = request.json or {}
        if 'smoothing' in data:
            tracker.smoothing = float(data['smoothing'])
        if 'pinch_trigger_dist' in data:
            tracker.pinch_trigger_dist = float(data['pinch_trigger_dist'])
        if 'finger_extended_ratio' in data:
            tracker.finger_extended_ratio = float(data['finger_extended_ratio'])
    return jsonify({
        "smoothing": tracker.smoothing,
        "pinch_trigger_dist": tracker.pinch_trigger_dist,
        "finger_extended_ratio": tracker.finger_extended_ratio,
        "control_real_mouse": tracker.control_real_mouse
    })


@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory(DOCS_DIR, path)


# ---------------------------------------------------------------------------
# Standalone Desktop Mode
# ---------------------------------------------------------------------------

def run_desktop():
    """Runs Gesconi in the original desktop-only window mode."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX.")
        return

    prev_screen_x, prev_screen_y = pyautogui.position()
    pinch_start_time = {"t": None}
    is_dragging = False
    last_action_time = {"click": 0.0, "right_click": 0.0, "screenshot": 0.0, "letter": 0.0}
    prev_gesture = "idle"

    print("Gesconi is running in DESKTOP mode. Focus the window and press 'q' to quit.")

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            status_text = "no hand detected"
            debug_text = ""

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                gesture, up = classify_gesture(landmarks, pinch_start_time, prev_gesture)
                status_text = gesture
                debug_text = " ".join(
                    f"{name}:{'up' if state else 'down'}" for name, state in up.items()
                )

                index_tip = landmarks[8]

                if gesture == "pause":
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                elif gesture == "move":
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                    x = min(max(index_tip.x * FRAME_WIDTH, FRAME_MARGIN), FRAME_WIDTH - FRAME_MARGIN)
                    y = min(max(index_tip.y * FRAME_HEIGHT, FRAME_MARGIN), FRAME_HEIGHT - FRAME_MARGIN)

                    screen_x = (x - FRAME_MARGIN) / (FRAME_WIDTH - 2 * FRAME_MARGIN) * SCREEN_W
                    screen_y = (y - FRAME_MARGIN) / (FRAME_HEIGHT - 2 * FRAME_MARGIN) * SCREEN_H

                    smooth_x = prev_screen_x + (screen_x - prev_screen_x) / SMOOTHING
                    smooth_y = prev_screen_y + (screen_y - prev_screen_y) / SMOOTHING
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_screen_x, prev_screen_y = smooth_x, smooth_y

                elif gesture == "click_pending":
                    pass

                elif gesture == "drag":
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True

                elif gesture == "right_click":
                    now = time.time()
                    if now - last_action_time["right_click"] > CLICK_COOLDOWN:
                        pyautogui.click(button="right")
                        last_action_time["right_click"] = now

                elif gesture == "scroll":
                    palm_y = landmarks[9].y
                    center = 0.5
                    if palm_y < center - SCROLL_ZONE_DEADZONE:
                        pyautogui.scroll(SCROLL_STEP)
                        status_text = "scroll up"
                    elif palm_y > center + SCROLL_ZONE_DEADZONE:
                        pyautogui.scroll(-SCROLL_STEP)
                        status_text = "scroll down"
                    else:
                        status_text = "scroll (neutral)"

                elif gesture == "screenshot":
                    now = time.time()
                    if now - last_action_time["screenshot"] > SCREENSHOT_COOLDOWN:
                        shot = pyautogui.screenshot()
                        filename = os.path.join(
                            SCREENSHOT_DIR,
                            f"gesconi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        )
                        shot.save(filename)
                        last_action_time["screenshot"] = now

                elif gesture in ("type_a", "type_b", "type_c"):
                    letter = gesture.split("_")[1]
                    now = time.time()
                    if now - last_action_time["letter"] > LETTER_COOLDOWN:
                        pyautogui.press(letter)
                        last_action_time["letter"] = now

                if prev_gesture == "click_pending" and gesture not in ("click_pending", "drag"):
                    now = time.time()
                    if now - last_action_time["click"] > CLICK_COOLDOWN:
                        pyautogui.click(button="left")
                        last_action_time["click"] = now

                prev_gesture = gesture
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                pinch_start_time["t"] = None
                prev_gesture = "idle"

            cv2.putText(
                frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            if debug_text:
                cv2.putText(
                    frame, debug_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1
                )
            cv2.imshow("Gesconi (press 'q' to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if is_dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesconi Gesture Control Mouse")
    parser.add_argument("--desktop", action="store_true", help="Run standalone desktop mode instead of web server")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default 5000)")
    args = parser.parse_args()

    if args.desktop:
        run_desktop()
    else:
        print(f"Starting Gesconi Web Dashboard on http://127.0.0.1:{args.port}")
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)