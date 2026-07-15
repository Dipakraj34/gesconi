"""
Gesconi — gesture-controlled mouse
------------------------------------
Uses your webcam + MediaPipe hand tracking to move the system cursor
and trigger clicks, drags, and scrolling using hand gestures.

Gestures (matches the website's gesture map):
  1. Index finger up, others down  -> move cursor
  2. Thumb + index pinch (tap)     -> left click
  3. Thumb + middle pinch (tap)    -> right click
  4. Thumb + index pinch (hold)    -> drag
  5. Index + middle both up        -> scroll (move hand up/down)
  6. Closed fist                   -> pause tracking (no cursor movement)

Run:
  python gesconi.py

Quit:
  press 'q' with the preview window focused, or Ctrl+C in the terminal.
"""

import time
import math

import cv2
import mediapipe as mp
import pyautogui

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAM_INDEX = 0                # change if you have multiple webcams
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Region of the webcam frame that maps to the full screen. Keeping a margin
# means you don't have to reach to the very edge of the camera's view to
# reach the edge of the screen.
FRAME_MARGIN = 100

SMOOTHING = 5                # higher = smoother but laggier cursor movement
PINCH_TRIGGER_DIST = 0.045   # normalized distance (0-1) that counts as a pinch
PINCH_HOLD_TIME = 0.35       # seconds a pinch must be held before it becomes a drag
CLICK_COOLDOWN = 0.4         # seconds between repeat clicks of the same type
SCROLL_SENSITIVITY = 400     # multiplier applied to vertical hand movement while scrolling

pyautogui.FAILSAFE = True    # move mouse to a screen corner to abort, as a safety net
pyautogui.PAUSE = 0          # don't let pyautogui add its own delay after every call

SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def landmark_dist(a, b):
    """Euclidean distance between two MediaPipe landmarks (normalized coords)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(landmarks):
    """
    Returns a dict of which fingers are extended, based on landmark y-position
    (tip above its own knuckle = extended). Thumb is handled separately since
    it moves sideways rather than up/down.
    """
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    mcps = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}

    up = {}
    for name in tips:
        tip = landmarks[tips[name]]
        mcp = landmarks[mcps[name]]
        up[name] = tip.y < mcp.y - 0.02
    return up


def classify_gesture(landmarks, pinch_start_time):
    """
    Looks at the current hand landmarks and returns one of:
    'move', 'click', 'right_click', 'drag', 'scroll', 'pause', 'idle'

    pinch_start_time: dict tracking how long the index-pinch has been held,
    passed in/out by the caller so state persists across frames.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    pinch_index_dist = landmark_dist(thumb_tip, index_tip)
    pinch_middle_dist = landmark_dist(thumb_tip, middle_tip)
    up = fingers_up(landmarks)

    is_fist = not any(up.values())
    if is_fist:
        return "pause", up

    # Thumb + middle pinch -> right click (checked before index pinch so the
    # two don't fight when your fingers are close together)
    if pinch_middle_dist < PINCH_TRIGGER_DIST and pinch_index_dist >= PINCH_TRIGGER_DIST:
        return "right_click", up

    # Thumb + index pinch -> click, or drag if held
    if pinch_index_dist < PINCH_TRIGGER_DIST:
        now = time.time()
        if pinch_start_time["t"] is None:
            pinch_start_time["t"] = now
        held_for = now - pinch_start_time["t"]
        if held_for >= PINCH_HOLD_TIME:
            return "drag", up
        return "click_pending", up
    else:
        pinch_start_time["t"] = None

    if up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
        return "scroll", up

    if up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
        return "move", up

    return "idle", up


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX and that no other app is using it.")
        return

    prev_screen_x, prev_screen_y = pyautogui.position()
    pinch_start_time = {"t": None}
    is_dragging = False
    last_click_time = {"click": 0.0, "right_click": 0.0}
    last_scroll_y = None
    prev_gesture = "idle"

    print("Gesconi is running. Focus the preview window and press 'q' to quit.")

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

            frame = cv2.flip(frame, 1)  # mirror, so movement feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            status_text = "no hand detected"

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                gesture, up = classify_gesture(landmarks, pinch_start_time)
                status_text = gesture

                index_tip = landmarks[8]

                if gesture == "pause":
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                elif gesture == "move":
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                    # Map the normalized fingertip position (with a margin) to
                    # full screen coordinates.
                    x = min(max(index_tip.x * FRAME_WIDTH, FRAME_MARGIN), FRAME_WIDTH - FRAME_MARGIN)
                    y = min(max(index_tip.y * FRAME_HEIGHT, FRAME_MARGIN), FRAME_HEIGHT - FRAME_MARGIN)

                    screen_x = (x - FRAME_MARGIN) / (FRAME_WIDTH - 2 * FRAME_MARGIN) * SCREEN_W
                    screen_y = (y - FRAME_MARGIN) / (FRAME_HEIGHT - 2 * FRAME_MARGIN) * SCREEN_H

                    # Smooth the movement so the cursor doesn't jitter.
                    smooth_x = prev_screen_x + (screen_x - prev_screen_x) / SMOOTHING
                    smooth_y = prev_screen_y + (screen_y - prev_screen_y) / SMOOTHING
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_screen_x, prev_screen_y = smooth_x, smooth_y

                elif gesture == "click_pending":
                    # Waiting to see if this pinch turns into a hold (drag) or
                    # a quick tap (click). Registered as a click on release.
                    pass

                elif gesture == "drag":
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True

                elif gesture == "right_click":
                    now = time.time()
                    if now - last_click_time["right_click"] > CLICK_COOLDOWN:
                        pyautogui.click(button="right")
                        last_click_time["right_click"] = now

                elif gesture == "scroll":
                    y_pos = index_tip.y
                    if last_scroll_y is not None:
                        delta = (last_scroll_y - y_pos) * SCROLL_SENSITIVITY
                        if abs(delta) > 1:
                            pyautogui.scroll(int(delta))
                    last_scroll_y = y_pos

                if gesture != "scroll":
                    last_scroll_y = None

                # A pinch that was released without becoming a drag counts as a
                # quick left click.
                if prev_gesture == "click_pending" and gesture not in ("click_pending", "drag"):
                    now = time.time()
                    if now - last_click_time["click"] > CLICK_COOLDOWN:
                        pyautogui.click(button="left")
                        last_click_time["click"] = now

                prev_gesture = gesture

            else:
                # No hand in frame: release any in-progress drag as a safety net.
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                pinch_start_time["t"] = None
                prev_gesture = "idle"

            cv2.putText(
                frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("Gesconi (press 'q' to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if is_dragging:
        pyautogui.mouseUp()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
