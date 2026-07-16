"""
Gesconi — gesture-controlled mouse
------------------------------------
Uses your webcam + MediaPipe hand tracking to move the system cursor
and trigger clicks, drags, scrolling, screenshots, and letter typing
using hand gestures.

Gestures:
  1. Index finger up, others down       -> move cursor
  2. Thumb + index pinch (tap)           -> left click
  3. Thumb + middle pinch (tap)          -> right click
  4. Thumb + index pinch (hold)          -> drag
  5. All four fingers up, upper zone     -> scroll up
     All four fingers up, lower zone     -> scroll down
  6. Thumb + pinky pinch                 -> take a screenshot
  7. Closed fist, then release:
       index only                        -> type "a"
       index + middle                    -> type "b"
       index + middle + ring             -> type "c"
  8. Closed fist (otherwise)             -> pause tracking

Run:
  python gesconi.py

Quit:
  press 'q' with the preview window focused, or Ctrl+C in the terminal.
"""

import os
import time
import math
from datetime import datetime

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
FINGER_EXTENDED_RATIO = 1.15 # how much farther (than the PIP joint) the tip must be from
                             # the wrist to count as "extended" - raise if fingers misread
                             # as up, lower if they misread as down

SCREENSHOT_PINCH_DIST = 0.08 # thumb-to-pinky is naturally a wider gap than thumb-to-index
SCREENSHOT_COOLDOWN = 1.5    # seconds between screenshots
SCREENSHOT_DIR = "screenshots"

LETTER_COOLDOWN = 0.6        # minimum seconds between letter presses (safety net)

SCROLL_ZONE_DEADZONE = 0.06  # fraction of frame height around vertical center that does nothing
SCROLL_STEP = 40             # scroll amount applied per frame while in a scroll zone

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
    Returns a dict of which fingers are extended.

    Compares each fingertip's distance from the wrist to its PIP joint's
    distance from the wrist. An extended finger's tip sits noticeably
    farther from the wrist than its own middle knuckle; a curled finger's
    tip sits close to (or closer than) that knuckle. This works regardless
    of how the hand is rotated in the frame, unlike a plain y-position
    check, which only works when the hand is held flat and facing the
    camera.
    """
    wrist = landmarks[0]
    tips = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pips = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

    up = {}
    for name in tips:
        tip_dist = landmark_dist(wrist, landmarks[tips[name]])
        pip_dist = landmark_dist(wrist, landmarks[pips[name]])
        up[name] = tip_dist > pip_dist * FINGER_EXTENDED_RATIO
    return up


def classify_gesture(landmarks, pinch_start_time, prev_gesture):
    """
    Looks at the current hand landmarks and returns one of:
    'move', 'click_pending', 'right_click', 'drag', 'scroll',
    'screenshot', 'type_a', 'type_b', 'type_c', 'pause', 'idle'

    pinch_start_time: dict tracking how long the index-pinch has been held,
    passed in/out by the caller so state persists across frames.
    prev_gesture: the gesture classified on the previous frame, used to
    edge-trigger the type-letter gestures only at the moment a fist opens.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    pinky_tip = landmarks[20]

    pinch_index_dist = landmark_dist(thumb_tip, index_tip)
    pinch_middle_dist = landmark_dist(thumb_tip, middle_tip)
    pinch_pinky_dist = landmark_dist(thumb_tip, pinky_tip)
    up = fingers_up(landmarks)

    is_fist = not any(up.values())
    if is_fist:
        return "pause", up

    # Thumb + pinky pinch -> screenshot. Checked early, and guarded so it
    # doesn't fire while you're also mid-pinch on index/middle.
    if (
        pinch_pinky_dist < SCREENSHOT_PINCH_DIST
        and pinch_index_dist >= PINCH_TRIGGER_DIST
        and pinch_middle_dist >= PINCH_TRIGGER_DIST
    ):
        return "screenshot", up

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

    # Type-letter combos: index/index+middle/index+middle+ring share their
    # finger shape with "move" and "scroll" respectively, so they only fire
    # in the single frame right after a closed fist opens into that shape.
    # Hold the shape past that frame and it falls through to the normal
    # move/idle behavior below instead of retyping every frame.
    if prev_gesture == "pause":
        if up["index"] and up["middle"] and up["ring"] and not up["pinky"]:
            return "type_c", up
        if up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
            return "type_b", up
        if up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
            return "type_a", up

    if up["index"] and up["middle"] and up["ring"] and up["pinky"]:
        return "scroll", up

    if up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
        return "move", up

    return "idle", up


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam. Check CAM_INDEX and that no other app is using it.")
        return

    prev_screen_x, prev_screen_y = pyautogui.position()
    pinch_start_time = {"t": None}
    is_dragging = False
    last_action_time = {"click": 0.0, "right_click": 0.0, "screenshot": 0.0, "letter": 0.0}
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
                    if now - last_action_time["right_click"] > CLICK_COOLDOWN:
                        pyautogui.click(button="right")
                        last_action_time["right_click"] = now

                elif gesture == "scroll":
                    # Zone-based: which half of the frame the palm sits in
                    # decides scroll direction, held continuously.
                    palm_y = landmarks[9].y  # middle-finger MCP, a stable palm-center point
                    center = 0.5
                    if palm_y < center - SCROLL_ZONE_DEADZONE:
                        pyautogui.scroll(SCROLL_STEP)
                        status_text = "scroll up"
                    elif palm_y > center + SCROLL_ZONE_DEADZONE:
                        pyautogui.scroll(-SCROLL_STEP)
                        status_text = "scroll down"
                    else:
                        status_text = "scroll (neutral zone)"

                elif gesture == "screenshot":
                    now = time.time()
                    if now - last_action_time["screenshot"] > SCREENSHOT_COOLDOWN:
                        shot = pyautogui.screenshot()
                        filename = os.path.join(
                            SCREENSHOT_DIR,
                            f"gesconi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        )
                        shot.save(filename)
                        print(f"Screenshot saved: {filename}")
                        last_action_time["screenshot"] = now

                elif gesture in ("type_a", "type_b", "type_c"):
                    letter = gesture.split("_")[1]
                    now = time.time()
                    if now - last_action_time["letter"] > LETTER_COOLDOWN:
                        pyautogui.press(letter)
                        last_action_time["letter"] = now

                # A pinch that was released without becoming a drag counts as a
                # quick left click.
                if prev_gesture == "click_pending" and gesture not in ("click_pending", "drag"):
                    now = time.time()
                    if now - last_action_time["click"] > CLICK_COOLDOWN:
                        pyautogui.click(button="left")
                        last_action_time["click"] = now

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


if __name__ == "__main__":
    main()