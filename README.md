# Gesconi

Control your mouse cursor with hand gestures, using nothing but a webcam.

Built with OpenCV (camera capture), MediaPipe (hand landmark tracking), and
PyAutoGUI (cursor, click, scroll, and keyboard control).

**Live demo:**
- GitHub Pages: https://Dipakraj34.github.io/gesconi/
- Vercel: https://gesconi.vercel.app

## Gestures

| Gesture                                          | Action              |
|---------------------------------------------------|---------------------|
| Index finger up, others down                      | Move cursor         |
| Thumb + index pinch (tap)                         | Left click          |
| Thumb + middle pinch (tap)                        | Right click         |
| Thumb + index pinch (hold)                        | Drag                |
| All four fingers up, hand in upper half of frame  | Scroll up           |
| All four fingers up, hand in lower half of frame  | Scroll down         |
| Thumb + pinky pinch                                | Take a screenshot   |
| Fist, then release index only                      | Type "a"            |
| Fist, then release index + middle                  | Type "b"            |
| Fist, then release index + middle + ring           | Type "c"            |
| Closed fist (otherwise)                            | Pause tracking      |

The letter gestures only fire in the instant a fist opens into that shape —
hold the shape afterward and it falls through to normal move/idle behavior,
so it never fights with the cursor-move gesture.

## Setup

Requires **Python 3.11** specifically. Newer versions (3.12+) can fail to
install `mediapipe==0.10.14`, or install a newer mediapipe that's missing
the legacy `mp.solutions` API this script relies on.

```bash
git clone https://github.com/Dipakraj34/gesconi.git
cd gesconi

# create the virtual environment with Python 3.11 explicitly
py -3.11 -m venv venv          # Windows
# python3.11 -m venv venv      # macOS/Linux

source venv/Scripts/activate   # Windows (Git Bash)
# venv\Scripts\Activate.ps1    # Windows (PowerShell)
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python gesconi.py
```

Press `q` with the preview window focused to quit.

## Project structure

```
gesconi/
├── gesconi.py           # the gesture-control script
├── requirements.txt      # pinned dependencies
├── docs/
│   └── index.html        # project website (served via GitHub Pages and Vercel)
├── screenshots/           # created automatically the first time you use the screenshot gesture
└── README.md
```

## Tuning

A few constants at the top of `gesconi.py` are worth adjusting if a gesture
feels too sensitive, not sensitive enough, or misreads:

- `FINGER_EXTENDED_RATIO` — how "straight" a finger must be to count as up.
- `PINCH_TRIGGER_DIST` / `SCREENSHOT_PINCH_DIST` — how close two fingertips
  must get to count as a pinch.
- `SCROLL_ZONE_DEADZONE` — how large the "do nothing" band is around the
  vertical center of the frame during scroll mode.
- `SMOOTHING` — higher values make cursor movement smoother but laggier.

If a gesture keeps misfiring, run the script and watch the small debug line
under the main status label — it shows each finger's detected up/down state
in real time, which is the fastest way to see what the classifier is
actually reading.

## Notes

- `pyautogui.FAILSAFE` is on — drag your mouse to a screen corner at any time
  to immediately abort if the cursor is doing something unexpected.
- Everything runs locally; no video or landmark data leaves your machine.
- Screenshots are saved locally to `screenshots/` and are not uploaded
  anywhere.
