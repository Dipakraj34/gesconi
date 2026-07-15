# Gesconi

Control your mouse cursor with hand gestures, using nothing but a webcam.

Built with OpenCV (camera capture), MediaPipe (hand landmark tracking), and
PyAutoGUI (cursor and click control).

**[Try the live in-browser demo →](https://YOUR_USERNAME.github.io/gesconi/)**

## Gestures

| Gesture                              | Action       |
|---------------------------------------|--------------|
| Index finger up, others down          | Move cursor  |
| Thumb + index pinch (tap)             | Left click   |
| Thumb + middle pinch (tap)            | Right click  |
| Thumb + index pinch (hold)            | Drag         |
| Index + middle both up, move hand     | Scroll       |
| Closed fist                           | Pause        |

## Setup

Requires Python 3.11 (newer versions can hit MediaPipe compatibility issues).

```bash
git clone https://github.com/YOUR_USERNAME/gesconi.git
cd gesconi
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python gesconi.py
```

Press `q` with the preview window focused to quit.

## Project structure

```
gesconi/
├── gesconi.py          # the gesture-control script
├── requirements.txt     # pinned dependencies
├── docs/
│   └── index.html       # project website (also served via GitHub Pages)
└── README.md
```

## Notes

- `pyautogui.FAILSAFE` is on — drag your mouse to a screen corner at any time
  to immediately abort if the cursor is doing something unexpected.
- Everything runs locally; no video or landmark data leaves your machine.
