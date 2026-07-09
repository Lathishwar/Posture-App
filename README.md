# PostureAI — Posture Analysis & Engagement Monitoring System

A real-time computer vision web app that uses your webcam to analyse posture and engagement using **MediaPipe** and **Flask**.

---

## Features

| Module | What it detects |
|--------|----------------|
| **Posture Analysis** | Shoulder tilt, head tilt, spine angle, slouch detection |
| **Engagement Monitoring** | Eye contact (gaze estimation), blink rate, attention state |
| **Live Dashboard** | Skeleton overlay, score gauges, alert feed, score-history sparkline |

---

## Project Structure

```
posture-app/
├── app.py               ← Flask backend + MediaPipe processing
├── requirements.txt
└── templates/
    └── index.html       ← Frontend (live feed + dashboard)
```

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## How it works

### Posture Analysis (MediaPipe Pose)
- Detects 33 body landmarks in real time
- Calculates **shoulder tilt** (angle from horizontal)
- Calculates **head tilt** (ear-to-shoulder vector angle)
- Calculates **spine angle** (hip → shoulder → ear)
- Deducts points from posture score based on deviation thresholds

### Engagement Monitoring (MediaPipe FaceMesh)
- Detects 468 face landmarks
- **Blink detection** via Eye Aspect Ratio (EAR < 0.22 threshold)
- **Blink rate** computed over a rolling 60-second window (normal: 15–20/min)
- **Head pose / gaze** estimated from nose-tip vs eye midpoint offset
- Engagement score drops if looking away > 3 seconds or blink rate is abnormal

### Scoring
| Score | Posture | Engagement |
|-------|---------|------------|
| ≥ 80  | Good    | Engaged    |
| 55–79 | Fair    | Moderate   |
| < 55  | Poor    | Disengaged |

---

## Notes
- Webcam index defaults to `0`. Change `cv2.VideoCapture(0)` in `app.py` if you have multiple cameras.
- Works best with good frontal lighting.
- Tested with Python 3.9–3.11.
