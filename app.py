from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
import json

app = Flask(__name__)

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_pose     = mp.solutions.pose
mp_face     = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

pose      = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ── Shared state ─────────────────────────────────────────────────────────────
state_lock = threading.Lock()
latest_metrics = {
    "posture_score":    100,
    "posture_status":   "Good",
    "head_tilt":        0.0,
    "shoulder_tilt":    0.0,
    "spine_angle":      0.0,
    "engagement_score": 100,
    "engagement_status":"Engaged",
    "eye_contact":      True,
    "blink_rate":       0.0,
    "attention":        "Focused",
    "session_duration": 0,
    "alerts":           []
}

cap = None
cap_lock  = threading.Lock()
start_time = time.time()

# ── Helpers ──────────────────────────────────────────────────────────────────
def angle_between(a, b, c):
    """Angle at vertex b formed by points a-b-c (degrees)."""
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))

def landmark_xy(lm, idx, w, h):
    p = lm[idx]
    return int(p.x * w), int(p.y * h)

# ── Blink detector state ─────────────────────────────────────────────────────
blink_state = {
    "count": 0, "closed": False,
    "window_start": time.time(), "rate": 0.0
}
EAR_THRESHOLD = 0.22

def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    # vertical distances
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    # horizontal distance
    hor = math.dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hor + 1e-6)

# left & right eye landmark indices for FaceMesh
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Gaze / head-pose state ───────────────────────────────────────────────────
NO_LOOK_TIMEOUT = 3.0   # seconds looking away → disengaged
last_contact_time = time.time()

def estimate_head_pose(face_lm, w, h):
    """
    Rough yaw from nose-tip vs left/right cheek midpoint,
    and pitch from nose-tip vs forehead/chin.
    Returns (yaw_deg, pitch_deg).
    """
    nose   = face_lm[1]
    l_eye  = face_lm[33]
    r_eye  = face_lm[263]
    chin   = face_lm[152]
    forehead = face_lm[10]

    eye_mid_x = (l_eye.x + r_eye.x) / 2
    yaw   = (nose.x - eye_mid_x) * 200          # positive = face right
    pitch = (nose.y - (forehead.y + chin.y)/2) * 200  # rough
    return yaw, pitch

# ── Frame processor ──────────────────────────────────────────────────────────
def process_frame(frame):
    global last_contact_time

    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res  = pose.process(rgb)
    face_res  = face_mesh.process(rgb)

    alerts = []
    posture_score    = 100
    engagement_score = 100

    head_tilt = shoulder_tilt = spine_angle = 0.0
    eye_contact = False
    attention   = "Focused"

    # ── POSE ─────────────────────────────────────────────────────────────────
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark

        # Key landmarks (normalised)
        nose       = (lm[0].x * w,  lm[0].y * h)
        l_shoulder = (lm[11].x * w, lm[11].y * h)
        r_shoulder = (lm[12].x * w, lm[12].y * h)
        l_ear      = (lm[7].x * w,  lm[7].y * h)
        r_ear      = (lm[8].x * w,  lm[8].y * h)
        l_hip      = (lm[23].x * w, lm[23].y * h)
        r_hip      = (lm[24].x * w, lm[24].y * h)

        mid_shoulder = ((l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2)
        mid_hip      = ((l_hip[0]+r_hip[0])/2,           (l_hip[1]+r_hip[1])/2)
        mid_ear      = ((l_ear[0]+r_ear[0])/2,           (l_ear[1]+r_ear[1])/2)

        # Shoulder tilt (degrees from horizontal)
        shoulder_tilt = abs(math.degrees(math.atan2(
            r_shoulder[1] - l_shoulder[1],
            r_shoulder[0] - l_shoulder[0])))
        if shoulder_tilt > 90: shoulder_tilt = 180 - shoulder_tilt

        # Head tilt (ear midpoint vs shoulder midpoint vs vertical)
        head_tilt = abs(math.degrees(math.atan2(
            mid_ear[0] - mid_shoulder[0],
            mid_shoulder[1] - mid_ear[1] + 1e-6)))

        # Spine angle (hip → shoulder → ear)
        spine_angle = angle_between(mid_hip, mid_shoulder, mid_ear)

        # Score deductions
        if shoulder_tilt > 10:
            deduct = min(30, int((shoulder_tilt - 10) * 2))
            posture_score -= deduct
            alerts.append(f"Uneven shoulders ({shoulder_tilt:.1f}°)")

        if head_tilt > 15:
            deduct = min(25, int((head_tilt - 15) * 1.5))
            posture_score -= deduct
            alerts.append(f"Head tilted ({head_tilt:.1f}°)")

        if spine_angle < 150:
            deduct = min(40, int((150 - spine_angle) * 1.2))
            posture_score -= deduct
            alerts.append("Slouching detected")

        posture_score = max(0, posture_score)

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            pose_res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 180), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 200, 255), thickness=2))

    # ── FACE / ENGAGEMENT ────────────────────────────────────────────────────
    if face_res.multi_face_landmarks:
        fl = face_res.multi_face_landmarks[0].landmark

        # Eye-aspect-ratio → blink
        left_ear  = eye_aspect_ratio(fl, LEFT_EYE,  w, h)
        right_ear = eye_aspect_ratio(fl, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        now = time.time()
        if avg_ear < EAR_THRESHOLD:
            if not blink_state["closed"]:
                blink_state["count"]  += 1
                blink_state["closed"]  = True
        else:
            blink_state["closed"] = False

        elapsed = now - blink_state["window_start"]
        if elapsed >= 60:
            blink_state["rate"]         = blink_state["count"]
            blink_state["count"]        = 0
            blink_state["window_start"] = now
        elif elapsed > 0:
            blink_state["rate"] = blink_state["count"] / elapsed * 60

        # Head pose → eye contact
        yaw, pitch = estimate_head_pose(fl, w, h)
        eye_contact = abs(yaw) < 20 and abs(pitch) < 20

        if eye_contact:
            last_contact_time = time.time()
            attention = "Focused"
        else:
            away = time.time() - last_contact_time
            if away > NO_LOOK_TIMEOUT:
                attention = "Distracted"
                engagement_score -= min(40, int(away * 5))
                alerts.append("Looking away from screen")
            else:
                attention = "Glancing Away"

        # Blink-rate engagement
        if blink_state["rate"] < 8:
            engagement_score -= 15
            alerts.append("Low blink rate (eye strain?)")
        elif blink_state["rate"] > 30:
            engagement_score -= 10
            alerts.append("High blink rate (fatigue?)")

        engagement_score = max(0, engagement_score)

        # Draw minimal face mesh (eyes + irises only)
        for eye_indices in [LEFT_EYE, RIGHT_EYE]:
            pts = [(int(fl[i].x * w), int(fl[i].y * h)) for i in eye_indices]
            for j in range(len(pts)):
                cv2.line(frame, pts[j], pts[(j+1) % len(pts)], (255, 220, 0), 1)

    # ── Status strings ────────────────────────────────────────────────────────
    if posture_score >= 80:   posture_status = "Good"
    elif posture_score >= 55: posture_status = "Fair"
    else:                     posture_status = "Poor"

    if engagement_score >= 75:  engagement_status = "Engaged"
    elif engagement_score >= 45: engagement_status = "Moderate"
    else:                        engagement_status = "Disengaged"

    # ── Update shared state ───────────────────────────────────────────────────
    with state_lock:
        latest_metrics.update({
            "posture_score":     posture_score,
            "posture_status":    posture_status,
            "head_tilt":         round(head_tilt, 1),
            "shoulder_tilt":     round(shoulder_tilt, 1),
            "spine_angle":       round(spine_angle, 1),
            "engagement_score":  engagement_score,
            "engagement_status": engagement_status,
            "eye_contact":       eye_contact,
            "blink_rate":        round(blink_state["rate"], 1),
            "attention":         attention,
            "session_duration":  int(time.time() - start_time),
            "alerts":            list(set(alerts))[:4]
        })

    return frame

# ── Video stream ─────────────────────────────────────────────────────────────
def gen_frames():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        with cap_lock:
            success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = process_frame(frame)

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    with state_lock:
        return jsonify(latest_metrics)

@app.route('/reset')
def reset():
    global start_time
    start_time = time.time()
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("🚀 Starting Posture & Engagement Monitor on http://localhost:5000")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
