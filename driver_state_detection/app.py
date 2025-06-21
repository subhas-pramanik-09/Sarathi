import streamlit as st
import time
import pygame
import os
import glob
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageGrab
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
from arg_parser import get_args
from posture import classify_posture


mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Initialize mixer and folders
pygame.mixer.init()
sound = None
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Streamlit UI Setup
st.set_page_config(page_title="Drowsiness & Posture Monitor", layout="wide")

# Load custom CSS
st.markdown("""
<style>
body { background-color: #f9fafb; font-family: 'Segoe UI', sans-serif; }
header, .reportview-container .main { padding: 2rem; }
.block-header {
    font-size: 2rem; font-weight: bold; color: #1f2937;
    margin-bottom: 1rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;
}
.metric-box {
    padding: 1rem; border-radius: 10px; background-color: #e0f2fe;
    color: #0369a1; font-weight: 600; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 0.5rem;
}
.alert-message {
    background-color: #fee2e2; color: #b91c1c;
    padding: 1rem; border-radius: 10px;
    font-weight: bold; text-align: center; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("📊 SARATHI")
menu = st.sidebar.radio("Go to", ["Front View (Face)", "Side View (Posture)", "Alerts Gallery"])

# Landmark groups for drawing
LANDMARK_GROUPS = {
    "left_eye": [33, 160, 158, 133, 153, 144, 33],
    "right_eye": [362, 385, 387, 263, 373, 380, 362],
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "nose_bridge": [168, 6, 197, 195, 5],
    "outer_lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 61],
}

@st.cache_resource
def load_models(camera_params_path):
    if camera_params_path:
        camera_matrix, dist_coeffs = load_camera_parameters(camera_params_path)
    else:
        camera_matrix, dist_coeffs = None, None
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )
    return detector, EyeDet(), HeadPoseEst(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

def play_alert():
    if sound:
        sound.play(loops=0)

def capture_screenshot(tag):
    try:
        image = ImageGrab.grab()
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"{tag}_{ts}.png")
        image.save(path)
        st.toast(f"📸 Saved: {path}")
    except Exception as e:
        st.error(f"Screenshot error: {e}")

last_screenshot_time = 0
screenshot_interval = 2

def draw_selected_landmarks(image, landmarks, frame_size):
    h, w = frame_size[1], frame_size[0]
    for group in LANDMARK_GROUPS.values():
        for i in range(len(group) - 1):
            pt1 = tuple((landmarks[group[i], :2] * [w, h]).astype(int))
            pt2 = tuple((landmarks[group[i + 1], :2] * [w, h]).astype(int))
            cv2.line(image, pt1, pt2, (0, 0, 255), 1)


def draw_pupil_centers(frame, landmarks, frame_size):
    h, w = frame_size[1], frame_size[0]
    left_pts = np.array([landmarks[i][:2] * [w, h] for i in [474, 475, 476, 477]], dtype=np.float32)
    right_pts = np.array([landmarks[i][:2] * [w, h] for i in [469, 470, 471, 472]], dtype=np.float32)
    left_center = tuple(np.mean(left_pts, axis=0).astype(int))
    right_center = tuple(np.mean(right_pts, axis=0).astype(int))
    cv2.circle(frame, left_center, 2, (255, 0, 0), -1)
    cv2.circle(frame, right_center, 2, (255, 0, 0), -1)

def process_frame(frame, detector, eye_det, head_pose, scorer, frame_size):
    alerts = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.repeat(gray[..., np.newaxis], 3, axis=-1)
    lms = detector.process(gray).multi_face_landmarks
    roll = pitch = yaw = None
    if lms:
        landmarks = get_landmarks(lms)
        draw_selected_landmarks(frame, landmarks, frame_size)
        draw_pupil_centers(frame, landmarks, frame_size)
        ear = eye_det.get_EAR(landmarks)
        tired, _ = scorer.get_rolling_PERCLOS(time.perf_counter(), ear)
        gaze = eye_det.get_Gaze_Score(frame, landmarks, frame_size)
        _, roll, pitch, yaw = head_pose.get_pose(frame, landmarks, frame_size)
        asleep, look_away, distracted = scorer.eval_scores(time.perf_counter(), ear, gaze, roll, pitch, yaw)
        for flag, label in zip([tired, asleep, look_away, distracted], ["DROWSY", "ASLEEP", "LOOK AWAY", "DISTRACTED"]):
            if flag:
                alerts.append(label)
                play_alert()
                capture_screenshot(label)
        if roll is not None:
            cv2.putText(frame, f"Roll: {roll[0]:.1f}", (10, frame_size[1] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        if pitch is not None:
            cv2.putText(frame, f"Pitch: {pitch[0]:.1f}", (10, frame_size[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        if yaw is not None:
            cv2.putText(frame, f"Yaw: {yaw[0]:.1f}", (10, frame_size[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return frame, alerts


def dashboard(detector, eye_det, head_pose, scorer, args):
    st.markdown("<div class='block-header'>🧠 Front View - Drowsiness and Fatigue Detection</div>", unsafe_allow_html=True)
    start_col, stop_col = st.columns([1, 1])
    video_placeholder = st.empty()
    cap = None

    if start_col.button("Start Camera"):
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            st.error("Failed to open camera")
            return

    if stop_col.button("Stop Camera"):
        if cap:
            cap.release()
        video_placeholder.empty()
        return

    if cap and cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 2)
            frame = cv2.resize(frame, (400, 200))
            frame_size = frame.shape[1], frame.shape[0]
            processed, messages = process_frame(frame, detector, eye_det, head_pose, scorer, frame_size)
            for msg in messages:
                cv2.putText(processed, msg, (10, 30 + 30 * messages.index(msg)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            video_placeholder.image(processed, channels="BGR", use_container_width=True)

def side_posture_monitor():
    st.markdown("<div class='block-header'>📸 Side View - Posture Detection</div>", unsafe_allow_html=True)
    start_col, stop_col = st.columns([1, 1])
    video_placeholder = st.empty()
    cap = None

    if start_col.button("Start Posture Camera"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera")
            return

    if stop_col.button("Stop Camera"):
        if cap:
            cap.release()
        video_placeholder.empty()
        return

    if cap and cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                posture, color = classify_posture(results.pose_landmarks.landmark)
            else:
                posture, color = "No person detected", (200, 200, 200)

            cv2.putText(frame, posture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

def gallery():
    st.markdown("<div class='block-header'>📂 Alert Screenshots</div>", unsafe_allow_html=True)
    imgs = sorted(glob.glob(os.path.join(SCREENSHOT_DIR, '*.png')), key=os.path.getmtime, reverse=True)
    if not imgs:
        st.warning("No screenshots yet.")
        return
    cols = st.columns(3)
    for i, img_path in enumerate(imgs):
        with cols[i % 3]:
            try:
                img = Image.open(img_path)
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
            except:
                st.error("Error loading image")

def main():
    global sound
    args = get_args()
    sound = pygame.mixer.Sound('driver_state_detection/640g_alarm-83662.mp3')
    detector, eye_det, head_pose = load_models(args.camera_params)
    scorer = AttScorer(
        t_now=time.perf_counter(), ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh, yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh, gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
        verbose=getattr(args, 'verbose', False)
    )
    if menu == "Front View (Face)":
        dashboard(detector, eye_det, head_pose, scorer, args)
    elif menu == "Side View (Posture)":
        side_posture_monitor()
    elif menu == "Alerts Gallery":
        gallery()

if __name__ == "__main__":
    main()