import numpy as np
import mediapipe as mp

# Initialize pose landmarks
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Returns the angle at point b given three points (used for back posture)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def classify_posture(landmarks):
    """
    Classifies side-view driver posture using key body landmarks.
    Returns (label, color) for display purposes.
    """
    try:
        # Key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        # Back posture
        back_angle = calculate_angle(left_shoulder, left_hip, left_knee)

        # Arm extension
        arm_extension = abs(left_elbow.x - left_wrist.x)

        # Thresholds and classification
        if back_angle < 140:
            return "Reclined Posture", (0, 0, 255)  # Red
        elif arm_extension > 0.25:
            return "Overextended Arm", (0, 0, 255)
        else:
            return "Right Posture", (0, 255, 0)  # Green
    except:
        return "No person detected", (255, 255, 255)