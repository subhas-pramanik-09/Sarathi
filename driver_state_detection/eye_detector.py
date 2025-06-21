import cv2
import numpy as np
from numpy import linalg as LA
from utils import resize

class EyeDetector:
    def __init__(self, show_processing: bool = False):
        self.show_processing = show_processing

        # Landmark groups for line connections
        self.FACE_CONNECTIONS = {
            "left_eye": [33, 160, 158, 133, 153, 144, 33],
            "right_eye": [362, 385, 387, 263, 373, 380, 362],
            "left_eyebrow": [70, 63, 105, 66, 107],
            "right_eyebrow": [336, 296, 334, 293, 300],
            # "jawline": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288],
            "nose_bridge": [168, 6, 197, 195, 5],
            "outer_lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 61]
        }

        self.LEFT_IRIS_NUM = 468
        self.RIGHT_IRIS_NUM = 473

        self.EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]

    @staticmethod
    def _calc_EAR_eye(eye_pts):
        return (
            LA.norm(eye_pts[2] - eye_pts[3]) + LA.norm(eye_pts[4] - eye_pts[5])
        ) / (2 * LA.norm(eye_pts[0] - eye_pts[1]))

    def show_eye_keypoints(self, color_frame, landmarks, frame_size):
        # Draw iris (white dot)
        for iris_idx in [self.LEFT_IRIS_NUM, self.RIGHT_IRIS_NUM]:
            cv2.circle(
                color_frame,
                (landmarks[iris_idx, :2] * frame_size).astype(np.uint32),
                3, (255, 255, 255), cv2.FILLED,
            )

        # Draw each defined face connection as a red line
        for feature, indices in self.FACE_CONNECTIONS.items():
            for i in range(len(indices) - 1):
                pt1 = (landmarks[indices[i], :2] * frame_size).astype(int)
                pt2 = (landmarks[indices[i+1], :2] * frame_size).astype(int)
                cv2.line(color_frame, pt1, pt2, (0, 0, 255), 1)

    def get_EAR(self, landmarks):
        eye_pts_l = np.zeros((6, 2))
        eye_pts_r = np.zeros((6, 2))

        for i in range(6):
            eye_pts_l[i] = landmarks[self.EYES_LMS_NUMS[i], :2]
            eye_pts_r[i] = landmarks[self.EYES_LMS_NUMS[i + 6], :2]

        ear_left = self._calc_EAR_eye(eye_pts_l)
        ear_right = self._calc_EAR_eye(eye_pts_r)
        return (ear_left + ear_right) / 2

    @staticmethod
    def _calc_1eye_score(landmarks, eye_lms_nums, eye_iris_num, frame_size, frame):
        iris = landmarks[eye_iris_num, :2]
        eye_x_min = landmarks[eye_lms_nums, 0].min()
        eye_y_min = landmarks[eye_lms_nums, 1].min()
        eye_x_max = landmarks[eye_lms_nums, 0].max()
        eye_y_max = landmarks[eye_lms_nums, 1].max()

        eye_center = np.array([(eye_x_min + eye_x_max) / 2, (eye_y_min + eye_y_max) / 2])
        gaze_score = LA.norm(iris - eye_center) / eye_center[0]

        x1 = int(eye_x_min * frame_size[0])
        y1 = int(eye_y_min * frame_size[1])
        x2 = int(eye_x_max * frame_size[0])
        y2 = int(eye_y_max * frame_size[1])
        eye = frame[y1:y2, x1:x2]

        return gaze_score, eye

    def get_Gaze_Score(self, frame, landmarks, frame_size):
        left_gaze_score, left_eye = self._calc_1eye_score(
            landmarks, self.EYES_LMS_NUMS[:6], self.LEFT_IRIS_NUM, frame_size, frame
        )
        right_gaze_score, right_eye = self._calc_1eye_score(
            landmarks, self.EYES_LMS_NUMS[6:], self.RIGHT_IRIS_NUM, frame_size, frame
        )

        avg_gaze_score = (left_gaze_score + right_gaze_score) / 2

        if self.show_processing and left_eye is not None and right_eye is not None:
            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("Left Eye", left_eye)
            cv2.imshow("Right Eye", right_eye)

        return avg_gaze_score
