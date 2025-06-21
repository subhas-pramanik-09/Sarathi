# WITHOUT SS
import time
import pprint
import pygame
import cv2
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters

pygame.mixer.init()
sound = None  

def load_sound(filepath):
    try:
        return pygame.mixer.Sound(filepath)
    except pygame.error as e:
        print(f"Error loading sound file '{filepath}': {e}")
        return None

def main():
    args = get_args()
    global sound
    sound = load_sound(r'driver_state_detection\640g_alarm-83662.mp3')

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  
        except Exception as e:
            print(
                f"OpenCV optimization could not be set to True, the script may be slower than expected.\nError: {e}"
            )

    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    """instantiation of mediapipe face mesh model. This model give back 478 landmarks
    if the rifine_landmarks parameter is set to True. 468 landmarks for the face and
    the last 10 landmarks for the irises
    """
    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    # instantiation of the Eye Detector and Head Pose estimator objects
    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(
        show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )

    # timing variables
    prev_time = time.perf_counter()
    fps = 0.0  

    t_now = time.perf_counter()

    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():  
        print("Cannot open camera")
        exit()

    while True:  
        t_now = time.perf_counter()

        elapsed_time = t_now - prev_time
        prev_time = t_now

        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        ret, frame = cap.read()  

        if not ret:  
            print("Can't receive frame from camera/stream end")
            break

        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        e1 = cv2.getTickCount()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_size = frame.shape[1], frame.shape[0]
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        lms = Detector.process(gray).multi_face_landmarks

        alert_messages = []
        if lms:  
            landmarks = get_landmarks(lms)
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            ear = Eye_det.get_EAR(landmarks=landmarks)

            tired, perclos_score = Scorer.get_rolling_PERCLOS(t_now, ear)

            # compute the Gaze Score
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size
            )

            # compute the head pose
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            # evaluate the scores for EAR, GAZE and HEAD POSE
            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )

            # if the head pose estimation is successful, show the results
            if frame_det is not None:
                frame = frame_det

            # Collect alert messages
            if tired:
                alert_messages.append("TIRED!")
            if asleep:
                alert_messages.append("ASLEEP!")
                sound.play(loops=0)
            if looking_away:
                alert_messages.append("LOOKING AWAY!")
                sound.play(loops=0)
            if distracted:
                alert_messages.append("DISTRACTED!")
                sound.play(loops=0)

            # Display alert messages on the top left in bold red
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color_alert = (0, 0, 255)  # Red
            y_position = 30
            for message in alert_messages:
                cv2.putText(
                    frame,
                    message,
                    (10, y_position),
                    font,
                    font_scale,
                    text_color_alert,
                    font_thickness,
                    cv2.LINE_AA,
                )
                y_position += 30

            # Display roll, pitch, yaw 
            font_pose = cv2.FONT_HERSHEY_PLAIN
            font_scale_pose = 1.5
            font_thickness_pose = 1
            text_color_pose = (0, 255, 0)  # Green
            x_position_pose = frame.shape[1] - 150
            y_position_pose = 40
            if roll is not None:
                cv2.putText(
                    frame,
                    f"Roll:{roll.round(1)[0]}",
                    (x_position_pose, y_position_pose),
                    font_pose,
                    font_scale_pose,
                    text_color_pose,
                    font_thickness_pose,
                    cv2.LINE_AA,
                )
                y_position_pose += 30
            if pitch is not None:
                cv2.putText(
                    frame,
                    f"Pitch:{pitch.round(1)[0]}",
                    (x_position_pose, y_position_pose),
                    font_pose,
                    font_scale_pose,
                    text_color_pose,
                    font_thickness_pose,
                    cv2.LINE_AA,
                )
                y_position_pose += 30
            if yaw is not None:
                cv2.putText(
                    frame,
                    f"Yaw:{yaw.round(1)[0]}",
                    (x_position_pose, y_position_pose),
                    font_pose,
                    font_scale_pose,
                    text_color_pose,
                    font_thickness_pose,
                    cv2.LINE_AA,
                )

        e2 = cv2.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

        # show the frame on screen
        cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
