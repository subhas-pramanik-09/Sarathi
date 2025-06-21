import json
import cv2
import numpy as np

def load_camera_parameters(file_path):
    try:
        with open(file_path, "r") as file:
            if file_path.endswith(".json"):
                data = json.load(file)
            else:
                raise ValueError("Unsupported file format. Use JSON or YAML.")
            return (
                np.array(data["camera_matrix"], dtype="double"),
                np.array(data["dist_coeffs"], dtype="double"),
            )
    except Exception as e:
        print(f"Failed to load camera parameters: {e}")
        return None, None

def resize(frame, scale_percent):
    
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    return resized

def get_landmarks(lms):
    surface = 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0.0, 0] = 0.0
        landmarks[landmarks[:, 0] > 1.0, 0] = 1.0
        landmarks[landmarks[:, 1] < 0.0, 1] = 0.0
        landmarks[landmarks[:, 1] > 1.0, 1] = 1.0

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks

    return biggest_face

def get_face_area(face):
    
    return abs((face.left() - face.right()) * (face.bottom() - face.top()))

def show_keypoints(keypoints, frame):
    
    for n in range(0, 68):
        x = keypoints.part(n).x
        y = keypoints.part(n).y
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        return frame

def midpoint(p1, p2):
    
    return np.array([int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)])

def get_array_keypoints(landmarks, dtype="int", verbose: bool = False):
    
    points_array = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        points_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

    if verbose:
        print(points_array)

    return points_array

def rot_mat_to_euler(rmat):

    rtr = np.transpose(rmat)
    r_identity = np.matmul(rtr, rmat)

    I = np.identity(3, dtype=rmat.dtype)
    if np.linalg.norm(r_identity - I) < 1e-6:
        sy = (rmat[:2, 0] ** 2).sum() ** 0.5
        singular = sy < 1e-6

        if not singular:  # check if it's a gimbal lock situation
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])

        else:  # if in gimbal lock, use different formula for yaw, pitch roll
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0

        if x > 0:
            x = np.pi - x
        else:
            x = -(np.pi + x)

        if z > 0:
            z = np.pi - z
        else:
            z = -(np.pi + z)

        return (np.array([x, y, z]) * 180.0 / np.pi).round(2)
    else:
        print("Isn't rotation matrix")

def draw_pose_info(frame, img_point, point_proj, roll=None, pitch=None, yaw=None):

    frame = cv2.line(
        frame, img_point, tuple(point_proj[0].ravel().astype(int)), (255, 0, 0), 3
    )
    frame = cv2.line(
        frame, img_point, tuple(point_proj[1].ravel().astype(int)), (0, 255, 0), 3
    )
    frame = cv2.line(
        frame, img_point, tuple(point_proj[2].ravel().astype(int)), (0, 0, 255), 3
    )

    if roll is not None and pitch is not None and yaw is not None:
        cv2.putText(
            frame,
            "Roll:" + str(round(roll, 0)),
            (500, 50),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Pitch:" + str(round(pitch, 0)),
            (500, 70),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Yaw:" + str(round(yaw, 0)),
            (500, 90),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame