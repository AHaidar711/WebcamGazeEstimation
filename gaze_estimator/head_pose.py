# gaze_estimator/head_pose.py

import cv2
import numpy as np
import math

def get_head_pose(landmarks, img_w, img_h):
    """
    Estimates the head pose (pitch, yaw, roll) using solvePnP.
    """
    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points
    image_points = np.array([
        (landmarks.landmark[1].x * img_w, landmarks.landmark[1].y * img_h),     # Nose tip
        (landmarks.landmark[152].x * img_w, landmarks.landmark[152].y * img_h), # Chin
        (landmarks.landmark[33].x * img_w, landmarks.landmark[33].y * img_h),   # Left eye corner
        (landmarks.landmark[263].x * img_w, landmarks.landmark[263].y * img_h), # Right eye corner
        (landmarks.landmark[61].x * img_w, landmarks.landmark[61].y * img_h),   # Left mouth corner
        (landmarks.landmark[291].x * img_w, landmarks.landmark[291].y * img_h)  # Right mouth corner
    ], dtype='double')

    # Camera parameters
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype='double'
    )

    # No distortion
    dist_coeffs = np.zeros((4,1))

    # Pose estimation
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = math.atan2(-rotation_matrix[2,0], sy)
        z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = math.atan2(-rotation_matrix[2,0], sy)
        z = 0

    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)

    return pitch, yaw, roll
