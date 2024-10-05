# gaze_estimator/utils.py

import numpy as np
import cv2

def calculate_focal_length(initial_depth, eye_distance_pixels, real_eye_distance):
    f = (initial_depth * eye_distance_pixels) / real_eye_distance
    return f

def calculate_depth(focal_length, real_eye_distance, eye_distance_pixels):
    if eye_distance_pixels == 0:
        return 0  # Avoid division by zero
    z = (focal_length * real_eye_distance) / eye_distance_pixels
    return z

def get_cm_per_pixel(screen_width, screen_height, screen_width_cm, screen_height_cm):
    """
    Calculates the conversion from pixels to centimeters.
    """
    cm_per_pixel = (screen_width_cm / screen_width + screen_height_cm / screen_height) / 2
    return cm_per_pixel

def get_eye_distance_pixels(landmarks, img_w, img_h, left_iris_center, right_iris_center):
    """
    Calculates the distance between the eyes in pixels.
    """
    # Get eye centers
    left_eye_center = (
        landmarks.landmark[left_iris_center].x * img_w,
        landmarks.landmark[left_iris_center].y * img_h
    )
    right_eye_center = (
        landmarks.landmark[right_iris_center].x * img_w,
        landmarks.landmark[right_iris_center].y * img_h
    )

    # Distance between eyes in pixels
    eye_distance_pixels = np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center))

    return eye_distance_pixels

def get_eye_midpoint_3d(landmarks, img_w, img_h, depth, focal_length, left_iris_center, right_iris_center):
    """
    Calculates the 3D midpoint between the eyes.
    """
    # Camera parameters
    cx = img_w / 2
    cy = img_h / 2

    # Eye centers in image coordinates
    left_eye_center_u = landmarks.landmark[left_iris_center].x * img_w
    left_eye_center_v = landmarks.landmark[left_iris_center].y * img_h
    right_eye_center_u = landmarks.landmark[right_iris_center].x * img_w
    right_eye_center_v = landmarks.landmark[right_iris_center].y * img_h

    # Compute midpoint
    u_mid = (left_eye_center_u + right_eye_center_u) / 2
    v_mid = (left_eye_center_v + right_eye_center_v) / 2

    # Compute 3D coordinates
    x_mid = (u_mid - cx) * depth / focal_length
    y_mid = (v_mid - cy) * depth / focal_length
    z_mid = depth

    return x_mid, y_mid, z_mid

def display_eye_patches(frame, left_eye, right_eye):
    """
    Adds the left and right eye patches to the main frame.
    The eye patches are displayed in the top left and right corners.
    """
    if left_eye is not None:
        # Left eye at top left
        frame[10:190, 10:310] = cv2.cvtColor(left_eye, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, 'Left Eye', (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if right_eye is not None:
        # Right eye at top right
        frame[10:190, frame.shape[1]-310:frame.shape[1]-10] = cv2.cvtColor(right_eye, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, 'Right Eye', (frame.shape[1]-310, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
