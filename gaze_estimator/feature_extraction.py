# gaze_estimator/feature_extraction.py

import cv2
import numpy as np

# MediaPipe FaceMesh landmark indices
LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155,
                      133, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390,
                       249, 263, 466, 388, 387, 386, 385, 384, 398]

def extract_eye_patch(image, landmarks, eye='both', output_size=(60, 36)):
    """
    Extracts and normalizes the eye region from the image.
    Inputs:
        image: The input image.
        landmarks: Face landmarks detected by MediaPipe.
        eye: 'left' or 'right' to specify which eye to process.
        output_size: Desired output size of the eye patch.
    Outputs:
        eye_patch_flat: Flattened and normalized eye patch.
        eye_patch_display: Eye patch image for display.
    """
    if eye == 'left':
        eye_landmarks = RIGHT_EYE_LANDMARKS  # Note: Right eye from mirrored image
    elif eye == 'right':
        eye_landmarks = LEFT_EYE_LANDMARKS
    else:
        raise ValueError("Eye must be 'left' or 'right'")

    h, w, _ = image.shape
    # Extract eye points
    eye_points = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in eye_landmarks], dtype=np.float32)
    # Center the eye
    eye_center = np.mean(eye_points, axis=0)
    # Compute the angle of the eye
    left_corner = eye_points[0]
    right_corner = eye_points[8]
    delta_x = right_corner[0] - left_corner[0]
    delta_y = right_corner[1] - left_corner[1]
    angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
    # Get rotated image
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Update eye points after rotation
    rotated_eye_points = np.array([cv2.transform(np.array([[pt]]), M)[0][0] for pt in eye_points])

    # Determine bounding box around the eye
    x_min, y_min = np.min(rotated_eye_points, axis=0).astype(int)
    x_max, y_max = np.max(rotated_eye_points, axis=0).astype(int)

    # Add padding
    padding = 5
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, w - 1)
    y_max = min(y_max + padding, h - 1)

    # Extract eye region
    eye_region = rotated_image[y_min:y_max, x_min:x_max]
    eye_region_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    # Equalize histogram
    eye_region_equalized = cv2.equalizeHist(eye_region_gray)
    # Resize and normalize
    eye_patch = cv2.resize(eye_region_equalized, output_size).astype(np.float32) / 255.0
    eye_patch_flat = eye_patch.flatten()

    # For display
    eye_patch_display = (eye_patch * 255).astype(np.uint8)
    eye_patch_display = cv2.resize(eye_patch_display, (300, 180))

    return eye_patch_flat, eye_patch_display

def get_iris_coordinates(landmarks, iris_landmark_indices, img_w, img_h):
    """
    Ermittelt die Koordinaten aller Iris-Landmarken.
    """
    iris_coords = []
    for idx in iris_landmark_indices:
        x = landmarks.landmark[idx].x * img_w
        y = landmarks.landmark[idx].y * img_h
        iris_coords.extend([x, y])  # x und y zur Liste hinzufügen
    return np.array(iris_coords)

