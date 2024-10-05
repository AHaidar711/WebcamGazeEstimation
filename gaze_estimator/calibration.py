# gaze_estimator/calibration.py

import cv2
import numpy as np
import time
from .feature_extraction import extract_eye_patch, get_iris_coordinates
from .utils import calculate_depth, calculate_focal_length, get_eye_distance_pixels
from .head_pose import get_head_pose

def generate_calibration_points(screen_width, screen_height, rows=3, cols=3, margin=0.05):
    """
    Generates symmetrically arranged calibration points in a grid pattern.
    """
    x_margin = int(screen_width * margin)
    y_margin = int(screen_height * margin)
    x_points = np.linspace(x_margin, screen_width - x_margin, cols)
    y_points = np.linspace(y_margin, screen_height - y_margin, rows)
    calibration_points = [(int(x), int(y)) for y in y_points for x in x_points]
    print(f"{len(calibration_points)} calibration points generated.")
    return calibration_points

def generate_evaluation_points(screen_width, screen_height):
    """
    Generates evaluation points symmetrically arranged:
    4 at the top, 4 in the middle, 4 at the bottom.
    """
    cols = 4
    x_margin = int(screen_width * 0.1)
    y_margin = int(screen_height * 0.1)

    x_points = np.linspace(x_margin, screen_width - x_margin, cols)

    # Positions for top, middle, and bottom rows
    y_positions = [
        y_margin,  # Top
        screen_height // 2,  # Middle
        screen_height - y_margin  # Bottom
    ]

    evaluation_points = []

    for y in y_positions:
        for x in x_points:
            evaluation_points.append((int(x), int(y)))

    print(f"{len(evaluation_points)} evaluation points generated.")
    return evaluation_points

def capture_calibration_data(self, point):
    """
    Captures calibration data for a given point.
    """
    for m in range(self.measurements_per_point):
        ret, frame = self.cap.read()
        if not ret:
            print("Error capturing frame from camera.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            feature_vector = self.get_feature_vector(frame, landmarks)
            if feature_vector is not None:
                self.X.append(feature_vector)
                self.Y.append(point)
                print(f"Measurement {m+1}/{self.measurements_per_point} for point ({point[0]}, {point[1]}) captured.")
            else:
                print("No face detected. Please try again.")
        else:
            print("No face detected. Please try again.")

def calculate_initial_focal_length(self):
    """
    Calculates the initial focal length based on the user's eye distance.
    """
    print("Calculating focal length...")
    while True:
        ret, frame = self.cap.read()
        if not ret:
            print("Error capturing frame from camera.")
            continue

        # Flip the frame for consistent representation
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            img_h, img_w, _ = frame.shape

            # Get eye distance in pixels
            eye_distance_pixels = get_eye_distance_pixels(landmarks, img_w, img_h, self.LEFT_IRIS_CENTER, self.RIGHT_IRIS_CENTER)

            # Calculate focal length
            self.initial_eye_distance_pixels = eye_distance_pixels
            self.focal_length = calculate_focal_length(self.initial_depth, eye_distance_pixels, self.real_eye_distance)
            print(f"Focal length calculated: {self.focal_length:.2f} units")
            break
        else:
            print("No face detected. Please look at the camera.")
