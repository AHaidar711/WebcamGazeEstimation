# gaze_estimator/evaluation.py

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from .feature_extraction import get_iris_coordinates, extract_eye_patch
from .model import predict_gaze_point
from .utils import get_eye_distance_pixels, calculate_depth
from .head_pose import get_head_pose

def evaluate_model(self):
    """
    Evaluates the gaze estimation model by displaying evaluation points
    and comparing predicted gaze points.
    """
    print("Evaluating model. Press the spacebar to begin.")
    evaluation_results = []
    angular_errors = []

    evaluation_image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
    cv2.namedWindow('Evaluation', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Evaluation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    started = False

    while True:
        ret, frame = self.cap.read()
        if not ret:
            print("Error capturing frame from camera.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        evaluation_image[:] = 0
        cv2.imshow('Evaluation', evaluation_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            feature_vector = self.get_feature_vector(frame, landmarks)
            if feature_vector is not None:
                predicted_point = predict_gaze_point(self, feature_vector)
                # Display gaze estimation
                cv2.circle(evaluation_image, tuple(predicted_point), 10, (0, 0, 255), -1)

        cv2.imshow('Evaluation', evaluation_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            started = True
            print("Evaluation starting...")
            break
        elif key == ord('q'):
            cv2.destroyWindow('Evaluation')
            return

    # Process evaluation points with 3-second intervals
    for idx, point in enumerate(self.evaluation_points):
        print(f"Evaluation point {idx+1}/{len(self.evaluation_points)}")
        start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing frame from camera.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            evaluation_image[:] = 0
            cv2.circle(evaluation_image, point, 10, (255, 255, 255), -1)

            if results.multi_face_landmarks: 
                landmarks = results.multi_face_landmarks[0]
                feature_vector = self.get_feature_vector(frame, landmarks)
                if feature_vector is not None:
                    predicted_point = predict_gaze_point(self, feature_vector)
                    # Display gaze estimation
                    cv2.circle(evaluation_image, tuple(predicted_point), 10, (0, 0, 255), -1)

                    # After 3 seconds, perform measurement
                    if time.time() - start_time >= 3:
                        # Error calculation
                        actual_point = np.array(point)
                        predicted_point = predicted_point
                        error_pixels = np.linalg.norm(actual_point - predicted_point)
                        error_cm = error_pixels * self.cm_per_pixel
                        angular_error_rad = np.arctan(error_cm / self.initial_depth)
                        angular_error_deg = np.degrees(angular_error_rad)
                        angular_errors.append(angular_error_deg)

                        evaluation_results.append((actual_point, predicted_point))
                        print(f"Point {idx+1}: Actual: {actual_point}, Predicted: {predicted_point}")
                        print(f"Error: {error_pixels:.2f} pixels, {error_cm:.2f} cm, Angle: {angular_error_deg:.2f}°")
                        break  # Move to the next point

                else:
                    print("No face detected. Please look at the camera.")
            else:
                print("No face detected. Please look at the camera.")

            cv2.imshow('Evaluation', evaluation_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('Evaluation')
                return

    cv2.destroyWindow('Evaluation')

    # Plot results
    actual_points = np.array([res[0] for res in evaluation_results])
    predicted_points = np.array([res[1] for res in evaluation_results])

    plt.figure(figsize=(10, 6))
    plt.scatter(actual_points[:, 0], actual_points[:, 1], c='g', label='Actual Points')
    plt.scatter(predicted_points[:, 0], predicted_points[:, 1], c='r', label='Predicted Points')

    # Calculate mean error
    errors = np.linalg.norm(actual_points - predicted_points, axis=1)
    mean_error_pixels = np.mean(errors)
    mean_error_cm = mean_error_pixels * self.cm_per_pixel
    mean_angular_error = np.mean(angular_errors)

    plt.title(f"Model Evaluation\nMean Error: {mean_error_pixels:.2f} pixels, {mean_error_cm:.2f} cm, {mean_angular_error:.2f}°")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.gca().invert_yaxis()  # Invert Y-axis for screen coordinates
    plt.show()
