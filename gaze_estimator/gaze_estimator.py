# gaze_estimator/gaze_estimator.py

import cv2
import numpy as np
import pyautogui
import time
from collections import deque
import mediapipe as mp

from .feature_extraction import extract_eye_patch, get_iris_coordinates
from .model import prepare_model, predict_gaze_point, save_calibration, load_calibration
from .calibration import generate_calibration_points, generate_evaluation_points, capture_calibration_data, calculate_initial_focal_length
from .evaluation import evaluate_model
from .utils import calculate_focal_length, get_cm_per_pixel, display_eye_patches, get_eye_distance_pixels, calculate_depth, get_eye_midpoint_3d
from .head_pose import get_head_pose

class GazeEstimator:
    # MediaPipe FaceMesh landmark indices
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
    RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]

    def __init__(self, smoothing_window=5, alpha=1e-4):
        # Automatically detect screen resolution
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Detected screen resolution: {self.screen_width}x{self.screen_height}")

        # Initialize MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Data storage
        self.X = []
        self.Y = []

        # Real eye distance for depth estimation (in cm)
        self.real_eye_distance = None  # To be input by the user

        # Attributes for initial values
        self.initial_eye_distance_pixels = None  # Will be calculated
        self.initial_depth = None  # To be input by the user
        self.focal_length = None  # Will be calculated

        # Screen dimensions in cm
        self.screen_width_cm = None
        self.screen_height_cm = None
        self.cm_per_pixel = None

        # Generate calibration and evaluation points
        self.calibration_points = generate_calibration_points(self.screen_width, self.screen_height)
        self.evaluation_points = generate_evaluation_points(self.screen_width, self.screen_height)

        # Start video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Error accessing the camera.")

        # Deque for smoothing predictions
        self.predicted_points = deque(maxlen=smoothing_window)
        self.smoothed_prediction = None  # For low-pass filter

        # Number of measurements per calibration point
        self.measurements_per_point = 50

        # Initialize model and scaler
        self.model = None
        self.scaler = None

    def show_live_feed(self):
        print("Live video feed is displayed. Press 'q' to return to the main menu.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing frame from camera.")
                break

            # Flip the frame for intuitive display
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            image_h, image_w, _ = frame.shape

            left_eye_display = None
            right_eye_display = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                # Extract and normalize eye patches
                left_eye_patch, left_eye_display = extract_eye_patch(frame, landmarks, eye='left')
                right_eye_patch, right_eye_display = extract_eye_patch(frame, landmarks, eye='right')

                # Get pupil coordinates
                left_pupil_x, left_pupil_y = get_iris_coordinates(landmarks, self.LEFT_IRIS_CENTER, image_w, image_h)
                right_pupil_x, right_pupil_y = get_iris_coordinates(landmarks, self.RIGHT_IRIS_CENTER, image_w, image_h)

                # Get eye distance in pixels
                eye_distance_pixels = get_eye_distance_pixels(landmarks, image_w, image_h, self.LEFT_IRIS_CENTER, self.RIGHT_IRIS_CENTER)

                # Get head pose and depth
                pitch, yaw, roll = get_head_pose(landmarks, image_w, image_h)
                depth = calculate_depth(self.focal_length, self.real_eye_distance, eye_distance_pixels)

                # Eye midpoint in 3D
                x_mid, y_mid, z_mid = get_eye_midpoint_3d(landmarks, image_w, image_h, depth, self.focal_length, self.LEFT_IRIS_CENTER, self.RIGHT_IRIS_CENTER)

                # Display calculated values on the frame
                info_text = [
                    f"Focal Length (F): {self.focal_length:.2f}" if self.focal_length else "Calculating focal length...",
                    f"Roll: {roll:.2f}",
                    f"Yaw: {yaw:.2f}",
                    f"Eye Midpoint (X, Y, Z): ({x_mid:.2f}, {y_mid:.2f}, {z_mid:.2f})"
                ]

                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (10, image_h - 10 - i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            display_eye_patches(frame, left_eye_display, right_eye_display)
            cv2.imshow('Live Feed', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('Live Feed')
                break

    def get_feature_vector(self, frame, landmarks):
        img_h, img_w, _ = frame.shape

        # Extrahieren und Normalisieren der Augen-Patches
        left_eye_patch, _ = extract_eye_patch(frame, landmarks, eye='left')
        right_eye_patch, _ = extract_eye_patch(frame, landmarks, eye='right')

        # Iris-Koordinaten für beide Augen erhalten
        left_iris_coords = get_iris_coordinates(landmarks, self.LEFT_IRIS_LANDMARKS, img_w, img_h)
        right_iris_coords = get_iris_coordinates(landmarks, self.RIGHT_IRIS_LANDMARKS, img_w, img_h)

        # Kopfpose erhalten (Yaw und Roll)
        yaw, roll = get_head_pose(landmarks, img_w, img_h)[1:]  # Nur yaw und roll

        # Feature-Vektor zusammenstellen
        feature_vector = np.concatenate((
            left_iris_coords,        # 10 Werte (5 Punkte x 2 Koordinaten)
            right_iris_coords,       # 10 Werte
            [yaw, roll],             # 2 Werte
            left_eye_patch,          # Größe je nach Ihrem Patch (z.B. 60x36 = 2160 Werte)
            right_eye_patch          # Gleiches wie oben
        ))
        return feature_vector


    def start_calibration(self):
        print("Calibration is starting. Follow the on-screen instructions.")
        calibration_image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Go through the calibration points twice
        calibration_sequence = self.calibration_points  # Duplicate the list

        for idx, point in enumerate(calibration_sequence):
            calibration_image[:] = 0
            cv2.circle(calibration_image, point, 10, (255, 255, 255), -1)
            cv2.imshow('Calibration', calibration_image)
            print(f"Calibration point {idx+1}/{len(calibration_sequence)}: Look at the point and press the spacebar. Press 'q' to return to the main menu.")

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    capture_calibration_data(self, point)
                    break
                elif key == ord('q'):
                    print("Calibration aborted.")
                    cv2.destroyWindow('Calibration')
                    return

        cv2.destroyWindow('Calibration')
        self.model, self.scaler = prepare_model(self.X, self.Y)
        print("Calibration completed.")

    def start_gaze_estimation(self):
        print("Starting real-time gaze estimation. Press 'q' to return to the main menu.")
        cv2.namedWindow('Gaze Estimation', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Gaze Estimation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing frame from camera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                feature_vector = self.get_feature_vector(frame, landmarks)
                if feature_vector is not None:
                    predicted_point = predict_gaze_point(self, feature_vector)

                    gaze_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                    cv2.circle(gaze_img, tuple(predicted_point), 10, (0, 0, 255), -1)
                    cv2.imshow('Gaze Estimation', gaze_img)
            else:
                print("No face detected.")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('Gaze Estimation')
                break

    def control_mouse_with_gaze(self):
        print("Controlling mouse with gaze estimation. Press 'q' to return to the main menu.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing frame from camera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                feature_vector = self.get_feature_vector(frame, landmarks)
                if feature_vector is not None:
                    predicted_point = predict_gaze_point(self, feature_vector)
                    # Move mouse cursor
                    pyautogui.moveTo(predicted_point[0], predicted_point[1])
            else:
                print("No face detected.")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    def evaluate_model(self):
        evaluate_model(self)

    def calculate_initial_focal_length(self):
        calculate_initial_focal_length(self)

    def save_calibration(self, filename='calibration_data.pkl'):
        save_calibration(self.model, self.scaler, filename)

    def load_calibration(self, filename='calibration_data.pkl'):
        self.model, self.scaler = load_calibration(filename)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        # Ask for initial values
        self.real_eye_distance = float(input("Please enter your eye distance in cm: "))
        self.initial_depth = float(input("Please enter the initial distance to the screen in cm: "))
        self.screen_width_cm = float(input("Please enter the width of your screen in cm: "))
        self.screen_height_cm = float(input("Please enter the height of your screen in cm: "))

        # Calculate pixel to cm conversion
        self.cm_per_pixel = get_cm_per_pixel(self.screen_width, self.screen_height, self.screen_width_cm, self.screen_height_cm)

        print("Please look directly into the camera to calculate the focal length.")
        self.calculate_initial_focal_length()

        while True:
            print("\nPlease choose an option:")
            print("1. Live video feed with info")
            print("2. Calibration")
            print("3. Live prediction")
            print("4. Save calibration")
            print("5. Load calibration")
            print("6. Control mouse with gaze")
            print("7. Evaluate model")
            print("8. Exit program")

            choice = input("Your choice: ")

            if choice == '1':
                self.show_live_feed()
            elif choice == '2':
                self.start_calibration()
            elif choice == '3':
                if self.model and self.scaler:
                    self.start_gaze_estimation()
                else:
                    print("Please perform calibration or load calibration first.")
            elif choice == '4':
                filename = input("Filename to save (default 'calibration_data.pkl'): ") or 'calibration_data.pkl'
                self.save_calibration(filename)
            elif choice == '5':
                filename = input("Filename to load (default 'calibration_data.pkl'): ") or 'calibration_data.pkl'
                self.load_calibration(filename)
            elif choice == '6':
                if self.model and self.scaler:
                    self.control_mouse_with_gaze()
                else:
                    print("Please perform calibration or load calibration first.")
            elif choice == '7':
                if self.model and self.scaler:
                    self.evaluate_model()
                else:
                    print("Please perform calibration or load calibration first.")
            elif choice == '8':
                print("Exiting program.")
                self.cleanup()
                break
            else:
                print("Invalid selection. Please try again.")
