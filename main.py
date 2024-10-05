# main.py

from gaze_estimator.gaze_estimator import GazeEstimator

if __name__ == "__main__":
    try:
        gaze_estimator = GazeEstimator(smoothing_window=5)
        gaze_estimator.run()
    except Exception as e:
        print(f"An error occurred: {e}")
