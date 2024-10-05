# gaze_estimator/model.py

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pickle

def prepare_model(X, Y):
    """
    Prepares and trains the gaze estimation model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lineare Regression
    #model = LinearRegression()

    # Ridge Regression
    model = Ridge(alpha=1e-10)
 
    # Elastic Net Regression
    #model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000)
   
    # Gradient Boosting Regression
    #model = MultiOutputRegressor(
    #    GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=5)
    #)
    
    model.fit(X_scaled, Y)
    
    return model, scaler

def predict_gaze_point(self, feature_vector):
    """
    Predicts the gaze point from the feature vector.
    Inputs:
        feature_vector: The extracted feature vector.
    Outputs:
        smoothed: The predicted gaze point as (x, y) coordinates.
    """
    feature_scaled = self.scaler.transform([feature_vector])
    predicted = self.model.predict(feature_scaled)[0]

    # Apply smoothing
    alpha = 0.1
    if self.smoothed_prediction is None:
        self.smoothed_prediction = predicted
    else:
        self.smoothed_prediction = alpha * predicted + (1 - alpha) * self.smoothed_prediction
    smoothed = self.smoothed_prediction.astype(int)
    smoothed = np.clip(smoothed, [0, 0], [self.screen_width - 1, self.screen_height - 1])

    return smoothed

def save_calibration(model, scaler, filename='calibration_data.pkl'):
    """
    Saves the calibration data to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"Calibration saved to '{filename}'.")

def load_calibration(filename='calibration_data.pkl'):
    """
    Loads the calibration data from a file.
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            scaler = data['scaler']
        print(f"Calibration loaded from '{filename}'.")
        return model, scaler
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None, None
