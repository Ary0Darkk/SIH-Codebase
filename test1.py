import serial
import numpy as np
import joblib
import time

# -------------------- LOAD MODEL + SCALER --------------------
model = joblib.load("gesture_rf_model_fft.pkl")
scaler = joblib.load("gesture_scaler_fft.pkl")

WINDOW_SIZE = 100     # must match training code
FS = 500              # must match training code

import os
print("Current working directory:", os.getcwd())

import sys
sys.path.append(r"C:\Users\rohit\OneDrive\Desktop\MTECH\Sem3\HACKATHON\Code\SIH\SIH-Codebase")

from Fourier_tansfrom import extract_features, feature_cols


# -------------------- SERIAL CONFIG --------------------
SERIAL_PORT = "COM3"      # update based on your system
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print(f"\n📡 Listening to EMG data on {SERIAL_PORT} ...")


# -------------------- REAL-TIME BUFFER --------------------
buffer = []

def predict_from_buffer(signal_buffer):
    """Takes at least 100 samples and predicts gesture."""
    signal_array = np.array(signal_buffer)

    # Extract features from the last window
    feat_df = extract_features(signal_array, window_size=WINDOW_SIZE, fs=FS)

    # Scale
    feat_scaled = scaler.transform(feat_df[feature_cols].values)

    # Predict
    preds = model.predict(feat_scaled)

    # Majority vote
    values, counts = np.unique(preds, return_counts=True)
    return values[np.argmax(counts)]

# -------------------- MAIN REAL-TIME LOOP --------------------
print("\n⚡ Real-time prediction started. Press Ctrl+C to stop.\n")

try:
    while True:
        line = ser.readline().decode().strip()

        if line == "":
            continue

        try:
            val = float(line)
            buffer.append(val)

            # keep buffer from growing too big
            if len(buffer) > WINDOW_SIZE * 3:
                buffer = buffer[-WINDOW_SIZE*3:]

            # predict only when enough samples collected
            if len(buffer) >= WINDOW_SIZE:
                gesture = predict_from_buffer(buffer)
                print(f"🟢 Predicted Gesture: {gesture}")

        except ValueError:
            # skip invalid lines
            continue


except KeyboardInterrupt:
    print("\n🛑 Stopped real-time prediction.")
    ser.close()