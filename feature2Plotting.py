import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# USER CONFIG
# ======================
CSV_PATH = r"C:\sih\SIH_GamingNexus\Gayatri\round1.csv"   # change to your file
WINDOW = 20   # sliding window size

# ======================
# Feature Functions
# ======================

def MAV(x):
    return np.mean(np.abs(x))

def RMS(x):
    return np.sqrt(np.mean(x**2))

def ZC(x, threshold=1e-3):
    count = 0
    for i in range(len(x)-1):
        if (x[i] > 0 and x[i+1] < 0) or (x[i] < 0 and x[i+1] > 0):
            if abs(x[i] - x[i+1]) > threshold:
                count += 1
    return count

def SSC(x, threshold=1e-3):
    count = 0
    for i in range(1, len(x)-1):
        if ((x[i] - x[i-1]) * (x[i] - x[i+1]) > threshold):
            count += 1
    return count

def WL(x):
    return np.sum(np.abs(np.diff(x)))


# ======================
# Load CSV
# ======================
df = pd.read_csv(CSV_PATH, header=None, names=["timestamp", "value"])



# ======================
# Sliding Window Feature Extraction
# ======================
feature_data = {
    "MAV": [],
    "RMS": [],
    "ZC": [],
    "SSC": [],
    "WL": [],
    "Index": []
}

for i in range(0, len(values) - WINDOW):
    window = values[i:i+WINDOW]

    feature_data["MAV"].append(MAV(window))
    feature_data["RMS"].append(RMS(window))
    feature_data["ZC"].append(ZC(window))
    feature_data["SSC"].append(SSC(window))
    feature_data["WL"].append(WL(window))
    feature_data["Index"].append(i)

features_df = pd.DataFrame(feature_data)

# ======================
# Plot the feature curves
# ======================
plt.figure(figsize=(14, 10))

plt.subplot(5, 1, 1)
plt.plot(features_df["Index"], features_df["MAV"])
plt.title("MAV (Mean Absolute Value)")
plt.ylabel("MAV")

plt.subplot(5, 1, 2)
plt.plot(features_df["Index"], features_df["RMS"])
plt.title("RMS")
plt.ylabel("RMS")

plt.subplot(5, 1, 3)
plt.plot(features_df["Index"], features_df["ZC"])
plt.title("Zero Crossing (ZC)")
plt.ylabel("ZC")

plt.subplot(5, 1, 4)
plt.plot(features_df["Index"], features_df["SSC"])
plt.title("Slope Sign Changes (SSC)")
plt.ylabel("SSC")

plt.subplot(5, 1, 5)
plt.plot(features_df["Index"], features_df["WL"])
plt.title("Waveform Length (WL)")
plt.ylabel("WL")
plt.xlabel("Window Index")

plt.tight_layout()
plt.show()

print("Feature extraction complete.")
