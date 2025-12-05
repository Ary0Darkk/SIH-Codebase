import os
import numpy as np
import pandas as pd
import glob

# ======== CONFIG ==========
RAW_FOLDER = r"processed_data/mixed_dataset.csv"
WINDOW_SIZE = 200    # samples per window
STEP = 100           # overlap
# ==========================

def extract_features(signal):
    """Extract EMG features from window."""
    x = np.array(signal)

    mav = np.mean(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    wl = np.sum(np.abs(np.diff(x)))
    var = np.var(x)

    # zero crossings
    zc = np.sum(np.diff(np.sign(x)) != 0)

    # slope sign changes
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    ssc = np.sum((diff1[:-1] * diff1[1:]) < 0)

    return [mav, rms, wl, var, zc, ssc]


def process_file(path, label):
    """Load CSV and convert into feature windows."""
    df = pd.read_csv(path)
    values = df["value"].values

    features = []

    for i in range(0, len(values) - WINDOW_SIZE, STEP):
        window = values[i:i+WINDOW_SIZE]
        feats = extract_features(window)
        features.append(feats + [label])

    return features


# ======== LOAD ALL GESTURE FILES ========
# map each file to its gesture
map_gesture = {
    "round1.csv":"round",
    "round3.csv":"round",
    "shoot3.csv":"shoot",
    "shoot_gesture1.csv":"shoot",
    "up_down1.csv":"updown",
    "up_down3.csv":"updown",
}


dataset = []

for file_path in glob.glob(os.path.join(RAW_FOLDER, "*.csv")):

    dataset += process_file(file_path, map_gesture[str(os.path.basename(file_path))])
    
    print(dataset)

# define columns in data
df = pd.DataFrame(dataset, columns=[
    "MAV", "RMS", "WL", "VAR", "ZC", "SSC", "label"
])

print(df.head(5))