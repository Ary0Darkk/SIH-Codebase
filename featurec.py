import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======== CONFIG ==========
DATA_DIR = r"C:\sih\SIH_GamingNexus\Gayatri"
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
dataset = []

dataset += process_file(os.path.join(DATA_DIR, "round1.csv"), "round")
dataset += process_file(os.path.join(DATA_DIR, "shoot_gesture1.csv"), "shoot")
dataset += process_file(os.path.join(DATA_DIR, "up_down1.csv"), "updown")

df = pd.DataFrame(dataset, columns=[
    "MAV", "RMS", "WL", "VAR", "ZC", "SSC", "label"
])

print(df.head())

# ======== TRAIN ML MODEL ========
X = df.iloc[:, :-1]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ====== ACCURACY ======
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
