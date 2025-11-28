import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)
from scipy.signal import welch
import joblib
import os

# ===================== USER CONFIGURATION =====================
BASE_DIR = r"C:\Users\rohit\OneDrive\Desktop\MTECH\Sem3\HACKATHON\Code\SIH"

GESTURE_FILES = {
    'round': ["round1.csv", "round3.csv"],
    'shoot': ["shoot3.csv", "shoot_gesture1.csv"],
    'up_down': ["up_down1.csv", "up_down3.csv"]
}

VALUE_COLUMN = 'value'
WINDOW_SIZE = 100    # samples per window
FS = 500             # assumed sampling frequency (Hz)
# ============================================================


# --- 1. Time Domain Feature Functions ---
def mean_absolute_value(segment):
    return np.mean(np.abs(segment))


def zero_crossings(segment):
    mean_val = np.mean(segment)
    return np.sum(np.diff(np.array(segment) > mean_val) != 0)


def waveform_length(segment):
    return np.sum(np.abs(np.diff(segment)))


def root_mean_square(segment):
    return np.sqrt(np.mean(segment**2))


def variance(segment):
    return np.var(segment)


# --- 2. Frequency Domain Features (Welch PSD based) ---
def mean_power_frequency(segment, fs=FS):
    freqs, psd = welch(
        segment,
        fs=fs,
        nperseg=len(segment),
        window='hann',
        scaling='density'
    )
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0
    return np.sum(freqs * psd) / total_power


def median_frequency(segment, fs=FS):
    freqs, psd = welch(
        segment,
        fs=fs,
        nperseg=len(segment),
        window='hann',
        scaling='density'
    )
    power_sum = np.sum(psd)
    if power_sum == 0:
        return 0.0
    cumulative_power = np.cumsum(psd)
    median_idx = np.where(cumulative_power >= power_sum / 2)[0][0]
    return freqs[median_idx]


# --- 3. NEW: Fourier (FFT) Based Features ---
def fourier_features(segment, fs=FS):
    """
    Compute FFT-based features:
    - Spectral Centroid
    - Spectral Spread
    - Spectral Entropy
    """
    segment = np.asarray(segment)
    N = len(segment)

    # Real FFT
    fft_vals = np.fft.rfft(segment)
    mag = np.abs(fft_vals)
    power = mag**2

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    total_power = np.sum(power) + 1e-12  # avoid div by zero
    norm_power = power / total_power

    # Spectral Centroid
    spectral_centroid = np.sum(freqs * norm_power)

    # Spectral Spread (standard deviation around centroid)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_power))

    # Spectral Entropy
    spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))

    return spectral_centroid, spectral_spread, spectral_entropy


def extract_features(segment_data, window_size=WINDOW_SIZE, fs=FS):
    """
    Extract 10 features per window:
    MAV, ZC, WL, RMS, VAR, MNF, MDF, SpecCent, SpecSpread, SpecEnt
    """
    features = []

    for i in range(0, len(segment_data) - window_size + 1, window_size // 2):
        window = segment_data[i:i + window_size]
        if len(window) == window_size:
            # Time-domain
            mav = mean_absolute_value(window)
            zc = zero_crossings(window)
            wl = waveform_length(window)
            rms = root_mean_square(window)
            var = variance(window)

            # Welch features
            mnf = mean_power_frequency(window, fs=fs)
            mdf = median_frequency(window, fs=fs)

            # FFT-based Fourier features
            spec_cent, spec_spread, spec_ent = fourier_features(window, fs=fs)

            features.append([
                mav, zc, wl, rms, var,
                mnf, mdf,
                spec_cent, spec_spread, spec_ent
            ])

    return pd.DataFrame(
        features,
        columns=[
            'MAV', 'ZC', 'WL', 'RMS', 'VAR',
            'MNF', 'MDF',
            'SpecCent', 'SpecSpread', 'SpecEnt'
        ]
    )


# --- 4. Data Loading and Feature Aggregation ---
print(f"Loading data and extracting features from: {BASE_DIR}")
all_data = []

for gesture, files in GESTURE_FILES.items():
    print(f"\nProcessing gesture: '{gesture}'")
    for file_name in files:
        file_path = os.path.join(BASE_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"  ❌ File not found: {file_name}. Skipping.")
            continue

        try:
            df = pd.read_csv(file_path)

            # Cleaning: ensure numeric
            df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors='coerce')
            df.dropna(subset=[VALUE_COLUMN], inplace=True)
            raw_signal = df[VALUE_COLUMN].values

            if len(raw_signal) < WINDOW_SIZE:
                print(f" ⚠ Not enough clean data in {file_name}. Skipping.")
                continue

            feature_df = extract_features(raw_signal, window_size=WINDOW_SIZE, fs=FS)
            feature_df['label'] = gesture
            all_data.append(feature_df)

            print(f"  ✅ Loaded and extracted {len(feature_df)} feature vectors from {file_name}")

        except Exception as e:
            print(f"  ❌ Error processing {file_name}: {e}")

if not all_data:
    print("\nFATAL: No data was successfully loaded. Check file paths and column names.")
    raise SystemExit

final_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal extracted feature vectors: {len(final_df)}")


# --- 5. Raw Signal Plotting ---
def plot_raw_signal(data_frame, gesture_name):
    plt.figure(figsize=(12, 4))

    gesture_df = data_frame[data_frame['label'] == gesture_name]
    if len(gesture_df) == 0:
        print(f"Cannot plot raw signal: No data for '{gesture_name}'.")
        return

    if len(gesture_df) >= 500:
        start_index = np.random.randint(0, len(gesture_df) - 500)
    else:
        start_index = 0

    sample_segment = gesture_df[VALUE_COLUMN].iloc[start_index:start_index + 500]

    plt.plot(sample_segment.values, label=f'Raw Signal ({gesture_name})', alpha=0.8)
    plt.title(f'Sample Raw EMG Signal for "{gesture_name}" (500 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sensor Value')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()


# Try plotting raw signal for one gesture
try:
    sample_file_path = os.path.join(BASE_DIR, GESTURE_FILES['shoot'][0])
    sample_raw_df = pd.read_csv(sample_file_path)
    sample_raw_df[VALUE_COLUMN] = pd.to_numeric(sample_raw_df[VALUE_COLUMN], errors='coerce')
    sample_raw_df.dropna(subset=[VALUE_COLUMN], inplace=True)
    sample_raw_df['label'] = 'shoot'
    plot_raw_signal(sample_raw_df, 'shoot')
except Exception as e:
    print(f"Raw signal plot error: {e}")


# --- 6. 3D Feature Plot (Using MAV, ZC, WL as before) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
labels = final_df['label'].unique()

for i, label in enumerate(labels):
    subset = final_df[final_df['label'] == label]
    ax.scatter(
        subset['MAV'],
        subset['ZC'],
        subset['WL'],
        alpha=0.6,
        s=50,
        label=label
    )

ax.set_xlabel('MAV')
ax.set_ylabel('ZC')
ax.set_zlabel('WL')
ax.set_title('3D EMG Features Visualization (MAV, ZC, WL)')
ax.legend()
plt.tight_layout()


# --- 7. MACHINE LEARNING MODEL (RANDOM FOREST) ---
print("\n================ MACHINE LEARNING: RANDOM FOREST ================")

feature_cols = [
    'MAV', 'ZC', 'WL', 'RMS', 'VAR',
    'MNF', 'MDF',
    'SpecCent', 'SpecSpread', 'SpecEnt'
]

X = final_df[feature_cols]
y = final_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

print("\nTraining Random Forest...")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Test Accuracy: {accuracy:.4f} ---")
print("\n" + "=" * 20 + " CLASSIFICATION REPORT " + "=" * 20)
print(classification_report(y_test, y_pred, target_names=labels))


# --- 8. Confusion Matrix Plot ---
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()


# --- 9. Feature Importance ---
importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print("\nTop 5 Important Features:")
for i in sorted_idx[:5]:
    print(f"  {feature_cols[i]}: {importance[i]:.4f}")

plt.figure(figsize=(8, 5))
plt.barh([feature_cols[i] for i in sorted_idx], importance[sorted_idx])
plt.gca().invert_yaxis()
plt.title('Feature Importance in Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()


# --- 10. Save Model and Scaler ---
joblib.dump(model, 'gesture_rf_model_fft.pkl')
joblib.dump(scaler, 'gesture_scaler_fft.pkl')
print("\nSaved model to 'gesture_rf_model_fft.pkl' and scaler to 'gesture_scaler_fft.pkl'")


# --- 11. Helper for Real-Time Prediction ---
def predict_gesture(new_signal_1d):
    """
    Predict gesture from a 1D raw EMG signal array.
    - new_signal_1d: 1D numpy array of EMG values.
    Uses same feature pipeline (time + Welch + FFT).
    """
    new_signal_1d = np.asarray(new_signal_1d)
    if len(new_signal_1d) < WINDOW_SIZE:
        raise ValueError(f"Signal too short, need at least {WINDOW_SIZE} samples")

    feat_df = extract_features(new_signal_1d, window_size=WINDOW_SIZE, fs=FS)
    feat_scaled = scaler.transform(feat_df[feature_cols].values)
    preds = model.predict(feat_scaled)

    # Majority vote over windows
    values, counts = np.unique(preds, return_counts=True)
    return values[np.argmax(counts)]


# Example usage (uncomment to test with dummy data)
# dummy_signal = np.random.randn(500)  # Replace with real EMG signal
# print("Predicted Gesture:", predict_gesture(dummy_signal))

plt.show()

print("\nML Pipeline with FFT features finished successfully!")