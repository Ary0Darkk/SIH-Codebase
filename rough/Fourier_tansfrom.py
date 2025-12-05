import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # help in plotting data in 3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from scipy.signal import welch  # TODO: what does this welch exactly does 
import joblib
import os

# ===================== USER CONFIGURATION =====================
BASE_DIR = r"raw_data"

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



plt.show()

print("\nML Pipeline with FFT features finished successfully!")


# -------------------------------------------------------------------------------------





"""
emg_gesture_game.py

Arcade runner controlled by EMG gestures (or keyboard demo).

Modes:
 - DEMO (keyboard): press 1/2/3 to emulate gestures
 - EMG  (csv stream): reads CSV(s) from BASE_DIR and predicts gestures using your trained pipeline

Usage:
  python emg_gesture_game.py   # defaults to DEMO mode
  python emg_gesture_game.py --mode emg   # use EMG mode (reads CSV files in BASE_DIR)

Make sure your trained model (gesture_rf_model_fft.pkl) and scaler (gesture_scaler_fft.pkl)
exist in the working directory when running in EMG mode.
"""








# import argparse
# import os
# import time
# import random
# from collections import deque

# import pygame
# import numpy as np
# import pandas as pd
# import joblib

# # ----------------- Import feature extraction functions from your pipeline -----------------
# # For simplicity we will copy minimal required functions here (adapted from your code).
# from scipy.signal import welch

# # ========== Config (adjust paths / params) ==========
# BASE_DIR = r"C:\Users\rohit\OneDrive\Desktop\MTECH\Sem3\HACKATHON\Code\SIH"
# VALUE_COLUMN = 'value'
# WINDOW_SIZE = 100
# FS = 500
# MODEL_PATH = "gesture_rf_model_fft.pkl"
# SCALER_PATH = "gesture_scaler_fft.pkl"

# # Gesture labels expected by the model (should match training)
# GESTURE_LABELS = ['round', 'shoot', 'up_down']  # same as training

# # ------------- Feature functions (trimmed) -------------
# def mean_absolute_value(segment):
#     return np.mean(np.abs(segment))

# def zero_crossings(segment):
#     mean_val = np.mean(segment)
#     return np.sum(np.diff(np.array(segment) > mean_val) != 0)

# def waveform_length(segment):
#     return np.sum(np.abs(np.diff(segment)))

# def root_mean_square(segment):
#     return np.sqrt(np.mean(segment**2))

# def variance(segment):
#     return np.var(segment)

# def mean_power_frequency(segment, fs=FS):
#     freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
#     total_power = np.sum(psd)
#     if total_power == 0:
#         return 0.0
#     return np.sum(freqs * psd) / total_power

# def median_frequency(segment, fs=FS):
#     freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
#     power_sum = np.sum(psd)
#     if power_sum == 0:
#         return 0.0
#     cumulative_power = np.cumsum(psd)
#     median_idx = np.where(cumulative_power >= power_sum / 2)[0][0]
#     return freqs[median_idx]

# def fourier_features(segment, fs=FS):
#     segment = np.asarray(segment)
#     N = len(segment)
#     fft_vals = np.fft.rfft(segment)
#     mag = np.abs(fft_vals)
#     power = mag**2
#     freqs = np.fft.rfftfreq(N, d=1.0 / fs)
#     total_power = np.sum(power) + 1e-12
#     norm_power = power / total_power
#     spectral_centroid = np.sum(freqs * norm_power)
#     spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_power))
#     spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))
#     return spectral_centroid, spectral_spread, spectral_entropy

# def extract_features(segment_data, window_size=WINDOW_SIZE, fs=FS):
#     features = []
#     for i in range(0, len(segment_data) - window_size + 1, window_size // 2):
#         window = segment_data[i:i + window_size]
#         if len(window) == window_size:
#             mav = mean_absolute_value(window)
#             zc = zero_crossings(window)
#             wl = waveform_length(window)
#             rms = root_mean_square(window)
#             var = variance(window)
#             mnf = mean_power_frequency(window, fs=fs)
#             mdf = median_frequency(window, fs=fs)
#             spec_cent, spec_spread, spec_ent = fourier_features(window, fs=fs)
#             features.append([mav, zc, wl, rms, var, mnf, mdf, spec_cent, spec_spread, spec_ent])
#     if len(features) == 0:
#         return pd.DataFrame(columns=['MAV','ZC','WL','RMS','VAR','MNF','MDF','SpecCent','SpecSpread','SpecEnt'])
#     return pd.DataFrame(features, columns=['MAV','ZC','WL','RMS','VAR','MNF','MDF','SpecCent','SpecSpread','SpecEnt'])

# # ------------------ EMG Stream Reader (CSV cycling) ------------------
# class CSVStreamSimulator:
#     """
#     Simulate a real-time EMG stream by reading CSV(s) in BASE_DIR and yielding
#     samples one-by-one. Assumes CSV has a 'value' column.
#     """
#     def __init__(self, base_dir, file_list=None, value_col='value'):
#         self.base_dir = base_dir
#         self.value_col = value_col
#         if file_list:
#             self.files = [os.path.join(base_dir, f) for f in file_list]
#         else:
#             # find csv files
#             self.files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.lower().endswith('.csv')]
#         if not self.files:
#             raise FileNotFoundError("No CSV files found in BASE_DIR for EMG streaming.")
#         self.file_idx = 0
#         self._load_current()
#         self.idx = 0

#     def _load_current(self):
#         path = self.files[self.file_idx]
#         df = pd.read_csv(path)
#         df[self.value_col] = pd.to_numeric(df[self.value_col], errors='coerce')
#         df = df.dropna(subset=[self.value_col])
#         self.current_values = df[self.value_col].values
#         self.idx = 0

#     def next(self):
#         if self.idx >= len(self.current_values):
#             # rotate to next file
#             self.file_idx = (self.file_idx + 1) % len(self.files)
#             self._load_current()
#         val = self.current_values[self.idx]
#         self.idx += 1
#         return val

# # ------------------ Game: pygame ------------------
# class EMGGame:
#     WIDTH = 800
#     HEIGHT = 400
#     GROUND_Y = 300

#     def __init__(self, mode='demo', emg_stream=None, model=None, scaler=None):
#         pygame.init()
#         pygame.display.set_caption("EMG Gesture Runner")
#         self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
#         self.clock = pygame.time.Clock()
#         self.font = pygame.font.SysFont(None, 24)

#         # Player state
#         self.player_x = 100
#         self.player_y = self.GROUND_Y
#         self.player_vy = 0
#         self.on_ground = True
#         self.color = (50, 150, 200)

#         # Obstacles
#         self.obstacles = []
#         self.obstacle_timer = 0

#         # Projectiles (for shoot)
#         self.projectiles = []

#         # Speed/dash
#         self.speed = 4
#         self.dash_timer = 0

#         # Mode & EMG
#         self.mode = mode
#         self.emg_stream = emg_stream
#         self.model = model
#         self.scaler = scaler

#         # sliding buffer of values for windowing
#         self.stream_buffer = deque(maxlen=WINDOW_SIZE * 4)  # keep some history

#         # last detected gesture and cooldown to avoid multiple triggers
#         self.last_gesture = None
#         self.gesture_cooldown = 0.6  # seconds
#         self.last_gesture_time = 0

#         # score / game state
#         self.score = 0
#         self.running = True

#     def spawn_obstacle(self):
#         w = random.randint(20, 40)
#         h = random.randint(30, 70)
#         x = self.WIDTH + 10
#         y = self.GROUND_Y - h
#         self.obstacles.append({'x': x, 'w': w, 'h': h})

#     def handle_gesture_action(self, gesture_label):
#         """Map gesture to action (jump/shoot/dash)."""
#         now = time.time()
#         if now - self.last_gesture_time < self.gesture_cooldown:
#             return
#         self.last_gesture_time = now
#         # mapping:
#         if gesture_label == 'round':  # jump
#             if self.on_ground:
#                 self.player_vy = -12
#                 self.on_ground = False
#         elif gesture_label == 'shoot':
#             # spawn projectile
#             self.projectiles.append({'x': self.player_x + 40, 'y': self.player_y + 10, 'vx': 12})
#         elif gesture_label == 'up_down':
#             # dash: temporary speed burst
#             self.dash_timer = 0.5
#             self.speed = 10

#     def predict_from_buffer(self):
#         """Run your pipeline on the recent stream buffer and return predicted gesture or None."""
#         if len(self.stream_buffer) < WINDOW_SIZE:
#             return None
#         arr = np.array(self.stream_buffer)
#         # Use sliding windows as in training: get last window_size samples
#         window = arr[-WINDOW_SIZE:]
#         feat_df = extract_features(window, window_size=WINDOW_SIZE, fs=FS)
#         if feat_df.empty:
#             return None
#         # scale & predict each window -> majority vote
#         X = feat_df[['MAV','ZC','WL','RMS','VAR','MNF','MDF','SpecCent','SpecSpread','SpecEnt']].values
#         if self.scaler is None or self.model is None:
#             return None
#         Xs = self.scaler.transform(X)
#         preds = self.model.predict(Xs)
#         vals, counts = np.unique(preds, return_counts=True)
#         return vals[np.argmax(counts)]

#     def run_frame(self):
#         # read EMG value if in EMG mode
#         if self.mode == 'emg' and self.emg_stream is not None:
#             # Simulated sampling: step a few values per frame to speed things up
#             for _ in range(4):  # push multiple samples per frame to fill buffer
#                 try:
#                     val = self.emg_stream.next()
#                     self.stream_buffer.append(val)
#                 except Exception:
#                     pass
#             predicted = self.predict_from_buffer()
#             if predicted is not None:
#                 self.handle_gesture_action(predicted)

#         # In DEMO mode keyboard events will be handled in event loop

#         # Update player physics
#         self.player_vy += 0.6  # gravity
#         self.player_y += self.player_vy
#         if self.player_y >= self.GROUND_Y:
#             self.player_y = self.GROUND_Y
#             self.player_vy = 0
#             self.on_ground = True

#         # Update dash
#         if self.dash_timer > 0:
#             self.dash_timer -= self.dt
#             if self.dash_timer <= 0:
#                 self.speed = 4

#         # Move obstacles left
#         for ob in self.obstacles:
#             ob['x'] -= self.speed
#         self.obstacles = [o for o in self.obstacles if o['x'] + o['w'] > -10]

#         # Spawn new obstacle occasionally
#         self.obstacle_timer += self.dt
#         if self.obstacle_timer > max(0.8, 2.5 - self.score * 0.01):
#             self.spawn_obstacle()
#             self.obstacle_timer = 0

#         # Update projectiles
#         for p in self.projectiles:
#             p['x'] += p['vx']
#         self.projectiles = [p for p in self.projectiles if p['x'] < self.WIDTH + 50]

#         # collision detection: player vs obstacles
#         player_rect = pygame.Rect(self.player_x, self.player_y - 40, 40, 40)
#         for ob in self.obstacles:
#             ob_rect = pygame.Rect(ob['x'], self.GROUND_Y - ob['h'], ob['w'], ob['h'])
#             if player_rect.colliderect(ob_rect):
#                 # game over (reset)
#                 self.running = False

#         # projectile vs obstacle
#         remaining_obs = []
#         for ob in self.obstacles:
#             ob_rect = pygame.Rect(ob['x'], self.GROUND_Y - ob['h'], ob['w'], ob['h'])
#             hit = False
#             for p in self.projectiles:
#                 p_rect = pygame.Rect(p['x'], p['y'], 8, 4)
#                 if p_rect.colliderect(ob_rect):
#                     hit = True
#                     self.score += 5
#             if not hit:
#                 remaining_obs.append(ob)
#         self.obstacles = remaining_obs

#         # increment score
#         self.score += 0.01

#     def draw(self):
#         self.screen.fill((30, 30, 40))
#         # ground
#         pygame.draw.rect(self.screen, (40, 90, 40), (0, self.GROUND_Y + 40, self.WIDTH, self.HEIGHT - self.GROUND_Y))
#         # player
#         pygame.draw.rect(self.screen, self.color, (self.player_x, self.player_y - 40, 40, 40))
#         # obstacles
#         for ob in self.obstacles:
#             pygame.draw.rect(self.screen, (180, 50, 60), (ob['x'], self.GROUND_Y - ob['h'], ob['w'], ob['h']))
#         # projectiles
#         for p in self.projectiles:
#             pygame.draw.rect(self.screen, (240, 200, 60), (p['x'], p['y'], 8, 4))
#         # HUD
#         mode_text = f"MODE: {self.mode.upper()}"
#         txt = self.font.render(mode_text + f"    SCORE: {int(self.score)}", True, (240,240,240))
#         self.screen.blit(txt, (8, 8))
#         if self.mode == 'emg':
#             st = f"Buffer: {len(self.stream_buffer)} samples"
#             t2 = self.font.render(st, True, (200,200,200))
#             self.screen.blit(t2, (8, 32))

#     def run(self):
#         prev = time.time()
#         while True:
#             self.dt = time.time() - prev
#             prev = time.time()
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     return
#                 if event.type == pygame.KEYDOWN:
#                     # DEMO keys map to gestures
#                     if self.mode == 'demo':
#                         if event.key == pygame.K_1:
#                             self.handle_gesture_action('round')
#                         elif event.key == pygame.K_2:
#                             self.handle_gesture_action('shoot')
#                         elif event.key == pygame.K_3:
#                             self.handle_gesture_action('up_down')
#                     # allow restart when stopped
#                     if not self.running and event.key == pygame.K_r:
#                         self.__init__(mode=self.mode, emg_stream=self.emg_stream, model=self.model, scaler=self.scaler)

#             if self.running:
#                 self.run_frame()
#             else:
#                 # show game over
#                 self.screen.fill((10,10,10))
#                 g = self.font.render("GAME OVER", True, (240,80,80))
#                 s = self.font.render(f"SCORE: {int(self.score)}   Press R to restart or ESC to quit", True, (240,240,240))
#                 self.screen.blit(g, (self.WIDTH//2 - 60, self.HEIGHT//2 - 30))
#                 self.screen.blit(s, (self.WIDTH//2 - 200, self.HEIGHT//2 + 10))
#                 for event in pygame.event.get():
#                     if event.type == pygame.QUIT:
#                         pygame.quit()
#                         return
#                     if event.type == pygame.KEYDOWN:
#                         if event.key == pygame.K_r:
#                             self.__init__(mode=self.mode, emg_stream=self.emg_stream, model=self.model, scaler=self.scaler)
#                         if event.key == pygame.K_ESCAPE:
#                             pygame.quit()
#                             return

#             self.draw()
#             pygame.display.flip()
#             self.clock.tick(60)

# # ------------------ Main runner ------------------
# def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
#     if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#         print("Model/scaler not found in current directory.")
#         return None, None
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
#     return model, scaler

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', choices=['demo','emg'], default='demo', help='demo=keyboard; emg=use CSV stream')
#     parser.add_argument('--base_dir', default=BASE_DIR, help='directory with CSVs for EMG streaming')
#     args = parser.parse_args()

#     if args.mode == 'emg':
#         try:
#             # load model & scaler
#             model, scaler = load_model_and_scaler()
#             if model is None:
#                 print("Switching to DEMO mode because model/scaler not found.")
#                 mode = 'demo'
#                 game = EMGGame(mode='demo')
#                 game.run()
#                 return
#             # create CSV stream
#             stream = CSVStreamSimulator(args.base_dir)
#             game = EMGGame(mode='emg', emg_stream=stream, model=model, scaler=scaler)
#         except Exception as e:
#             print("EMG mode failed:", e)
#             print("Falling back to DEMO mode.")
#             game = EMGGame(mode='demo')
#     else:
#         game = EMGGame(mode='demo')

#     game.run()

# if __name__ == "__main__":
#     main()