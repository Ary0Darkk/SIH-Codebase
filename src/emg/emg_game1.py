# """
# Real-time EMG-controlled simple pygame game.

# Requirements:
#   pip install pyserial pygame numpy pandas joblib tensorflow scipy

# Usage:
#   1) Ensure your trained model is available:
#      - classical ML pipeline saved via joblib (e.g. best_emg_model.pkl) OR
#      - deep model saved as Keras .keras or .h5 (best_cnn_model.keras / best_lstm_model.keras)

#   2) Put example gesture CSV files in `emg_dataset/` (used only to infer labels if needed).

#   3) Run:
#      python emg_game_realtime.py

# Notes:
#  - If no serial/Arduino found, script falls back to simulation mode (use keys to simulate gestures).
#  - WINDOW_SIZE must match what you used during training.
# """

# import os
# import time
# import glob
# import joblib
# import threading
# import numpy as np
# from collections import deque, Counter

# # serial + pygame
# try:
#     import serial
#     import serial.tools.list_ports
#     SERIAL_AVAILABLE = True
# except Exception:
#     SERIAL_AVAILABLE = False

# import pygame
# from scipy.fft import rfft, rfftfreq
# from scipy.stats import entropy

# # ------------------ USER CONFIG ------------------
# COM_PORT = "COM7"           # change to your Arduino COM port if needed
# BAUD = 115200
# WINDOW_SIZE = 50            # MUST match training window size
# STEP = 1
# SMOOTH_K = 7                # majority-vote window for smoothing predictions
# MODEL_PATHS = {
#     "ml": "best_emg_model.pkl",
#     "cnn": "best_cnn_model.keras",
#     "lstm": "best_lstm_model.keras"
# }
# EMG_DATASET_DIR = "./emg_dataset"  # optional, for label mapping fallback
# SIMULATE_IF_NO_SERIAL = True
# # -----------------------------------------------

# # ---------------- feature extractor (same as training) ----------------
# def extract_features(window):
#     x = np.array(window).astype(float)
#     N = len(x) if len(x)>0 else 1

#     mean = np.mean(x)
#     std = np.std(x)
#     var = np.var(x)
#     rms = np.sqrt(np.mean(x**2))
#     mav = np.mean(np.abs(x))
#     wl = np.sum(np.abs(np.diff(x))) if len(x)>1 else 0.0
#     zc = int(np.sum(np.diff(np.sign(x)) != 0))

#     ssc = int(np.sum(np.diff(np.sign(np.diff(x))) != 0)) if len(x)>2 else 0

#     yf = np.abs(rfft(x)) if len(x)>0 else np.array([0.])
#     xf = rfftfreq(N, d=1)

#     spectral_energy = float(np.sum(yf**2))
#     spectral_centroid = float(np.sum(xf * yf) / (np.sum(yf) + 1e-8)) if yf.sum() != 0 else 0.0
#     spectral_entropy = float(entropy((yf + 1e-8) / (np.sum(yf) + 1e-8)))

#     return np.array([
#         mean, std, var, rms, mav,
#         wl, zc, ssc,
#         spectral_energy, spectral_centroid, spectral_entropy
#     ]).reshape(1, -1)
# # ----------------------------------------------------------------------

# # ---------------- load label map from dataset folder if needed -----------
# def infer_label_list(folder=EMG_DATASET_DIR):
#     labels = []
#     if os.path.isdir(folder):
#         for f in glob.glob(os.path.join(folder, "*.csv")):
#             labels.append(os.path.basename(f).replace(".csv", ""))
#     if labels:
#         labels = sorted(list(set(labels)))
#     else:
#         labels = ["round", "shoot", "up_down"]
#     return labels

# LABELS = infer_label_list()

# # ---------------- load model (try ML then deep) -------------------------
# model = None
# model_type = None
# if os.path.exists(MODEL_PATHS["ml"]):
#     try:
#         model = joblib.load(MODEL_PATHS["ml"])
#         model_type = "ml"
#         print("Loaded ML model:", MODEL_PATHS["ml"])
#     except Exception as e:
#         print("Failed to load ML model:", e)

# if model is None and os.path.exists(MODEL_PATHS["cnn"]):
#     try:
#         from tensorflow.keras.models import load_model
#         model = load_model(MODEL_PATHS["cnn"])
#         model_type = "cnn"
#         print("Loaded CNN model:", MODEL_PATHS["cnn"])
#     except Exception as e:
#         print("Failed to load CNN model:", e)

# if model is None and os.path.exists(MODEL_PATHS["lstm"]):
#     try:
#         from tensorflow.keras.models import load_model
#         model = load_model(MODEL_PATHS["lstm"])
#         model_type = "lstm"
#         print("Loaded LSTM model:", MODEL_PATHS["lstm"])
#     except Exception as e:
#         print("Failed to load LSTM model:", e)

# if model is None:
#     raise FileNotFoundError("No model files found. Train and place model in working directory.")

# # If ML pipeline returns numeric labels, try to get string labels:
# # joblib pipeline may output string labels directly. We'll detect below in predict step.

# # ---------------- Serial helper (thread) --------------------------------
# class SerialReader(threading.Thread):
#     def __init__(self, port, baud, window_size, simulate=False):
#         super().__init__()
#         self.port = port
#         self.baud = baud
#         self.window_size = window_size
#         self.simulate = simulate
#         self.buffer = deque(maxlen=window_size)
#         self.pred_history = deque(maxlen=SMOOTH_K)
#         self.current_gesture = None
#         self.running = True
#         self.lock = threading.Lock()

#         if not simulate:
#             try:
#                 self.ser = serial.Serial(self.port, self.baud, timeout=1)
#                 time.sleep(1.5)
#                 print(f"Serial connected to {self.port} @ {self.baud}")
#             except Exception as e:
#                 print("Serial open failed:", e)
#                 if SIMULATE_IF_NO_SERIAL:
#                     print("Falling back to simulation mode.")
#                     self.simulate = True
#                 else:
#                     raise
#         else:
#             self.ser = None

#     def run(self):
#         # continuous read loop
#         while self.running:
#             try:
#                 if self.simulate:
#                     # produce simulated EMG: random baseline + occasional bursts mapped to gestures via keyboard simulation
#                     # but for simplicity produce random noise — game can be tested with keyboard simulation mode below
#                     import random, math
#                     val = 200 + 100 * math.sin(time.time()*5) + random.gauss(0, 30)
#                     val = int(max(0, min(1023, val)))
#                     self._append(val)
#                     time.sleep(0.01)
#                     continue

#                 line = self.ser.readline().decode(errors="ignore").strip()
#                 if not line:
#                     continue
#                 # try parse numeric
#                 try:
#                     # sometimes Arduino prints other text; accept floats/ints
#                     if ',' in line:
#                         # if multi-channel, take first
#                         parts = line.split(',')
#                         val = float(parts[0])
#                     else:
#                         val = float(line)
#                     val = int(max(0, min(1023, val)))
#                 except:
#                     continue

#                 self._append(val)
#             except Exception as e:
#                 print("Serial read error:", e)
#                 time.sleep(0.1)

#     def _append(self, val):
#         with self.lock:
#             self.buffer.append(val)
#             # if enough for a window, predict
#             if len(self.buffer) == self.window_size:
#                 arr = np.array(self.buffer)
#                 if model_type == "ml":
#                     feat = extract_features(arr)
#                     try:
#                         pred = model.predict(feat)[0]
#                         # if model returns numeric index, map using LABELS
#                         if isinstance(pred, (int, np.integer)):
#                             if 0 <= pred < len(LABELS):
#                                 label = LABELS[pred]
#                             else:
#                                 label = str(pred)
#                         else:
#                             label = str(pred)
#                     except Exception as e:
#                         label = "unknown"
#                 else:  # cnn or lstm
#                     X = arr.reshape(1, self.window_size, 1).astype(np.float32)
#                     probs = model.predict(X, verbose=0)
#                     pred_idx = int(np.argmax(probs, axis=1)[0])
#                     # map index to label using LABELS (assumes same order used in training)
#                     if pred_idx < len(LABELS):
#                         label = LABELS[pred_idx]
#                     else:
#                         label = str(pred_idx)

#                 # smoothing via majority vote
#                 self.pred_history.append(label)
#                 most = Counter(self.pred_history).most_common(1)[0][0]
#                 self.current_gesture = most

#     def stop(self):
#         self.running = False
#         try:
#             if self.ser:
#                 self.ser.close()
#         except:
#             pass

# # ---------------- Pygame Game -------------------------------
# pygame.init()
# SCREEN_W, SCREEN_H = 900, 500
# screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
# pygame.display.set_caption("EMG Gesture Game (Real-time)")
# clock = pygame.time.Clock()
# FONT = pygame.font.SysFont("Arial", 18)

# # Player properties
# player_w, player_h = 50, 50
# player_x = SCREEN_W//2 - player_w//2
# player_y = SCREEN_H - player_h - 30
# player_vy = 0
# on_ground = True

# # Game state
# bullets = []   # list of (x,y,vx)
# GRAVITY = 0.8

# # Controls mapping:
# # 'round' -> move left
# # 'shoot' -> spawn bullet (rightwards)
# # 'up_down' -> jump
# def apply_gesture_action(label):
#     global player_x, bullets, player_vy, on_ground
#     if label == "round":
#         player_x -= 8
#     elif label == "shoot":
#         # spawn bullet at player's front
#         bx = player_x + player_w
#         by = player_y + player_h//2
#         bullets.append([bx, by, 12])
#     elif label == "up_down":
#         if on_ground:
#             player_vy = -14
#             on_ground = False

# # ---------------- Main program ------------------------------
# def main():
#     # find serial port if not given
#     port = COM_PORT
#     simulate = False
#     if SERIAL_AVAILABLE:
#         # if COM port exists? check ports
#         ports = [p.device for p in serial.tools.list_ports.comports()]
#         if COM_PORT not in ports:
#             # try auto-detect
#             if ports:
#                 print("Auto-detected ports:", ports)
#                 port = ports[0]
#                 print("Using port:", port)
#             else:
#                 print("No serial ports found.")
#                 if SIMULATE_IF_NO_SERIAL:
#                     simulate = True
#                 else:
#                     raise RuntimeError("No serial port and simulation disabled.")
#     else:
#         print("pyserial not available; running in simulation mode.")
#         simulate = True

#     reader = SerialReader(port, BAUD, WINDOW_SIZE, simulate=simulate)
#     reader.daemon = True
#     reader.start()

#     running = True
#     last_label = None
#     sim_label = None  # for keyboard simulation

#     try:
#         while running:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False

#                 # Keyboard simulation (if no real sensor)
#                 if event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_1:
#                         sim_label = "round"
#                     elif event.key == pygame.K_2:
#                         sim_label = "shoot"
#                     elif event.key == pygame.K_3:
#                         sim_label = "up_down"
#                     elif event.key == pygame.K_ESCAPE:
#                         running = False
#                 if event.type == pygame.KEYUP:
#                     # clear simulation when key released
#                     if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
#                         sim_label = None

#             # If simulating gestures via keyboard, override current_gesture
#             if sim_label is not None:
#                 gesture = sim_label
#             else:
#                 gesture = reader.current_gesture

#             # Apply action if gesture changed or every frame? we'll apply continuously while held
#             if gesture:
#                 apply_gesture_action(gesture)

#             # update physics
#             global player_vy, player_y, on_ground
#             player_vy += GRAVITY
#             player_y += player_vy
#             if player_y >= SCREEN_H - player_h - 30:
#                 player_y = SCREEN_H - player_h - 30
#                 player_vy = 0
#                 on_ground = True

#             # bullets update
#             for b in bullets[:]:
#                 b[0] += b[2]
#                 # remove off-screen
#                 if b[0] > SCREEN_W + 50:
#                     bullets.remove(b)

#             # draw
#             screen.fill((30, 30, 30))
#             # ground
#             pygame.draw.rect(screen, (50, 50, 50), (0, SCREEN_H-30, SCREEN_W, 30))

#             # player
#             pygame.draw.rect(screen, (0, 180, 255), (int(player_x), int(player_y), player_w, player_h))

#             # bullets
#             for b in bullets:
#                 pygame.draw.circle(screen, (255, 200, 0), (int(b[0]), int(b[1])), 6)

#             # HUD
#             label_text = f"Gesture: {gesture}" if gesture else "Gesture: (none)"
#             txt = FONT.render(label_text, True, (220,220,220))
#             screen.blit(txt, (10, 10))

#             inst = FONT.render("Simulate: 1=round(left), 2=shoot, 3=up_down(jump)", True, (200,200,200))
#             screen.blit(inst, (10, 40))

#             pygame.display.flip()
#             clock.tick(60)

#     finally:
#         print("Stopping reader...")
#         reader.stop()
#         reader.join(timeout=1)
#         pygame.quit()
#         print("Exited cleanly.")

# if __name__ == "__main__":
#     main()




# import pygame
from collections import deque, Counter
import numpy as np
import joblib
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy
import tensorflow
from  keras.models import load_model
import random

# ---------------- Feature extractor (same as training) ----------------
def extract_features(window):
    x = np.array(window).astype(float)
    N = len(x) if len(x)>0 else 1
    mean = np.mean(x)
    std = np.std(x)
    var = np.var(x)
    rms = np.sqrt(np.mean(x**2))
    mav = np.mean(np.abs(x))
    wl = np.sum(np.abs(np.diff(x))) if len(x)>1 else 0
    zc = int(np.sum(np.diff(np.sign(x)) != 0))
    ssc = int(np.sum(np.diff(np.sign(np.diff(x))) != 0)) if len(x)>2 else 0
    yf = np.abs(rfft(x)) if len(x)>0 else np.array([0.])
    xf = rfftfreq(N, d=1)
    spectral_energy = float(np.sum(yf**2))
    spectral_centroid = float(np.sum(xf*yf)/(np.sum(yf)+1e-8)) if yf.sum()!=0 else 0
    spectral_entropy = float(entropy((yf+1e-8)/(np.sum(yf)+1e-8)))
    return np.array([mean,std,var,rms,mav,wl,zc,ssc,spectral_energy,spectral_centroid,spectral_entropy]).reshape(1,-1)

# ---------------- Load ML model ----------------
# model = joblib.load("best_emg_model.pkl")  # make sure your model is here

model = load_model("models/best_cnn_model.keras")

print(model.input_shape)

emg_window = deque(maxlen=50)

def get_emg_value():
    val = 512 + random.randint(-50,50)
    emg_window.append(val)
    return list(emg_window)

while True:
    
    win = get_emg_value()
    print('Printing win...')
    print(win)
    print('After printing win...')
    if len(win)==50:
        feat = extract_features(win)
        # Suppose feat is currently a 1D or 2D array of 50 values
        feat = np.asarray(feat, dtype="float32")

        # If feat is shape (50,) -> make it (1, 50, 1)
        if feat.shape == (50,):
            feat = feat.reshape(1, 50, 1)

        # If feat is shape (1, 50) -> make it (1, 50, 1)
        elif feat.shape == (1, 50):
            feat = feat.reshape(1, 50, 1)
        
        pred = model.predict(feat)[0]
        
        print(f'Prediction : {pred}')
        # pred_history.append(pred)
        # current_gesture = Counter(pred_history).most_common(1)[0][0]

# ---------------- Pygame Setup ----------------
# pygame.init()
# SCREEN_W, SCREEN_H = 800, 400
# screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
# pygame.display.set_caption("Simple EMG Game")
# clock = pygame.time.Clock()
# FONT = pygame.font.SysFont("Arial", 18)

# # Player
# player_w, player_h = 50, 50
# player_x = SCREEN_W//2 - player_w//2
# player_y = SCREEN_H - player_h - 20
# player_vy = 0
# on_ground = True

# # Bullets
# bullets = []
# GRAVITY = 0.8

# # Prediction smoothing
# SMOOTH_K = 5
# pred_history = deque(maxlen=SMOOTH_K)
# current_gesture = None

# # Simulated EMG input (replace with real Arduino readings)
# emg_window = deque(maxlen=50)
# import random
# def get_emg_value():
#     val = 512 + random.randint(-50,50)
#     emg_window.append(val)
#     return list(emg_window)

# # ---------------- Apply gesture action ----------------
# def apply_gesture_action(label):
#     global player_x, bullets, player_vy, on_ground
#     if label=="round":
#         player_x -= 6
#     elif label=="shoot":
#         bx = player_x + player_w
#         by = player_y + player_h//2
#         bullets.append([bx, by, 12])
#     elif label=="up_down":
#         if on_ground:
#             player_vy = -12
#             on_ground = False

# # ---------------- Main Loop ----------------
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type==pygame.QUIT:
#             running=False

#     # ---------------- Read EMG & predict ----------------
#     win = get_emg_value()
#     if len(win)==50:
#         feat = extract_features(win)
#         pred = model.predict(feat)[0]
#         pred_history.append(pred)
#         current_gesture = Counter(pred_history).most_common(1)[0][0]

#     # ---------------- Apply gesture ----------------
#     if current_gesture:
#         apply_gesture_action(current_gesture)

#     # ---------------- Update physics ----------------
#     player_vy += GRAVITY
#     player_y += player_vy
#     if player_y >= SCREEN_H - player_h - 20:
#         player_y = SCREEN_H - player_h - 20
#         player_vy = 0
#         on_ground = True

#     # update bullets
#     for b in bullets[:]:
#         b[0] += b[2]
#         if b[0] > SCREEN_W+50:
#             bullets.remove(b)

#     # ---------------- Draw ----------------
#     screen.fill((30,30,30))
#     pygame.draw.rect(screen,(50,50,50),(0,SCREEN_H-20,SCREEN_W,20))  # ground
#     pygame.draw.rect(screen,(0,180,255),(int(player_x),int(player_y),player_w,player_h))  # player
#     for b in bullets:
#         pygame.draw.circle(screen,(255,200,0),(int(b[0]),int(b[1])),6)
#     txt = FONT.render(f"Gesture: {current_gesture}",True,(255,255,255))
#     screen.blit(txt,(10,10))
#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()
