"""
gesture_game.py
A simple pygame game driven by EMG gestures predicted by your saved RandomForest model.

Requirements:
- Python 3.8+
- pygame
- numpy, pandas, scipy, scikit-learn, joblib
- pyserial (if using a physical COM port)

Place gesture_rf_model_fft.pkl and gesture_scaler_fft.pkl in the same directory.

Run:
    python gesture_game.py
"""


import threading
import time
from collections import deque
import os
import sys


# ---- ML / signal libs ----
import numpy as np
from scipy.signal import welch
import joblib


# ---- Pygame for game ----
import pygame


# ---- Serial (optional) ----
try:
    import serial
    HAS_SERIAL = True
except Exception:
    HAS_SERIAL = False

# ----------------- CONFIG -----------------
MODEL_PATH = "gesture_rf_model_fft.pkl"
SCALER_PATH = "gesture_scaler_fft.pkl"

SERIAL_PORT = "COM3"      # change if required
BAUD_RATE = 115200

WINDOW_SIZE = 100
FS = 500
PREDICTION_VOTE_WINDOWS = 4  # how many windows to consider for majority (approx)

# mapping gestures to game actions
# Make sure labels match your training labels exactly
GESTURE_ACTION_MAP = {
    "shoot": "shoot",
    "up_down": "jump",
    "round": "shield"
}

# Fallback mode if serial not present or if chosen by user
FALLBACK_SIMULATION = False  # set True to start in keyboard simulation mode



# ----------------- MODEL LOADING -----------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Model or scaler not found. Expecting '{MODEL_PATH}' and '{SCALER_PATH}' in current folder.")


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ----------------- FEATURE PIPELINE (self-contained) -----------------
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


def mean_power_frequency(segment, fs=FS):
    freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0
    return np.sum(freqs * psd) / total_power


def median_frequency(segment, fs=FS):
    freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
    power_sum = np.sum(psd)
    if power_sum == 0:
        return 0.0
    cumulative_power = np.cumsum(psd)
    median_idx = np.where(cumulative_power >= power_sum / 2)[0][0]
    return freqs[median_idx]


def fourier_features(segment, fs=FS):
    segment = np.asarray(segment)
    N = len(segment)
    fft_vals = np.fft.rfft(segment)
    mag = np.abs(fft_vals)
    power = mag**2 + 1e-12
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    total_power = np.sum(power)
    norm_power = power / (total_power + 1e-12)
    spectral_centroid = np.sum(freqs * norm_power)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_power))
    spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))
    return spectral_centroid, spectral_spread, spectral_entropy


feature_cols = [
    'MAV', 'ZC', 'WL', 'RMS', 'VAR',
    'MNF', 'MDF',
    'SpecCent', 'SpecSpread', 'SpecEnt'
]


def extract_features(segment_data, window_size=WINDOW_SIZE, fs=FS):
    features = []
    seg = np.asarray(segment_data)
    # sliding with 50% overlap
    step = window_size // 2
    for i in range(0, len(seg) - window_size + 1, step):
        window = seg[i:i + window_size]
        if len(window) != window_size:
            continue
        mav = mean_absolute_value(window)
        zc = zero_crossings(window)
        wl = waveform_length(window)
        rms = root_mean_square(window)
        var = variance(window)
        mnf = mean_power_frequency(window, fs=fs)
        mdf = median_frequency(window, fs=fs)
        spec_cent, spec_spread, spec_ent = fourier_features(window, fs=fs)
        features.append([mav, zc, wl, rms, var, mnf, mdf, spec_cent, spec_spread, spec_ent])
    return np.array(features)  # rows x features



# ----------------- SERIAL READING THREAD -----------------
data_buffer = deque(maxlen=WINDOW_SIZE * 6)  # rolling buffer of raw samples
prediction_history = deque(maxlen=PREDICTION_VOTE_WINDOWS)

serial_running = False
serial_obj = None

def serial_reader_thread(port, baud):
    global serial_running, serial_obj
    try:
        serial_obj = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f"[Serial] Could not open {port}: {e}")
        serial_running = False
        return
    print(f"[Serial] Opened {port} at {baud} baud.")
    serial_running = True
    while serial_running:
        try:
            line = serial_obj.readline().decode(errors='ignore').strip()
            if not line:
                continue
            # try parse float
            try:
                val = float(line)
                data_buffer.append(val)
            except ValueError:
                # ignore non-numeric
                continue
        except Exception:
            break
    if serial_obj and serial_obj.is_open:
        serial_obj.close()
    print("[Serial] Reader thread stopped.")

# ----------------- PREDICTION HELPER -----------------
def predict_gesture_from_buffer(buffer_deque):
    arr = np.array(buffer_deque)
    if len(arr) < WINDOW_SIZE:
        return None
    feats = extract_features(arr, window_size=WINDOW_SIZE, fs=FS)  # shape (n_windows, n_features)
    if feats.size == 0:
        return None
    # scale
    try:
        feats_scaled = scaler.transform(feats)
    except Exception as e:
        print("[Predict] Scaling failed:", e)
        return None
    # predict windows
    preds = model.predict(feats_scaled)
    # majority vote across window predictions
    unique, counts = np.unique(preds, return_counts=True)
    pred = unique[np.argmax(counts)]
    return pred

# ----------------- PYGAME GAME LOGIC -----------------
pygame.init()
WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EMG Gesture Game")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 20)

# Player attributes
player_w, player_h = 50, 60
player_x, player_y = 100, HEIGHT - player_h - 20
player_vy = 0
gravity = 1.2
on_ground = True
shield_timer = 0.0
SHIELD_DURATION = 2.0  # seconds

# Bullets
bullets = []

# Enemies (simple)
enemies = [{'x': 800, 'y': HEIGHT - 40, 'w': 40, 'h': 40, 'vx': -3}]

score = 0
lives = 3

# Visual helpers
def draw_text(surf, text, x, y, color=(255, 255, 255)):
    surf.blit(FONT.render(text, True, color), (x, y))

def spawn_enemy():
    enemies.append({'x': WIDTH + 50, 'y': HEIGHT - 40, 'w': 40, 'h': 40, 'vx': -3 - np.random.rand()*2})

def player_shoot():
    bullets.append({'x': player_x + player_w, 'y': player_y + player_h // 2, 'vx': 8})

def player_jump():
    global player_vy, on_ground
    if on_ground:
        player_vy = -16
        on_ground = False

def use_shield():
    global shield_timer
    shield_timer = SHIELD_DURATION

# ----------------- Fallback keyboard simulation controls -----------------
# Keys: 1 -> simulate 'shoot', 2 -> 'up_down' (jump), 3 -> 'round' (shield)
SIM_KEY_MAP = {
    pygame.K_1: 'shoot',
    pygame.K_2: 'up_down',
    pygame.K_3: 'round'
}

# ----------------- Start serial thread if available and desired -----------------
reader_thread = None
if HAS_SERIAL and not FALLBACK_SIMULATION:
    try:
        reader_thread = threading.Thread(target=serial_reader_thread, args=(SERIAL_PORT, BAUD_RATE), daemon=True)
        reader_thread.start()
    except Exception as e:
        print("[Main] Couldn't start serial thread:", e)
        FALLBACK_SIMULATION = True
else:
    if not HAS_SERIAL:
        print("[Main] pyserial not installed or import failed — using keyboard fallback.")
    if FALLBACK_SIMULATION:
        print("[Main] Using keyboard simulation mode. Press 1,2,3 to simulate gestures.")

last_prediction_time = 0.0
PRED_INTERVAL = 0.5  # seconds between predictions (to avoid over-predicting)

running = True
try:
    spawn_timer = 0.0
    while running:
        dt = clock.tick(60) / 1000.0  # seconds since last tick
        spawn_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if FALLBACK_SIMULATION and event.type == pygame.KEYDOWN:
                if event.key in SIM_KEY_MAP:
                    g = SIM_KEY_MAP[event.key]
                    # directly trigger action
                    action = GESTURE_ACTION_MAP.get(g)
                    if action == "shoot":
                        player_shoot()
                    elif action == "jump":
                        player_jump()
                    elif action == "shield":
                        use_shield()

        # Periodic prediction from buffer (if not fallback)
        if not FALLBACK_SIMULATION:
            now = time.time()
            if now - last_prediction_time > PRED_INTERVAL:
                last_prediction_time = now
                pred = predict_gesture_from_buffer(data_buffer)
                if pred is not None:
                    # keep short history and majority across history windows
                    prediction_history.append(pred)
                    # get majority across history
                    u, c = np.unique(np.array(prediction_history), return_counts=True)
                    if len(u) > 0:
                        maj = u[np.argmax(c)]
                        action = GESTURE_ACTION_MAP.get(maj)
                        if action == "shoot":
                            player_shoot()
                        elif action == "jump":
                            player_jump()
                        elif action == "shield":
                            use_shield()

        # physics
        if not on_ground:
            player_vy += gravity
            player_y += player_vy
            if player_y >= HEIGHT - player_h - 20:
                player_y = HEIGHT - player_h - 20
                player_vy = 0
                on_ground = True

        # bullets update
        for b in bullets[:]:
            b['x'] += b['vx']
            if b['x'] > WIDTH + 100:
                bullets.remove(b)

        # enemies update
        for e in enemies[:]:
            e['x'] += e['vx']
            if e['x'] < -100:
                enemies.remove(e)
            # collision with player (if not shielded)
            if (player_x < e['x'] + e['w'] and player_x + player_w > e['x'] and
                player_y < e['y'] + e['h'] and player_y + player_h > e['y']):
                if shield_timer <= 0:
                    lives -= 1
                    enemies.remove(e)
                    if lives <= 0:
                        running = False
                else:
                    # shield protects
                    enemies.remove(e)
                    score += 5

        # bullet / enemy collisions
        for b in bullets[:]:
            for e in enemies[:]:
                if (b['x'] < e['x'] + e['w'] and b['x'] + 8 > e['x'] and
                    b['y'] < e['y'] + e['h'] and b['y'] + 4 > e['y']):
                    try:
                        bullets.remove(b)
                    except ValueError:
                        pass
                    try:
                        enemies.remove(e)
                    except ValueError:
                        pass
                    score += 10
                    break

        # spawn enemies occasionally
        if spawn_timer > 1.2:
            spawn_timer = 0.0
            spawn_enemy()

        # shield countdown
        if shield_timer > 0:
            shield_timer -= dt
            if shield_timer < 0:
                shield_timer = 0

        # draw
        screen.fill((30, 30, 40))
        # ground
        pygame.draw.rect(screen, (60, 200, 60), (0, HEIGHT - 20, WIDTH, 20))

        # player (simple rectangle)
        player_color = (200, 200, 255) if shield_timer <= 0 else (255, 215, 0)
        pygame.draw.rect(screen, player_color, (int(player_x), int(player_y), player_w, player_h))
        # small shield visual when active
        if shield_timer > 0:
            pygame.draw.circle(screen, (255, 255, 100), (int(player_x + player_w/2), int(player_y + player_h/2)), 50, 3)

        # bullets
        for b in bullets:
            pygame.draw.rect(screen, (255, 80, 80), (int(b['x']), int(b['y']), 8, 4))

        # enemies
        for e in enemies:
            pygame.draw.rect(screen, (180, 50, 50), (int(e['x']), int(e['y']), e['w'], e['h']))

        # HUD
        draw_text(screen, f"Score: {score}", 10, 10)
        draw_text(screen, f"Lives: {lives}", 10, 36)
        draw_text(screen, f"Shield: {'ON' if shield_timer>0 else 'OFF'}", 10, 62)
        draw_text(screen, f"Buffer samples: {len(data_buffer)}", 10, 88)
        draw_text(screen, "Simulation keys: 1=shoot 2=jump 3=shield (if serial unavailable)", 10, HEIGHT-40)

        pygame.display.flip()

except Exception as e:
    print("[Main] Exception:", e)
finally:
    # cleanup
    serial_running = False
    try:
        if serial_obj and serial_obj.is_open:
            serial_obj.close()
    except Exception:
        pass
    pygame.quit()
    print("Game ended. Score:", score)