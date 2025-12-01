"""
gesture_game_debug.py
Debuggable version of your EMG-driven pygame game.
- Shows last raw samples count and last prediction on screen
- Prints model.classes_ and debug info to console
- Falls back to keyboard simulation if serial fails

Usage:
    python gesture_game_debug.py
"""

import threading
import time
from collections import deque
import os
import sys
import numpy as np
from scipy.signal import welch
import joblib
import pygame

# try import pyserial
try:
    import serial
    HAS_SERIAL = True
except Exception:
    HAS_SERIAL = False

# -------- CONFIG --------
MODEL_PATH = "gesture_rf_model_fft.pkl"
SCALER_PATH = "gesture_scaler_fft.pkl"

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

WINDOW_SIZE = 100
FS = 500
PRED_INTERVAL = 0.5
PREDICTION_VOTE_WINDOWS = 6

FALLBACK_SIMULATION = False  # set True to force keyboard mode

# gesture -> action mapping (strings expected)
GESTURE_ACTION_MAP = {
    "shoot": "shoot",
    "up_down": "jump",
    "round": "shield"
}

# If model uses integers, also accept numeric keys
INT_GESTURE_MAP = {
    1: "shoot",
    2: "up_down",
    3: "round"
}

# -------- load model/scaler --------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler not found in cwd.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[DEBUG] Loaded model and scaler.")
print("[DEBUG] model.classes_ ->", getattr(model, "classes_", "NO CLASSES ATTRIBUTE"))

# -------- feature pipeline (same as training) --------
def mean_absolute_value(segment): return np.mean(np.abs(segment))
def zero_crossings(segment):
    m = np.mean(segment)
    return np.sum(np.diff(np.array(segment) > m) != 0)
def waveform_length(segment): return np.sum(np.abs(np.diff(segment)))
def root_mean_square(segment): return np.sqrt(np.mean(segment**2))
def variance(segment): return np.var(segment)
def mean_power_frequency(segment, fs=FS):
    freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
    tp = np.sum(psd)
    return 0.0 if tp == 0 else np.sum(freqs * psd) / tp
def median_frequency(segment, fs=FS):
    freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
    ps = np.sum(psd)
    if ps == 0:
        return 0.0
    cum = np.cumsum(psd)
    idx = np.where(cum >= ps/2)[0][0]
    return freqs[idx]
def fourier_features(segment, fs=FS):
    seg = np.asarray(segment)
    N = len(seg)
    fft_vals = np.fft.rfft(seg)
    mag = np.abs(fft_vals)
    power = mag**2 + 1e-12
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    total = np.sum(power)
    norm = power/(total+1e-12)
    centroid = np.sum(freqs * norm)
    spread = np.sqrt(np.sum(((freqs-centroid)**2) * norm))
    entropy = -np.sum(norm * np.log2(norm + 1e-12))
    return centroid, spread, entropy

def extract_features(segment_data, window_size=WINDOW_SIZE, fs=FS):
    seg = np.asarray(segment_data)
    step = window_size // 2
    feats = []
    for i in range(0, len(seg) - window_size + 1, step):
        w = seg[i:i+window_size]
        if len(w) != window_size:
            continue
        mav = mean_absolute_value(w); zc = zero_crossings(w); wl = waveform_length(w)
        rms = root_mean_square(w); var = variance(w)
        mnf = mean_power_frequency(w, fs=fs); mdf = median_frequency(w, fs=fs)
        sc, ss, se = fourier_features(w, fs=fs)
        feats.append([mav, zc, wl, rms, var, mnf, mdf, sc, ss, se])
    return np.array(feats)

# -------- serial thread --------
data_buffer = deque(maxlen=WINDOW_SIZE * 6)
prediction_history = deque(maxlen=PREDICTION_VOTE_WINDOWS)
serial_running = False
serial_obj = None

def serial_reader_thread(port, baud):
    global serial_running, serial_obj
    try:
        serial_obj = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f"[Serial] failed to open {port}: {e}")
        serial_running = False
        return
    print(f"[Serial] Opened {port} at {baud}")
    serial_running = True
    while serial_running:
        try:
            line = serial_obj.readline().decode(errors='ignore').strip()
            if not line:
                continue
            try:
                val = float(line)
                data_buffer.append(val)
            except ValueError:
                # print non-numeric occasionally
                if len(line) > 0:
                    print("[Serial] non-numeric line:", repr(line))
                continue
        except Exception as e:
            print("[Serial] exception reading:", e)
            break
    if serial_obj and getattr(serial_obj, "is_open", False):
        serial_obj.close()
    print("[Serial] stopped.")

# -------- prediction helper with debug prints --------
def predict_gesture_from_buffer(buffer_deque):
    arr = np.array(buffer_deque)
    if len(arr) < WINDOW_SIZE:
        return None
    feats = extract_features(arr, window_size=WINDOW_SIZE, fs=FS)
    if feats.size == 0:
        print("[Predict DEBUG] extract_features returned empty array.")
        return None
    # debug shape
    print(f"[Predict DEBUG] features shape: {feats.shape}")
    try:
        feats_scaled = scaler.transform(feats)
    except Exception as e:
        print("[Predict DEBUG] scaler.transform failed:", e)
        return None
    try:
        preds = model.predict(feats_scaled)
    except Exception as e:
        print("[Predict DEBUG] model.predict failed:", e)
        return None
    unique, counts = np.unique(preds, return_counts=True)
    pred = unique[np.argmax(counts)]
    print(f"[Predict DEBUG] window preds unique: {dict(zip(unique, counts))} -> majority: {pred}")
    return pred

# -------- pygame simple game --------
pygame.init()
WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EMG Gesture Game (DEBUG)")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 18)

# simple player
player_w, player_h = 50, 50
player_x, player_y = 100, HEIGHT - player_h - 60
player_vy = 0
gravity = 1.2
on_ground = True
shield_timer = 0.0
SHIELD_DURATION = 2.0

bullets = []
enemies = [{'x': 800, 'y': HEIGHT - 40, 'w': 40, 'h': 40, 'vx': -3}]
score = 0; lives = 3

def draw_text(x,y,s): screen.blit(FONT.render(s, True, (255,255,255)), (x,y))
def spawn_enemy(): enemies.append({'x': WIDTH + 50, 'y': HEIGHT - 40, 'w': 40, 'h': 40, 'vx': -3 - np.random.rand()*2})
def player_shoot(): bullets.append({'x': player_x + player_w, 'y': player_y + player_h//2, 'vx': 8})
def player_jump():
    global player_vy, on_ground
    if on_ground: player_vy = -16; on_ground = False
def use_shield(): global shield_timer; shield_timer = SHIELD_DURATION

SIM_KEY_MAP = {pygame.K_1: 'shoot', pygame.K_2: 'up_down', pygame.K_3: 'round'}

# start serial reader if available
reader_thread = None
if HAS_SERIAL and not FALLBACK_SIMULATION:
    try:
        reader_thread = threading.Thread(target=serial_reader_thread, args=(SERIAL_PORT, BAUD_RATE), daemon=True)
        reader_thread.start()
    except Exception as e:
        print("[Main] couldn't start serial thread:", e)
        FALLBACK_SIMULATION = True
else:
    print("[Main] pyserial not available or forced fallback.")

last_pred_time = 0.0
spawn_timer = 0.0

print("[Main] Starting game. Press 1/2/3 for fallback simulation if serial unavailable.")
try:
    while True:
        dt = clock.tick(60)/1000.0
        spawn_timer += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt()
            if FALLBACK_SIMULATION and event.type == pygame.KEYDOWN:
                if event.key in SIM_KEY_MAP:
                    g = SIM_KEY_MAP[event.key]; action = GESTURE_ACTION_MAP.get(g)
                    if action == "shoot": player_shoot()
                    elif action == "jump": player_jump()
                    elif action == "shield": use_shield()

        # periodic prediction
        if not FALLBACK_SIMULATION:
            now = time.time()
            if now - last_pred_time > PRED_INTERVAL:
                last_pred_time = now
                pred = predict_gesture_from_buffer(data_buffer)
                # if prediction returned as bytes or numerical, normalize:
                if pred is not None:
                    # convert bytes->str if needed
                    if isinstance(pred, bytes):
                        try: pred = pred.decode()
                        except: pred = str(pred)
                    # if numeric returned (like 1), map using INT_GESTURE_MAP
                    if isinstance(pred, (int, np.integer)):
                        pred_label = INT_GESTURE_MAP.get(int(pred), None)
                    else:
                        pred_label = str(pred)
                    prediction_history.append(pred_label)
                    # majority across history
                    u, c = np.unique(np.array(prediction_history), return_counts=True)
                    if len(u)>0:
                        maj = u[np.argmax(c)]
                        draw_action = GESTURE_ACTION_MAP.get(maj, None)
                        print(f"[Main] Majority prediction over history: {maj} -> action: {draw_action}")
                        if draw_action == "shoot": player_shoot()
                        elif draw_action == "jump": player_jump()
                        elif draw_action == "shield": use_shield()
                        last_shown_pred = maj
                else:
                    last_shown_pred = "None"
        else:
            last_shown_pred = "FALLBACK"

        # physics
        if not on_ground:
            player_vy += gravity
            player_y += player_vy
            if player_y >= HEIGHT - player_h - 60:
                player_y = HEIGHT - player_h - 60; player_vy = 0; on_ground = True

        # update bullets/enemies
        for b in bullets[:]:
            b['x'] += b['vx']
            if b['x'] > WIDTH + 100: bullets.remove(b)
        for e in enemies[:]:
            e['x'] += e['vx']
            if e['x'] < -100: enemies.remove(e)

        # spawn logic
        if spawn_timer > 1.2:
            spawn_timer = 0.0; spawn_enemy()

        # shield countdown
        if shield_timer > 0:
            shield_timer -= dt
            if shield_timer < 0: shield_timer = 0

        # draw
        screen.fill((30,30,40))
        pygame.draw.rect(screen, (60,200,60), (0, HEIGHT - 20, WIDTH, 20))
        pc = (200,200,255) if shield_timer<=0 else (255,215,0)
        pygame.draw.rect(screen, pc, (int(player_x), int(player_y), player_w, player_h))
        if shield_timer>0:
            pygame.draw.circle(screen, (255,255,100), (int(player_x+player_w/2), int(player_y+player_h/2)), 50, 3)
        for b in bullets: pygame.draw.rect(screen, (255,80,80), (int(b['x']), int(b['y']), 8, 4))
        for e in enemies: pygame.draw.rect(screen, (180,50,50), (int(e['x']), int(e['y']), e['w'], e['h']))

        # overlay debug
        draw_text(8,8, f"Buffer samples: {len(data_buffer)}")
        draw_text(8,28, f"Last prediction (history majority): {prediction_history[-1] if prediction_history else 'None'}")
        draw_text(8,48, f"Model classes: {getattr(model, 'classes_', 'None')}")
        draw_text(8,68, f"Serial running: {serial_running}")
        draw_text(8,88, "If serial fails, set FALLBACK_SIMULATION=True and use keys 1/2/3")

        pygame.display.flip()

except KeyboardInterrupt:
    print("[Main] KeyboardInterrupt - quitting.")
except Exception as e:
    print("[Main] Exception:", e)
finally:
    serial_running = False
    try:
        if serial_obj and getattr(serial_obj, "is_open", False): serial_obj.close()
    except Exception:
        pass
    pygame.quit()
    print("Game ended.")
