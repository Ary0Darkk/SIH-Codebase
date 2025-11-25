import serial
import time
import os
import threading
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# ===================== USER CONFIG =====================
PORT = "COM5"
BAUD = 115200
SAVE_DIR = "data"
MAX_SAMPLES = 800
# =======================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# -------- SERIAL INIT ----------
ser = serial.Serial(PORT, BAUD, timeout=0.05)
time.sleep(1)
print("Serial connected.")

# -------- STATE VARIABLES --------
current_label = "rest"
recording = False
segment_data = []
segment_time = []
data_buffer = []
running = True

# -------- RECORDING FUNCTIONS --------
def start_recording(label):
    global recording, current_label, segment_data, segment_time
    current_label = label
    recording = True
    segment_data = []
    segment_time = []
    print(f"\n▶ START: {label.upper()}")


def stop_recording():
    global recording, segment_data, segment_time

    if not recording:
        return
    
    recording = False
    print(f"⏹ STOP: {current_label.upper()}")

    if len(segment_data) < 10:
        print("⚠ Segment too short. Not saving.")
        return

    label_dir = os.path.join(SAVE_DIR, current_label)
    os.makedirs(label_dir, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    filepath = os.path.join(label_dir, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value"])
        for t, v in zip(segment_time, segment_data):
            writer.writerow([t, v])

    print(f"💾 SAVED: {filepath}")


# -------- SERIAL READER THREAD --------
def serial_reader():
    global recording, running
    while running:
        try:
            raw = ser.readline().decode(errors="ignore").strip()
            if raw == "":
                continue
            try:
                value = float(raw)
            except:
                continue

            t = time.time()

            # plot buffer
            if len(data_buffer) > MAX_SAMPLES:
                del data_buffer[0]
            data_buffer.append(value)

            # recording buffer
            if recording:
                segment_data.append(value)
                segment_time.append(t)

        except Exception as e:
            print("Serial Error:", e)
            time.sleep(0.01)


# -------- LIVE PLOT SETUP --------
fig, ax = plt.subplots(figsize=(8, 3))
line, = ax.plot([], [], lw=1)

ax.set_ylim(0, 1024)
ax.set_xlim(0, MAX_SAMPLES)
ax.set_title("Live EMG Signal")
ax.set_xlabel("Samples")
ax.set_ylabel("Value")
ax.grid(True)


def update_plot(frame):
    if len(data_buffer) == 0:
        return line,
    y = np.array(data_buffer)
    x = np.arange(len(y))
    line.set_data(x, y)
    return line,


ani = animation.FuncAnimation(fig, update_plot, interval=40, blit=True)


# -------- KEYBOARD HANDLER --------
def handle_key(event):
    key = event.key.lower()

    if key == "r":
        start_recording("rest")

    elif key == "u":
        start_recording("up")

    elif key == "d":
        start_recording("down")

    elif key == "o":
        start_recording("rotate")

    elif key == "s":
        stop_recording()

    elif key == "q":
        print("Exiting...")
        plt.close()


fig.canvas.mpl_connect("key_press_event", handle_key)

# -------- START THREAD --------
threading.Thread(target=serial_reader, daemon=True).start()

# -------- HELP MENU --------
print("\n========== CONTROLS ==========")
print(" r → Record REST")
print(" u → Record UP")
print(" d → Record DOWN")
print(" o → Record ROTATE")
print(" s → STOP and SAVE")
print(" q → Quit program")
print("=================================\n")

# -------- START PLOTTING --------
plt.show()

# -------- CLEAN EXIT --------
running = False
time.sleep(0.2)
ser.close()
print("Serial closed. Done.")
