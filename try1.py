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
SAVE_DIR = r"C:\sih\SIH_GamingNexus\Gayatri"
MAX_SAMPLES = 800
# =======================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- CREATE RAW CSV FILE (NEW) ----------
raw_csv_path = os.path.join(SAVE_DIR, "raw_data.csv")     # <-- NEW
raw_csv = open(raw_csv_path, "w", newline="")             # <-- NEW
raw_writer = csv.writer(raw_csv)                           # <-- NEW
raw_writer.writerow(["timestamp", "value"])                # <-- NEW
# --------------------------------------------------------

# ---------- SERIAL INIT ----------
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.02)
    time.sleep(1)
    print("Serial connected.\n")
except Exception as e:
    print("ERROR: Serial port issue →", e)
    exit()

# ---------- STATE VARIABLES ----------
current_label = ""
recording = False
segment_data = []
segment_time = []
data_buffer = []
running = True


# ---------- RECORDING ----------
def start_recording(label):
    global recording, current_label, segment_data, segment_time
    current_label = label
    recording = True
    segment_data = []
    segment_time = []
    print(f"\n▶ START LABEL: {label}")


def stop_recording():
    global recording, segment_data, segment_time

    if not recording:
        return

    recording = False
    print(f"⏹ STOP LABEL: {current_label}")

    if len(segment_data) < 10:
        print("⚠ Not enough data. Segment not saved.")
        return

    folder = os.path.join(SAVE_DIR, current_label)
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    path = os.path.join(folder, filename)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value"])
        for t, v in zip(segment_time, segment_data):
            writer.writerow([t, v])

    print(f"💾 SAVED: {path}")


# ---------- SERIAL THREAD ----------
def serial_reader():
    global running, recording

    while running:
        try:
            raw = ser.readline().decode(errors="ignore").strip()
            if raw == "":
                continue

            try:
                value = float(raw)
            except:
                continue

            now = time.time()

            # ---------- WRITE TO RAW CSV FILE (NEW) ----------
            raw_writer.writerow([now, value])     # <-- NEW
            raw_csv.flush()                       # <-- NEW
            # --------------------------------------------------

            # Live buffer
            if len(data_buffer) >= MAX_SAMPLES:
                del data_buffer[0]
            data_buffer.append(value)

            # Segment recording buffer
            if recording:
                segment_data.append(value)
                segment_time.append(now)

        except:
            time.sleep(0.005)


# ---------- LIVE PLOT ----------
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [], lw=1)
ax.set_ylim(0, 1024)
ax.set_xlim(0, MAX_SAMPLES)
ax.set_title("Live EMG Signal")
ax.set_xlabel("Samples")
ax.set_ylabel("Value")
ax.grid(True)

def update_plot(frame):
    if len(data_buffer) > 0:
        y = np.array(data_buffer)
        x = np.arange(len(y))
        line.set_data(x, y)
    return line,


ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=True)


# ---------- KEYBOARD CONTROL ----------
def key_handler(event):
    key = event.key.lower()

    if key == "1":
        start_recording("rest")

    elif key == "2":
        start_recording("up")

    elif key == "3":
        start_recording("down")

    elif key == "4":
        start_recording("rotate")

    elif key == "s":
        stop_recording()

    elif key == "q":
        print("Exiting…")
        plt.close()

fig.canvas.mpl_connect("key_press_event", key_handler)

# ---------- START THREAD ----------
threading.Thread(target=serial_reader, daemon=True).start()

# ---------- HELP ----------
print("=========== CONTROLS ===========")
print(" 1 → REST")
print(" 2 → UP")
print(" 3 → DOWN")
print(" 4 → ROTATE")
print(" s → STOP + SAVE")
print(" q → QUIT")
print("================================\n")

# ---------- RUN ----------
plt.show()

# ---------- EXIT CLEANLY ----------
running = False
time.sleep(0.2)
ser.close()

raw_csv.close()   # <-- IMPORTANT NEW LINE
print("\nSerial Closed. Program Ended.")
