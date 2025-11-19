import serial
import time
from datetime import datetime
import csv

PORT = "COM4"       # <- change to your Arduino port
BAUD = 115200
FS = 250
CSV_FILE = "ecg_data.csv"

ser = serial.Serial(PORT, BAUD)
time.sleep(2)  # let Arduino reset

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "sample_index", "raw_input"])

    print("Starting ECG acquisition... Ctrl+C to stop.")
    n = 0
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            raw = int(line)
            timestamp = time.time()
            t = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
            # t = time.time()
            writer.writerow([t, n, raw])
            print(f"{n}\t{raw}")
            n += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"Saved {n} samples to {CSV_FILE}")

ser.close()
