import serial
import time
import csv
from collections import deque

# ---------------------------
# CONFIG
# ---------------------------
PORT = "COM5"          # Windows example. On Linux/Mac: "/dev/ttyACM0" or "/dev/ttyUSB0"
BAUD = 115200
LOG_TO_CSV = True
CSV_FILENAME = "imu_log.csv"

# how many recent samples to keep in memory (for smoothing / debug)
BUFFER_SIZE = 200


def parse_line(line: str):
    """
    Expecting CSV:
    ax,ay,az,gx,gy,gz
    Returns tuple of floats or None if invalid.
    """
    try:
        parts = line.strip().split(",")
        if len(parts) < 6:
            return None
        ax, ay, az, gx, gy, gz = map(float, parts[:6])
        return ax, ay, az, gx, gy, gz
    except ValueError:
        return None


def main():
    print(f"Opening serial port {PORT} @ {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # let Arduino reset

    buffer = deque(maxlen=BUFFER_SIZE)

    csv_file = None
    csv_writer = None
    if LOG_TO_CSV:
        csv_file = open(CSV_FILENAME, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
        print(f"Logging enabled -> {CSV_FILENAME}")

    print("Reading IMU data. Press Ctrl+C to stop.")
    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore")
            if not raw:
                continue

            data = parse_line(raw)
            if data is None:
                # uncomment to debug bad lines
                # print("Bad line:", raw)
                continue

            ax, ay, az, gx, gy, gz = data
            ts = time.time()

            buffer.append((ts, ax, ay, az, gx, gy, gz))

            # ---- print latest sample ----
            print(
                f"t={ts:.3f} | "
                f"A=({ax: .3f},{ay: .3f},{az: .3f}) "
                f"G=({gx: .3f},{gy: .3f},{gz: .3f})"
            )

            # ---- optionally log to CSV ----
            if csv_writer:
                csv_writer.writerow([ts, ax, ay, az, gx, gy, gz])

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        if csv_file:
            csv_file.close()
        print("Serial closed, file saved.")


if __name__ == "__main__":
    main()
