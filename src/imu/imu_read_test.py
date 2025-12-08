import serial
from pynput.keyboard import Controller, Key
from collections import deque
import time

keyboard = Controller()

# ===== SERIAL =====
PORT = "COM8"
BAUD = 115200
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ===== SMOOTHING =====
ax_buf = deque(maxlen=5)
ay_buf = deque(maxlen=5)

# ===== THRESHOLDS (Stable + Fast) =====
LEFT_AY  = -6500
RIGHT_AY =  6500

UP_AX    = -8000
DOWN_AX  =  5500

# ===== Dead zone =====
REST_AX_MIN = -3500
REST_AX_MAX =  3500
REST_AY_MIN = -3500
REST_AY_MAX =  3500

# Track currently held key
current_key = None

def hold(key):
    """Hold a key (press but no release)."""
    keyboard.press(key)

def release_all():
    """Release all directional keys."""
    for k in [Key.left, Key.right, Key.up, Key.down]:
        keyboard.release(k)

print("\nPLONKY IMU - HOLD CONTROL MODE READY...\n")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if not line:
        continue

    try:
        ax, ay, az = map(float, line.split(","))
    except:
        continue

    # Smooth
    ax_buf.append(ax)
    ay_buf.append(ay)

    ax_s = sum(ax_buf) / len(ax_buf)
    ay_s = sum(ay_buf) / len(ay_buf)

    # ===== Detect direction =====
    if REST_AX_MIN < ax_s < REST_AX_MAX and REST_AY_MIN < ay_s < REST_AY_MAX:
        state = "rest"
    elif ay_s < LEFT_AY:
        state = "left"
    elif ay_s > RIGHT_AY:
        state = "right"
    elif ax_s < UP_AX:
        state = "up"
    elif ax_s > DOWN_AX:
        state = "down"
    else:
        state = "rest"

    # ===== Handle key holding =====
    if state == "rest":
        if current_key is not None:
            release_all()
            current_key = None
            print("→ REST (release all)")
            
    else:
        key_map = {
            "left": Key.left,
            "right": Key.right,
            "up": Key.up,
            "down": Key.down
        }
        new_key = key_map[state]
        
        if new_key != current_key:
            release_all()
            hold(new_key)
            current_key = new_key
            print(f"HOLD → {state.upper()}  Ax={ax_s:.0f} Ay={ay_s:.0f}")
