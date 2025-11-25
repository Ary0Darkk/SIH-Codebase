# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # ======================
# # USER CONFIG
# # ======================
# CSV_PATH = r"C:\sih\SIH_GamingNexus\Gayatri\round1.csv"   # change to your file
# WINDOW = 20   # sliding window size

# # ======================
# # Feature Functions
# # ======================

# def MAV(x):
#     return np.mean(np.abs(x))

# def RMS(x):
#     return np.sqrt(np.mean(x**2))

# def ZC(x, threshold=1e-3):
#     count = 0
#     for i in range(len(x)-1):
#         if (x[i] > 0 and x[i+1] < 0) or (x[i] < 0 and x[i+1] > 0):
#             if abs(x[i] - x[i+1]) > threshold:
#                 count += 1
#     return count

# def SSC(x, threshold=1e-3):
#     count = 0
#     for i in range(1, len(x)-1):
#         if ((x[i] - x[i-1]) * (x[i] - x[i+1]) > threshold):
#             count += 1
#     return count

# def WL(x):
#     return np.sum(np.abs(np.diff(x)))


# # ======================
# # Load CSV
# # ======================
# df = pd.read_csv(CSV_PATH, header=None, names=["timestamp", "value"])



# # ======================
# # Sliding Window Feature Extraction
# # ======================
# feature_data = {
#     "MAV": [],
#     "RMS": [],
#     "ZC": [],
#     "SSC": [],
#     "WL": [],
#     "Index": []
# }

# for i in range(0, len(values) - WINDOW):
#     window = values[i:i+WINDOW]

#     feature_data["MAV"].append(MAV(window))
#     feature_data["RMS"].append(RMS(window))
#     feature_data["ZC"].append(ZC(window))
#     feature_data["SSC"].append(SSC(window))
#     feature_data["WL"].append(WL(window))
#     feature_data["Index"].append(i)

# features_df = pd.DataFrame(feature_data)

# # ======================
# # Plot the feature curves
# # ======================
# plt.figure(figsize=(14, 10))

# plt.subplot(5, 1, 1)
# plt.plot(features_df["Index"], features_df["MAV"])
# plt.title("MAV (Mean Absolute Value)")
# plt.ylabel("MAV")

# plt.subplot(5, 1, 2)
# plt.plot(features_df["Index"], features_df["RMS"])
# plt.title("RMS")
# plt.ylabel("RMS")

# plt.subplot(5, 1, 3)
# plt.plot(features_df["Index"], features_df["ZC"])
# plt.title("Zero Crossing (ZC)")
# plt.ylabel("ZC")

# plt.subplot(5, 1, 4)
# plt.plot(features_df["Index"], features_df["SSC"])
# plt.title("Slope Sign Changes (SSC)")
# plt.ylabel("SSC")

# plt.subplot(5, 1, 5)
# plt.plot(features_df["Index"], features_df["WL"])
# plt.title("Waveform Length (WL)")
# plt.ylabel("WL")
# plt.xlabel("Window Index")

# plt.tight_layout()
# plt.show()

# print("Feature extraction complete.")





# ---------------------------------------------------------------








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# ======================
# USER CONFIG
# ======================
# --- 1. Define the paths for your three CSV files ---
# NOTE: Update these paths to match your actual file locations (e.g., in the 'Gayatri' folder)



CSV_PATHS = [
    r"C:\Users\rohit\OneDrive\Desktop\MTECH\Sem3\HACKATHON\Code\SIH\shoot_gesture1.csv",
    r"C:\Users\rohit\OneDrive\Desktop\MTECH\Sem3\HACKATHON\Code\SIH\round1.csv",
    r"C:\Users\rohit\OneDrive\Desktop\MTECH\Sem3\HACKATHON\Code\SIH\up_down1.csv",
]



# --- 2. Labels for the plots ---
LABELS = ["REST", "UP Movement", "DOWN Movement"]

# --- 3. Feature Extraction Settings ---
WINDOW = 50  # Sliding window size (increased from 20 for smoother features)
ZC_THRESHOLD = 1e-3
SSC_THRESHOLD = 1e-3

# ======================
# Feature Functions
# ======================

def MAV(x):
    """Mean Absolute Value"""
    return np.mean(np.abs(x))

def RMS(x):
    """Root Mean Square"""
    return np.sqrt(np.mean(x**2))

def ZC(x, threshold=ZC_THRESHOLD):
    """Zero Crossing (ZC)"""
    count = 0
    for i in range(len(x)-1):
        # Checks if a zero crossing occurred AND the change is greater than the threshold
        if (x[i] * x[i+1] < 0) and (np.abs(x[i] - x[i+1]) > threshold):
            count += 1
    return count

def SSC(x, threshold=SSC_THRESHOLD):
    """Slope Sign Changes (SSC)"""
    count = 0
    # Needs at least three points (i-1, i, i+1)
    for i in range(1, len(x)-1):
        # Checks for a change in sign of the slope (a peak or trough)
        if ((x[i] - x[i-1]) * (x[i] - x[i+1]) >= threshold):
            count += 1
    return count

def WL(x):
    """Waveform Length (WL)"""
    return np.sum(np.abs(np.diff(x)))

# ======================
# Processing Function
# ======================

def process_file(file_path, label, window_size):
    """Loads a CSV, performs feature extraction, and returns results."""
    # Load the CSV file. Assuming the structure is: timestamp,value
    df = pd.read_csv(file_path, header=0, names=["timestamp", "value"])

    # Extract the 'value' column as a NumPy array for fast calculation
    values = df["value"].values

    feature_data = {
        "MAV": [],
        "RMS": [],
        "ZC": [],
        "SSC": [],
        "WL": [],
        "Index": [],
        "Timestamp": [] # Store the timestamp corresponding to the start of the window
    }

    # Sliding Window Feature Extraction Loop
    for i in range(0, len(values) - window_size, window_size // 2): # Using 50% overlap
        window = values[i:i+window_size]

        feature_data["MAV"].append(MAV(window))
        feature_data["RMS"].append(RMS(window))
        feature_data["ZC"].append(ZC(window))
        feature_data["SSC"].append(SSC(window))
        feature_data["WL"].append(WL(window))

        # Use the timestamp of the middle of the window for better alignment
        middle_index = i + window_size // 2
        feature_data["Timestamp"].append(df["timestamp"].iloc[middle_index])
        feature_data["Index"].append(middle_index)

    features_df = pd.DataFrame(feature_data)

    return df, features_df, label

# ======================
# MAIN EXECUTION
# ======================
all_raw_data = []
all_feature_data = []

# Process all files
for i, path in enumerate(CSV_PATHS):
    if not os.path.exists(path):
        print(f"ERROR: File not found at path: {path}. Skipping.")
        continue
    try:
        raw_df, features_df, label = process_file(path, LABELS[i], WINDOW)
        all_raw_data.append((raw_df, label))
        all_feature_data.append((features_df, label))
    except Exception as e:
        print(f"An error occurred while processing {path}: {e}")

if not all_raw_data:
    print("No data files were successfully loaded. Exiting plot.")
    exit()

# ====================================================================
# PLOT 1: RAW TIME-SERIES DATA (Timestamp vs. Value) - Requested Plot
# ====================================================================
plt.figure(figsize=(14, 6))
plt.style.use('seaborn-v0_8-darkgrid')

for raw_df, label in all_raw_data:
    # Convert Unix timestamp to readable datetime objects for better x-axis labeling
    time_index = pd.to_datetime(raw_df["timestamp"], unit='s')
    plt.plot(time_index, raw_df["value"], label=label, alpha=0.7, linewidth=1)

plt.title("Comparison of Raw EMG Signals (Timestamp vs. Value)", fontsize=16)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("ADC Value", fontsize=12)
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nRaw data plotting complete.")

# ====================================================================
# PLOT 2: Feature Curves Comparison
# ====================================================================
if all_feature_data:
    plt.figure(figsize=(14, 15))
    feature_names = ["MAV", "RMS", "ZC", "SSC", "WL"]

    for i, feature_name in enumerate(feature_names):
        plt.subplot(len(feature_names), 1, i + 1)
        
        for features_df, label in all_feature_data:
            # Plot features against their calculated timestamp
            plt.plot(features_df["Timestamp"], features_df[feature_name], label=label, linewidth=2)

        plt.title(f"{feature_name} Feature Progression (Window={WINDOW})", fontsize=14)
        plt.ylabel(feature_name, fontsize=12)
        plt.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right')
        plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    plt.xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.show()

    print("Feature curve plotting complete.")