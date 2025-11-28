import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os



# ===================== USER CONFIGURATION =====================
# Directory containing your CSV files (relative to where you run the script)
# NOTE: This must match the directory in your initial prompt output!
BASE_DIR = r"C:\Users\rohit\OneDrive\Desktop\MTECH\Sem3\HACKATHON\Code\SIH"

# Define the gestures and the CSV files associated with each one
# Assign a list of all relevant files for each gesture category.

GESTURE_FILES = {
    'round': ["round1.csv", "round3.csv"],
    'shoot': ["shoot3.csv", "shoot_gesture1.csv"],
    'up_down': ["up_down1.csv", "up_down3.csv"]
}
# Column name in the CSV that contains the raw sensor value.
# Based on your initial script, we assume the data is in the second column (index 1), 
# which we'll name 'value' after loading.
VALUE_COLUMN = 'value'
# ============================================================

# --- 1. Enhanced Feature Extraction Functions (Time Domain) ---

def mean_absolute_value(segment):
    """Calculates the Mean Absolute Value (MAV) of the segment."""
    return np.mean(np.abs(segment))


def zero_crossings(segment):
    """Calculates the number of times the signal crosses the mean-axis."""
    # We count crossings of the mean value instead of true zero for filtering robustness
    mean_val = np.mean(segment)
    return np.sum(np.diff(np.array(segment) > mean_val) != 0)


def waveform_length(segment):
    """Calculates the Waveform Length (WL), sum of absolute differences between samples."""
    return np.sum(np.abs(np.diff(segment)))


def root_mean_square(segment):
    """Calculates the Root Mean Square (RMS) of the segment."""
    return np.sqrt(np.mean(segment**2))



def variance(segment):
    """Calculates the Variance (VAR) of the segment."""
    return np.var(segment)



def extract_features(segment_data, window_size=100):
    """
    Extracts 5 time-domain features from the time series using a sliding window.
    
    Returns a DataFrame where each row is a feature vector for one window.
    """
    features = []
    
    # Iterate over the data using a sliding window (50% overlap)
    for i in range(0, len(segment_data) - window_size + 1, window_size // 2): 
        window = segment_data[i:i + window_size]
        
        # Ensure the window is full size
        if len(window) == window_size:
            mav = mean_absolute_value(window)
            zc = zero_crossings(window)
            wl = waveform_length(window)
            rms = root_mean_square(window) # NEW FEATURE
            var = variance(window)         # NEW FEATURE
            
            features.append([mav, zc, wl, rms, var])
            
    return pd.DataFrame(features, columns=['MAV', 'ZC', 'WL', 'RMS', 'VAR']) # Updated columns

# --- 2. Data Loading and Feature Aggregation ---

print(f"Loading data and extracting features from: {BASE_DIR}")
all_data = [] # To store the feature vectors and labels


for gesture, files in GESTURE_FILES.items():
    print(f"\nProcessing gesture: '{gesture}'")
    
    for file_name in files:
        file_path = os.path.join(BASE_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f"  ❌ File not found: {file_name}. Skipping.")
            continue
            
        try:
            # Load the CSV.
            df = pd.read_csv(file_path)
            
            # --- START DATA CLEANING FIX (Addresses 'str' error) ---
            # Convert 'value' column to numeric, forcing errors (like strings) to NaN.
            df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors='coerce') 
            
            # Drop rows where the 'value' is now NaN (i.e., remove non-numeric data)
            df.dropna(subset=[VALUE_COLUMN], inplace=True)
            
            # Use the cleaned 'value' column for raw signal extraction
            raw_signal = df[VALUE_COLUMN].values
            # --- END DATA CLEANING FIX ---
            
            # Check if there is enough data after cleaning
            if len(raw_signal) < 100:
                print(f"  ⚠ Not enough clean data in {file_name}. Skipping.")
                continue

            # Extract 5 features from this raw signal segment
            feature_df = extract_features(raw_signal, window_size=100)
            
            # Assign the label to all feature vectors from this file
            feature_df['label'] = gesture
            all_data.append(feature_df)
            
            print(f"  ✅ Loaded and extracted {len(feature_df)} feature vectors from {file_name}")
            
        except Exception as e:
            # Print a detailed error if cleaning failed or other loading issues occurred
            print(f"  ❌ Error processing {file_name} after initial clean: {e}")

# Combine all feature DataFrames into one
if not all_data:
    print("\nFATAL: No data was successfully loaded. Check file paths and column names.")
    exit()

final_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal extracted feature vectors: {len(final_df)}")


# --- 3. Data Visualization (Raw Signal and Features) ---

# A. Raw Signal Visualization (Plotting)
def plot_raw_signal(data_frame, gesture_name):
    """Plots a 1-second segment of the raw signal for inspection."""
    plt.figure(figsize=(12, 4))
    
    # Filter for the specific gesture
    gesture_df = data_frame[data_frame['label'] == gesture_name]
    
    if len(gesture_df) == 0:
        print(f"Cannot plot raw signal: No data found for '{gesture_name}'.")
        return
        
    # Pick a random 500-sample segment (approx 1 second if sampling rate is 500Hz)
    # Ensure this segment is available in the loaded file.
    start_index = np.random.randint(0, len(gesture_df) - 500) if len(gesture_df) >= 500 else 0
    sample_segment = gesture_df[VALUE_COLUMN].iloc[start_index : start_index + 500]

    plt.plot(sample_segment.values, label=f'Raw Signal Segment ({gesture_name})', color='purple', alpha=0.8)
    plt.title(f'Sample Raw EMG Signal for "{gesture_name}" Gesture (500 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sensor Value')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()

# Run the raw plotting function for one of the gestures
try:
    sample_file_path = os.path.join(BASE_DIR, GESTURE_FILES['shoot'][0])
    sample_raw_df = pd.read_csv(sample_file_path)
    sample_raw_df['label'] = 'shoot' # Add label temporarily for plotting function
    plot_raw_signal(sample_raw_df, 'shoot')
except Exception as e:
    print(f"Could not perform raw signal plot due to error: {e}")


# B. Feature Visualization (3D Plot of MAV, ZC, WL)
print("\nGenerating 3D feature visualization...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Get unique labels
labels = final_df['label'].unique()
# FIXED Matplotlib Deprecation Warning AND TypeError: Removed the second argument
cmap = plt.colormaps.get_cmap('viridis') 

# Plot each gesture's features (Only MAV, ZC, WL are plotted for 3D visualization)
for i, label in enumerate(labels):
    subset = final_df[final_df['label'] == label]
    
    # Calculate a normalized index (0 to 1) for the color mapping
    # This correctly maps the discrete loop index 'i' to a position on the continuous colormap 'cmap'.
    normalized_index = i / (len(labels) - 1) if len(labels) > 1 else 0

    ax.scatter(subset['MAV'], subset['ZC'], subset['WL'], 
                label=label, 
                olor=cmap(normalized_index), # Use cmap as the function with the normalized index
                alpha=0.6,
                s=50) # s is marker size

ax.set_xlabel('Mean Absolute Value (MAV)')
ax.set_ylabel('Zero Crossings (ZC)')
ax.set_zlabel('Waveform Length (WL)')
ax.set_title('3D Visualization of Extracted EMG Features (MAV, ZC, WL)')
ax.legend()
plt.tight_layout()
# --- 4. Machine Learning Model Training (Now uses 5 features) ---

print("\n" + "="*50)
print("STARTING MACHINE LEARNING CLASSIFICATION")
print("   Using 5 Time-Domain Features: MAV, ZC, WL, RMS, VAR")
print("="*50)

# Define feature matrix (X) and target vector (y). UPDATED to include RMS and VAR.
X = final_df[['MAV', 'ZC', 'WL', 'RMS', 'VAR']]
y = final_df['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Total feature vectors: {len(X)}")
print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# Feature Scaling (Crucial for distance-based algorithms like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Hyperparameter Tuning to Find the Optimal K ---
max_k = 20
k_range = range(1, max_k + 1, 2) # Test odd K values from 1 to 21 (or max_k)
scores = {}
best_k = 5
best_accuracy = 0

print("\n--- Tuning K-Value (1 to 20) for KNN ---")
for k in k_range:
    # Train the model for the current K
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred_k = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_k)
    scores[k] = accuracy
    print(f"  K={k}: Accuracy = {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Plotting the K-value results
plt.figure(figsize=(10, 5))
plt.plot(list(scores.keys()), list(scores.values()), marker='o', linestyle='--', color='blue')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Classification Accuracy')
plt.xticks(list(k_range))
plt.grid(True)
plt.tight_layout()

print(f"\n--- Best K Found: {best_k} (Accuracy: {best_accuracy:.4f}) ---")
# --- Retrain the final model using the best K ---
final_k = best_k
model = KNeighborsClassifier(n_neighbors=final_k)
print(f"\nTraining FINAL {model.__class__.__name__} with OPTIMAL K={final_k}...")
model.fit(X_train_scaled, y_train)
print("Final training complete.")


# --- 5. Model Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Print the performance report
print("\n" + "="*20 + " CLASSIFICATION REPORT (K={}) ".format(final_k) + "="*20)
print(classification_report(y_test, y_pred, target_names=labels))
print("="*63)

# Show all generated plots
plt.show()

print("\nML Pipeline finished. Review the plots and classification report.")