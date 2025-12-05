# # emg_model_trainer.py
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import joblib
# import warnings
# warnings.filterwarnings('ignore')

# # Feature extraction functions (same as your original)
# from scipy.signal import welch

# class EMGModelTrainer:
#     """
#     Train machine learning model for EMG gesture classification
#     """
    
#     def __init__(self, data_dir="emg_training_data", window_size=100, fs=500):
#         self.data_dir = data_dir
#         self.window_size = window_size
#         self.fs = fs
#         self.scaler = StandardScaler()
#         self.model = None
#         self.feature_names = None
#         self.gesture_labels = None
        
#         # Feature extraction functions
#         self.feature_functions = {
#             'MAV': self.mean_absolute_value,
#             'ZC': self.zero_crossings,
#             'WL': self.waveform_length,
#             'RMS': self.root_mean_square,
#             'VAR': self.variance,
#             'MNF': self.mean_power_frequency,
#             'MDF': self.median_frequency,
#             'SpecCent': lambda x: self.fourier_features(x)[0],
#             'SpecSpread': lambda x: self.fourier_features(x)[1],
#             'SpecEnt': lambda x: self.fourier_features(x)[2]
#         }
    
#     # Feature extraction methods (same as your original)
#     def mean_absolute_value(self, segment):
#         return np.mean(np.abs(segment))
    
#     def zero_crossings(self, segment):
#         mean_val = np.mean(segment)
#         return np.sum(np.diff(np.array(segment) > mean_val) != 0)
    
#     def waveform_length(self, segment):
#         return np.sum(np.abs(np.diff(segment)))
    
#     def root_mean_square(self, segment):
#         return np.sqrt(np.mean(segment**2))
    
#     def variance(self, segment):
#         return np.var(segment)
    
#     def mean_power_frequency(self, segment, fs=None):
#         fs = fs or self.fs
#         freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
#         total_power = np.sum(psd)
#         if total_power == 0:
#             return 0.0
#         return np.sum(freqs * psd) / total_power
    
#     def median_frequency(self, segment, fs=None):
#         fs = fs or self.fs
#         freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
#         power_sum = np.sum(psd)
#         if power_sum == 0:
#             return 0.0
#         cumulative_power = np.cumsum(psd)
#         median_idx = np.where(cumulative_power >= power_sum / 2)[0][0]
#         return freqs[median_idx]
    
#     def fourier_features(self, segment, fs=None):
#         fs = fs or self.fs
#         segment = np.asarray(segment)
#         N = len(segment)
#         fft_vals = np.fft.rfft(segment)
#         mag = np.abs(fft_vals)
#         power = mag**2
#         freqs = np.fft.rfftfreq(N, d=1.0 / fs)
#         total_power = np.sum(power) + 1e-12
#         norm_power = power / total_power
#         spectral_centroid = np.sum(freqs * norm_power)
#         spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_power))
#         spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))
#         return spectral_centroid, spectral_spread, spectral_entropy
    
#     def extract_features_from_segment(self, segment):
#         """Extract all features from a single segment"""
#         features = []
#         for name, func in self.feature_functions.items():
#             if name in ['SpecCent', 'SpecSpread', 'SpecEnt']:
#                 # These are handled together in fourier_features
#                 continue
#             elif name in ['MNF', 'MDF']:
#                 features.append(func(segment, self.fs))
#             else:
#                 features.append(func(segment))
        
#         # Add Fourier features
#         spec_cent, spec_spread, spec_ent = self.fourier_features(segment, self.fs)
#         features.extend([spec_cent, spec_spread, spec_ent])
        
#         return features
    
#     def load_and_preprocess_data(self):
#         """Load all CSV files and preprocess into features"""
#         print("Loading training data...")
        
#         all_features = []
#         all_labels = []
        
#         # Get all CSV files
#         csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
#         if not csv_files:
#             raise ValueError(f"No CSV files found in {self.data_dir}")
        
#         print(f"Found {len(csv_files)} data files")
        
#         for csv_file in csv_files:
#             filepath = os.path.join(self.data_dir, csv_file)
#             df = pd.read_csv(filepath)
            
#             # Get gesture label (should be same for entire file)
#             if 'gesture' in df.columns:
#                 gesture_label = df['gesture'].iloc[0]
#             else:
#                 # Infer from filename
#                 gesture_label = csv_file.split('_')[0]
            
#             # Get EMG values
#             if 'emg_value' in df.columns:
#                 emg_data = df['emg_value'].values
#             elif 'value' in df.columns:
#                 emg_data = df['value'].values
#             else:
#                 # Try to find any numeric column
#                 numeric_cols = df.select_dtypes(include=[np.number]).columns
#                 if len(numeric_cols) > 0:
#                     emg_data = df[numeric_cols[0]].values
#                 else:
#                     print(f"Skipping {csv_file}: No EMG data found")
#                     continue
            
#             # Apply sliding window
#             for i in range(0, len(emg_data) - self.window_size + 1, self.window_size // 2):
#                 segment = emg_data[i:i + self.window_size]
                
#                 # Extract features
#                 features = self.extract_features_from_segment(segment)
#                 all_features.append(features)
#                 all_labels.append(gesture_label)
            
#             print(f"  Processed {csv_file}: {len(emg_data)} samples -> {len(df) // self.window_size * 2} windows")
        
        
#         # Convert to numpy arrays
#         X = np.array(all_features)
#         y = np.array(all_labels)
        
        
#         # Get unique labels
#         self.gesture_labels = np.unique(y)
#         print(f"\nLoaded {len(X)} samples with {len(self.gesture_labels)} gestures: {list(self.gesture_labels)}")
        
#         return X, y
    
#     def train_model(self, X, y, test_size=0.2, random_state=42):
#         """Train and evaluate the model"""
#         print("\n" + "="*60)
#         print("TRAINING GESTURE CLASSIFICATION MODEL")
#         print("="*60)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=random_state, stratify=y
#         )
        
#         print(f"Training set: {X_train.shape[0]} samples")
#         print(f"Test set: {X_test.shape[0]} samples")
        
#         # Scale features
#         print("\nScaling features...")
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
        
#         # Define models to try
#         models = {
#             'Random Forest': RandomForestClassifier(
#                 n_estimators=100,
#                 max_depth=10,
#                 random_state=random_state,
#                 class_weight='balanced'
#             ),
#             'SVM': SVC(
#                 kernel='rbf',
#                 C=1.0,
#                 gamma='scale',
#                 random_state=random_state,
#                 class_weight='balanced'
#             ),
#             'Neural Network': MLPClassifier(
#                 hidden_layer_sizes=(64, 32),
#                 activation='relu',
#                 solver='adam',
#                 max_iter=500,
#                 random_state=random_state,
#                 early_stopping=True
#             )
#         }
        
        
#         best_model = None
#         best_score = 0
#         best_model_name = ""
        
        
#         # Train and evaluate each model
#         for name, model in models.items():
#             print(f"\nTraining {name}...")
            
#             # Cross-validation
#             cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
#             print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
#             # Train on full training set
#             model.fit(X_train_scaled, y_train)
            
#             # Test accuracy
#             y_pred = model.predict(X_test_scaled)
#             accuracy = accuracy_score(y_test, y_pred)
#             print(f"  Test accuracy: {accuracy:.3f}")
            
#             # Update best model
#             if accuracy > best_score:
#                 best_score = accuracy
#                 best_model = model
#                 best_model_name = name
        
#         print(f"\n{'='*60}")
#         print(f"BEST MODEL: {best_model_name} with accuracy: {best_score:.3f}")
#         print("="*60)
        
#         # Detailed evaluation of best model
#         print("\nDetailed Classification Report:")
#         print(classification_report(y_test, best_model.predict(X_test_scaled), target_names=self.gesture_labels))
        
#         # Confusion matrix
#         self.plot_confusion_matrix(y_test, best_model.predict(X_test_scaled))
        
#         # Feature importance for Random Forest
#         if hasattr(best_model, 'feature_importances_'):
#             self.plot_feature_importance(best_model)
        
#         # Set the best model
#         self.model = best_model
        
#         # Save model and scaler
#         self.save_model()
        
#         return best_model, best_score
    
#     def plot_confusion_matrix(self, y_true, y_pred):
#         """Plot confusion matrix"""
#         cm = confusion_matrix(y_true, y_pred, labels=self.gesture_labels)
        
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                    xticklabels=self.gesture_labels,
#                    yticklabels=self.gesture_labels)
#         plt.title('Confusion Matrix')
#         plt.ylabel('True Label')
#         plt.xlabel('Predicted Label')
#         plt.tight_layout()
#         plt.savefig('confusion_matrix.png', dpi=300)
#         plt.show()
    
    
#     def plot_feature_importance(self, model):
#         """Plot feature importance for Random Forest"""
#         if hasattr(model, 'feature_importances_'):
#             feature_names = list(self.feature_functions.keys())
#             importances = model.feature_importances_
#             indices = np.argsort(importances)[::-1]
            
#             plt.figure(figsize=(10, 6))
#             plt.title('Feature Importance')
#             plt.bar(range(len(importances)), importances[indices])
#             plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
#             plt.xlabel('Features')
#             plt.ylabel('Importance')
#             plt.tight_layout()
#             plt.savefig('feature_importance.png', dpi=300)
#             plt.show()
    
#     def save_model(self, model_path='gesture_rf_model.pkl', scaler_path='gesture_scaler.pkl'):
#         """Save trained model and scaler"""
#         if self.model is None:
#             print("No model trained yet!")
#             return
        
#         # Save model
#         joblib.dump(self.model, model_path)
#         print(f"Model saved to {model_path}")
        
#         # Save scaler
#         joblib.dump(self.scaler, scaler_path)
#         print(f"Scaler saved to {scaler_path}")
        
#         # Save metadata
#         metadata = {
#             'window_size': self.window_size,
#             'sample_rate': self.fs,
#             'gesture_labels': list(self.gesture_labels),
#             'feature_names': list(self.feature_functions.keys()),
#             'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         import json
#         with open('model_metadata.json', 'w') as f:
#             json.dump(metadata, f, indent=2)
#         print(f"Metadata saved to model_metadata.json")
    
#     def load_model(self, model_path='gesture_rf_model.pkl', scaler_path='gesture_scaler.pkl'):
#         """Load trained model and scaler"""
#         if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#             print("Model or scaler file not found!")
#             return False
        
#         self.model = joblib.load(model_path)
#         self.scaler = joblib.load(scaler_path)
        
#         # Load metadata
#         if os.path.exists('model_metadata.json'):
#             import json
#             with open('model_metadata.json', 'r') as f:
#                 metadata = json.load(f)
#             self.gesture_labels = metadata['gesture_labels']
        
#         print(f"Model loaded from {model_path}")
#         print(f"Gestures: {self.gesture_labels}")
#         return True
    
#     def test_real_time_prediction(self, sample_data):
#         """Test prediction on sample data"""
#         if self.model is None or self.scaler is None:
#             print("Model not loaded. Please train or load a model first.")
#             return None
        
#         # Ensure sample_data is the right length
#         if len(sample_data) < self.window_size:
#             print(f"Need at least {self.window_size} samples, got {len(sample_data)}")
#             return None
        
#         # Use last window_size samples
#         segment = sample_data[-self.window_size:]
        
#         # Extract features
#         features = self.extract_features_from_segment(segment)
#         features = np.array(features).reshape(1, -1)
        
#         # Scale features
#         features_scaled = self.scaler.transform(features)
        
#         # Predict
#         prediction = self.model.predict(features_scaled)[0]
#         probabilities = self.model.predict_proba(features_scaled)[0]
        
#         # Get confidence
#         confidence = np.max(probabilities)
        
#         return prediction, confidence

# def main():
#     # Configuration
#     CONFIG = {
#         'data_dir': 'emg_training_data',
#         'window_size': 100,
#         'sample_rate': 500,
#         'test_size': 0.2
#     }
    
#     print("\n" + "="*60)
#     print("EMG GESTURE CLASSIFICATION MODEL TRAINER")
#     print("="*60)
    
#     # Create trainer
#     trainer = EMGModelTrainer(
#         data_dir=CONFIG['data_dir'],
#         window_size=CONFIG['window_size'],
#         fs=CONFIG['sample_rate']
#     )
    
#     # Ask user what to do
#     print("\nOptions:")
#     print("  1: Train new model")
#     print("  2: Test existing model")
#     print("  3: Load and view model info")
    
#     choice = input("\nEnter choice (1-3): ").strip()
    
#     if choice == '1':
#         # Train new model
#         X, y = trainer.load_and_preprocess_data()
#         trainer.train_model(X, y, test_size=CONFIG['test_size'])
    
#     elif choice == '2':
#         # Test existing model
#         if trainer.load_model():
#             # Generate test data for each gesture
#             gestures = trainer.gesture_labels if trainer.gesture_labels else ['round', 'shoot', 'up_down', 'rest']
            
#             for gesture in gestures:
#                 print(f"\nTesting {gesture} gesture...")
                
#                 # Generate simulated data for this gesture
#                 if gesture == 'round':
#                     test_data = 100 + 50 * np.sin(2 * np.pi * 2 * np.linspace(0, 1, 200)) + np.random.normal(0, 10, 200)
#                 elif gesture == 'shoot':
#                     test_data = np.concatenate([
#                         50 * np.ones(100),
#                         200 * np.ones(50),
#                         50 * np.ones(50)
#                     ]) + np.random.normal(0, 15, 200)
#                 elif gesture == 'up_down':
#                     test_data = 80 + 40 * np.sin(2 * np.pi * 1 * np.linspace(0, 2, 200)) + np.random.normal(0, 15, 200)
#                 else:  # rest
#                     test_data = 20 + np.random.normal(0, 5, 200)
                
#                 # Test prediction
#                 result = trainer.test_real_time_prediction(test_data)
#                 if result:
#                     prediction, confidence = result
#                     print(f"  Predicted: {prediction} (True: {gesture})")
#                     print(f"  Confidence: {confidence:.2%}")
    
#     elif choice == '3':
#         # Load and view model info
#         if trainer.load_model():
#             print("\nModel Information:")
#             print(f"  Model type: {type(trainer.model).__name__}")
#             print(f"  Gestures: {trainer.gesture_labels}")
#             print(f"  Window size: {trainer.window_size}")
#             print(f"  Sample rate: {trainer.fs}")
    
#     else:
#         print("Invalid choice!")

# if __name__ == "__main__":
#     main()








# # -------------------------------------------------------------------------------




# # # emg_model_trainer.py
# # import os
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split, cross_val_score
# # from sklearn.preprocessing import StandardScaler, LabelEncoder # <-- ADDED LabelEncoder
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.svm import SVC
# # from sklearn.neural_network import MLPClassifier
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # import seaborn as sns
# # import joblib
# # import warnings
# # import json
# # from scipy.signal import welch

# # warnings.filterwarnings('ignore')

# # # Feature extraction functions (same as your original)
# # class EMGModelTrainer:
# #     """
# #     Train machine learning model for EMG gesture classification
# #     """
    
# #     def __init__(self, data_dir="emg_training_data", window_size=100, fs=500):
# #         self.data_dir = data_dir
# #         self.window_size = window_size
# #         self.fs = fs
# #         self.scaler = StandardScaler()
# #         self.label_encoder = LabelEncoder() # <-- INITIALIZE ENCODER
# #         self.model = None
# #         self.feature_names = None
# #         self.gesture_labels = None
        
# #         # Feature extraction functions
# #         self.feature_functions = {
# #             'MAV': self.mean_absolute_value,
# #             'ZC': self.zero_crossings,
# #             'WL': self.waveform_length,
# #             'RMS': self.root_mean_square,
# #             'VAR': self.variance,
# #             'MNF': self.mean_power_frequency,
# #             'MDF': self.median_frequency,
# #             'SpecCent': lambda x: self.fourier_features(x)[0],
# #             'SpecSpread': lambda x: self.fourier_features(x)[1],
# #             'SpecEnt': lambda x: self.fourier_features(x)[2]
# #         }
    
# #     # Feature extraction methods (same as your original)
# #     def mean_absolute_value(self, segment):
# #         return np.mean(np.abs(segment))
    
# #     def zero_crossings(self, segment):
# #         mean_val = np.mean(segment)
# #         return np.sum(np.diff(np.array(segment) > mean_val) != 0)
    
# #     def waveform_length(self, segment):
# #         return np.sum(np.abs(np.diff(segment)))
    
# #     def root_mean_square(self, segment):
# #         return np.sqrt(np.mean(segment**2))
    
# #     def variance(self, segment):
# #         return np.var(segment)
    
# #     def mean_power_frequency(self, segment, fs=None):
# #         fs = fs or self.fs
# #         freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
# #         total_power = np.sum(psd)
# #         if total_power == 0:
# #             return 0.0
# #         return np.sum(freqs * psd) / total_power
    
# #     def median_frequency(self, segment, fs=None):
# #         fs = fs or self.fs
# #         freqs, psd = welch(segment, fs=fs, nperseg=len(segment), window='hann', scaling='density')
# #         power_sum = np.sum(psd)
# #         if power_sum == 0:
# #             return 0.0
# #         cumulative_power = np.cumsum(psd)
# #         median_idx = np.where(cumulative_power >= power_sum / 2)[0][0]
# #         return freqs[median_idx]
    
# #     def fourier_features(self, segment, fs=None):
# #         fs = fs or self.fs
# #         segment = np.asarray(segment)
# #         N = len(segment)
# #         fft_vals = np.fft.rfft(segment)
# #         mag = np.abs(fft_vals)
# #         power = mag**2
# #         freqs = np.fft.rfftfreq(N, d=1.0 / fs)
# #         total_power = np.sum(power) + 1e-12
# #         norm_power = power / total_power
# #         spectral_centroid = np.sum(freqs * norm_power)
# #         spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_power))
# #         spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))
# #         return spectral_centroid, spectral_spread, spectral_entropy
    
# #     def extract_features_from_segment(self, segment):
# #         """Extract all features from a single segment"""
# #         features = []
# #         # Get feature names in the order they are extracted
# #         feature_names = list(self.feature_functions.keys())
        
# #         for name, func in self.feature_functions.items():
# #             if name in ['SpecCent', 'SpecSpread', 'SpecEnt']:
# #                 # These are handled together in fourier_features
# #                 continue
# #             elif name in ['MNF', 'MDF']:
# #                 features.append(func(segment, self.fs))
# #             else:
# #                 features.append(func(segment))
        
# #         # Add Fourier features
# #         spec_cent, spec_spread, spec_ent = self.fourier_features(segment, self.fs)
# #         features.extend([spec_cent, spec_spread, spec_ent])
        
# #         # Set feature names only once (assuming consistent feature extraction)
# #         if self.feature_names is None:
# #             self.feature_names = feature_names
            
# #         return features
    
# #     def load_and_preprocess_data(self):
# #         """Load all CSV files and preprocess into features"""
# #         print("Loading training data...")
        
# #         all_features = []
# #         all_labels = []
        
# #         # Get all CSV files
# #         csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
# #         if not csv_files:
# #             raise ValueError(f"No CSV files found in {self.data_dir}")
        
# #         print(f"Found {len(csv_files)} data files")
        
# #         for csv_file in csv_files:
# #             filepath = os.path.join(self.data_dir, csv_file)
# #             df = pd.read_csv(filepath)
            
# #             # Get gesture label (should be same for entire file)
# #             if 'gesture' in df.columns:
# #                 gesture_label = str(df['gesture'].iloc[0])
# #             else:
# #                 # Infer from filename
# #                 gesture_label = csv_file.split('_')[0]
            
# #             # Get EMG values
# #             if 'emg_value' in df.columns:
# #                 emg_data = df['emg_value'].values
# #             elif 'value' in df.columns:
# #                 emg_data = df['value'].values
# #             else:
# #                 # Try to find any numeric column
# #                 numeric_cols = df.select_dtypes(include=[np.number]).columns
# #                 if len(numeric_cols) > 0:
# #                     emg_data = df[numeric_cols[0]].values
# #                 else:
# #                     print(f"Skipping {csv_file}: No EMG data found")
# #                     continue
            
# #             # Apply sliding window
# #             for i in range(0, len(emg_data) - self.window_size + 1, self.window_size // 2):
# #                 segment = emg_data[i:i + self.window_size]
                
# #                 # Extract features
# #                 features = self.extract_features_from_segment(segment)
# #                 all_features.append(features)
# #                 all_labels.append(gesture_label)
            
# #             print(f"  Processed {csv_file}: {len(emg_data)} samples -> {len(all_features) - len(all_labels) + (len(df) // self.window_size * 2)} windows")
        
# #         # Convert to numpy arrays
# #         X = np.array(all_features)
# #         y = np.array(all_labels)
        
# #         # Get unique labels (for display before encoding)
# #         self.gesture_labels = np.unique(y)
# #         print(f"\nLoaded {len(X)} samples with {len(self.gesture_labels)} gestures: {list(self.gesture_labels)}")
        
# #         return X, y
    
# #     def train_model(self, X, y, test_size=0.2, random_state=42):
# #         """Train and evaluate the model"""
# #         print("\n" + "="*60)
# #         print("TRAINING GESTURE CLASSIFICATION MODEL")
# #         print("="*60)
        
# #         # --- START: LABEL ENCODING SECTION ---
# #         # Fit and transform string labels (y) into numerical integers (y_encoded)
# #         print("\nEncoding string labels to integers for training...")
# #         y_encoded = self.label_encoder.fit_transform(y)
# #         self.gesture_labels = self.label_encoder.classes_
# #         print(f"Gesture labels (0 to {len(self.gesture_labels) - 1}): {list(self.gesture_labels)}")
# #         # --- END: LABEL ENCODING SECTION ---
        
# #         # Split data using the ENCODED labels
# #         X_train, X_test, y_train, y_test = train_test_split(
# #             X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
# #         )
        
# #         print(f"Training set: {X_train.shape[0]} samples")
# #         print(f"Test set: {X_test.shape[0]} samples")
        
# #         # Scale features
# #         print("\nScaling features...")
# #         X_train_scaled = self.scaler.fit_transform(X_train)
# #         X_test_scaled = self.scaler.transform(X_test)
        
# #         # Define models to try
# #         models = {
# #             'Random Forest': RandomForestClassifier(
# #                 n_estimators=100,
# #                 max_depth=10,
# #                 random_state=random_state,
# #                 class_weight='balanced'
# #             ),
# #             'SVM': SVC(
# #                 kernel='rbf',
# #                 C=1.0,
# #                 gamma='scale',
# #                 random_state=random_state,
# #                 class_weight='balanced',
# #                 probability=True # Required for predict_proba later
# #             ),
# #             'Neural Network': MLPClassifier(
# #                 hidden_layer_sizes=(64, 32),
# #                 activation='relu',
# #                 solver='adam',
# #                 max_iter=500,
# #                 random_state=random_state,
# #                 early_stopping=True,
# #                 verbose=False
# #             )
# #         }
        
# #         best_model = None
# #         best_score = 0
# #         best_model_name = ""
        
# #         # Train and evaluate each model
# #         for name, model in models.items():
# #             print(f"\nTraining {name}...")
            
# #             # Cross-validation
# #             # y_train is now numerical, fixing the MLPClassifier ValueError
# #             cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5) 
# #             print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
# #             # Train on full training set
# #             model.fit(X_train_scaled, y_train)
            
# #             # Test accuracy
# #             y_pred = model.predict(X_test_scaled)
# #             accuracy = accuracy_score(y_test, y_pred)
# #             print(f"  Test accuracy: {accuracy:.3f}")
            
# #             # Update best model
# #             if accuracy > best_score:
# #                 best_score = accuracy
# #                 best_model = model
# #                 best_model_name = name
        
# #         print(f"\n{'='*60}")
# #         print(f"BEST MODEL: {best_model_name} with accuracy: {best_score:.3f}")
# #         print("="*60)
        
# #         # Detailed evaluation of best model
# #         y_test_pred = best_model.predict(X_test_scaled)
        
# #         print("\nDetailed Classification Report:")
# #         # Use inverse_transform to get original gesture names for the report
# #         report_y_true = self.label_encoder.inverse_transform(y_test)
# #         report_y_pred = self.label_encoder.inverse_transform(y_test_pred)
# #         print(classification_report(report_y_true, report_y_pred, target_names=self.gesture_labels))
        
# #         # Confusion matrix
# #         self.plot_confusion_matrix(y_test, y_test_pred)
        
# #         # Feature importance for Random Forest
# #         if hasattr(best_model, 'feature_importances_'):
# #             self.plot_feature_importance(best_model)
        
# #         # Set the best model
# #         self.model = best_model
        
# #         # Save model and scaler
# #         self.save_model()
        
# #         return best_model, best_score
    
# #     def plot_confusion_matrix(self, y_true_encoded, y_pred_encoded):
# #         """Plot confusion matrix"""
# #         # Confusion matrix is generated using the ENCODED integers
# #         cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=self.label_encoder.transform(self.gesture_labels))
        
# #         plt.figure(figsize=(8, 6))
# #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #                     # Use original gesture labels for display
# #                     xticklabels=self.gesture_labels,
# #                     yticklabels=self.gesture_labels)
# #         plt.title('Confusion Matrix')
# #         plt.ylabel('True Label')
# #         plt.xlabel('Predicted Label')
# #         plt.tight_layout()
# #         plt.savefig('confusion_matrix.png', dpi=300)
# #         plt.show()
    
# #     def plot_feature_importance(self, model):
# #         """Plot feature importance for Random Forest"""
# #         if hasattr(model, 'feature_importances_'):
# #             # self.feature_names is set during feature extraction
# #             feature_names = list(self.feature_functions.keys())
# #             importances = model.feature_importances_
# #             indices = np.argsort(importances)[::-1]
            
# #             plt.figure(figsize=(10, 6))
# #             plt.title('Feature Importance')
# #             plt.bar(range(len(importances)), importances[indices])
# #             plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
# #             plt.xlabel('Features')
# #             plt.ylabel('Importance')
# #             plt.tight_layout()
# #             plt.savefig('feature_importance.png', dpi=300)
# #             plt.show()
    
# #     def save_model(self, model_path='gesture_rf_model.pkl', scaler_path='gesture_scaler.pkl', encoder_path='gesture_encoder.pkl'):
# #         """Save trained model, scaler, and encoder"""
# #         if self.model is None:
# #             print("No model trained yet!")
# #             return
        
# #         # Save model
# #         joblib.dump(self.model, model_path)
# #         print(f"Model saved to {model_path}")
        
# #         # Save scaler
# #         joblib.dump(self.scaler, scaler_path)
# #         print(f"Scaler saved to {scaler_path}")
        
# #         # Save encoder
# #         joblib.dump(self.label_encoder, encoder_path) # <-- SAVING THE ENCODER
# #         print(f"Label Encoder saved to {encoder_path}")
        
# #         # Save metadata
# #         metadata = {
# #             'window_size': self.window_size,
# #             'sample_rate': self.fs,
# #             'gesture_labels': list(self.gesture_labels),
# #             'feature_names': list(self.feature_functions.keys()),
# #             'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
# #         }
        
# #         with open('model_metadata.json', 'w') as f:
# #             json.dump(metadata, f, indent=2)
# #         print(f"Metadata saved to model_metadata.json")
    
# #     def load_model(self, model_path='gesture_rf_model.pkl', scaler_path='gesture_scaler.pkl', encoder_path='gesture_encoder.pkl'):
# #         """Load trained model, scaler, and encoder"""
# #         if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
# #             print("Model, scaler, or encoder file not found!")
# #             return False
        
# #         self.model = joblib.load(model_path)
# #         self.scaler = joblib.load(scaler_path)
# #         self.label_encoder = joblib.load(encoder_path) # <-- LOADING THE ENCODER
        
# #         # Load metadata
# #         if os.path.exists('model_metadata.json'):
# #             with open('model_metadata.json', 'r') as f:
# #                 metadata = json.load(f)
# #             self.gesture_labels = metadata['gesture_labels']
        
# #         print(f"Model loaded from {model_path}")
# #         print(f"Gestures: {self.gesture_labels}")
# #         return True
    
# #     def test_real_time_prediction(self, sample_data):
# #         """Test prediction on sample data"""
# #         if self.model is None or self.scaler is None or self.label_encoder is None:
# #             print("Model or components (scaler/encoder) not loaded. Please train or load a model first.")
# #             return None
        
# #         # Ensure sample_data is the right length
# #         if len(sample_data) < self.window_size:
# #             print(f"Need at least {self.window_size} samples, got {len(sample_data)}")
# #             return None
        
# #         # Use last window_size samples
# #         segment = sample_data[-self.window_size:]
        
# #         # Extract features
# #         features = self.extract_features_from_segment(segment)
# #         features = np.array(features).reshape(1, -1)
        
# #         # Scale features
# #         features_scaled = self.scaler.transform(features)
        
# #         # Predict the ENCODED integer
# #         prediction_encoded = self.model.predict(features_scaled)[0]
        
# #         # Get the original string label
# #         prediction = self.label_encoder.inverse_transform([prediction_encoded])[0] # <-- INVERSE TRANSFORM
        
# #         # Get confidence (requires model to support predict_proba, like RF or SVC with probability=True)
# #         try:
# #             probabilities = self.model.predict_proba(features_scaled)[0]
# #             confidence = np.max(probabilities)
# #         except AttributeError:
# #             confidence = 0.0 # Default if model doesn't support probabilities
        
# #         return prediction, confidence

# # def main():
# #     # Configuration
# #     CONFIG = {
# #         'data_dir': 'emg_training_data',
# #         'window_size': 100,
# #         'sample_rate': 500,
# #         'test_size': 0.2
# #     }
    
# #     print("\n" + "="*60)
# #     print("EMG GESTURE CLASSIFICATION MODEL TRAINER")
# #     print("="*60)
    
# #     # Create trainer
# #     trainer = EMGModelTrainer(
# #         data_dir=CONFIG['data_dir'],
# #         window_size=CONFIG['window_size'],
# #         fs=CONFIG['sample_rate']
# #     )
    
# #     # Ask user what to do
# #     print("\nOptions:")
# #     print("  1: Train new model")
# #     print("  2: Test existing model")
# #     print("  3: Load and view model info")
    
# #     choice = input("\nEnter choice (1-3): ").strip()
    
# #     if choice == '1':
# #         # Train new model
# #         try:
# #             X, y = trainer.load_and_preprocess_data()
# #             trainer.train_model(X, y, test_size=CONFIG['test_size'])
# #         except ValueError as e:
# #             print(f"Error: {e}")
    
# #     elif choice == '2':
# #         # Test existing model
# #         if trainer.load_model():
# #             # Generate test data for each gesture
# #             gestures = trainer.gesture_labels if trainer.gesture_labels else ['round', 'shoot', 'up_down', 'rest']
            
# #             for gesture in gestures:
# #                 print(f"\nTesting {gesture} gesture...")
                
# #                 # Generate simulated data for this gesture
# #                 if gesture == 'round':
# #                     # High power, medium frequency (simulating circular motion)
# #                     test_data = 100 + 50 * np.sin(2 * np.pi * 2 * np.linspace(0, 1, 200)) + np.random.normal(0, 10, 200)
# #                 elif gesture == 'shoot':
# #                     # Sudden spike (simulating a quick extension/contraction)
# #                     test_data = np.concatenate([
# #                         50 * np.ones(100),
# #                         200 * np.ones(50),
# #                         50 * np.ones(50)
# #                     ]) + np.random.normal(0, 15, 200)
# #                 elif gesture == 'up_down':
# #                     # Medium power, lower frequency oscillation
# #                     test_data = 80 + 40 * np.sin(2 * np.pi * 1 * np.linspace(0, 2, 200)) + np.random.normal(0, 15, 200)
# #                 else: # rest
# #                     # Low power, flat signal
# #                     test_data = 20 + np.random.normal(0, 5, 200)
                
# #                 # Test prediction
# #                 result = trainer.test_real_time_prediction(test_data)
# #                 if result:
# #                     prediction, confidence = result
# #                     print(f"  Predicted: {prediction} (Simulated True: {gesture})")
# #                     print(f"  Confidence: {confidence:.2%}")
    
# #     elif choice == '3':
# #         # Load and view model info
# #         if trainer.load_model():
# #             print("\nModel Information:")
# #             print(f"  Model type: {type(trainer.model).__name__}")
# #             print(f"  Gestures: {trainer.gesture_labels}")
# #             print(f"  Window size: {trainer.window_size}")
# #             print(f"  Sample rate: {trainer.fs}")
    
# #     else:
# #         print("Invalid choice!")

# # if __name__ == "__main__":
# #     main()







# ---------------------------------------------------------------------------








# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
# import seaborn as sns
# import joblib
# import warnings
# from scipy.signal import welch
# from scipy import stats
# import json
# from datetime import datetime

# warnings.filterwarnings('ignore')

# class EMGModelTrainer:
#     """
#     Machine learning model trainer for EMG gesture classification
#     Simplified version - no complex filtering that causes errors
#     """
    
#     def __init__(self, data_dir="custom_emg_gestures", window_size=150, overlap=0.5, fs=500):
#         self.data_dir = data_dir
#         self.window_size = window_size
#         self.overlap = overlap
#         self.fs = fs
#         self.step_size = int(window_size * (1 - overlap))
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.model = None
#         self.feature_names = None
#         self.gesture_labels = None
        
#         # Feature extraction functions (simplified - no filtering)
#         self.feature_functions = {
#             'MAV': self.mean_absolute_value,
#             'ZC': self.zero_crossings,
#             'WL': self.waveform_length,
#             'RMS': self.root_mean_square,
#             'VAR': self.variance,
#             'IEMG': self.integrated_emg,
#             'Skewness': self.skewness,
#             'Kurtosis': self.kurtosis,
#             'Percentile_25': lambda x: np.percentile(x, 25),
#             'Percentile_75': lambda x: np.percentile(x, 75),
#             'Min': lambda x: np.min(x),
#             'Max': lambda x: np.max(x),
#             'Range': lambda x: np.max(x) - np.min(x),
#             'Mean': lambda x: np.mean(x),
#             'Median': lambda x: np.median(x),
#             'Std': lambda x: np.std(x)
#         }
    
#     # Feature extraction methods
#     def mean_absolute_value(self, segment):
#         return np.mean(np.abs(segment))
    
#     def zero_crossings(self, segment, threshold=10):
#         segment = np.array(segment)
#         mean_val = np.mean(segment)
#         diff = segment - mean_val
#         return np.sum((diff[1:] * diff[:-1] < 0) & (np.abs(diff[1:] - diff[:-1]) > threshold))
    
#     def waveform_length(self, segment):
#         return np.sum(np.abs(np.diff(segment)))
    
#     def root_mean_square(self, segment):
#         return np.sqrt(np.mean(segment ** 2))
    
#     def variance(self, segment):
#         return np.var(segment)
    
#     def integrated_emg(self, segment):
#         return np.sum(np.abs(segment))
    
#     def skewness(self, segment):
#         return stats.skew(segment)
    
#     def kurtosis(self, segment):
#         return stats.kurtosis(segment)
    
#     def fourier_features(self, segment):
#         """Simplified Fourier features without complex filtering"""
#         N = len(segment)
        
#         if N < 10:  # Not enough samples for FFT
#             return 0.0, 0.0, 0.0
        
#         try:
#             fft_vals = np.fft.rfft(segment)
#             mag = np.abs(fft_vals)
#             power = mag ** 2
#             freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
            
#             if len(power) == 0 or np.sum(power) == 0:
#                 return 0.0, 0.0, 0.0
            
#             total_power = np.sum(power) + 1e-12
#             norm_power = power / total_power
            
#             spectral_centroid = np.sum(freqs * norm_power)
#             spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_power))
#             spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))
            
#             return spectral_centroid, spectral_spread, spectral_entropy
#         except:
#             return 0.0, 0.0, 0.0
    
#     def extract_features_from_segment(self, segment):
#         """Extract all features from a single segment"""
#         features = []
#         feature_names = []
        
#         # Basic statistical features
#         for name, func in self.feature_functions.items():
#             try:
#                 features.append(func(segment))
#                 feature_names.append(name)
#             except:
#                 features.append(0.0)
#                 feature_names.append(name)
        
#         # Add Fourier features
#         try:
#             spec_cent, spec_spread, spec_ent = self.fourier_features(segment)
#             features.extend([spec_cent, spec_spread, spec_ent])
#             feature_names.extend(['SpecCent', 'SpecSpread', 'SpecEnt'])
#         except:
#             features.extend([0.0, 0.0, 0.0])
#             feature_names.extend(['SpecCent', 'SpecSpread', 'SpecEnt'])
        
#         # Set feature names (only once)
#         if self.feature_names is None:
#             self.feature_names = feature_names
        
#         return features
    
#     def load_and_preprocess_data(self, limit_files=None):
#         """Load all CSV files and preprocess into features"""
#         print("Loading and preprocessing training data...")
        
#         all_features = []
#         all_labels = []
#         file_count = 0
        
        
#         # Get all CSV files
#         csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
#         if not csv_files:
#             raise ValueError(f"No CSV files found in {self.data_dir}")
        
        
#         if limit_files:
#             csv_files = csv_files[:limit_files]
#             print(f"Limiting to {limit_files} files")
        
#         print(f"Found {len(csv_files)} data files")
        
        
#         for csv_file in csv_files:
#             filepath = os.path.join(self.data_dir, csv_file)
#             try:
#                 df = pd.read_csv(filepath)
                
#                 # Get gesture label from filename or column
#                 gesture_label = None
                
#                 # First try to get from 'gesture' column
#                 if 'gesture' in df.columns:
#                     gesture_label = str(df['gesture'].iloc[0])
#                 # Otherwise extract from filename
#                 else:
#                     # Filename format: gesture_session_repeats_samples_timestamp.csv
#                     filename_parts = csv_file.split('_')
#                     if filename_parts:
#                         gesture_label = filename_parts[0]
                
#                 if not gesture_label:
#                     print(f"  Warning: Could not determine gesture for {csv_file}")
#                     continue
                
#                 # Get EMG values
#                 emg_data = None
#                 if 'emg_value' in df.columns:
#                     emg_data = df['emg_value'].values
#                 elif 'value' in df.columns:
#                     emg_data = df['value'].values
#                 else:
#                     # Try to find any numeric column
#                     numeric_cols = df.select_dtypes(include=[np.number]).columns
#                     if len(numeric_cols) > 0:
#                         emg_data = df[numeric_cols[0]].values
#                     else:
#                         print(f"  Warning: No EMG data found in {csv_file}")
#                         continue
                
#                 # Simple preprocessing - just remove DC offset
#                 emg_data = emg_data - np.mean(emg_data)
                
#                 # Apply sliding window with overlap
#                 windows = []
#                 labels = []
                
#                 # Make sure we have enough data
#                 if len(emg_data) >= self.window_size:
#                     for i in range(0, len(emg_data) - self.window_size + 1, self.step_size):
#                         segment = emg_data[i:i + self.window_size]
#                         windows.append(segment)
#                         labels.append(gesture_label)
                
#                 # Extract features from each window
#                 for segment in windows:
#                     features = self.extract_features_from_segment(segment)
#                     all_features.append(features)
#                     all_labels.append(gesture_label)
                
#                 file_count += 1
#                 print(f"  Processed {csv_file[:30]:30} -> {len(windows):4} windows, gesture: {gesture_label}")
                
#             except Exception as e:
#                 print(f"  Error processing {csv_file}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         if not all_features:
#             raise ValueError("No features extracted from any files!")
        
        
#         # Convert to numpy arrays
#         X = np.array(all_features)
#         y = np.array(all_labels)
        
        
#         # Get unique labels
#         unique_labels = np.unique(y)
#         print(f"\n✓ Successfully loaded {len(X)} samples")
#         print(f"  Gestures: {list(unique_labels)}")
#         print(f"  Samples per gesture:")
#         for label in unique_labels:
#             count = np.sum(y == label)
#             print(f"    {label:10}: {count:6,} samples")
        
#         return X, y
    
#     def train_model(self, X, y, test_size=0.2, random_state=42):
#         """Train and evaluate multiple models"""
#         print("\n" + "="*70)
#         print("EMG GESTURE CLASSIFICATION MODEL TRAINING")
#         print("="*70)
        
#         # Encode labels
#         print("\nEncoding labels...")
#         y_encoded = self.label_encoder.fit_transform(y)
#         self.gesture_labels = self.label_encoder.classes_
#         print(f"Labels encoded: {list(self.gesture_labels)}")
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
#         )
        
#         print(f"\nDataset split:")
#         print(f"  Training set: {X_train.shape[0]} samples")
#         print(f"  Test set: {X_test.shape[0]} samples")
#         print(f"  Features per sample: {X_train.shape[1]}")
        
#         # Scale features
#         print("\nScaling features...")
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Define models to try (simpler configurations)
#         models = {
#             'Random Forest': RandomForestClassifier(
#                 n_estimators=100,
#                 max_depth=10,
#                 min_samples_split=2,
#                 min_samples_leaf=1,
#                 class_weight='balanced',
#                 random_state=random_state,
#                 n_jobs=-1
#             ),
#             'Gradient Boosting': GradientBoostingClassifier(
#                 n_estimators=50,
#                 learning_rate=0.1,
#                 max_depth=3,
#                 random_state=random_state
#             ),
#             'SVM': SVC(
#                 kernel='rbf',
#                 C=1.0,
#                 gamma='scale',
#                 class_weight='balanced',
#                 random_state=random_state,
#                 probability=True
#             ),
#             'K-NN': KNeighborsClassifier(
#                 n_neighbors=3,
#                 weights='uniform',
#                 metric='euclidean'
#             )
#         }
        
#         best_model = None
#         best_score = 0
#         best_model_name = ""
#         results = {}
        
#         # Train and evaluate each model
#         for name, model in models.items():
#             print(f"\n{'─' * 40}")
#             print(f"Training {name}...")
            
#             try:
#                 # Cross-validation
#                 cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
#                 cv_mean = cv_scores.mean()
#                 cv_std = cv_scores.std()
#                 print(f"  Cross-validation accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
                
#                 # Train on full training set
#                 model.fit(X_train_scaled, y_train)
                
#                 # Test accuracy
#                 y_pred = model.predict(X_test_scaled)
#                 accuracy = accuracy_score(y_test, y_pred)
#                 print(f"  Test accuracy: {accuracy:.4f}")
                
#                 # Additional metrics
#                 precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
#                 print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
                
#                 # Store results
#                 results[name] = {
#                     'cv_mean': cv_mean,
#                     'cv_std': cv_std,
#                     'test_accuracy': accuracy,
#                     'precision': precision,
#                     'recall': recall,
#                     'f1': f1,
#                     'model': model
#                 }
                
#                 # Update best model
#                 if accuracy > best_score:
#                     best_score = accuracy
#                     best_model = model
#                     best_model_name = name
                    
#             except Exception as e:
#                 print(f"  Error training {name}: {e}")
#                 results[name] = None
        
#         # Display results summary
#         print(f"\n{'='*70}")
#         print("TRAINING RESULTS SUMMARY")
#         print("="*70)
#         print(f"{'Model':20} {'CV Acc':10} {'Test Acc':10} {'F1-Score':10}")
#         print("-" * 70)
        
#         for name, result in results.items():
#             if result:
#                 print(f"{name:20} {result['cv_mean']:9.4f} ± {result['cv_std']:.4f} "
#                       f"{result['test_accuracy']:9.4f} {result['f1']:9.4f}")
        
#         print(f"\n{'='*70}")
#         print(f"🎯 BEST MODEL: {best_model_name}")
#         print(f"🏆 Test Accuracy: {best_score:.4f}")
#         print("="*70)
        
#         if best_model is not None:
#             # Detailed evaluation of best model
#             y_pred = best_model.predict(X_test_scaled)
#             y_true_decoded = self.label_encoder.inverse_transform(y_test)
#             y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
            
#             print("\nDetailed Classification Report:")
#             print(classification_report(y_true_decoded, y_pred_decoded, target_names=self.gesture_labels))
            
#             # Confusion matrix
#             self.plot_confusion_matrix(y_test, y_pred)
            
#             # Feature importance for tree-based models
#             if hasattr(best_model, 'feature_importances_'):
#                 self.plot_feature_importance(best_model)
            
#             # Set the best model
#             self.model = best_model
            
#             # Save model and components
#             model_dir = self.save_model()
            
#             print(f"\n✅ Model training completed successfully!")
#             print(f"📁 Model saved in: {model_dir}")
            
#             return best_model, best_score
#         else:
#             print("\n❌ No model could be trained successfully!")
#             return None, 0
    
#     def plot_confusion_matrix(self, y_true, y_pred):
#         """Plot confusion matrix"""
#         try:
#             cm = confusion_matrix(y_true, y_pred)
            
#             plt.figure(figsize=(10, 8))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                        xticklabels=self.gesture_labels,
#                        yticklabels=self.gesture_labels)
#             plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
#             plt.ylabel('True Label', fontsize=12)
#             plt.xlabel('Predicted Label', fontsize=12)
#             plt.tight_layout()
#             plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
#             print("✓ Confusion matrix saved as 'confusion_matrix.png'")
#             plt.show()
#         except Exception as e:
#             print(f"  Could not create confusion matrix: {e}")
    
#     def plot_feature_importance(self, model):
#         """Plot feature importance for tree-based models"""
#         try:
#             if hasattr(model, 'feature_importances_'):
#                 importances = model.feature_importances_
#                 indices = np.argsort(importances)[::-1]
                
#                 # Get top 15 features
#                 top_n = min(15, len(self.feature_names))
                
#                 plt.figure(figsize=(10, 6))
#                 plt.title('Top Feature Importances', fontsize=14, fontweight='bold')
#                 bars = plt.bar(range(top_n), importances[indices[:top_n]])
#                 plt.xticks(range(top_n), [self.feature_names[i] for i in indices[:top_n]], 
#                           rotation=45, ha='right', fontsize=10)
#                 plt.xlabel('Features', fontsize=12)
#                 plt.ylabel('Importance', fontsize=12)
                
#                 plt.tight_layout()
#                 plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
#                 print("✓ Feature importance plot saved as 'feature_importance.png'")
#                 plt.show()
#         except Exception as e:
#             print(f"  Could not create feature importance plot: {e}")
    
#     def save_model(self, model_name='emg_gesture_model'):
#         """Save trained model and all components"""
#         if self.model is None:
#             print("No model trained yet!")
#             return
        
#         # Create model directory
#         model_dir = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         os.makedirs(model_dir, exist_ok=True)
        
#         # Save components
#         components = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_names': self.feature_names
#         }
        
#         for name, component in components.items():
#             if component is not None:
#                 path = os.path.join(model_dir, f'{name}.pkl')
#                 joblib.dump(component, path)
#                 print(f"  ✓ {name} saved to {path}")
        
#         # Save metadata
#         metadata = {
#             'model_type': type(self.model).__name__,
#             'gesture_labels': list(self.gesture_labels),
#             'feature_names': list(self.feature_names) if self.feature_names else [],
#             'window_size': self.window_size,
#             'overlap': self.overlap,
#             'sample_rate': self.fs,
#             'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         metadata_path = os.path.join(model_dir, 'metadata.json')
#         with open(metadata_path, 'w') as f:
#             json.dump(metadata, f, indent=2)
#         print(f"  ✓ Metadata saved to {metadata_path}")
        
#         return model_dir
    
#     def load_model(self, model_dir=None):
#         """Load trained model from directory"""
#         if model_dir is None:
#             # Find the most recent model directory
#             model_dirs = [d for d in os.listdir('.') if d.startswith('emg_gesture_model_') and os.path.isdir(d)]
#             if not model_dirs:
#                 print("No trained models found!")
#                 return False
#             model_dir = sorted(model_dirs, reverse=True)[0]  # Get most recent
        
#         try:
#             self.model = joblib.load(os.path.join(model_dir, 'model.pkl'))
#             self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
#             self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
#             self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
            
#             with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
#                 metadata = json.load(f)
#             self.gesture_labels = metadata['gesture_labels']
            
#             print(f"✓ Model loaded from {model_dir}")
#             print(f"  Model type: {type(self.model).__name__}")
#             print(f"  Gestures: {self.gesture_labels}")
#             print(f"  Features: {len(self.feature_names)}")
            
#             return True
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return False
    
#     def real_time_predict(self, emg_segment):
#         """Predict gesture from EMG segment in real-time"""
#         if self.model is None or self.scaler is None or self.label_encoder is None:
#             print("Model not loaded!")
#             return None, 0.0
        
#         if len(emg_segment) < self.window_size:
#             # Pad with zeros if needed
#             padded_segment = np.zeros(self.window_size)
#             padded_segment[-len(emg_segment):] = emg_segment
#             emg_segment = padded_segment
        
#         try:
#             # Simple preprocessing
#             segment = emg_segment[-self.window_size:]
#             segment = segment - np.mean(segment)  # Remove DC offset
            
#             # Extract features
#             features = self.extract_features_from_segment(segment)
#             features = np.array(features).reshape(1, -1)
            
#             # Scale features
#             features_scaled = self.scaler.transform(features)
            
#             # Predict
#             pred_encoded = self.model.predict(features_scaled)[0]
#             gesture = self.label_encoder.inverse_transform([pred_encoded])[0]
            
#             # Get confidence
#             if hasattr(self.model, 'predict_proba'):
#                 probs = self.model.predict_proba(features_scaled)[0]
#                 confidence = float(np.max(probs))
#             else:
#                 confidence = 0.8
            
#             return gesture, confidence
#         except Exception as e:
#             # print(f"Prediction error: {e}")
#             return None, 0.0
    
#     def test_on_sample_files(self):
#         """Test model on actual CSV files"""
#         print("\n" + "="*60)
#         print("TESTING ON SAMPLE FILES")
#         print("="*60)
        
#         if not self.model:
#             print("Please train or load a model first!")
#             return
        
#         # Get CSV files
#         csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
#         if not csv_files:
#             print("No CSV files found in data directory!")
#             return
        
#         # Test on 3 random files (one per gesture)
#         import random
#         test_files = random.sample(csv_files, min(3, len(csv_files)))
        
#         results = []
        
#         for test_file in test_files:
#             try:
#                 filepath = os.path.join(self.data_dir, test_file)
#                 df = pd.read_csv(filepath)
                
#                 # Get true gesture
#                 if 'gesture' in df.columns:
#                     true_gesture = str(df['gesture'].iloc[0])
#                 else:
#                     # Extract from filename
#                     true_gesture = test_file.split('_')[0]
                
#                 # Get EMG data
#                 if 'emg_value' in df.columns:
#                     emg_data = df['emg_value'].values
#                 else:
#                     numeric_cols = df.select_dtypes(include=[np.number]).columns
#                     emg_data = df[numeric_cols[0]].values
                
#                 # Make prediction
#                 gesture, confidence = self.real_time_predict(emg_data)
                
#                 if gesture:
#                     correct = (gesture == true_gesture)
#                     symbol = "✓" if correct else "✗"
#                     color = '\033[92m' if correct else '\033[91m'  # Green/Red
                    
#                     results.append({
#                         'file': test_file,
#                         'true': true_gesture,
#                         'predicted': gesture,
#                         'correct': correct,
#                         'confidence': confidence
#                     })
                    
#                     print(f"{color}{symbol}\033[0m {test_file[:30]:30} "
#                           f"True: {true_gesture:10} Pred: {gesture:10} "
#                           f"Conf: {confidence:.2%}")
                    
#             except Exception as e:
#                 print(f"  Error testing {test_file}: {e}")
        
#         # Calculate overall accuracy
#         if results:
#             correct_count = sum(1 for r in results if r['correct'])
#             total_count = len(results)
#             accuracy = correct_count / total_count
            
#             print(f"\nOverall Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
#     def analyze_dataset(self):
#         """Analyze the dataset before training"""
#         print("\n" + "="*60)
#         print("DATASET ANALYSIS")
#         print("="*60)
        
#         # Load data without preprocessing
#         csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
#         if not csv_files:
#             print("No CSV files found!")
#             return
        
#         gesture_counts = {}
#         total_samples = 0
        
#         print(f"Found {len(csv_files)} CSV files")
#         print("\nGesture Distribution:")
#         print("-" * 40)
        
#         for csv_file in csv_files:
#             try:
#                 filepath = os.path.join(self.data_dir, csv_file)
#                 df = pd.read_csv(filepath)
                
#                 # Get gesture
#                 gesture = None
#                 if 'gesture' in df.columns:
#                     gesture = str(df['gesture'].iloc[0])
#                 else:
#                     gesture = csv_file.split('_')[0]
                
#                 # Count samples
#                 num_samples = len(df)
                
#                 if gesture not in gesture_counts:
#                     gesture_counts[gesture] = 0
#                 gesture_counts[gesture] += num_samples
#                 total_samples += num_samples
                
#                 print(f"  {csv_file[:30]:30} → {gesture:10} ({num_samples:,} samples)")
                
#             except Exception as e:
#                 print(f"  Error reading {csv_file}: {e}")
        
#         print("\nSummary:")
#         print("-" * 40)
#         for gesture, count in sorted(gesture_counts.items()):
#             percentage = (count / total_samples) * 100
#             print(f"  {gesture:15}: {count:10,} samples ({percentage:6.1f}%)")
        
#         print(f"\nTotal samples: {total_samples:,}")
        
#         # Check if we have enough data
#         MIN_SAMPLES_PER_GESTURE = 1000
#         issues = []
#         for gesture, count in gesture_counts.items():
#             if count < MIN_SAMPLES_PER_GESTURE:
#                 issues.append(f"  - {gesture}: Only {count:,} samples (recommended: {MIN_SAMPLES_PER_GESTURE:,}+)")
        
#         if issues:
#             print(f"\n⚠ Potential issues:")
#             for issue in issues:
#                 print(issue)
#             print("\nConsider collecting more data before training.")
#         else:
#             print("\n✅ Dataset looks good for training!")


# def main():
#     print("\n" + "="*70)
#     print("EMG GESTURE CLASSIFICATION MODEL TRAINER")
#     print("="*70)
#     print("For gestures: round, shoot, up_down, rest")
#     print("="*70)
    
#     # Configuration
#     CONFIG = {
#         'data_dir': 'custom_emg_gestures',  # Directory with your CSV files
#         'window_size': 100,                  # Smaller window for less data
#         'overlap': 0.5,                      # Overlap between windows
#         'sample_rate': 500,                  # EMG sampling rate in Hz
#         'test_size': 0.2                     # Test set size
#     }
    
#     # Create trainer
#     trainer = EMGModelTrainer(
#         data_dir=CONFIG['data_dir'],
#         window_size=CONFIG['window_size'],
#         overlap=CONFIG['overlap'],
#         fs=CONFIG['sample_rate']
#     )
    
#     while True:
#         print("\n" + "="*70)
#         print("MAIN MENU")
#         print("="*70)
#         print("  1: Analyze dataset")
#         print("  2: Train new model")
#         print("  3: Test existing model")
#         print("  4: Test on sample files")
#         print("  5: Real-time prediction demo")
#         print("  6: List available models")
#         print("  7: Exit")
#         print("="*70)
        
#         choice = input("\nEnter choice (1-7): ").strip()
        
#         if choice == '1':
#             # Analyze dataset
#             trainer.analyze_dataset()
        
#         elif choice == '2':
#             # Train new model
#             try:
#                 print("\nLoading data...")
#                 X, y = trainer.load_and_preprocess_data()
#                 print(f"\nDataset loaded successfully!")
#                 print(f"  Samples: {X.shape[0]}")
#                 print(f"  Features: {X.shape[1]}")
#                 print(f"  Gestures: {np.unique(y)}")
                
#                 confirm = input("\nProceed with training? (y/n): ").strip().lower()
#                 if confirm == 'y':
#                     trainer.train_model(X, y, test_size=CONFIG['test_size'])
#                 else:
#                     print("Training cancelled.")
                    
#             except ValueError as e:
#                 print(f"\n❌ Error: {e}")
#                 print("\nMake sure you have collected data using the data collector first!")
#                 print("Data should be in: custom_emg_gestures/")
#                 print("\nCheck that your CSV files have the correct format:")
#                 print("1. Should have 'gesture' column or gesture name in filename")
#                 print("2. Should have 'emg_value' column or numeric data")
#                 print("3. Should have at least 1000 samples per file")
        
#         elif choice == '3':
#             # Test existing model
#             if trainer.load_model():
#                 print("\nTesting model with simulated signals...")
                
#                 # Create simulated signals for each gesture
#                 t = np.linspace(0, 1, 500)
                
#                 test_signals = {
#                     'round': 250 + 150 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 25, 500),
#                     'shoot': np.concatenate([
#                         100 + np.random.normal(0, 15, 200),
#                         350 + np.random.normal(0, 40, 100),
#                         100 + np.random.normal(0, 15, 200)
#                     ]),
#                     'up_down': 200 + 120 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 20, 500),
#                     'rest': 50 + np.random.normal(0, 10, 500)
#                 }
                
#                 for gesture_name, signal in test_signals.items():
#                     pred, conf = trainer.real_time_predict(signal)
#                     if pred:
#                         correct = (pred == gesture_name)
#                         symbol = "✓" if correct else "✗"
#                         color = '\033[92m' if correct else '\033[91m'
#                         print(f"{color}{symbol}\033[0m True: {gesture_name:10} → Pred: {pred:10} (Conf: {conf:.2%})")
        
#         elif choice == '4':
#             # Test on sample files
#             if trainer.load_model():
#                 trainer.test_on_sample_files()
        
#         elif choice == '5':
#             # Real-time prediction demo
#             print("\n" + "="*60)
#             print("REAL-TIME PREDICTION DEMO")
#             print("="*60)
#             print("This will simulate real-time EMG signals for testing.")
#             print("Press Ctrl+C to stop.")
#             print("="*60)
            
#             import time
#             from collections import deque
            
#             if not trainer.load_model():
#                 continue
            
#             buffer = deque(maxlen=trainer.window_size * 2)
#             gestures_cycle = ['round', 'shoot', 'up_down']
            
#             try:
#                 for i in range(50):  # 50 prediction cycles
#                     # Generate simulated signal based on current gesture
#                     current_gesture = gestures_cycle[i % len(gestures_cycle)]
#                     t = time.time()
                    
#                     if current_gesture == 'round':
#                         sample = 250 + 150 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 25)
#                     elif current_gesture == 'shoot':
#                         if np.random.random() < 0.2:  # 20% chance of burst
#                             sample = 350 + np.random.normal(0, 40)
#                         else:
#                             sample = 100 + np.random.normal(0, 15)
#                     else:  # up_down
#                         sample = 200 + 120 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 20)
                    
#                     buffer.append(sample)
                    
#                     # Make prediction if we have enough data
#                     if len(buffer) >= trainer.window_size:
#                         gesture, confidence = trainer.real_time_predict(list(buffer))
#                         if gesture:
#                             correct = (gesture == current_gesture)
#                             symbol = "✓" if correct else "✗"
#                             color = '\033[92m' if correct else '\033[91m'
#                             print(f"{color}{symbol}\033[0m True: {current_gesture:10} → Pred: {gesture:10} (Conf: {confidence:.2%})")
                    
#                     time.sleep(0.2)  # 5 Hz prediction rate
                    
#             except KeyboardInterrupt:
#                 print("\nDemo stopped.")
        
#         elif choice == '6':
#             # List available models
#             model_dirs = [d for d in os.listdir('.') if d.startswith('emg_gesture_model_') and os.path.isdir(d)]
#             if model_dirs:
#                 print("\nAvailable trained models:")
#                 for i, dir_name in enumerate(sorted(model_dirs, reverse=True)[:5]):
#                     try:
#                         with open(os.path.join(dir_name, 'metadata.json'), 'r') as f:
#                             metadata = json.load(f)
                        
#                         print(f"\n{i+1}. {dir_name}")
#                         print(f"   Model: {metadata['model_type']}")
#                         print(f"   Gestures: {metadata['gesture_labels']}")
#                         print(f"   Date: {metadata['training_date']}")
#                     except:
#                         print(f"{i+1}. {dir_name}")
#             else:
#                 print("\nNo trained models found!")
        
#         elif choice == '7':
#             print("\nThank you for using EMG Model Trainer!")
#             print("Goodbye!")
#             break
        
#         else:
#             print("\nInvalid choice! Please try again.")

# if __name__ == "__main__":
#     main()




# ----------------------------------------------------------------------------------------------






# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import pickle
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # Import ML models
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# import xgboost as xgb

# class EMGModelTrainer:
#     def __init__(self, data_paths=None):
#         """Initialize trainer with EMG data paths"""
#         self.data_paths = data_paths or []
#         self.models = {}
#         self.scaler = StandardScaler()
#         self.feature_importance = {}
        
#     def load_and_preprocess_data(self, window_size=50, overlap=25):
#         """Load EMG data and extract features"""
        
#         all_features = []
#         all_labels = []
        
#         for data_path in self.data_paths:
#             if not os.path.exists(data_path):
#                 print(f"Warning: {data_path} not found")
#                 continue
            
#             print(f"Loading data from: {data_path}")
#             df = pd.read_csv(data_path)
            
#             # Extract features from sliding windows
#             features, labels = self.extract_features(df, window_size, overlap)
#             all_features.extend(features)
#             all_labels.extend(labels)
        
#         all_features = np.array(all_features)
#         all_labels = np.array(all_labels)
        
#         print(f"\nTotal samples: {len(all_labels)}")
#         print(f"Feature vector shape: {all_features.shape}")
        
#         return all_features, all_labels
    
#     def extract_features(self, df, window_size, overlap):
#         """Extract features from EMG signal windows"""
        
#         emg_signal = df['emg_value'].values
#         labels = df['gesture_label'].values
        
#         features = []
#         window_labels = []
        
#         step = window_size - overlap
        
#         for start in range(0, len(emg_signal) - window_size + 1, step):
#             window = emg_signal[start:start + window_size]
#             window_label = labels[start + window_size // 2]  # Use middle label
            
#             # Statistical features
#             mean_val = np.mean(window)
#             std_val = np.std(window)
#             var_val = np.var(window)
#             rms_val = np.sqrt(np.mean(window**2))
#             max_val = np.max(window)
#             min_val = np.min(window)
#             peak_to_peak = max_val - min_val
            
#             # Frequency domain features (simplified)
#             fft_vals = np.abs(np.fft.fft(window))
#             spectral_centroid = np.sum(fft_vals * np.arange(len(fft_vals))) / np.sum(fft_vals)
            
#             features.append([
#                 mean_val, std_val, var_val, rms_val, max_val, min_val,
#                 peak_to_peak, spectral_centroid
#             ])
#             window_labels.append(window_label)
        
#         return features, window_labels
    
#     def train_models(self, X_train, y_train):
#         """Train multiple ML models and compare performance"""
        
#         models_to_train = {
#             'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
#             'SVM': SVC(kernel='rbf', probability=True, random_state=42),
#             'KNN': KNeighborsClassifier(n_neighbors=5),
#             'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
#             'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
#             'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
#         }
        
#         print("Training models...")
#         print("=" * 60)
        
#         results = {}
        
#         for name, model in models_to_train.items():
#             print(f"Training {name}...")
            
#             # Scale features for models that need it
#             if name in ['SVM', 'KNN', 'MLP']:
#                 X_train_scaled = self.scaler.fit_transform(X_train)
#                 model.fit(X_train_scaled, y_train)
#             else:
#                 model.fit(X_train, y_train)
            
#             self.models[name] = model
            
#             # Cross-validation score
#             cv_scores = cross_val_score(model, X_train, y_train, cv=5)
#             results[name] = {
#                 'model': model,
#                 'cv_mean': cv_scores.mean(),
#                 'cv_std': cv_scores.std()
#             }
            
#             print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
#         # Sort models by performance
#         sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        
#         print("\n" + "=" * 60)
#         print("MODEL PERFORMANCE RANKING:")
#         print("=" * 60)
#         for rank, (name, result) in enumerate(sorted_results, 1):
#             print(f"{rank}. {name}: {result['cv_mean']:.3f} (+/- {result['cv_std']:.3f})")
        
#         return sorted_results
    
#     def evaluate_best_model(self, X_test, y_test, best_model_name):
#         """Evaluate the best model on test data"""
        
#         best_model = self.models[best_model_name]
        
#         if best_model_name in ['SVM', 'KNN', 'MLP']:
#             X_test_scaled = self.scaler.transform(X_test)
#             y_pred = best_model.predict(X_test_scaled)
#         else:
#             y_pred = best_model.predict(X_test)
        
#         accuracy = accuracy_score(y_test, y_pred)
        
#         print("\n" + "=" * 60)
#         print(f"EVALUATION OF BEST MODEL: {best_model_name}")
#         print("=" * 60)
#         print(f"Test Accuracy: {accuracy:.3f}")
#         print("\nClassification Report:")
#         print(classification_report(y_test, y_pred))
        
#         return accuracy
    
#     def save_best_model(self, model_name, features, labels):
#         """Save the best model and feature extractor"""
        
#         if model_name not in self.models:
#             print(f"Model {model_name} not found!")
#             return
        
#         # Create models directory
#         model_dir = "trained_models"
#         if not os.path.exists(model_dir):
#             os.makedirs(model_dir)
        
#         # Save the model
#         model_path = f"{model_dir}/best_emg_model.pkl"
#         with open(model_path, 'wb') as f:
#             pickle.dump({
#                 'model': self.models[model_name],
#                 'scaler': self.scaler if model_name in ['SVM', 'KNN', 'MLP'] else None,
#                 'window_size': 50,
#                 'overlap': 25,
#                 'feature_names': ['mean', 'std', 'var', 'rms', 'max', 'min', 'peak_to_peak', 'spectral_centroid'],
#                 'class_names': ["relax", "fist", "open_hand", "wrist_up", "wrist_down"],
#                 'training_samples': len(labels)
#             }, f)
        
#         print(f"\nModel saved to: {model_path}")
        
#         # Also save feature importance if available
#         if hasattr(self.models[model_name], 'feature_importances_'):
#             feature_importance = self.models[model_name].feature_importances_
#             importance_df = pd.DataFrame({
#                 'feature': ['mean', 'std', 'var', 'rms', 'max', 'min', 'peak_to_peak', 'spectral_centroid'],
#                 'importance': feature_importance
#             })
#             importance_df = importance_df.sort_values('importance', ascending=False)
#             print("\nFeature Importance:")
#             print(importance_df)

# def main_training():
#     """Main training pipeline"""
    
#     # Find all CSV files in emg_data directory
#     data_dir = "emg_data"
#     if not os.path.exists(data_dir):
#         print(f"Error: Directory '{data_dir}' not found!")
#         print("Please run Part 1 first to collect data.")
#         return
    
#     data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
#     if not data_files:
#         print("No CSV files found in emg_data directory!")
#         print("Please run Part 1 first to collect data.")
#         return
    
#     print(f"Found {len(data_files)} data file(s)")
    
#     # Initialize trainer
#     trainer = EMGModelTrainer(data_files)
    
#     # Load and preprocess data
#     X, y = trainer.load_and_preprocess_data()
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     print(f"\nTraining set: {len(X_train)} samples")
#     print(f"Test set: {len(X_test)} samples")
    
#     # Train models
#     results = trainer.train_models(X_train, y_train)
    
#     # Get best model
#     best_model_name = results[0][0]
    
#     # Evaluate best model
#     trainer.evaluate_best_model(X_test, y_test, best_model_name)
    
#     # Save best model
#     trainer.save_best_model(best_model_name, X, y)
    
#     return trainer, best_model_name

# if __name__ == "__main__":
#     trainer, best_model = main_training()






# ---------------------------------------------------------------------





# import os
# import glob
# import numpy as np
# import pandas as pd
# import joblib
# from scipy.fft import rfft, rfftfreq
# from scipy.stats import entropy

# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression


# # ==========================================================
# # ADVANCED FEATURE EXTRACTION
# # ==========================================================

# def extract_features(df):
#     x = df["emg_value"].values.astype(float)
#     N = len(x)

#     # ------------- Time Domain Features -------------
#     mean = np.mean(x)
#     std = np.std(x)
#     var = np.var(x)
#     rms = np.sqrt(np.mean(x ** 2))
#     median = np.median(x)
#     maximum = np.max(x)
#     minimum = np.min(x)
#     mav = np.mean(np.abs(x))
#     abs_energy = np.sum(x ** 2)
#     waveform_length = np.sum(np.abs(np.diff(x)))

#     # ------------- Zero Crossing Rate -------------
#     zero_crossings = np.sum(np.diff(np.sign(x)) != 0)

#     # ------------- Slope Sign Changes -------------
#     diff_x = np.diff(x)
#     slope_changes = np.sum(np.diff(np.sign(diff_x)) != 0)

#     # ------------- First Derivative Features -------------
#     dx = np.diff(x)
#     dx_mean = np.mean(dx)
#     dx_rms = np.sqrt(np.mean(dx ** 2))
#     dx_zero_cross = np.sum(np.diff(np.sign(dx)) != 0)

#     # ------------- Frequency Domain Features (FFT) -------------
#     xf = rfftfreq(N, d=1)       # frequency axis (assuming sampling interval = 1)
#     yf = np.abs(rfft(x))        # FFT magnitude

#     spectral_energy = np.sum(yf ** 2)
#     spectral_centroid = np.sum(xf * yf) / np.sum(yf)
#     spectral_entropy = entropy(yf / np.sum(yf))
#     dominant_freq = xf[np.argmax(yf)]
#     mean_freq = np.sum(xf * yf) / np.sum(yf)
#     median_freq = xf[np.argsort(np.cumsum(yf) / np.sum(yf) > 0.5)][0]

#     features = [
#         mean, std, var, rms, median, maximum, minimum, mav,
#         abs_energy, waveform_length, zero_crossings, slope_changes,
#         dx_mean, dx_rms, dx_zero_cross,
#         spectral_energy, spectral_centroid, spectral_entropy,
#         dominant_freq, mean_freq, median_freq
#     ]

#     return features


# # ==========================================================
# # LOAD ALL GESTURE FILES
# # ==========================================================

# def load_dataset(folder="emg_dataset"):
#     X = []
#     y = []

#     for file in glob.glob(os.path.join(folder, "*.csv")):
#         df = pd.read_csv(file)
#         gesture = os.path.basename(file).replace(".csv", "")

#         print(f"📥 Loaded {gesture}")

#         feat = extract_features(df)
#         X.append(feat)
#         y.append(gesture)

#     return np.array(X), np.array(y)


# # ==========================================================
# # EVALUATE MULTIPLE MODELS (Pick Best One)
# # ==========================================================

# def evaluate_models(X, y):
#     models = {
#         "RandomForest": RandomForestClassifier(n_estimators=250),
#         "SVM": SVC(kernel="rbf"),
#         "KNN": KNeighborsClassifier(n_neighbors=3),
#         "LogisticRegression": LogisticRegression(max_iter=2000)
#     }

#     scores = {}

#     for name, model in models.items():
#         pipe = Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", model)
#         ])

#         score = cross_val_score(pipe, X, y, cv=3).mean()
#         scores[name] = score

#         print(f"✔ {name} Accuracy: {score:.4f}")

#     return scores


# # ==========================================================
# # TRAIN BEST MODEL
# # ==========================================================

# def train_best_model(X, y, best_name):

#     print(f"\n🏆 Best Model → {best_name}")

#     if best_name == "RandomForest":
#         model = RandomForestClassifier(n_estimators=250)
#     elif best_name == "SVM":
#         model = SVC(kernel="rbf", probability=True)
#     elif best_name == "KNN":
#         model = KNeighborsClassifier(n_neighbors=3)
#     else:
#         model = LogisticRegression(max_iter=2000)

#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("model", model)
#     ])

#     pipe.fit(X, y)

#     joblib.dump(pipe, "best_emg_model.pkl")
#     print("💾 Model saved as best_emg_model.pkl")

#     return pipe


# # ==========================================================
# # MAIN
# # ==========================================================

# if __name__ == "__main__":
#     print("\n====================================")
#     print("  ADVANCED EMG ML TRAINER")
#     print("====================================")

#     X, y = load_dataset()

#     print("\n🔥 Testing models...")
#     results = evaluate_models(X, y)

#     best_model = max(results, key=results.get)

#     train_best_model(X, y, best_model)

#     print("\n🎉 Training Completed Successfully!")





# -----------------------------------------------------------------




# import os
# import glob
# import numpy as np
# import pandas as pd
# import joblib
# from scipy.fft import rfft, rfftfreq
# from scipy.stats import entropy
# import warnings

# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression

# # Suppress warnings for cleaner output
# warnings.filterwarnings('ignore')

# # ==========================================================
# # ADVANCED FEATURE EXTRACTION
# # ==========================================================

# def extract_features(df):
#     """
#     Extract advanced features from EMG signal
#     """
#     # Ensure we have the correct column name
#     if 'emg_value' not in df.columns:
#         # Try to find EMG data column
#         possible_cols = [col for col in df.columns if 'emg' in col.lower() or 'value' in col.lower()]
#         if possible_cols:
#             x = df[possible_cols[0]].values.astype(float)
#         else:
#             # Use first numeric column
#             numeric_cols = df.select_dtypes(include=[np.number]).columns
#             if len(numeric_cols) > 0:
#                 x = df[numeric_cols[0]].values.astype(float)
#             else:
#                 raise ValueError("No suitable EMG data column found in the dataframe")
#     else:
#         x = df["emg_value"].values.astype(float)
    
#     N = len(x)
    
#     if N < 10:
#         raise ValueError(f"Signal too short (N={N}) for feature extraction")

#     # ------------- Time Domain Features -------------
#     mean = np.mean(x)
#     std = np.std(x)
#     var = np.var(x)
#     rms = np.sqrt(np.mean(x ** 2))
#     median = np.median(x)
#     maximum = np.max(x)
#     minimum = np.min(x)
#     mav = np.mean(np.abs(x))
#     abs_energy = np.sum(x ** 2)
#     waveform_length = np.sum(np.abs(np.diff(x)))

#     # ------------- Zero Crossing Rate -------------
#     zero_crossings = np.sum(np.diff(np.sign(x)) != 0)

#     # ------------- Slope Sign Changes -------------
#     diff_x = np.diff(x)
#     slope_changes = np.sum(np.diff(np.sign(diff_x)) != 0)

#     # ------------- First Derivative Features -------------
#     dx = np.diff(x)
#     dx_mean = np.mean(dx)
#     dx_rms = np.sqrt(np.mean(dx ** 2))
#     dx_zero_cross = np.sum(np.diff(np.sign(dx)) != 0)

#     # ------------- Frequency Domain Features (FFT) -------------
#     try:
#         xf = rfftfreq(N, d=1)       # frequency axis (assuming sampling interval = 1)
#         yf = np.abs(rfft(x))        # FFT magnitude
        
#         if np.sum(yf) > 0:
#             spectral_energy = np.sum(yf ** 2)
#             spectral_centroid = np.sum(xf * yf) / np.sum(yf)
#             spectral_entropy = entropy(yf / np.sum(yf))
#             dominant_freq = xf[np.argmax(yf)]
#             mean_freq = np.sum(xf * yf) / np.sum(yf)
#             # Find median frequency (frequency below which 50% of power resides)
#             cumulative_power = np.cumsum(yf) / np.sum(yf)
#             median_freq_idx = np.where(cumulative_power >= 0.5)[0][0]
#             median_freq = xf[median_freq_idx]
#         else:
#             spectral_energy = spectral_centroid = spectral_entropy = 0
#             dominant_freq = mean_freq = median_freq = 0
#     except:
#         # Fallback in case of FFT issues
#         spectral_energy = spectral_centroid = spectral_entropy = 0
#         dominant_freq = mean_freq = median_freq = 0

#     features = [
#         mean, std, var, rms, median, maximum, minimum, mav,
#         abs_energy, waveform_length, zero_crossings, slope_changes,
#         dx_mean, dx_rms, dx_zero_cross,
#         spectral_energy, spectral_centroid, spectral_entropy,
#         dominant_freq, mean_freq, median_freq
#     ]

#     return features


# # ==========================================================
# # LOAD ALL GESTURE FILES
# # ==========================================================

# def load_dataset(folder="emg_dataset"):
#     """
#     Load all CSV files from the specified folder and extract features
#     """
#     X = []
#     y = []
#     file_count = 0
#     error_files = []
    
#     # Get all CSV files in the folder
#     csv_files = glob.glob(os.path.join(folder, "*.csv"))
    
#     if not csv_files:
#         print(f"❌ No CSV files found in '{folder}' directory")
#         print(f"📁 Current working directory: {os.getcwd()}")
#         print(f"📁 Contents of '{folder}':")
#         if os.path.exists(folder):
#             print(os.listdir(folder))
#         else:
#             print("Directory does not exist!")
#         return np.array(X), np.array(y)
    
#     print(f"📂 Found {len(csv_files)} CSV files in '{folder}'")
    
#     for file in csv_files:
#         try:
#             df = pd.read_csv(file)
#             gesture = os.path.splitext(os.path.basename(file))[0]
            
#             # Skip empty files
#             if df.empty:
#                 print(f"⚠️  Skipping empty file: {gesture}")
#                 error_files.append(file)
#                 continue
                
#             feat = extract_features(df)
#             X.append(feat)
#             y.append(gesture)
#             file_count += 1
            
#             print(f"✅ Loaded {gesture} - {len(df)} samples")
            
#         except Exception as e:
#             print(f"❌ Error loading {os.path.basename(file)}: {str(e)}")
#             error_files.append(file)
    
#     print(f"\n📊 Summary:")
#     print(f"   Successfully loaded: {file_count} files")
#     print(f"   Failed to load: {len(error_files)} files")
    
#     if error_files:
#         print(f"   Error files: {error_files}")
    
#     # Check if we have data
#     if len(X) == 0:
#         print("❌ No valid data loaded!")
#         return np.array(X), np.array(y)
    
#     print(f"\n📈 Feature matrix shape: ({len(X)}, {len(X[0])})")
#     print(f"🎯 Classes found: {set(y)}")
#     print(f"📊 Samples per class:")
#     for gesture in set(y):
#         count = y.count(gesture)
#         print(f"   {gesture}: {count}")
    
#     return np.array(X), np.array(y)


# # ==========================================================
# # EVALUATE MULTIPLE MODELS (Pick Best One)
# # ==========================================================

# def evaluate_models(X, y):
#     """
#     Evaluate multiple classifiers and return their scores
#     """
#     if len(X) == 0:
#         print("❌ No data to train models!")
#         return {}
    
#     # Check for sufficient data for 3-fold CV
#     if len(X) < 3:
#         print(f"⚠️  Insufficient samples ({len(X)}) for 3-fold cross-validation")
#         return {}
    
#     # Use stratified K-fold for imbalanced classes
#     cv = StratifiedKFold(n_splits=min(3, len(set(y))), shuffle=True, random_state=42)
    
#     models = {
#         "RandomForest": RandomForestClassifier(n_estimators=250, random_state=42),
#         "SVM": SVC(kernel="rbf", random_state=42),
#         "KNN": KNeighborsClassifier(n_neighbors=3),
#         "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42)
#     }
    
#     scores = {}
#     stds = {}
    
#     print("\n🔍 Model Evaluation Results:")
#     print("=" * 50)
    
#     for name, model in models.items():
#         try:
#             pipe = Pipeline([
#                 ("scaler", StandardScaler()),
#                 ("model", model)
#             ])
            
#             # Perform cross-validation
#             cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
#             mean_score = cv_scores.mean()
#             std_score = cv_scores.std()
            
#             scores[name] = mean_score
#             stds[name] = std_score
            
#             print(f"📊 {name:20s} Accuracy: {mean_score:.4f} (±{std_score:.4f})")
            
#         except Exception as e:
#             print(f"❌ Error evaluating {name}: {str(e)}")
#             scores[name] = 0
    
#     return scores


# # ==========================================================
# # TRAIN BEST MODEL
# # ==========================================================

# def train_best_model(X, y, best_name):
#     """
#     Train the best performing model on the entire dataset
#     """
#     if len(X) == 0:
#         print("❌ No data to train the model!")
#         return None
    
#     print(f"\n🏆 Training Best Model → {best_name}")
#     print("=" * 50)
    
#     # Define models with parameters
#     if best_name == "RandomForest":
#         model = RandomForestClassifier(n_estimators=250, random_state=42)
#     elif best_name == "SVM":
#         model = SVC(kernel="rbf", probability=True, random_state=42)
#     elif best_name == "KNN":
#         model = KNeighborsClassifier(n_neighbors=3)
#     elif best_name == "LogisticRegression":
#         model = LogisticRegression(max_iter=2000, random_state=42)
#     else:
#         print(f"❌ Unknown model: {best_name}")
#         return None
    
#     # Create pipeline
#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("model", model)
#     ])
    
#     # Train the model
#     pipe.fit(X, y)
    
#     # Calculate training accuracy
#     train_accuracy = pipe.score(X, y)
#     print(f"✅ Training accuracy: {train_accuracy:.4f}")
    
#     # Save the model
#     try:
#         joblib.dump(pipe, "best_emg_model.pkl")
#         print("💾 Model saved as 'best_emg_model.pkl'")
        
#         # Also save the class labels
#         class_labels = list(set(y))
#         joblib.dump(class_labels, "class_labels.pkl")
#         print(f"📝 Class labels saved: {class_labels}")
        
#     except Exception as e:
#         print(f"❌ Error saving model: {str(e)}")
    
#     return pipe


# # ==========================================================
# # MAIN
# # ==========================================================

# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("           ADVANCED EMG ML TRAINER")
#     print("=" * 60)
    
#     # Create dataset folder if it doesn't exist
#     dataset_folder = "emg_dataset"
#     if not os.path.exists(dataset_folder):
#         print(f"📁 Creating dataset folder: {dataset_folder}")
#         os.makedirs(dataset_folder)
#         print(f"⚠️  Please place your CSV files in the '{dataset_folder}' folder and run again.")
#         exit(0)
    
#     # Load dataset
#     print(f"\n📥 Loading data from '{dataset_folder}'...")
#     X, y = load_dataset(dataset_folder)
    
#     if len(X) == 0:
#         print("\n❌ No data available. Exiting...")
#         exit(1)
    
#     # Evaluate models
#     print("\n🔥 Testing models with 3-fold cross-validation...")
#     results = evaluate_models(X, y)
    
#     if not results:
#         print("\n❌ No models were evaluated successfully.")
#         exit(1)
    
#     # Find best model
#     best_model = max(results, key=results.get)
#     best_score = results[best_model]
    
#     print(f"\n" + "=" * 50)
#     print(f"🎯 BEST MODEL: {best_model} (Accuracy: {best_score:.4f})")
#     print("=" * 50)
    
#     # Train the best model
#     model_pipeline = train_best_model(X, y, best_model)
    
#     if model_pipeline:
#         print("\n✅ Training Completed Successfully!")
#         print(f"\n📋 Model Summary:")
#         print(f"   - Best algorithm: {best_model}")
#         print(f"   - Number of features: {X.shape[1]}")
#         print(f"   - Number of samples: {X.shape[0]}")
#         print(f"   - Classes: {sorted(set(y))}")
        
#         # Test prediction on training data
#         print(f"\n🧪 Test Prediction on first sample:")
#         sample_features = X[0].reshape(1, -1)
#         prediction = model_pipeline.predict(sample_features)
#         probability = model_pipeline.predict_proba(sample_features)
        
#         print(f"   Features shape: {sample_features.shape}")
#         print(f"   Predicted class: {prediction[0]}")
#         print(f"   True class: {y[0]}")
#         print(f"   Prediction confidence: {np.max(probability[0]):.4f}")
#     else:
#         print("\n❌ Model training failed!")






# -------------------------------------------------------------------------------






# import os
# import glob
# import numpy as np
# import pandas as pd
# import joblib
# from scipy.fft import rfft, rfftfreq
# from scipy.stats import entropy

# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression


# # ==========================================================
# # ADVANCED FEATURE EXTRACTION
# # ==========================================================

# def extract_features(df):
#     x = df["emg_value"].values.astype(float)
#     N = len(x)

#     # Time domain
#     mean = np.mean(x)
#     std = np.std(x)
#     var = np.var(x)
#     rms = np.sqrt(np.mean(x ** 2))
#     median = np.median(x)
#     maximum = np.max(x)
#     minimum = np.min(x)
#     mav = np.mean(np.abs(x))
#     abs_energy = np.sum(x ** 2)
#     waveform_length = np.sum(np.abs(np.diff(x)))

#     # Zero crossing
#     zero_crossings = np.sum(np.diff(np.sign(x)) != 0)

#     # Slope changes
#     diff_x = np.diff(x)
#     slope_changes = np.sum(np.diff(np.sign(diff_x)) != 0)

#     # First derivative features
#     dx = diff_x
#     dx_mean = np.mean(dx)
#     dx_rms = np.sqrt(np.mean(dx ** 2))
#     dx_zero_cross = np.sum(np.diff(np.sign(dx)) != 0)

#     # Frequency domain
#     xf = rfftfreq(N, d=1)
#     yf = np.abs(rfft(x))

#     spectral_energy = np.sum(yf ** 2)
#     spectral_centroid = np.sum(xf * yf) / np.sum(yf)
#     spectral_entropy = entropy(yf / np.sum(yf))
#     dominant_freq = xf[np.argmax(yf)]
#     mean_freq = np.sum(xf * yf) / np.sum(yf)

#     # Median frequency
#     cumulative = np.cumsum(yf)
#     median_freq = xf[np.searchsorted(cumulative, cumulative[-1] / 2)]

#     return [
#         mean, std, var, rms, median, maximum, minimum, mav,
#         abs_energy, waveform_length, zero_crossings, slope_changes,
#         dx_mean, dx_rms, dx_zero_cross,
#         spectral_energy, spectral_centroid, spectral_entropy,
#         dominant_freq, mean_freq, median_freq
#     ]


# # ==========================================================
# # LOAD DATASET
# # ==========================================================

# def load_dataset(folder="emg_dataset"):
#     X, y = [], []

#     for file in glob.glob(os.path.join(folder, "*.csv")):
#         df = pd.read_csv(file)
#         gesture = os.path.basename(file).replace(".csv", "")

#         print(f"📥 Loaded {gesture}")

#         feat = extract_features(df)
#         X.append(feat)
#         y.append(gesture)

#     return np.array(X), np.array(y)


# # ==========================================================
# # MODEL EVALUATION (with auto CV fix)
# # ==========================================================

# def evaluate_models(X, y):
#     class_counts = pd.Series(y).value_counts()
#     min_class = class_counts.min()

#     cv = min(3, min_class)

#     if cv < 2:
#         print("\n⚠ WARNING: Not enough samples for CV → using CV=1 (no validation)")
#         cv = 1
#     else:
#         print(f"\n🔧 Auto CV selected = {cv}")

#     models = {
#         "RandomForest": RandomForestClassifier(n_estimators=200),
#         "SVM": SVC(kernel="rbf"),
#         "KNN": KNeighborsClassifier(n_neighbors=3),
#         "LogisticRegression": LogisticRegression(max_iter=2000)
#     }

#     scores = {}

#     for name, model in models.items():
#         pipe = Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", model)
#         ])

#         if cv == 1:
#             # No cross-validation possible → just fit & score = 1.0
#             pipe.fit(X, y)
#             score = 1.0
#         else:
#             score = cross_val_score(pipe, X, y, cv=cv).mean()

#         scores[name] = score
#         print(f"✔ {name} Accuracy: {score:.4f}")

#     return scores


# # ==========================================================
# # TRAIN BEST MODEL
# # ==========================================================

# def train_best_model(X, y, best_name):

#     print(f"\n🏆 Best Model → {best_name}")

#     if best_name == "RandomForest":
#         model = RandomForestClassifier(n_estimators=200)
#     elif best_name == "SVM":
#         model = SVC(kernel="rbf", probability=True)
#     elif best_name == "KNN":
#         model = KNeighborsClassifier(n_neighbors=3)
#     else:
#         model = LogisticRegression(max_iter=2000)

#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("model", model)
#     ])

#     pipe.fit(X, y)
#     joblib.dump(pipe, "best_emg_model.pkl")

#     print("💾 Saved model → best_emg_model.pkl")


# # ==========================================================
# # MAIN
# # ==========================================================

# if __name__ == "__main__":
#     print("\n====================================")
#     print("  ADVANCED EMG ML TRAINER")
#     print("====================================\n")

#     X, y = load_dataset()

#     print("\n🔥 Testing models...")
#     results = evaluate_models(X, y)

#     best_model = max(results, key=results.get)

#     train_best_model(X, y, best_model)

#     print("\n🎉 Training Completed Successfully!\n")






# ---------------------------------------------------------------------------






# import os
# import glob
# import numpy as np
# import pandas as pd
# import joblib
# from scipy.fft import rfft, rfftfreq
# from scipy.stats import entropy

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression

# # ==========================================================
# # WINDOW SETTINGS
# # ==========================================================
# WINDOW = 20          # number of EMG samples per training sample
# STEP = 10            # overlap step


# # ==========================================================
# # ADVANCED FEATURE EXTRACTION
# # ==========================================================
# def extract_features(window):
#     x = np.array(window).astype(float)
#     N = len(x)

#     mean = np.mean(x)
#     std = np.std(x)
#     var = np.var(x)
#     rms = np.sqrt(np.mean(x ** 2))
#     median = np.median(x)
#     maximum = np.max(x)
#     minimum = np.min(x)
#     mav = np.mean(np.abs(x))
#     abs_energy = np.sum(x ** 2)
#     waveform_length = np.sum(np.abs(np.diff(x)))
#     zero_crossings = np.sum(np.diff(np.sign(x)) != 0)

#     diff_x = np.diff(x)
#     slope_changes = np.sum(np.diff(np.sign(diff_x)) != 0)

#     dx_mean = np.mean(diff_x)
#     dx_rms = np.sqrt(np.mean(diff_x ** 2))
#     dx_zero_cross = np.sum(np.diff(np.sign(diff_x)) != 0)

#     xf = rfftfreq(N, d=1)
#     yf = np.abs(rfft(x))

#     spectral_energy = np.sum(yf ** 2)
#     spectral_centroid = np.sum(xf * yf) / (np.sum(yf) + 1e-6)
#     spectral_entropy = entropy((yf + 1e-10) / np.sum(yf))
#     dominant_freq = xf[np.argmax(yf)]
#     mean_freq = np.sum(xf * yf) / (np.sum(yf) + 1e-6)

#     cum = np.cumsum(yf)
#     median_freq = xf[np.searchsorted(cum, cum[-1] * 0.5)]

#     features = [
#         mean, std, var, rms, median, maximum, minimum, mav,
#         abs_energy, waveform_length, zero_crossings, slope_changes,
#         dx_mean, dx_rms, dx_zero_cross,
#         spectral_energy, spectral_centroid, spectral_entropy,
#         dominant_freq, mean_freq, median_freq
#     ]
#     return features


# # ==========================================================
# # LOAD AND CREATE MULTIPLE TRAINING SAMPLES PER CSV
# # ==========================================================
# def load_dataset(folder="emg_dataset"):
#     X = []
#     y = []

#     files = glob.glob(os.path.join(folder, "*.csv"))
#     print(f"\n📂 Found {len(files)} gesture files\n")

#     for file in files:
#         df = pd.read_csv(file)
#         gesture = os.path.basename(file).replace(".csv", "")

#         print(f"📥 Loaded {gesture}")

#         signal = df["emg_value"].values

#         # sliding window
#         for i in range(0, len(signal) - WINDOW, STEP):
#             window = signal[i:i + WINDOW]

#             feat = extract_features(window)
#             X.append(feat)
#             y.append(gesture)

#     return np.array(X), np.array(y)


# # ==========================================================
# # TRAIN WITH MULTIPLE MODELS & PICK BEST
# # ==========================================================
# def train_models(X, y):

#     models = {
#         "RandomForest": RandomForestClassifier(n_estimators=300),
#         "SVM": SVC(kernel="rbf", probability=True),
#         "KNN": KNeighborsClassifier(n_neighbors=3),
#         "LogisticRegression": LogisticRegression(max_iter=3000)
#     }

#     best_name = None
#     best_score = 0
#     best_model = None

#     # split the dataset
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     for name, model in models.items():

#         pipe = Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", model)
#         ])

#         pipe.fit(X_train, y_train)
#         score = pipe.score(X_test, y_test)

#         print(f"✔ {name} → Accuracy: {score:.4f}")

#         if score > best_score:
#             best_score = score
#             best_name = name
#             best_model = pipe

#     print(f"\n🏆 Best Model: {best_name} (Accuracy={best_score:.4f})")
#     joblib.dump(best_model, "best_emg_model.pkl")
#     print("💾 Saved → best_emg_model.pkl")

#     return best_model


# # ==========================================================
# # MAIN
# # ==========================================================
# if __name__ == "__main__":
#     print("\n====================================")
#     print("  ADVANCED EMG ML TRAINER (WINDOWED)")
#     print("====================================")

#     X, y = load_dataset()

#     print(f"\n📊 Total training samples created: {len(X)}")
#     print("🔍 Training models...\n")

#     train_models(X, y)

#     print("\n🎉 Training Completed Successfully!")






# ------------------------------------------------------------------------------






# import os
# import glob
# import numpy as np
# import pandas as pd
# import joblib
# from scipy.fft import rfft, rfftfreq
# from scipy.stats import entropy

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression

# # ------------------ Deep Learning ------------------
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import warnings
# warnings.filterwarnings("ignore")


# # ==========================================================
# # FEATURE EXTRACTION PER WINDOW
# # ==========================================================

# def extract_features(window):
#     x = np.array(window)
#     N = len(x)

#     # ----- Time-domain features -----
#     mean = np.mean(x)
#     std = np.std(x)
#     var = np.var(x)
#     rms = np.sqrt(np.mean(x ** 2))
#     mav = np.mean(np.abs(x))
#     wl = np.sum(np.abs(np.diff(x)))
#     zc = np.sum(np.diff(np.sign(x)) != 0)

#     # ----- Frequency domain -----
#     yf = np.abs(rfft(x))
#     xf = rfftfreq(N, d=1)

#     spectral_energy = np.sum(yf ** 2)
#     spectral_centroid = np.sum(xf * yf) / (np.sum(yf) + 1e-8)
#     spectral_entropy = entropy((yf + 1e-8) / np.sum(yf))

#     features = [
#         mean, std, var, rms, mav,
#         wl, zc,
#         spectral_energy, spectral_centroid, spectral_entropy
#     ]
#     return features


# # ==========================================================
# # LOAD DATA & SLIDING WINDOW
# # ==========================================================

# def load_dataset(folder="emg_dataset", win_size=50, step=25):
#     X_feat = []
#     X_raw = []
#     y = []

#     files = glob.glob(os.path.join(folder, "*.csv"))
#     print(f"\n📂 Found {len(files)} gesture files")

#     for file in files:
#         df = pd.read_csv(file)
#         gesture = os.path.basename(file).split(".")[0]
#         print(f"📥 Loaded {gesture}")

#         sig = df["emg_value"].values

#         # Sliding windows
#         for i in range(0, len(sig) - win_size, step):
#             win = sig[i:i + win_size]

#             X_feat.append(extract_features(win))
#             X_raw.append(win)         # for neural network
#             y.append(gesture)

#     X_feat = np.array(X_feat)
#     X_raw = np.array(X_raw)
#     y = np.array(y)

#     print(f"\n📊 Total training samples created: {len(X_feat)}")
#     return X_feat, X_raw, y


# # ==========================================================
# # ML MODELS
# # ==========================================================

# def evaluate_ml_models(X, y, cv_folds):
#     models = {
#         "RandomForest": RandomForestClassifier(n_estimators=200),
#         "SVM": SVC(kernel="rbf"),
#         "KNN": KNeighborsClassifier(n_neighbors=3),
#         "LogisticRegression": LogisticRegression(max_iter=2000)
#     }

#     results = {}

#     for name, model in models.items():

#         pipe = Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", model)
#         ])

#         score = cross_val_score(pipe, X, y, cv=cv_folds).mean()
#         results[name] = score

#         print(f"✔ {name} → Accuracy: {score:.4f}")

#     return results


# # ==========================================================
# # 1D-CNN MODEL
# # ==========================================================

# def build_cnn(input_len, num_classes):

#     model = models.Sequential([
#         layers.Input(shape=(input_len, 1)),
#         layers.Conv1D(32, 5, activation='relu'),
#         layers.MaxPooling1D(2),
#         layers.Conv1D(64, 5, activation='relu'),
#         layers.MaxPooling1D(2),
#         layers.Flatten(),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(num_classes, activation="softmax")
#     ])

#     model.compile(optimizer="adam",
#                   loss="sparse_categorical_crossentropy",
#                   metrics=["accuracy"])

#     return model


# def evaluate_cnn(X_raw, y, epochs=25):

#     X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)

#     label_to_int = {lbl: i for i, lbl in enumerate(np.unique(y))}
#     y_int = np.array([label_to_int[l] for l in y])

#     model = build_cnn(input_len=X_raw.shape[1], num_classes=len(label_to_int))

#     history = model.fit(
#         X_raw, y_int,
#         epochs=epochs,
#         batch_size=8,
#         verbose=0
#     )

#     acc = history.history['accuracy'][-1]
#     print(f"✔ 1D-CNN → Accuracy: {acc:.4f}")

#     model.save("best_emg_cnn.h5")
#     print("💾 Saved CNN model → best_emg_cnn.h5")

#     return acc


# # ==========================================================
# # MAIN PROGRAM
# # ==========================================================

# if __name__ == "__main__":

#     print("\n====================================")
#     print("  ADVANCED EMG ML TRAINER + CNN")
#     print("====================================")

#     X_feat, X_raw, y = load_dataset()

#     # Determine CV folds
#     min_samples = min(pd.Series(y).value_counts())
#     cv_folds = max(2, min(5, min_samples))

#     print(f"\n🔍 Using CV={cv_folds}")

#     print("\n🔥 Training ML models...")
#     ml_results = evaluate_ml_models(X_feat, y, cv_folds)

#     print("\n🔥 Training Neural Network (1D-CNN)...")
#     cnn_acc = evaluate_cnn(X_raw, y, epochs=35)

#     ml_results["1D-CNN"] = cnn_acc

#     # Choose best model
#     best_model = max(ml_results, key=ml_results.get)
#     print(f"\n🏆 Best Model: {best_model} (Accuracy={ml_results[best_model]:.4f})")

#     if best_model != "1D-CNN":
#         final_model = Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", {
#                 "RandomForest": RandomForestClassifier(200),
#                 "SVM": SVC(kernel="rbf", probability=True),
#                 "KNN": KNeighborsClassifier(3),
#                 "LogisticRegression": LogisticRegression(max_iter=2000)
#             }[best_model])
#         ])
#         final_model.fit(X_feat, y)
#         joblib.dump(final_model, "best_emg_model.pkl")
#         print("💾 Saved best ML model → best_emg_model.pkl")

#     print("\n🎉 Training Completed Successfully!")







# ------------------------------------------------------------------------------------------------





import os
import glob
import numpy as np
import pandas as pd
import joblib

from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models

import warnings
warnings.filterwarnings("ignore")


# ==============================================================
# FEATURE EXTRACTION
# ==============================================================

def extract_features(window):
    x = np.array(window)
    N = len(x)

    # Time-domain
    mean = np.mean(x)
    std = np.std(x)
    var = np.var(x)
    rms = np.sqrt(np.mean(x**2))
    mav = np.mean(np.abs(x))
    wl = np.sum(np.abs(np.diff(x)))
    zc = np.sum(np.diff(np.sign(x)) != 0)

    # "Slope sign change"
    ssc = np.sum(np.diff(np.sign(np.diff(x))) != 0)

    # Frequency-domain
    yf = np.abs(rfft(x))
    xf = rfftfreq(N, d=1)

    spectral_energy = np.sum(yf**2)
    spectral_centroid = np.sum(xf * yf) / (np.sum(yf) + 1e-8)
    spectral_entropy = entropy((yf + 1e-8) / np.sum(yf))

    return [
        mean, std, var, rms, mav,
        wl, zc, ssc,
        spectral_energy, spectral_centroid, spectral_entropy
    ]


# ==============================================================
# LOAD DATA + SLIDING WINDOWS
# ==============================================================

def load_dataset(folder="emg_dataset", win_size=50, step=25):

    X_feat, X_raw, y = [], [], []

    files = glob.glob(os.path.join(folder, "*.csv"))
    print(f"\n📂 Found {len(files)} CSV gesture files\n")

    for file in files:
        df = pd.read_csv(file)
        gesture = os.path.basename(file).split(".")[0]
        print(f"📥 Loaded: {gesture}")

        sig = df["emg_value"].values

        for i in range(0, len(sig) - win_size, step):
            win = sig[i:i+win_size]

            X_feat.append(extract_features(win))
            X_raw.append(win)
            y.append(gesture)

    X_feat = np.array(X_feat)
    X_raw = np.array(X_raw)
    y = np.array(y)

    print(f"\n📊 Total window samples: {len(X_feat)}")

    return X_feat, X_raw, y


# ==============================================================
# TRAIN ML MODELS
# ==============================================================

def evaluate_ml_models(X, y, cv_folds):
    models = {
        "RandomForest": RandomForestClassifier(300),
        "SVM": SVC(kernel="rbf"),
        "KNN": KNeighborsClassifier(3),
        "LogisticRegression": LogisticRegression(max_iter=3000)
    }

    results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        score = cross_val_score(pipe, X, y, cv=cv_folds).mean()
        results[name] = score

        print(f"✔ {name} Accuracy → {score:.4f}")

    return results


# ==============================================================
# 1D-CNN MODEL
# ==============================================================

def build_cnn(input_len, num_classes):

    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(32, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(),
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_cnn(X_raw, y, epochs=35):
    X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)

    # Encode labels
    labels = sorted(list(np.unique(y)))
    label_to_int = {l: i for i, l in enumerate(labels)}
    y_int = np.array([label_to_int[x] for x in y])

    model = build_cnn(input_len=X_raw.shape[1], num_classes=len(labels))

    history = model.fit(
        X_raw, y_int,
        epochs=epochs,
        batch_size=8,
        verbose=0
    )

    acc = history.history["accuracy"][-1]
    print(f"✔ 1D-CNN Accuracy → {acc:.4f}")

    model.save("best_cnn_model.keras")
    print("💾 Saved CNN → best_cnn_model.keras")

    return acc


# ==============================================================
# LSTM MODEL (powerful for EMG time series)
# ==============================================================

def build_lstm(input_len, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.1),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_lstm(X_raw, y, epochs=40):
    X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)

    labels = sorted(list(np.unique(y)))
    label_to_int = {l: i for i, l in enumerate(labels)}
    y_int = np.array([label_to_int[x] for x in y])

    model = build_lstm(input_len=X_raw.shape[1], num_classes=len(labels))

    history = model.fit(
        X_raw, y_int,
        epochs=epochs,
        batch_size=8,
        verbose=0
    )

    acc = history.history["accuracy"][-1]
    print(f"✔ LSTM Accuracy → {acc:.4f}")

    model.save("best_lstm_model.keras")
    print("💾 Saved LSTM → best_lstm_model.keras")

    return acc


# ==============================================================
# MAIN PROGRAM
# ==============================================================

if __name__ == "__main__":

    print("\n==================================================")
    print("            EMG SUPER TRAINER (ML + CNN + LSTM)")
    print("==================================================\n")

    X_feat, X_raw, y = load_dataset()

    min_class = min(pd.Series(y).value_counts())
    cv_folds = max(2, min(5, min_class))

    print(f"\n🔍 Using CV={cv_folds}\n")

    # ML Models
    print("🔥 Training ML Models...")
    ml_results = evaluate_ml_models(X_feat, y, cv_folds)

    # CNN
    print("\n🔥 Training CNN...")
    cnn_acc = evaluate_cnn(X_raw, y, epochs=40)
    ml_results["CNN"] = cnn_acc

    # LSTM
    print("\n🔥 Training LSTM...")
    lstm_acc = evaluate_lstm(X_raw, y, epochs=40)
    ml_results["LSTM"] = lstm_acc

    # Best Model
    best_model = max(ml_results, key=ml_results.get)
    print(f"\n🏆 BEST MODEL → {best_model} (Accuracy={ml_results[best_model]:.4f})")

    # Save ML model if best
    if best_model not in ["CNN", "LSTM"]:
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", {
                "RandomForest": RandomForestClassifier(300),
                "SVM": SVC(kernel="rbf", probability=True),
                "KNN": KNeighborsClassifier(3),
                "LogisticRegression": LogisticRegression(max_iter=3000)
            }[best_model])
        ])
        final_model.fit(X_feat, y)
        joblib.dump(final_model, "best_ml_model.pkl")
        print("💾 Saved best ML model → best_ml_model.pkl")

    print("\n🎉 TRAINING COMPLETE!")
