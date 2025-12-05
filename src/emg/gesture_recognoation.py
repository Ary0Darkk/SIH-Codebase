# import pygame
# import numpy as np
# import joblib
# import time
# import threading
# from collections import deque
# import random
# import os
# import sys
# import json
# from datetime import datetime
# import serial
# import serial.tools.list_ports

# # ---------- Configuration ----------
# WINDOW_WIDTH = 800
# WINDOW_HEIGHT = 600
# FPS = 60

# # Colors
# BACKGROUND = (20, 25, 40)
# TEXT_COLOR = (240, 240, 240)
# GESTURE_COLORS = {
#     'round': (255, 100, 100),      # Red
#     'shoot': (100, 255, 100),      # Green
#     'up_down': (255, 200, 50),     # Yellow
#     'rest': (150, 150, 150)        # Gray
# }

# # EMG Configuration
# WINDOW_SIZE = 100
# FS = 500

# # ---------- Feature Extraction (Same as Training) ----------
# def extract_simple_features(segment):
#     """Extract features (must match training features)"""
#     segment = np.asarray(segment)
    
#     features = []
    
#     # Basic statistical features
#     features.append(np.mean(np.abs(segment)))                    # MAV
#     features.append(np.mean(segment))                           # Mean
#     features.append(np.median(segment))                         # Median
#     features.append(np.std(segment))                            # Std
#     features.append(np.var(segment))                            # VAR
#     features.append(np.max(segment) - np.min(segment))          # Range
#     features.append(np.max(segment))                            # Max
#     features.append(np.min(segment))                            # Min
    
#     # Zero crossings
#     mean_val = np.mean(segment)
#     diff = segment - mean_val
#     features.append(np.sum((diff[1:] * diff[:-1] < 0) & (np.abs(diff[1:] - diff[:-1]) > 10)))  # ZC
    
#     # Waveform length
#     features.append(np.sum(np.abs(np.diff(segment))))           # WL
    
#     # Root mean square
#     features.append(np.sqrt(np.mean(segment ** 2)))             # RMS
    
#     # Integrated EMG
#     features.append(np.sum(np.abs(segment)))                    # IEMG
    
#     # Percentiles
#     features.append(np.percentile(segment, 25))                 # P25
#     features.append(np.percentile(segment, 75))                 # P75
    
#     # Skewness and Kurtosis
#     try:
#         from scipy import stats
#         features.append(stats.skew(segment))                    # Skewness
#         features.append(stats.kurtosis(segment))                # Kurtosis
#     except:
#         features.extend([0.0, 0.0])
    
#     # Fourier features
#     N = len(segment)
#     if N >= 10:
#         try:
#             fft_vals = np.fft.rfft(segment)
#             mag = np.abs(fft_vals)
#             power = mag ** 2
#             freqs = np.fft.rfftfreq(N, d=1.0 / FS)
            
#             if len(power) > 0 and np.sum(power) > 0:
#                 total_power = np.sum(power) + 1e-12
#                 norm_power = power / total_power
                
#                 # Spectral Centroid
#                 spec_cent = np.sum(freqs * norm_power)
#                 features.append(spec_cent)                     # SpecCent
                
#                 # Spectral Spread
#                 features.append(np.sqrt(np.sum(((freqs - spec_cent) ** 2) * norm_power)))  # SpecSpread
                
#                 # Spectral Entropy
#                 features.append(-np.sum(norm_power * np.log2(norm_power + 1e-12)))  # SpecEnt
#             else:
#                 features.extend([0.0, 0.0, 0.0])
#         except:
#             features.extend([0.0, 0.0, 0.0])
#     else:
#         features.extend([0.0, 0.0, 0.0])
    
#     return features

# # ---------- Gesture Tester Class ----------
# class GestureTester:
#     def __init__(self, model_dir=None):
#         pygame.init()
#         self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
#         pygame.display.set_caption("EMG Gesture Recognition Test")
#         self.clock = pygame.time.Clock()
#         self.running = True
        
#         # Fonts
#         self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
#         self.font = pygame.font.SysFont('Arial', 24)
#         self.small_font = pygame.font.SysFont('Arial', 18)
        
#         # Load model
#         self.model = None
#         self.scaler = None
#         self.encoder = None
#         self.feature_names = None
#         self.gesture_labels = []
        
#         if model_dir and os.path.exists(model_dir):
#             self.load_model(model_dir)
#         else:
#             # Find most recent model
#             model_dirs = [d for d in os.listdir('.') if d.startswith('emg_gesture_model_') and os.path.isdir(d)]
#             if model_dirs:
#                 model_dir = sorted(model_dirs, reverse=True)[0]
#                 self.load_model(model_dir)
#             else:
#                 print("❌ No trained model found!")
#                 print("Please train a model first using emg_model_trainer.py")
#                 pygame.quit()
#                 sys.exit()
        
#         # Data buffer
#         self.data_buffer = deque(maxlen=WINDOW_SIZE * 3)
#         self.predictions = deque(maxlen=50)  # Store last 50 predictions
#         self.confidences = deque(maxlen=50)
        
#         # Current state
#         self.current_gesture = 'rest'
#         self.current_confidence = 0.0
#         self.emg_history = deque(maxlen=200)  # For visualization
        
#         # EMG source
#         self.use_real_emg = False
#         self.serial_conn = None
        
#         # Simulation
#         self.simulated_gesture = 'rest'
#         self.manual_gesture = None
        
#         # Statistics
#         self.prediction_count = 0
#         self.correct_count = 0
        
#         # Start EMG thread
#         self.emg_thread = threading.Thread(target=self.emg_sampling_loop, daemon=True)
#         self.emg_thread.start()
        
#         print("\n" + "="*70)
#         print("EMG GESTURE RECOGNITION TESTER")
#         print("="*70)
#         print(f"Model: {os.path.basename(model_dir)}")
#         print(f"Gestures: {self.gesture_labels}")
#         print(f"Sampling rate: {FS} Hz")
#         print(f"Window size: {WINDOW_SIZE} samples")
#         print("\nControls:")
#         print("  1: Simulate ROUND gesture")
#         print("  2: Simulate SHOOT gesture")
#         print("  3: Simulate UP-DOWN gesture")
#         print("  0: Simulate REST")
#         print("  C: Connect to real EMG")
#         print("  SPACE: Toggle auto-simulation")
#         print("  ESC: Exit")
#         print("="*70)
    
#     def load_model(self, model_dir):
#         """Load the trained model"""
#         try:
#             self.model = joblib.load(os.path.join(model_dir, 'model.pkl'))
#             self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
#             self.encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
#             self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
            
#             with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
#                 metadata = json.load(f)
#             self.gesture_labels = metadata['gesture_labels']
            
#             print(f"✅ Model loaded successfully!")
#             print(f"   Model type: {type(self.model).__name__}")
#             print(f"   Gestures: {self.gesture_labels}")
#             print(f"   Features: {len(self.feature_names)}")
            
#             return True
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             return False
    
#     def connect_real_emg(self, port=None, baudrate=115200):
#         """Connect to real EMG hardware"""
#         try:
#             if port is None:
#                 ports = [p.device for p in serial.tools.list_ports.comports()]
#                 if not ports:
#                     print("No serial ports found!")
#                     return False
#                 port = ports[0]
            
#             self.serial_conn = serial.Serial(port=port, baudrate=baudrate, timeout=1)
#             time.sleep(2)
#             self.serial_conn.reset_input_buffer()
#             self.use_real_emg = True
            
#             print(f"✅ Connected to EMG sensor on {port}")
#             return True
            
#         except Exception as e:
#             print(f"❌ EMG connection failed: {e}")
#             self.use_real_emg = False
#             return False
    
#     def read_real_emg(self):
#         """Read real EMG data"""
#         if self.serial_conn and self.serial_conn.in_waiting > 0:
#             try:
#                 raw = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
#                 if not raw:
#                     return 50.0
                
#                 if ':' in raw:
#                     raw = raw.split(':')[-1].strip()
                
#                 try:
#                     return float(raw)
#                 except ValueError:
#                     return 50.0
#             except:
#                 return 50.0
#         return 50.0
    
#     def simulate_emg_sample(self):
#         """Generate simulated EMG based on current gesture"""
#         t = time.time()
        
#         # Use manual gesture if set, otherwise auto-cycle
#         if self.manual_gesture:
#             gesture = self.manual_gesture
#         else:
#             # Auto-cycle every 4 seconds
#             gestures = self.gesture_labels
#             idx = int(t / 4) % len(gestures)
#             gesture = gestures[idx]
        
#         # Generate pattern based on gesture
#         if gesture == 'round':
#             return 250 + 150 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 25)
#         elif gesture == 'shoot':
#             if np.random.random() < 0.2:
#                 return 350 + np.random.normal(0, 40)
#             else:
#                 return 100 + np.random.normal(0, 15)
#         elif gesture == 'up_down':
#             return 200 + 120 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 20)
#         else:  # rest
#             return 50 + np.random.normal(0, 10)
    
#     def emg_sampling_loop(self):
#         """Continuously sample EMG data"""
#         while self.running:
#             if self.use_real_emg:
#                 sample = self.read_real_emg()
#             else:
#                 sample = self.simulate_emg_sample()
            
#             self.data_buffer.append(sample)
#             self.emg_history.append(sample)
#             time.sleep(1.0 / FS)
    
#     def predict_gesture(self):
#         """Predict gesture from current EMG buffer"""
#         if len(self.data_buffer) < WINDOW_SIZE:
#             return 'rest', 0.0
        
#         try:
#             # Get latest window
#             segment = list(self.data_buffer)[-WINDOW_SIZE:]
#             segment = np.array(segment)
            
#             # Remove DC offset
#             segment = segment - np.mean(segment)
            
#             # Extract features
#             features = extract_simple_features(segment)
#             features = np.array(features).reshape(1, -1)
            
#             # Scale features
#             features_scaled = self.scaler.transform(features)
            
#             # Predict
#             pred_encoded = self.model.predict(features_scaled)[0]
#             gesture = self.encoder.inverse_transform([pred_encoded])[0]
            
#             # Get confidence
#             if hasattr(self.model, 'predict_proba'):
#                 probs = self.model.predict_proba(features_scaled)[0]
#                 confidence = float(np.max(probs))
#             else:
#                 confidence = 0.9
            
#             # Update statistics
#             self.prediction_count += 1
            
#             # Check if prediction matches simulated gesture
#             if not self.use_real_emg and self.manual_gesture:
#                 if gesture == self.manual_gesture:
#                     self.correct_count += 1
            
#             return gesture, confidence
            
#         except Exception as e:
#             # print(f"Prediction error: {e}")
#             return 'rest', 0.0
    
#     def draw_emg_signal(self):
#         """Draw the EMG signal waveform"""
#         if not self.emg_history:
#             return
        
#         # Create surface for EMG plot
#         plot_width = WINDOW_WIDTH - 40
#         plot_height = 150
#         plot_x = 20
#         plot_y = 100
        
#         # Draw background
#         pygame.draw.rect(self.screen, (30, 35, 50), 
#                         (plot_x, plot_y, plot_width, plot_height))
#         pygame.draw.rect(self.screen, (60, 70, 90), 
#                         (plot_x, plot_y, plot_width, plot_height), 2)
        
#         # Draw EMG signal
#         if len(self.emg_history) > 1:
#             points = []
#             for i, value in enumerate(self.emg_history):
#                 x = plot_x + (i / len(self.emg_history)) * plot_width
#                 y = plot_y + plot_height // 2 - (value - 512) / 1024 * plot_height
#                 points.append((x, y))
            
#             if len(points) > 1:
#                 pygame.draw.lines(self.screen, (0, 200, 255), False, points, 2)
        
#         # Draw zero line
#         zero_y = plot_y + plot_height // 2
#         pygame.draw.line(self.screen, (100, 100, 100), 
#                         (plot_x, zero_y), (plot_x + plot_width, zero_y), 1)
        
#         # Draw label
#         label = self.small_font.render("EMG Signal (Real-time)", True, (200, 200, 200))
#         self.screen.blit(label, (plot_x, plot_y - 25))
    
#     def draw_gesture_display(self):
#         """Draw the current gesture prediction"""
#         # Main gesture display
#         display_radius = 80
#         center_x = WINDOW_WIDTH // 2
#         center_y = WINDOW_HEIGHT // 2 + 50
        
#         # Draw circle background
#         pygame.draw.circle(self.screen, (40, 45, 60), 
#                           (center_x, center_y), display_radius)
#         pygame.draw.circle(self.screen, (70, 80, 100), 
#                           (center_x, center_y), display_radius, 3)
        
#         # Draw gesture circle with confidence-based alpha
#         gesture_color = GESTURE_COLORS.get(self.current_gesture, (150, 150, 150))
#         alpha_color = (*gesture_color, int(self.current_confidence * 255))
        
#         # Create surface for alpha blending
#         gesture_surf = pygame.Surface((display_radius * 2, display_radius * 2), pygame.SRCALPHA)
#         pygame.draw.circle(gesture_surf, alpha_color, 
#                           (display_radius, display_radius), 
#                           int(display_radius * self.current_confidence))
#         self.screen.blit(gesture_surf, (center_x - display_radius, center_y - display_radius))
        
#         # Draw gesture text
#         gesture_text = self.title_font.render(self.current_gesture.upper(), True, gesture_color)
#         self.screen.blit(gesture_text, 
#                         (center_x - gesture_text.get_width() // 2, 
#                          center_y - gesture_text.get_height() // 2))
        
#         # Draw confidence
#         conf_text = self.font.render(f"{self.current_confidence:.1%}", True, TEXT_COLOR)
#         self.screen.blit(conf_text, 
#                         (center_x - conf_text.get_width() // 2, 
#                          center_y + display_radius + 10))
    
#     def draw_prediction_history(self):
#         """Draw history of recent predictions"""
#         if not self.predictions:
#             return
        
#         history_width = WINDOW_WIDTH - 40
#         history_height = 100
#         history_x = 20
#         history_y = WINDOW_HEIGHT - history_height - 20
        
#         # Draw background
#         pygame.draw.rect(self.screen, (30, 35, 50), 
#                         (history_x, history_y, history_width, history_height))
#         pygame.draw.rect(self.screen, (60, 70, 90), 
#                         (history_x, history_y, history_width, history_height), 2)
        
#         # Draw prediction bars
#         bar_width = history_width / len(self.predictions)
#         for i, (gesture, confidence) in enumerate(zip(self.predictions, self.confidences)):
#             color = GESTURE_COLORS.get(gesture, (150, 150, 150))
#             bar_height = confidence * history_height
            
#             x = history_x + i * bar_width
#             y = history_y + history_height - bar_height
            
#             pygame.draw.rect(self.screen, color, 
#                             (x, y, bar_width, bar_height))
        
#         # Draw label
#         label = self.small_font.render("Prediction History (Last 50)", True, (200, 200, 200))
#         self.screen.blit(label, (history_x, history_y - 25))
    
#     def draw_statistics(self):
#         """Draw prediction statistics"""
#         stats_y = 280
        
#         # Mode indicator
#         mode_text = self.font.render(
#             f"Mode: {'REAL EMG' if self.use_real_emg else 'SIMULATION'}", 
#             True, (100, 200, 255) if self.use_real_emg else (255, 200, 100))
#         self.screen.blit(mode_text, (20, stats_y))
        
#         # Prediction count
#         count_text = self.font.render(f"Predictions: {self.prediction_count}", True, TEXT_COLOR)
#         self.screen.blit(count_text, (20, stats_y + 35))
        
#         # Accuracy (only in simulation with manual gestures)
#         if not self.use_real_emg and self.manual_gesture and self.prediction_count > 0:
#             accuracy = self.correct_count / self.prediction_count
#             acc_text = self.font.render(f"Accuracy: {accuracy:.1%}", True, 
#                                        (100, 255, 100) if accuracy > 0.9 else (255, 200, 100))
#             self.screen.blit(acc_text, (20, stats_y + 70))
        
#         # Current simulated gesture
#         if not self.use_real_emg:
#             sim_text = self.font.render(
#                 f"Simulated: {self.manual_gesture if self.manual_gesture else 'AUTO-CYCLE'}", 
#                 True, (200, 200, 200))
#             self.screen.blit(sim_text, (20, stats_y + 105))
    
#     def draw_legend(self):
#         """Draw gesture color legend"""
#         legend_x = WINDOW_WIDTH - 220
#         legend_y = 100
        
#         # Draw legend background
#         pygame.draw.rect(self.screen, (30, 35, 50), 
#                         (legend_x - 10, legend_y - 10, 210, 160))
#         pygame.draw.rect(self.screen, (60, 70, 90), 
#                         (legend_x - 10, legend_y - 10, 210, 160), 2)
        
#         title = self.small_font.render("GESTURE LEGEND", True, (200, 200, 200))
#         self.screen.blit(title, (legend_x, legend_y))
        
#         # Draw each gesture
#         y_offset = legend_y + 35
#         for gesture, color in GESTURE_COLORS.items():
#             if gesture in self.gesture_labels:
#                 # Color square
#                 pygame.draw.rect(self.screen, color, (legend_x, y_offset, 20, 20))
#                 pygame.draw.rect(self.screen, (100, 100, 100), (legend_x, y_offset, 20, 20), 1)
                
#                 # Gesture name
#                 name_text = self.small_font.render(f"{gesture.upper()}", True, color)
#                 self.screen.blit(name_text, (legend_x + 30, y_offset + 2))
                
#                 y_offset += 30
    
#     def draw_controls(self):
#         """Draw control instructions"""
#         controls = [
#             "CONTROLS:",
#             "1: Simulate ROUND gesture",
#             "2: Simulate SHOOT gesture", 
#             "3: Simulate UP-DOWN gesture",
#             "0: Simulate REST",
#             "C: Connect to real EMG",
#             "SPACE: Toggle auto-simulation",
#             "ESC: Exit"
#         ]
        
#         y_offset = WINDOW_HEIGHT - 200
#         for i, control in enumerate(controls):
#             color = TEXT_COLOR if i == 0 else (180, 180, 180)
#             font = self.font if i == 0 else self.small_font
#             text = font.render(control, True, color)
#             self.screen.blit(text, (20, y_offset))
#             y_offset += 30 if i == 0 else 25
    
#     def draw(self):
#         """Draw everything"""
#         # Clear screen
#         self.screen.fill(BACKGROUND)
        
#         # Draw title
#         title = self.title_font.render("EMG GESTURE RECOGNITION TEST", True, (255, 255, 255))
#         self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
        
#         # Draw all components
#         self.draw_emg_signal()
#         self.draw_gesture_display()
#         self.draw_prediction_history()
#         self.draw_statistics()
#         self.draw_legend()
#         self.draw_controls()
        
#         pygame.display.flip()
    
#     def handle_events(self):
#         """Handle pygame events"""
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 self.running = False
            
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     self.running = False
                
#                 elif event.key == pygame.K_1:
#                     self.manual_gesture = 'round'
#                     print("Simulating: ROUND gesture")
#                 elif event.key == pygame.K_2:
#                     self.manual_gesture = 'shoot'
#                     print("Simulating: SHOOT gesture")
#                 elif event.key == pygame.K_3:
#                     self.manual_gesture = 'up_down'
#                     print("Simulating: UP-DOWN gesture")
#                 elif event.key == pygame.K_0:
#                     self.manual_gesture = 'rest'
#                     print("Simulating: REST")
                
#                 elif event.key == pygame.K_c:
#                     if not self.use_real_emg:
#                         self.connect_real_emg()
                
#                 elif event.key == pygame.K_SPACE:
#                     if self.manual_gesture:
#                         self.manual_gesture = None
#                         print("Auto-simulation: ON")
#                     else:
#                         self.manual_gesture = 'rest'
#                         print("Manual simulation: ON")
    
#     def run(self):
#         """Main loop"""
#         last_prediction_time = 0
        
#         while self.running:
#             # Handle events
#             self.handle_events()
            
#             # Make predictions at 10 Hz
#             current_time = time.time()
#             if current_time - last_prediction_time > 0.1:  # 10 Hz
#                 gesture, confidence = self.predict_gesture()
#                 self.current_gesture = gesture
#                 self.current_confidence = confidence
#                 self.predictions.append(gesture)
#                 self.confidences.append(confidence)
#                 last_prediction_time = current_time
            
#             # Draw everything
#             self.draw()
            
#             # Control frame rate
#             self.clock.tick(FPS)
        
#         # Cleanup
#         pygame.quit()
#         print("\n" + "="*70)
#         print("TEST COMPLETED")
#         print("="*70)
#         if self.prediction_count > 0:
#             print(f"Total predictions: {self.prediction_count}")
#             if self.correct_count > 0:
#                 accuracy = self.correct_count / self.prediction_count
#                 print(f"Accuracy: {accuracy:.1%}")
#         print("="*70)

# # ---------- Main Function ----------
# def main():
#     print("\n" + "="*70)
#     print("EMG GESTURE RECOGNITION TESTER")
#     print("="*70)
    
#     # Find trained model
#     model_dirs = [d for d in os.listdir('.') if d.startswith('emg_gesture_model_') and os.path.isdir(d)]
    
#     if not model_dirs:
#         print("❌ No trained models found!")
#         print("Please train a model first:")
#         print("  1. Run: python emg_custom_gesture_collector.py")
#         print("  2. Collect data for round, shoot, up_down gestures")
#         print("  3. Run: python emg_model_trainer.py")
#         print("  4. Train the model")
#         return
    
#     # Use most recent model
#     model_dir = sorted(model_dirs, reverse=True)[0]
    
#     try:
#         with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
#             metadata = json.load(f)
        
#         print(f"✅ Found model: {model_dir}")
#         print(f"   Gestures: {metadata['gesture_labels']}")
#         print(f"   Model: {metadata['model_type']}")
#         print(f"   Trained: {metadata['training_date']}")
        
#         print("\n" + "="*70)
#         print("Starting gesture recognition test...")
#         print("The tester will show you exactly what gestures the model detects.")
#         print("You can simulate gestures with keyboard keys 1, 2, 3, 0")
#         print("="*70)
        
#         input("\nPress Enter to start the test...")
        
#         # Run the tester
#         tester = GestureTester(model_dir)
#         tester.run()
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         print("Make sure you have collected and trained data for your gestures.")

# if __name__ == "__main__":
#     main()





# ----------------------------------------------------------------





import numpy as np
import joblib
import time
import threading
from collections import deque
import os
import json
import serial
import serial.tools.list_ports
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

# ---------- Configuration ----------
WINDOW_SIZE = 100  # Must match your trained model
FS = 500           # Sampling rate (Hz)
BUFFER_SIZE = WINDOW_SIZE * 3
PREDICTION_RATE = 10  # Predictions per second
PLOT_HISTORY = 100    # Number of samples to plot

# Gesture colors for terminal
GESTURE_COLORS = {
    'round': '\033[91m',    # Red
    'shoot': '\033[92m',    # Green
    'up_down': '\033[93m',  # Yellow
    'rest': '\033[90m'      # Gray
}
RESET_COLOR = '\033[0m'

# ASCII art for gestures
GESTURE_ART = {
    'round': """
      ╭──────╮
     ╭│  ○○  │╮
    ╭││      ││╮
    ││  ROUND  ││
    ╰││      ││╯
     ╰│  ○○  │╯
      ╰──────╯""",
    
    'shoot': """
      ▄▄▄▄▄▄▄▄▄
      █ SHOOT █
      ▀▀▀▀▀▀▀▀▀
         / \\
        /   \\
       /  |  \\
      /   |   \\
     ▔▔▔▔▔▔▔▔▔▔▔""",
    
    'up_down': """
      ╭─────╮
      │ UP  │
      │ DOWN│
      ╰─────╯
        │ │
        │ │
       ╭╯ ╰╮
      ╭╯   ╰╮""",
    
    'rest': """
      ╭─────╮
      │     │
      │ REST│
      │     │
      ╰─────╯
        ───
        ───
        ───"""
}

# ---------- Feature Extraction (Same as Training) ----------
def extract_features(segment):
    """Extract features from EMG segment (same as training)"""
    segment = np.asarray(segment)
    
    features = []
    
    # Basic statistical features
    features.append(np.mean(np.abs(segment)))                    # MAV
    features.append(np.mean(segment))                           # Mean
    features.append(np.median(segment))                         # Median
    features.append(np.std(segment))                            # Std
    features.append(np.var(segment))                            # VAR
    features.append(np.max(segment) - np.min(segment))          # Range
    features.append(np.max(segment))                            # Max
    features.append(np.min(segment))                            # Min
    
    # Zero crossings
    mean_val = np.mean(segment)
    diff = segment - mean_val
    features.append(np.sum((diff[1:] * diff[:-1] < 0) & (np.abs(diff[1:] - diff[:-1]) > 10)))  # ZC
    
    # Waveform length
    features.append(np.sum(np.abs(np.diff(segment))))           # WL
    
    # Root mean square
    features.append(np.sqrt(np.mean(segment ** 2)))             # RMS
    
    # Integrated EMG
    features.append(np.sum(np.abs(segment)))                    # IEMG
    
    # Percentiles
    features.append(np.percentile(segment, 25))                 # P25
    features.append(np.percentile(segment, 75))                 # P75
    
    # Skewness and Kurtosis
    features.append(stats.skew(segment))                       # Skewness
    features.append(stats.kurtosis(segment))                   # Kurtosis
    
    # Fourier features
    N = len(segment)
    if N >= 10:
        try:
            fft_vals = np.fft.rfft(segment)
            mag = np.abs(fft_vals)
            power = mag ** 2
            freqs = np.fft.rfftfreq(N, d=1.0 / FS)
            
            if len(power) > 0 and np.sum(power) > 0:
                total_power = np.sum(power) + 1e-12
                norm_power = power / total_power
                
                # Spectral Centroid
                spec_cent = np.sum(freqs * norm_power)
                features.append(spec_cent)                     # SpecCent
                
                # Spectral Spread
                features.append(np.sqrt(np.sum(((freqs - spec_cent) ** 2) * norm_power)))  # SpecSpread
                
                # Spectral Entropy
                features.append(-np.sum(norm_power * np.log2(norm_power + 1e-12)))  # SpecEnt
            else:
                features.extend([0.0, 0.0, 0.0])
        except:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])
    
    return features

# ---------- EMG Gesture Detector Class ----------
class EMGGestureDetector:
    def __init__(self, model_dir=None):
        # Load model
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.gesture_labels = []
        
        if model_dir and os.path.exists(model_dir):
            self.load_model(model_dir)
        else:
            # Find most recent model
            model_dirs = [d for d in os.listdir('.') if d.startswith('emg_gesture_model_') and os.path.isdir(d)]
            if model_dirs:
                model_dir = sorted(model_dirs, reverse=True)[0]
                self.load_model(model_dir)
            else:
                print("❌ No trained model found!")
                print("Train a model first: python emg_model_trainer.py")
                sys.exit(1)
        
        # Data buffers
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.prediction_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)
        self.emg_history = deque(maxlen=PLOT_HISTORY)
        
        # Current state
        self.current_gesture = 'rest'
        self.current_confidence = 0.0
        self.last_prediction = None
        self.prediction_count = 0
        self.correct_count = 0
        
        # EMG connection
        self.serial_conn = None
        self.connected = False
        
        # Thread control
        self.running = True
        self.prediction_thread = None
        
        # For ASCII visualization
        self.gesture_changes = 0
        
        print("\n" + "="*70)
        print("REAL-TIME EMG GESTURE DETECTOR")
        print("="*70)
        print(f"Model: {os.path.basename(model_dir)}")
        print(f"Gestures: {self.gesture_labels}")
        print(f"Sampling: {FS} Hz, Window: {WINDOW_SIZE} samples")
        print("="*70)
    
    def load_model(self, model_dir):
        """Load the trained model"""
        try:
            self.model = joblib.load(os.path.join(model_dir, 'model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            self.encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
            
            with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            self.gesture_labels = metadata['gesture_labels']
            
            print(f"✅ Model loaded: {metadata['model_type']}")
            print(f"   Gestures: {self.gesture_labels}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Trained: {metadata['training_date']}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def connect_emg(self, port=None, baudrate=115200):
        """Connect to EMG sensor"""
        try:
            if port is None:
                ports = [p.device for p in serial.tools.list_ports.comports()]
                if not ports:
                    print("No serial ports found!")
                    return False
                port = ports[0]
            
            print(f"Connecting to {port}...")
            self.serial_conn = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            self.connected = True
            
            print(f"✅ Connected to EMG sensor on {port}")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    def read_emg_sample(self):
        """Read EMG sample from serial port"""
        if not self.connected or not self.serial_conn:
            return None
        
        try:
            if self.serial_conn.in_waiting > 0:
                raw = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    return 0.0
                
                # Parse different formats
                if ':' in raw:
                    raw = raw.split(':')[-1].strip()
                
                try:
                    value = float(raw)
                    # Normalize to reasonable range
                    return max(0, min(1023, value))
                except ValueError:
                    return 0.0
        except Exception as e:
            print(f"Read error: {e}")
            return None
        
        return 0.0
    
    def sampling_loop(self):
        """Continuously read EMG samples"""
        print("📡 Starting EMG sampling...")
        
        while self.running:
            if self.connected:
                sample = self.read_emg_sample()
                if sample is not None:
                    self.data_buffer.append(sample)
                    self.emg_history.append(sample)
            else:
                # If not connected, simulate data for demo
                time.sleep(0.002)  # 500 Hz
                self.data_buffer.append(50 + np.random.normal(0, 10))
                self.emg_history.append(50 + np.random.normal(0, 10))
    
    def predict_gesture(self):
        """Predict gesture from current EMG buffer"""
        if len(self.data_buffer) < WINDOW_SIZE:
            return 'rest', 0.0
        
        try:
            # Get latest window
            segment = list(self.data_buffer)[-WINDOW_SIZE:]
            segment = np.array(segment)
            
            # Remove DC offset
            segment = segment - np.mean(segment)
            
            # Extract features
            features = extract_features(segment)
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            pred_encoded = self.model.predict(features_scaled)[0]
            gesture = self.encoder.inverse_transform([pred_encoded])[0]
            
            # Get confidence
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probs))
            else:
                confidence = 0.9
            
            self.prediction_count += 1
            
            return gesture, confidence
            
        except Exception as e:
            # print(f"Prediction error: {e}")
            return 'rest', 0.0
    
    def prediction_loop(self):
        """Continuously make predictions"""
        print("🤖 Starting gesture prediction...")
        
        while self.running:
            gesture, confidence = self.predict_gesture()
            
            # Update if gesture changed or confidence is high
            if gesture != self.last_prediction or confidence > 0.8:
                self.current_gesture = gesture
                self.current_confidence = confidence
                self.last_prediction = gesture
                
                self.prediction_history.append(gesture)
                self.confidence_history.append(confidence)
                
                # Count gesture changes
                if len(self.prediction_history) > 1 and self.prediction_history[-1] != self.prediction_history[-2]:
                    self.gesture_changes += 1
            
            time.sleep(1.0 / PREDICTION_RATE)
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def draw_ascii_meter(self, value, width=50):
        """Draw ASCII progress bar"""
        filled = int(width * value)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {value:.1%}"
    
    def draw_ascii_graph(self, data, height=10, width=50):
        """Draw ASCII graph of EMG signal"""
        if len(data) < 2:
            return "No data"
        
        # Normalize data to fit in height
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            return "─" * width
        
        normalized = [(val - min_val) / (max_val - min_val) * (height - 1) for val in data[-width:]]
        
        # Create ASCII graph
        graph_lines = [''] * height
        for i in range(height):
            for j, val in enumerate(normalized):
                if int(val) == (height - 1 - i):
                    graph_lines[i] += '●'
                elif int(val) > (height - 1 - i):
                    graph_lines[i] += '│'
                else:
                    graph_lines[i] += ' '
        
        return '\n'.join(graph_lines)
    
    def draw_gesture_probabilities(self, segment):
        """Draw probability bars for each gesture"""
        try:
            # Get probabilities for all gestures
            features = extract_features(segment)
            features = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features_scaled)[0]
                
                bars = []
                for i, gesture in enumerate(self.encoder.classes_):
                    prob = probs[i]
                    color = GESTURE_COLORS.get(gesture, RESET_COLOR)
                    bar = self.draw_ascii_meter(prob, 20)
                    bars.append(f"{color}{gesture:10}{RESET_COLOR} {bar}")
                
                return '\n'.join(bars)
        except:
            pass
        
        return "Cannot display probabilities"
    
    def display_dashboard(self):
        """Display real-time dashboard in terminal"""
        self.clear_screen()
        
        # Header
        print("\n" + "="*80)
        print(f"{'EMG GESTURE DETECTION - REAL TIME':^80}")
        print("="*80)
        
        # Connection status
        status_color = '\033[92m' if self.connected else '\033[91m'
        status_text = "CONNECTED" if self.connected else "DISCONNECTED"
        print(f"\n📡 Status: {status_color}{status_text}{RESET_COLOR}")
        print(f"📊 Samples: {len(self.data_buffer)}/{BUFFER_SIZE}")
        print(f"🔢 Predictions: {self.prediction_count}")
        print(f"🔄 Gesture changes: {self.gesture_changes}")
        print("-"*80)
        
        # Current gesture with ASCII art
        color = GESTURE_COLORS.get(self.current_gesture, RESET_COLOR)
        print(f"\n🎯 CURRENT GESTURE: {color}{self.current_gesture.upper()}{RESET_COLOR}")
        print(f"📈 Confidence: {self.draw_ascii_meter(self.current_confidence)}")
        
        # Display ASCII art for current gesture
        if self.current_gesture in GESTURE_ART:
            art_lines = GESTURE_ART[self.current_gesture].split('\n')
            for line in art_lines:
                print(f"{color}{line}{RESET_COLOR}")
        
        print("\n" + "-"*80)
        
        # EMG Signal Graph
        if len(self.emg_history) > 10:
            print("📈 EMG SIGNAL (Last 50 samples):")
            print(self.draw_ascii_graph(list(self.emg_history), height=8, width=50))
            print(f"Min: {min(list(self.emg_history)[-50:]):6.1f} | "
                  f"Max: {max(list(self.emg_history)[-50:]):6.1f} | "
                  f"Avg: {np.mean(list(self.emg_history)[-50:]):6.1f}")
        
        print("\n" + "-"*80)
        
        # Gesture Probabilities (if available)
        if len(self.data_buffer) >= WINDOW_SIZE:
            segment = list(self.data_buffer)[-WINDOW_SIZE:]
            print("📊 GESTURE PROBABILITIES:")
            print(self.draw_gesture_probabilities(segment))
        
        print("\n" + "-"*80)
        
        # Recent predictions history
        if self.prediction_history:
            print("🕐 RECENT PREDICTIONS (Last 10):")
            recent_preds = list(self.prediction_history)[-10:]
            recent_confs = list(self.confidence_history)[-10:]
            
            for i, (gesture, conf) in enumerate(zip(recent_preds, recent_confs)):
                color = GESTURE_COLORS.get(gesture, RESET_COLOR)
                time_ago = f"{9-i}s ago" if i < 9 else "just now"
                bar = self.draw_ascii_meter(conf, 20)
                print(f"  {time_ago:10} → {color}{gesture:10}{RESET_COLOR} {bar}")
        
        print("\n" + "="*80)
        print("🎮 Perform gestures: ROUND | SHOOT | UP-DOWN | REST")
        print("⏹️  Press Ctrl+C to stop")
        print("="*80)
    
    def start_matplotlib_plot(self):
        """Start real-time matplotlib plot"""
        plt.ion()  # Interactive mode on
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # EMG signal plot
        emg_line, = ax1.plot([], [], 'b-', linewidth=2)
        ax1.set_title('Real-time EMG Signal', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('EMG Value')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1023)
        
        # Gesture probability plot
        x_pos = np.arange(len(self.gesture_labels))
        bars = ax2.bar(x_pos, [0]*len(self.gesture_labels), 
                      color=['red', 'green', 'yellow', 'gray'][:len(self.gesture_labels)])
        ax2.set_title('Gesture Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Gesture')
        ax2.set_ylabel('Probability')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.gesture_labels)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        return fig, ax1, ax2, emg_line, bars
    
    def update_matplotlib_plot(self, fig, ax1, ax2, emg_line, bars):
        """Update matplotlib plot with new data"""
        if not self.running:
            return
        
        # Update EMG signal
        if self.emg_history:
            emg_data = list(self.emg_history)
            emg_line.set_data(range(len(emg_data)), emg_data)
            ax1.set_xlim(0, len(emg_data))
            ax1.set_ylim(min(emg_data) - 10, max(emg_data) + 10)
        
        # Update gesture probabilities
        if len(self.data_buffer) >= WINDOW_SIZE:
            try:
                segment = list(self.data_buffer)[-WINDOW_SIZE:]
                features = extract_features(segment)
                features = np.array(features).reshape(1, -1)
                features_scaled = self.scaler.transform(features)
                
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(features_scaled)[0]
                    
                    for i, bar in enumerate(bars):
                        if i < len(probs):
                            bar.set_height(probs[i])
                            # Color based on probability
                            if probs[i] > 0.7:
                                bar.set_color('green')
                            elif probs[i] > 0.4:
                                bar.set_color('orange')
                            else:
                                bar.set_color('red')
            
            except:
                pass
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    def run_terminal_mode(self):
        """Run in terminal-only mode"""
        print("\n🚀 Starting real-time gesture detection...")
        print("📡 Connect your EMG band to USB port")
        print("🎮 Perform gestures: ROUND, SHOOT, UP-DOWN, REST")
        print("⏳ Loading... (collecting initial data)")
        
        # Start threads
        sampling_thread = threading.Thread(target=self.sampling_loop, daemon=True)
        prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
        
        sampling_thread.start()
        time.sleep(0.5)  # Let some data accumulate
        prediction_thread.start()
        
        try:
            # Initial wait to fill buffer
            while len(self.data_buffer) < WINDOW_SIZE and self.running:
                print(f"  Collecting data: {len(self.data_buffer)}/{WINDOW_SIZE} samples", end='\r')
                time.sleep(0.1)
            
            print("\n✅ Ready! Starting real-time detection...")
            time.sleep(1)
            
            # Main display loop
            while self.running:
                self.display_dashboard()
                time.sleep(0.5)  # Update twice per second
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")
            self.running = False
            time.sleep(0.5)
        
        self.show_summary()
    
    def run_with_matplotlib(self):
        """Run with matplotlib visualization"""
        print("\n📊 Starting with Matplotlib visualization...")
        
        # Setup plot
        fig, ax1, ax2, emg_line, bars = self.start_matplotlib_plot()
        
        # Start threads
        sampling_thread = threading.Thread(target=self.sampling_loop, daemon=True)
        prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
        
        sampling_thread.start()
        time.sleep(0.5)
        prediction_thread.start()
        
        try:
            # Wait for initial data
            while len(self.data_buffer) < WINDOW_SIZE and self.running:
                print(f"Collecting data: {len(self.data_buffer)}/{WINDOW_SIZE}", end='\r')
                time.sleep(0.1)
            
            print("\n✅ Ready! Close plot window to stop.")
            
            # Update plot in main thread
            while self.running and plt.fignum_exists(fig.number):
                self.update_matplotlib_plot(fig, ax1, ax2, emg_line, bars)
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            self.running = False
            plt.close('all')
        
        self.show_summary()
    
    def show_summary(self):
        """Show summary statistics"""
        print("\n" + "="*80)
        print("📊 DETECTION SUMMARY")
        print("="*80)
        
        if self.prediction_count > 0:
            # Calculate gesture distribution
            gesture_counts = {}
            for gesture in self.prediction_history:
                if gesture in gesture_counts:
                    gesture_counts[gesture] += 1
                else:
                    gesture_counts[gesture] = 1
            
            print(f"\nTotal predictions: {self.prediction_count}")
            print(f"Total gesture changes: {self.gesture_changes}")
            
            print("\nGesture distribution:")
            for gesture, count in gesture_counts.items():
                percentage = (count / len(self.prediction_history)) * 100
                color = GESTURE_COLORS.get(gesture, RESET_COLOR)
                print(f"  {color}{gesture:10}{RESET_COLOR}: {count:4} times ({percentage:5.1f}%)")
        
        print("\n" + "="*80)
        print("✅ Detection completed!")
        print("="*80)

# ---------- Main Function ----------
def main():
    print("\n" + "="*80)
    print(f"{'REAL-TIME EMG GESTURE DETECTOR':^80}")
    print("="*80)
    print("This program reads EMG data from your band and predicts gestures")
    print("using your trained machine learning model.")
    print("="*80)
    
    # Find trained model
    model_dirs = [d for d in os.listdir('.') if d.startswith('emg_gesture_model_') and os.path.isdir(d)]
    
    if not model_dirs:
        print("❌ No trained models found!")
        print("\nPlease train a model first:")
        print("1. Run: python emg_custom_gesture_collector.py")
        print("2. Collect data for gestures")
        print("3. Run: python emg_model_trainer.py")
        print("4. Train the model")
        return
    
    # Use most recent model
    model_dir = sorted(model_dirs, reverse=True)[0]
    
    print(f"\n✅ Found model: {model_dir}")
    
    # Create detector
    detector = EMGGestureDetector(model_dir)
    
    # Try to connect to EMG
    print("\n🔌 Attempting to connect to EMG sensor...")
    if detector.connect_emg():
        print("✅ Connected successfully!")
    else:
        print("⚠  Could not connect to EMG sensor")
        print("   Running in simulation mode")
        print("   Connect EMG band and restart for real detection")
    
    # Choose display mode
    print("\n" + "="*80)
    print("SELECT DISPLAY MODE:")
    print("1. Terminal only (ASCII graphics)")
    print("2. Terminal + Matplotlib plots")
    print("3. Exit")
    print("="*80)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        detector.run_terminal_mode()
    elif choice == '2':
        detector.run_with_matplotlib()
    elif choice == '3':
        print("\n👋 Goodbye!")
        return
    else:
        print("\n❌ Invalid choice!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")