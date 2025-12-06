# import os
# import time
# import numpy as np
# import pandas as pd
# import serial
# import serial.tools.list_ports
# from datetime import datetime
# import json
# import threading
# import keyboard
# import warnings
# import sys
# warnings.filterwarnings('ignore')

# class EMGDataCollectorLong:
#     """
#     Collect EMG data for long durations with 3 gestures
#     """
    
#     def __init__(self, port=None, baudrate=115200, save_dir="emg_training_data_v2"):
#         self.port = port
#         self.baudrate = baudrate
#         self.save_dir = save_dir
#         self.serial_conn = None
#         self.is_recording = False
#         self.current_gesture = None
        
#         # Create save directory
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Gesture configurations - 3 gestures only
#         self.gestures = {
#             '1': 'fist',        # Fist gesture (close hand)
#             '2': 'wave_in',     # Wave in gesture (palm facing you, move inward)
#             '3': 'wave_out',    # Wave out gesture (palm facing away, move outward)
#             '4': 'rest'         # Rest/relaxed state
#         }
        
#         # Recording settings for long duration
#         self.sample_rate = 500  # Hz
#         self.recording_duration = 10  # seconds per gesture (longer)
#         self.samples_per_gesture = self.sample_rate * self.recording_duration
        
#         # Initialize serial port
#         self._setup_serial_port()
        
#     def _setup_serial_port(self):
#         """Find and setup serial port for EMG sensor"""
#         if self.port:
#             ports = [self.port]
#         else:
#             ports = [p.device for p in serial.tools.list_ports.comports()]
        
#         for port in ports:
#             try:
#                 print(f"Trying to connect to {port}...")
#                 self.serial_conn = serial.Serial(
#                     port=port,
#                     baudrate=self.baudrate,
#                     timeout=1
#                 )
#                 time.sleep(2)  # Wait for connection to stabilize
                
#                 # Test communication
#                 self.serial_conn.write(b'?')
#                 time.sleep(0.1)
#                 if self.serial_conn.in_waiting > 0:
#                     response = self.serial_conn.read(self.serial_conn.in_waiting)
#                     print(f"Connected to {port}. Response: {response}")
#                     self.port = port
#                     return True
#             except Exception as e:
#                 print(f"Failed to connect to {port}: {e}")
#                 if self.serial_conn:
#                     self.serial_conn.close()
        
#         print("No valid serial port found. Using simulation mode.")
#         self.serial_conn = None
#         return False
    
#     def read_emg_data(self):
#         """Read EMG data from serial port or simulate"""
#         if self.serial_conn and self.serial_conn.in_waiting > 0:
#             try:
#                 line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
#                 if line:
#                     try:
#                         if ':' in line:
#                             value = float(line.split(':')[-1].strip())
#                         else:
#                             value = float(line)
#                         return max(0, min(1023, value))  # Normalize to 0-1023 range
#                     except:
#                         return 0
#             except Exception as e:
#                 print(f"Error reading serial: {e}")
#                 return 0
        
#         # Simulation mode
#         elif self.current_gesture:
#             t = time.time()
#             if self.current_gesture == 'fist':
#                 # Strong contraction pattern
#                 base = 300
#                 variation = 200 * np.sin(2 * np.pi * 1.5 * t)
#                 noise = np.random.normal(0, 30)
#             elif self.current_gesture == 'wave_in':
#                 # Medium contraction pattern
#                 base = 200
#                 variation = 150 * np.sin(2 * np.pi * 2 * t)
#                 noise = np.random.normal(0, 20)
#             elif self.current_gesture == 'wave_out':
#                 # Light contraction pattern
#                 base = 150
#                 variation = 100 * np.sin(2 * np.pi * 2.5 * t)
#                 noise = np.random.normal(0, 15)
#             else:  # rest
#                 base = 50
#                 variation = 20 * np.sin(2 * np.pi * 0.5 * t)
#                 noise = np.random.normal(0, 10)
            
#             return max(0, min(1023, base + variation + noise))
        
#         return 50 + np.random.normal(0, 5)  # Default rest state
    
#     def record_gesture_session(self, gesture_key, session_name="session", num_repeats=5, break_time=2):
#         """Record multiple repeats of a gesture with breaks"""
#         if gesture_key not in self.gestures:
#             print(f"Invalid gesture key: {gesture_key}")
#             return
        
#         gesture_name = self.gestures[gesture_key]
#         self.current_gesture = gesture_name
        
#         print(f"\n{'='*60}")
#         print(f"Recording session: {session_name}")
#         print(f"Gesture: {gesture_name}")
#         print(f"Repeats: {num_repeats}")
#         print(f"Duration per repeat: {self.recording_duration} seconds")
#         print(f"Break between repeats: {break_time} seconds")
#         print("="*60)
        
#         all_data = []
#         all_timestamps = []
#         all_labels = []
        
#         try:
#             for repeat in range(num_repeats):
#                 print(f"\nRepeat {repeat + 1}/{num_repeats}")
#                 print(f"Get ready in 3 seconds...")
                
#                 for countdown in range(3, 0, -1):
#                     print(f"{countdown}...")
#                     time.sleep(1)
                
#                 print("START RECORDING!")
#                 print("Perform the gesture naturally...")
                
#                 repeat_data = []
#                 repeat_timestamps = []
#                 repeat_labels = []
                
#                 start_time = time.time()
#                 samples_collected = 0
                
#                 # Recording loop
#                 while samples_collected < self.samples_per_gesture:
#                     if keyboard.is_pressed('esc'):
#                         print("\nRecording cancelled!")
#                         self.is_recording = False
#                         self.current_gesture = None
#                         return
                    
#                     # Read EMG data
#                     emg_value = self.read_emg_data()
#                     timestamp = time.time() - start_time
                    
#                     # Store data
#                     repeat_data.append(emg_value)
#                     repeat_timestamps.append(timestamp)
#                     repeat_labels.append(gesture_name)
                    
#                     samples_collected += 1
                    
#                     # Show progress
#                     if samples_collected % 50 == 0:
#                         progress = (samples_collected / self.samples_per_gesture) * 100
#                         print(f"Progress: {progress:.1f}%", end='\r')
                    
#                     time.sleep(1 / self.sample_rate)
                
#                 # Add this repeat's data to overall data
#                 all_data.extend(repeat_data)
#                 all_timestamps.extend(repeat_timestamps)
#                 all_labels.extend(repeat_labels)
                
#                 print(f"\n✓ Repeat {repeat + 1} complete!")
                
#                 # Break between repeats (except after last one)
#                 if repeat < num_repeats - 1:
#                     print(f"\nRest for {break_time} seconds...")
#                     for i in range(break_time, 0, -1):
#                         print(f"{i}...", end=' ')
#                         time.sleep(1)
#                     print()
            
#             # Save all data from this session
#             self.save_session_data(all_data, all_timestamps, all_labels, session_name, gesture_name)
#             print(f"\n✓ Session '{session_name}' completed successfully!")
            
#         except KeyboardInterrupt:
#             print("\n\nSession interrupted!")
#         except Exception as e:
#             print(f"\nError during recording: {e}")
#         finally:
#             self.current_gesture = None
    
#     def save_session_data(self, data, timestamps, labels, session_name, gesture_name):
#         """Save the recorded session data to CSV"""
#         if not data:
#             print("No data to save!")
#             return
        
#         # Create DataFrame
#         df = pd.DataFrame({
#             'timestamp': timestamps,
#             'emg_value': data,
#             'gesture': labels,
#             'session': session_name
#         })
        
#         # Generate filename
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{gesture_name}_{session_name}_{timestamp}.csv"
#         filepath = os.path.join(self.save_dir, filename)
        
#         # Save to CSV
#         df.to_csv(filepath, index=False)
#         print(f"\nData saved to: {filepath}")
#         print(f"Total samples: {len(data):,}")
#         print(f"Duration: {timestamps[-1]:.1f} seconds")
        
#         # Update dataset summary
#         self.update_dataset_summary(filepath, len(data))
        
#         return filepath
    
#     def update_dataset_summary(self, new_file, sample_count):
#         """Update or create dataset summary JSON"""
#         summary_file = os.path.join(self.save_dir, "dataset_summary.json")
        
#         if os.path.exists(summary_file):
#             with open(summary_file, 'r') as f:
#                 summary = json.load(f)
#         else:
#             summary = {
#                 "total_samples": 0,
#                 "gesture_counts": {},
#                 "sessions": {},
#                 "files": [],
#                 "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
        
#         # Extract info from filename
#         basename = os.path.basename(new_file)
#         parts = basename.split('_')
#         gesture = parts[0]
#         session = parts[1] if len(parts) > 1 else "unknown"
        
#         # Update counts
#         if gesture not in summary['gesture_counts']:
#             summary['gesture_counts'][gesture] = 0
#         summary['gesture_counts'][gesture] += sample_count
        
#         if session not in summary['sessions']:
#             summary['sessions'][session] = {}
#         if gesture not in summary['sessions'][session]:
#             summary['sessions'][session][gesture] = 0
#         summary['sessions'][session][gesture] += sample_count
        
#         # Update total
#         summary['total_samples'] += sample_count
        
#         # Add file info
#         summary['files'].append({
#             'filename': basename,
#             'gesture': gesture,
#             'session': session,
#             'samples': sample_count,
#             'recorded': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         })
        
#         summary['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         # Save updated summary
#         with open(summary_file, 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         print(f"\nDataset summary updated.")
#         print(f"Total samples: {summary['total_samples']:,}")
    
#     def guided_data_collection(self):
#         """Interactive guided data collection for all gestures"""
#         print("\n" + "="*60)
#         print("EMG DATA COLLECTION FOR 3 GESTURES")
#         print("="*60)
#         print("\nWe'll collect data for 3 gestures:")
#         print("  1. FIST - Make a strong fist")
#         print("  2. WAVE IN - Move hand toward you (palm facing you)")
#         print("  3. WAVE OUT - Move hand away from you (palm facing away)")
#         print("  4. REST - Relaxed hand")
#         print("\nEach gesture will be recorded multiple times.")
#         print("\nCommands:")
#         print("  s: Start guided collection")
#         print("  v: View current dataset")
#         print("  q: Quit")
#         print("="*60)
        
#         while True:
#             command = input("\nEnter command (s/v/q): ").strip().lower()
            
#             if command == 'q':
#                 print("Exiting data collector...")
#                 break
            
#             elif command == 'v':
#                 self.view_dataset_summary()
            
#             elif command == 's':
#                 # Get session name
#                 session_name = input("Enter session name (e.g., morning_session1): ").strip()
#                 if not session_name:
#                     session_name = "session"
                
#                 # Get number of repeats
#                 try:
#                     num_repeats = int(input("Number of repeats per gesture (default 10): ") or "10")
#                 except:
#                     num_repeats = 10
                
#                 # Collect data for each gesture
#                 gestures_to_collect = ['1', '2', '3', '4']
#                 gesture_names = {k: self.gestures[k] for k in gestures_to_collect}
                
#                 print(f"\nStarting collection session: {session_name}")
#                 print(f"Repeats per gesture: {num_repeats}")
#                 print(f"Total estimated time: {num_repeats * len(gestures_to_collect) * (self.recording_duration + 5) / 60:.1f} minutes")
                
#                 confirm = input("\nReady to start? (y/n): ").strip().lower()
#                 if confirm != 'y':
#                     print("Cancelled.")
#                     continue
                
#                 for gesture_key in gestures_to_collect:
#                     self.record_gesture_session(
#                         gesture_key=gesture_key,
#                         session_name=session_name,
#                         num_repeats=num_repeats,
#                         break_time=3
#                     )
                    
#                     # Ask if user wants to continue
#                     if gesture_key != gestures_to_collect[-1]:
#                         cont = input(f"\nContinue with next gesture? (y/n): ").strip().lower()
#                         if cont != 'y':
#                             print("Stopping collection...")
#                             break
    
#     def view_dataset_summary(self):
#         """View current dataset statistics"""
#         summary_file = os.path.join(self.save_dir, "dataset_summary.json")
        
#         if not os.path.exists(summary_file):
#             print("No data collected yet!")
#             return
        
#         with open(summary_file, 'r') as f:
#             summary = json.load(f)
        
#         print("\n" + "="*60)
#         print("DATASET SUMMARY")
#         print("="*60)
#         print(f"Total Samples: {summary['total_samples']:,}")
#         print(f"Total Files: {len(summary['files'])}")
#         print(f"Created: {summary['created']}")
#         print(f"Last Updated: {summary['last_updated']}")
        
#         print("\nGesture Distribution:")
#         total = summary['total_samples']
#         for gesture, count in sorted(summary['gesture_counts'].items()):
#             percentage = (count / total) * 100 if total > 0 else 0
#             print(f"  {gesture:10}: {count:8,} samples ({percentage:5.1f}%)")
        
#         print("\nSessions:")
#         for session, gestures in summary['sessions'].items():
#             session_total = sum(gestures.values())
#             print(f"  {session}: {session_total:,} samples")
        
#         print("="*60)
    
#     def collect_large_dataset(self, samples_per_gesture=10000):
#         """Collect a large dataset for training"""
#         print(f"\nCollecting large dataset (~{samples_per_gesture} samples per gesture)")
        
#         # Calculate needed repeats
#         samples_per_repeat = self.samples_per_gesture
#         num_repeats = max(1, samples_per_gesture // samples_per_repeat)
        
#         print(f"Will perform {num_repeats} repeats per gesture")
#         print(f"Estimated time: {num_repeats * 4 * (self.recording_duration + 5) / 60:.1f} minutes")
        
#         confirm = input("\nStart large dataset collection? (y/n): ").strip().lower()
#         if confirm != 'y':
#             return
        
#         # Create sessions for different times/conditions
#         sessions = ["morning", "afternoon", "evening"]
        
#         for session in sessions:
#             print(f"\n{'='*60}")
#             print(f"Starting {session} session")
#             print("="*60)
            
#             for gesture_key in ['1', '2', '3', '4']:
#                 gesture_name = self.gestures[gesture_key]
#                 print(f"\nCollecting {gesture_name} data...")
                
#                 self.record_gesture_session(
#                     gesture_key=gesture_key,
#                     session_name=f"{session}_large",
#                     num_repeats=num_repeats // len(sessions),
#                     break_time=2
#                 )

# def main():
#     print("\n" + "="*60)
#     print("EMG LONG-DURATION DATA COLLECTOR")
#     print("="*60)
    
#     # Configuration
#     CONFIG = {
#         'port': None,  # Auto-detect (COM3, /dev/ttyUSB0, etc.)
#         'baudrate': 115200,
#         'save_dir': 'emg_training_data_long'
#     }
    
#     # Create collector
#     collector = EMGDataCollectorLong(
#         port=CONFIG['port'],
#         baudrate=CONFIG['baudrate'],
#         save_dir=CONFIG['save_dir']
#     )
    
#     # Menu
#     while True:
#         print("\nOptions:")
#         print("  1: Guided data collection (recommended)")
#         print("  2: Collect large dataset for training")
#         print("  3: View dataset statistics")
#         print("  4: Exit")
        
#         choice = input("\nEnter choice (1-4): ").strip()
        
#         if choice == '1':
#             collector.guided_data_collection()
#         elif choice == '2':
#             collector.collect_large_dataset(samples_per_gesture=10000)
#         elif choice == '3':
#             collector.view_dataset_summary()
#         elif choice == '4':
#             print("Exiting...")
#             break
#         else:
#             print("Invalid choice!")

# if __name__ == "__main__":
#     main()




# --------------------------------------------------------------------------------------------







# import os
# import time
# import numpy as np
# import pandas as pd
# import serial
# import serial.tools.list_ports
# from datetime import datetime
# import json
# import threading
# import keyboard
# import warnings
# import sys
# warnings.filterwarnings('ignore')

# class CustomGestureCollector:
#     """
#     Collect EMG data for custom gestures with user-defined names
#     """
    
#     def __init__(self, port=None, baudrate=115200, save_dir="custom_emg_data"):
#         self.port = port
#         self.baudrate = baudrate
#         self.save_dir = save_dir
#         self.serial_conn = None
#         self.is_recording = False
#         self.current_gesture = None
        
#         # Create save directory
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Default gestures (can be customized by user)
#         self.gestures = {
#             '1': {'name': 'round', 'description': 'Make circular motion with hand'},
#             '2': {'name': 'shoot', 'description': 'Quick extension like shooting'},
#             '3': {'name': 'up_down', 'description': 'Move hand up and down'},
#             '4': {'name': 'rest', 'description': 'Relaxed hand position'}
#         }
        
#         # Recording settings
#         self.sample_rate = 500  # Hz
#         self.recording_duration = 5  # seconds per repeat
#         self.samples_per_repeat = self.sample_rate * self.recording_duration
        
#         # Initialize serial port
#         self._setup_serial_port()
        
#     def _setup_serial_port(self):
#         """Find and setup serial port for EMG sensor"""
#         if self.port:
#             ports = [self.port]
#         else:
#             ports = [p.device for p in serial.tools.list_ports.comports()]
        
#         for port in ports:
#             try:
#                 print(f"Trying to connect to {port}...")
#                 self.serial_conn = serial.Serial(
#                     port=port,
#                     baudrate=self.baudrate,
#                     timeout=1
#                 )
#                 time.sleep(2)  # Wait for connection to stabilize
                
#                 # Test communication
#                 self.serial_conn.write(b'?')
#                 time.sleep(0.1)
#                 if self.serial_conn.in_waiting > 0:
#                     response = self.serial_conn.read(self.serial_conn.in_waiting)
#                     print(f"✓ Connected to {port}")
#                     self.port = port
#                     return True
#             except Exception as e:
#                 print(f"Failed to connect to {port}: {e}")
#                 if self.serial_conn:
#                     self.serial_conn.close()
        
#         print("No valid serial port found. Using simulation mode.")
#         self.serial_conn = None
#         return False
    
#     def read_emg_data(self):
#         """Read EMG data from serial port or simulate"""
#         if self.serial_conn and self.serial_conn.in_waiting > 0:
#             try:
#                 line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
#                 if line:
#                     try:
#                         if ':' in line:
#                             value = float(line.split(':')[-1].strip())
#                         else:
#                             value = float(line)
#                         return max(0, min(1023, value))
#                     except:
#                         return 0
#             except Exception as e:
#                 print(f"Error reading serial: {e}")
#                 return 0
        
#         # Simulation mode based on current gesture
#         elif self.current_gesture:
#             t = time.time()
#             gesture_name = self.current_gesture.lower()
            
#             if 'round' in gesture_name:
#                 # Circular motion pattern
#                 base = 250
#                 variation = 150 * np.sin(2 * np.pi * 2 * t)
#                 noise = np.random.normal(0, 25)
#             elif 'shoot' in gesture_name:
#                 # Quick burst pattern
#                 if np.random.random() < 0.15:  # 15% chance of burst
#                     base = 350 + np.random.normal(0, 40)
#                 else:
#                     base = 100 + np.random.normal(0, 15)
#                 variation = 0
#                 noise = 0
#             elif 'up' in gesture_name and 'down' in gesture_name:
#                 # Up-down pattern
#                 base = 200
#                 variation = 120 * np.sin(2 * np.pi * 1.2 * t)
#                 noise = np.random.normal(0, 20)
#             else:  # Default pattern for custom gestures
#                 base = 180
#                 variation = 100 * np.sin(2 * np.pi * 1.8 * t)
#                 noise = np.random.normal(0, 18)
            
#             return max(0, min(1023, base + variation + noise))
        
#         return 50 + np.random.normal(0, 8)  # Default rest state
    
#     def define_custom_gesture(self):
#         """Let user define a new custom gesture"""
#         print("\n" + "="*60)
#         print("DEFINE NEW GESTURE")
#         print("="*60)
        
#         # Get gesture name
#         while True:
#             gesture_name = input("\nEnter gesture name (e.g., 'fist', 'wave', 'grab'): ").strip()
#             if gesture_name:
#                 break
#             print("Please enter a valid name!")
        
#         # Get gesture description
#         description = input("Enter gesture description (optional): ").strip()
#         if not description:
#             description = f"Custom gesture: {gesture_name}"
        
#         # Find next available key
#         existing_keys = list(self.gestures.keys())
#         for i in range(1, 10):
#             if str(i) not in existing_keys:
#                 new_key = str(i)
#                 break
#         else:
#             new_key = str(len(existing_keys) + 1)
        
#         # Add to gestures dictionary
#         self.gestures[new_key] = {
#             'name': gesture_name.lower(),
#             'description': description
#         }
        
#         print(f"\n✓ Gesture '{gesture_name}' added with key '{new_key}'")
#         print(f"  Description: {description}")
        
#         return new_key
    
#     def record_gesture(self, gesture_key, session_name="session", num_repeats=10, break_time=2):
#         """Record multiple repeats of a gesture"""
#         if gesture_key not in self.gestures:
#             print(f"Invalid gesture key: {gesture_key}")
#             return False
        
#         gesture_info = self.gestures[gesture_key]
#         gesture_name = gesture_info['name']
#         self.current_gesture = gesture_name
        
#         print(f"\n{'='*60}")
#         print(f"RECORDING GESTURE: {gesture_name.upper()}")
#         if gesture_info['description']:
#             print(f"Description: {gesture_info['description']}")
#         print(f"Session: {session_name}")
#         print(f"Repeats: {num_repeats}")
#         print(f"Duration per repeat: {self.recording_duration} seconds")
#         print("="*60)
        
#         all_data = []
#         all_timestamps = []
#         all_labels = []
        
#         try:
#             for repeat in range(num_repeats):
#                 print(f"\n\nRepeat {repeat + 1}/{num_repeats}")
#                 print("-" * 40)
#                 print(f"Get ready in 3 seconds...")
                
#                 # Countdown
#                 for countdown in range(3, 0, -1):
#                     print(f"{countdown}...", end=' ', flush=True)
#                     time.sleep(1)
#                 print("GO!")
                
#                 print(f"\nPerform: {gesture_name}")
#                 print("Recording... (Press ESC to cancel)")
                
#                 repeat_data = []
#                 repeat_timestamps = []
#                 repeat_labels = []
                
#                 start_time = time.time()
#                 samples_collected = 0
                
#                 # Recording loop
#                 while samples_collected < self.samples_per_repeat:
#                     if keyboard.is_pressed('esc'):
#                         print("\n\nRecording cancelled!")
#                         self.current_gesture = None
#                         return False
                    
#                     # Read EMG data
#                     emg_value = self.read_emg_data()
#                     timestamp = time.time() - start_time
                    
#                     # Store data
#                     repeat_data.append(emg_value)
#                     repeat_timestamps.append(timestamp)
#                     repeat_labels.append(gesture_name)
                    
#                     samples_collected += 1
                    
#                     # Show progress bar
#                     if samples_collected % 50 == 0:
#                         progress = samples_collected / self.samples_per_repeat
#                         bar_length = 30
#                         filled = int(bar_length * progress)
#                         bar = '█' * filled + '░' * (bar_length - filled)
#                         print(f"\r[{bar}] {progress*100:.1f}%", end='')
                    
#                     time.sleep(1 / self.sample_rate)
                
#                 # Add this repeat's data
#                 all_data.extend(repeat_data)
#                 all_timestamps.extend(repeat_timestamps)
#                 all_labels.extend(repeat_labels)
                
#                 print(f"\r[{'█' * 30}] 100.0%")
#                 print(f"✓ Repeat {repeat + 1} complete!")
                
#                 # Break between repeats
#                 if repeat < num_repeats - 1:
#                     print(f"\nRest for {break_time} seconds...")
#                     for i in range(break_time, 0, -1):
#                         print(f"{i}...", end=' ', flush=True)
#                         time.sleep(1)
#                     print()
            
#             # Save the data
#             filename = self.save_gesture_data(
#                 all_data, all_timestamps, all_labels, 
#                 gesture_name, session_name, num_repeats
#             )
            
#             print(f"\n{'='*60}")
#             print(f"✓ COMPLETED: {gesture_name.upper()}")
#             print(f"✓ Total samples: {len(all_data):,}")
#             print(f"✓ File saved: {filename}")
#             print("="*60)
            
#             return True
            
#         except KeyboardInterrupt:
#             print("\n\nRecording interrupted by user!")
#             return False
#         except Exception as e:
#             print(f"\nError during recording: {e}")
#             return False
#         finally:
#             self.current_gesture = None
    
#     def save_gesture_data(self, data, timestamps, labels, gesture_name, session_name, num_repeats):
#         """Save gesture data to CSV file"""
#         if not data:
#             return None
        
#         # Create DataFrame
#         df = pd.DataFrame({
#             'timestamp': timestamps,
#             'emg_value': data,
#             'gesture': labels,
#             'session': session_name,
#             'sample_rate': self.sample_rate
#         })
        
#         # Add metadata
#         metadata = {
#             'gesture_name': gesture_name,
#             'session_name': session_name,
#             'num_repeats': num_repeats,
#             'total_samples': len(data),
#             'duration_seconds': timestamps[-1],
#             'recording_date': datetime.now().strftime("%Y-%m-%d"),
#             'recording_time': datetime.now().strftime("%H:%M:%S"),
#             'sample_rate': self.sample_rate
#         }
        
#         # Create metadata string for filename
#         meta_str = f"{gesture_name}_{session_name}_{num_repeats}reps_{len(data)}samples"
        
#         # Clean filename
#         import re
#         clean_name = re.sub(r'[^\w\-_]', '_', meta_str)
        
#         # Generate filename
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{clean_name}_{timestamp}.csv"
#         filepath = os.path.join(self.save_dir, filename)
        
#         # Save to CSV
#         df.to_csv(filepath, index=False)
        
#         # Save metadata as JSON
#         metadata_file = filepath.replace('.csv', '_metadata.json')
#         with open(metadata_file, 'w') as f:
#             json.dump(metadata, f, indent=2)
        
#         # Update dataset summary
#         self.update_dataset_summary(filepath, metadata)
        
#         return filename
    
#     def update_dataset_summary(self, filepath, metadata):
#         """Update dataset summary file"""
#         summary_file = os.path.join(self.save_dir, "dataset_summary.json")
        
#         if os.path.exists(summary_file):
#             with open(summary_file, 'r') as f:
#                 summary = json.load(f)
#         else:
#             summary = {
#                 "total_gestures": 0,
#                 "total_samples": 0,
#                 "total_sessions": 0,
#                 "gestures": {},
#                 "sessions": [],
#                 "files": [],
#                 "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
        
#         # Update summary
#         gesture_name = metadata['gesture_name']
        
#         if gesture_name not in summary['gestures']:
#             summary['gestures'][gesture_name] = {
#                 "total_samples": 0,
#                 "total_sessions": 0,
#                 "sessions": []
#             }
        
#         summary['gestures'][gesture_name]["total_samples"] += metadata['total_samples']
#         summary['gestures'][gesture_name]["total_sessions"] += 1
#         summary['gestures'][gesture_name]["sessions"].append({
#             "session": metadata['session_name'],
#             "samples": metadata['total_samples'],
#             "repeats": metadata['num_repeats'],
#             "date": metadata['recording_date']
#         })
        
#         summary['total_samples'] += metadata['total_samples']
        
#         # Check if session is new
#         session_info = {
#             "name": metadata['session_name'],
#             "gesture": gesture_name,
#             "samples": metadata['total_samples'],
#             "date": metadata['recording_date']
#         }
        
#         if session_info not in summary['sessions']:
#             summary['sessions'].append(session_info)
#             summary['total_sessions'] += 1
        
#         # Add file info
#         summary['files'].append({
#             "filename": os.path.basename(filepath),
#             "gesture": gesture_name,
#             "session": metadata['session_name'],
#             "samples": metadata['total_samples'],
#             "date": metadata['recording_date']
#         })
        
#         summary['total_gestures'] = len(summary['gestures'])
#         summary['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         # Save updated summary
#         with open(summary_file, 'w') as f:
#             json.dump(summary, f, indent=2)
    
#     def view_gestures(self):
#         """View all defined gestures"""
#         print("\n" + "="*60)
#         print("DEFINED GESTURES")
#         print("="*60)
        
#         if not self.gestures:
#             print("No gestures defined yet!")
#             return
        
#         print(f"\n{'Key':<6} {'Gesture Name':<20} {'Description':<40}")
#         print("-" * 70)
        
#         for key, info in self.gestures.items():
#             name = info['name']
#             desc = info.get('description', 'No description')
#             print(f"{key:<6} {name:<20} {desc[:38]:<40}")
        
#         print("="*60)
    
#     def view_dataset_summary(self):
#         """View dataset statistics"""
#         summary_file = os.path.join(self.save_dir, "dataset_summary.json")
        
#         if not os.path.exists(summary_file):
#             print("\nNo data collected yet!")
#             return
        
#         with open(summary_file, 'r') as f:
#             summary = json.load(f)
        
#         print("\n" + "="*60)
#         print("DATASET SUMMARY")
#         print("="*60)
#         print(f"Total Gestures: {summary['total_gestures']}")
#         print(f"Total Samples: {summary['total_samples']:,}")
#         print(f"Total Sessions: {summary['total_sessions']}")
#         print(f"Created: {summary['created']}")
#         print(f"Last Updated: {summary['last_updated']}")
        
#         if summary['gestures']:
#             print("\nGesture Statistics:")
#             print(f"{'Gesture':<20} {'Samples':<12} {'Sessions':<10}")
#             print("-" * 45)
            
#             for gesture, data in sorted(summary['gestures'].items()):
#                 samples = data['total_samples']
#                 sessions = data['total_sessions']
#                 print(f"{gesture:<20} {samples:<12,} {sessions:<10}")
        
#         print("="*60)
    
#     def collect_gesture_session(self):
#         """Interactive session to collect data for a gesture"""
#         print("\n" + "="*60)
#         print("GESTURE DATA COLLECTION")
#         print("="*60)
        
#         # Step 1: Select or define gesture
#         self.view_gestures()
        
#         print("\nOptions:")
#         print("  1-9: Select existing gesture by key")
#         print("  n: Define new gesture")
#         print("  b: Back to main menu")
        
#         while True:
#             choice = input("\nEnter choice: ").strip().lower()
            
#             if choice == 'b':
#                 return
#             elif choice == 'n':
#                 gesture_key = self.define_custom_gesture()
#                 break
#             elif choice in self.gestures:
#                 gesture_key = choice
#                 break
#             else:
#                 print("Invalid choice! Please try again.")
        
#         # Step 2: Get session details
#         gesture_name = self.gestures[gesture_key]['name']
        
#         print(f"\nSelected gesture: {gesture_name}")
        
#         session_name = input("Enter session name (e.g., 'training1', 'afternoon'): ").strip()
#         if not session_name:
#             session_name = "session"
        
#         # Get number of repeats
#         while True:
#             try:
#                 repeats_input = input("Number of repeats (default 10, max 50): ").strip()
#                 if not repeats_input:
#                     num_repeats = 10
#                 else:
#                     num_repeats = int(repeats_input)
                
#                 if 1 <= num_repeats <= 50:
#                     break
#                 else:
#                     print("Please enter a number between 1 and 50")
#             except ValueError:
#                 print("Please enter a valid number")
        
#         # Step 3: Confirm and start recording
#         print(f"\n{'='*60}")
#         print(f"READY TO RECORD")
#         print("="*60)
#         print(f"Gesture: {gesture_name}")
#         print(f"Session: {session_name}")
#         print(f"Repeats: {num_repeats}")
#         print(f"Total time: ~{num_repeats * (self.recording_duration + 3) / 60:.1f} minutes")
        
#         confirm = input("\nStart recording? (y/n): ").strip().lower()
#         if confirm != 'y':
#             print("Recording cancelled.")
#             return
        
#         # Step 4: Record the gesture
#         success = self.record_gesture(
#             gesture_key=gesture_key,
#             session_name=session_name,
#             num_repeats=num_repeats,
#             break_time=3
#         )
        
#         if success:
#             print(f"\n✓ Successfully recorded {gesture_name}!")
            
#             # Ask if user wants to record another gesture
#             another = input("\nRecord another gesture? (y/n): ").strip().lower()
#             if another == 'y':
#                 self.collect_gesture_session()
#         else:
#             print("\n✗ Recording failed or was cancelled.")
    
#     def batch_collection(self):
#         """Collect data for all defined gestures in batch"""
#         print("\n" + "="*60)
#         print("BATCH DATA COLLECTION")
#         print("="*60)
        
#         self.view_gestures()
        
#         if not self.gestures:
#             print("\nNo gestures defined! Please define gestures first.")
#             return
        
#         # Get session name
#         session_name = input("\nEnter batch session name: ").strip()
#         if not session_name:
#             session_name = "batch_session"
        
#         # Get repeats per gesture
#         while True:
#             try:
#                 repeats_input = input("Repeats per gesture (default 5): ").strip()
#                 num_repeats = int(repeats_input) if repeats_input else 5
#                 if num_repeats > 0:
#                     break
#                 print("Please enter a positive number")
#             except ValueError:
#                 print("Please enter a valid number")
        
#         print(f"\nBatch Collection Settings:")
#         print(f"Session: {session_name}")
#         print(f"Repeats per gesture: {num_repeats}")
#         print(f"Total gestures: {len(self.gestures)}")
#         print(f"Estimated time: ~{len(self.gestures) * num_repeats * (self.recording_duration + 3) / 60:.1f} minutes")
        
#         confirm = input("\nStart batch collection? (y/n): ").strip().lower()
#         if confirm != 'y':
#             print("Batch collection cancelled.")
#             return
        
#         # Collect data for each gesture
#         for i, (key, gesture_info) in enumerate(self.gestures.items(), 1):
#             print(f"\n\n{'='*60}")
#             print(f"Gesture {i}/{len(self.gestures)}: {gesture_info['name'].upper()}")
#             print("="*60)
            
#             success = self.record_gesture(
#                 gesture_key=key,
#                 session_name=session_name,
#                 num_repeats=num_repeats,
#                 break_time=2
#             )
            
#             if not success:
#                 cont = input("\nContinue with next gesture? (y/n): ").strip().lower()
#                 if cont != 'y':
#                     print("Batch collection stopped.")
#                     break
            
#             if i < len(self.gestures):
#                 print("\nPreparing for next gesture...")
#                 time.sleep(2)
        
#         print("\n✓ Batch collection completed!")
    
#     def edit_gestures(self):
#         """Edit or delete gestures"""
#         print("\n" + "="*60)
#         print("EDIT GESTURES")
#         print("="*60)
        
#         self.view_gestures()
        
#         if not self.gestures:
#             print("\nNo gestures to edit!")
#             return
        
#         print("\nOptions:")
#         print("  e: Edit gesture description")
#         print("  d: Delete gesture")
#         print("  b: Back")
        
#         choice = input("\nEnter choice: ").strip().lower()
        
#         if choice == 'b':
#             return
        
#         elif choice == 'e':
#             # Edit gesture description
#             gesture_key = input("Enter gesture key to edit: ").strip()
#             if gesture_key in self.gestures:
#                 new_desc = input("Enter new description: ").strip()
#                 if new_desc:
#                     self.gestures[gesture_key]['description'] = new_desc
#                     print(f"✓ Description updated for {self.gestures[gesture_key]['name']}")
#                 else:
#                     print("Description not changed")
#             else:
#                 print("Invalid gesture key!")
        
#         elif choice == 'd':
#             # Delete gesture
#             gesture_key = input("Enter gesture key to delete: ").strip()
#             if gesture_key in self.gestures:
#                 gesture_name = self.gestures[gesture_key]['name']
#                 confirm = input(f"Delete '{gesture_name}'? This cannot be undone! (y/n): ").strip().lower()
#                 if confirm == 'y':
#                     del self.gestures[gesture_key]
#                     print(f"✓ Gesture '{gesture_name}' deleted")
#             else:
#                 print("Invalid gesture key!")
    
#     def run(self):
#         """Main application loop"""
#         print("\n" + "="*60)
#         print("CUSTOM EMG GESTURE DATA COLLECTOR")
#         print("="*60)
#         print("\nWelcome! This tool helps you collect EMG data for custom gestures.")
#         print("You can define your own gestures (like 'round', 'shoot', 'up_down')")
#         print("and collect training data for machine learning.")
        
#         while True:
#             print("\n" + "="*60)
#             print("MAIN MENU")
#             print("="*60)
#             print("  1: View defined gestures")
#             print("  2: Define new gesture")
#             print("  3: Collect data for a gesture")
#             print("  4: Batch collect for all gestures")
#             print("  5: Edit/Delete gestures")
#             print("  6: View dataset summary")
#             print("  7: Test EMG sensor")
#             print("  8: Exit")
#             print("="*60)
            
#             choice = input("\nEnter choice (1-8): ").strip()
            
#             if choice == '1':
#                 self.view_gestures()
            
#             elif choice == '2':
#                 self.define_custom_gesture()
            
#             elif choice == '3':
#                 self.collect_gesture_session()
            
#             elif choice == '4':
#                 self.batch_collection()
            
#             elif choice == '5':
#                 self.edit_gestures()
            
#             elif choice == '6':
#                 self.view_dataset_summary()
            
#             elif choice == '7':
#                 self.test_sensor()
            
#             elif choice == '8':
#                 print("\nThank you for using EMG Gesture Collector!")
#                 print("Goodbye!")
#                 break
            
#             else:
#                 print("Invalid choice! Please try again.")
    
#     def test_sensor(self):
#         """Test EMG sensor connection and readings"""
#         print("\n" + "="*60)
#         print("EMG SENSOR TEST")
#         print("="*60)
        
#         if self.serial_conn:
#             print("✓ Connected to EMG sensor")
#             print(f"Port: {self.port}")
            
#             print("\nReading EMG values... (Press ESC to stop)")
#             print("-" * 40)
            
#             try:
#                 for i in range(1, 101):  # Read 100 samples
#                     if keyboard.is_pressed('esc'):
#                         print("\nTest stopped.")
#                         break
                    
#                     value = self.read_emg_data()
                    
#                     # Create simple bar visualization
#                     bar_length = 50
#                     normalized = min(1.0, value / 1023)
#                     filled = int(bar_length * normalized)
#                     bar = '█' * filled + '░' * (bar_length - filled)
                    
#                     print(f"\rSample {i:3d}: {value:6.1f} [{bar}]", end='')
                    
#                     time.sleep(0.05)  # 20 Hz for testing
                
#                 print("\n\n✓ Sensor test completed!")
                
#             except KeyboardInterrupt:
#                 print("\n\nTest interrupted!")
#         else:
#             print("⚠ No EMG sensor connected")
#             print("Using simulation mode. Real EMG values will be simulated.")
            
#             print("\nSimulated EMG values...")
#             for i in range(1, 51):
#                 value = self.read_emg_data()
#                 bar_length = 30
#                 normalized = min(1.0, value / 500)
#                 filled = int(bar_length * normalized)
#                 bar = '█' * filled + '░' * (bar_length - filled)
#                 print(f"Sample {i:2d}: {value:6.1f} [{bar}]")
#                 time.sleep(0.1)

# def main():
#     print("\n" + "="*60)
#     print("CUSTOM EMG GESTURE DATA COLLECTOR")
#     print("="*60)
    
#     # Configuration
#     CONFIG = {
#         'port': None,  # Auto-detect (COM3, /dev/ttyUSB0, etc.)
#         'baudrate': 115200,
#         'save_dir': 'custom_emg_gestures'
#     }
    
#     # Create collector instance
#     collector = CustomGestureCollector(
#         port=CONFIG['port'],
#         baudrate=CONFIG['baudrate'],
#         save_dir=CONFIG['save_dir']
#     )
    
#     # Start the application
#     collector.run()

# if __name__ == "__main__":
#     main()







# --------------------------------------------






# import serial
# import time
# import csv
# import os
# from datetime import datetime
# import numpy as np

# # ==========================
# # USER SETTINGS
# # ==========================
# PORT = "COM3"      # Change this to your port
# BAUD = 115200
# SAMPLES_PER_GESTURE = 200  # samples to collect per gesture
# # GESTURES list will be created based on user input
# # ==========================

# def get_gesture_input():
#     """Get gesture names from user input"""
#     print("\n" + "=" * 60)
#     print("GESTURE INPUT CONFIGURATION")
#     print("=" * 60)
    
#     gestures = []
#     print("\nEnter the gesture names you want to record.")
#     print("Enter 'done' when finished, or 'cancel' to exit.")
#     print("-" * 50)
    
#     while True:
#         gesture_name = input(f"\nEnter gesture name #{len(gestures) + 1}: ").strip()
        
#         if gesture_name.lower() == 'done':
#             if len(gestures) < 2:
#                 print("Please enter at least 2 gestures.")
#                 continue
#             break
#         elif gesture_name.lower() == 'cancel':
#             print("Canceling data collection.")
#             return None
#         elif not gesture_name:
#             print("Gesture name cannot be empty. Please enter a valid name.")
#             continue
#         elif gesture_name in gestures:
#             print(f"Gesture '{gesture_name}' already exists. Please enter a different name.")
#             continue
#         else:
#             gestures.append(gesture_name)
#             print(f"Added gesture: {gesture_name}")
#             print(f"Current gestures: {gestures}")
    
#     # Ask for number of samples per gesture
#     print("\n" + "-" * 50)
#     while True:
#         try:
#             samples_input = input(f"Enter number of samples per gesture (default: {SAMPLES_PER_GESTURE}): ").strip()
#             if not samples_input:
#                 samples = SAMPLES_PER_GESTURE
#                 break
#             samples = int(samples_input)
#             if samples < 10:
#                 print("Please enter at least 10 samples.")
#                 continue
#             break
#         except ValueError:
#             print("Please enter a valid number.")
    
#     return gestures, samples

# def collect_gesture_data():
#     """Collect EMG data for different gestures and save to CSV"""
    
#     # Get gesture input from user
#     user_input = get_gesture_input()
#     if user_input is None:
#         return
    
#     GESTURES, samples_per_gesture = user_input
    
#     print("\n" + "=" * 60)
#     print("STARTING DATA COLLECTION")
#     print("=" * 60)
#     print(f"Gestures to record: {GESTURES}")
#     print(f"Samples per gesture: {samples_per_gesture}")
#     print("=" * 60)
    
#     # Connect to Arduino
#     try:
#         ser = serial.Serial(PORT, BAUD, timeout=1)
#         time.sleep(2)  # wait for Arduino
#         print(f"Connected to {PORT}")
#     except Exception as e:
#         print(f"Error connecting to {PORT}: {e}")
#         print("Please check your port and try again.")
#         return
    
#     # Create data directory
#     data_dir = "emg_data"
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
    
#     # Create CSV file with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{data_dir}/emg_dataset_{timestamp}.csv"
    
#     print(f"\nData will be saved to: {filename}")
#     print("-" * 50)
    
#     # Ask for session description
#     session_desc = input("\nEnter a description for this session (optional): ").strip()
#     if not session_desc:
#         session_desc = "No description provided"
    
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         # Write header with session info as comment
#         writer.writerow(['# Session:', session_desc])
#         writer.writerow(['# Gestures:', ', '.join(GESTURES)])
#         writer.writerow(['# Samples per gesture:', str(samples_per_gesture)])
#         writer.writerow(['# Collection time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
#         writer.writerow(['timestamp', 'emg_value', 'gesture_label', 'gesture_name'])
        
#         total_gestures = len(GESTURES)
        
#         for gesture_idx, gesture_name in enumerate(GESTURES):
#             print(f"\n" + "=" * 50)
#             print(f"Gesture {gesture_idx + 1} of {total_gestures}: {gesture_name}")
#             print("=" * 50)
            
#             # Ask if user wants to skip this gesture
#             while True:
#                 choice = input(f"Record '{gesture_name}'? (y/n): ").strip().lower()
#                 if choice in ['y', 'yes']:
#                     break
#                 elif choice in ['n', 'no']:
#                     print(f"Skipping '{gesture_name}'...")
#                     # Add placeholder for skipped gesture
#                     for _ in range(samples_per_gesture):
#                         writer.writerow([0, 0, gesture_idx, gesture_name + "_skipped"])
#                     print(f"Added placeholder data for '{gesture_name}'")
#                     break
#                 else:
#                     print("Please enter 'y' or 'n'.")
            
#             if choice in ['n', 'no']:
#                 continue
            
#             # Ask for preparation time
#             prep_time = 3
#             try:
#                 prep_input = input(f"Preparation time in seconds (default: {prep_time}): ").strip()
#                 if prep_input:
#                     prep_time = int(prep_input)
#             except ValueError:
#                 print(f"Using default preparation time: {prep_time} seconds")
            
#             print(f"\nPreparing to record: {gesture_name}")
#             print(f"Get ready in {prep_time} seconds...")
            
#             for countdown in range(prep_time, 0, -1):
#                 print(f"{countdown}...")
#                 time.sleep(1)
            
#             print(f"\n🎯 RECORDING '{gesture_name.upper()}' NOW! 🎯")
#             print("Perform the gesture and hold it...")
#             print("-" * 60)
#             print(f"Gesture: {gesture_name}")
#             print("Values being written to CSV:")
#             print("Timestamp(ms) | EMG Value | Gesture ID | Gesture Name")
#             print("-" * 60)
            
#             samples_collected = 0
#             start_time = time.time()
            
#             # Buffer to store recent values for display
#             recent_values = []
            
#             while samples_collected < samples_per_gesture:
#                 try:
#                     raw = ser.readline().decode().strip()
                    
#                     if raw.isdigit():
#                         value = int(raw)
#                         timestamp_ms = int(time.time() * 1000)
                        
#                         # Write to CSV
#                         writer.writerow([timestamp_ms, value, gesture_idx, gesture_name])
                        
#                         # Store recent values (keep last 5)
#                         recent_values.append(value)
#                         if len(recent_values) > 5:
#                             recent_values.pop(0)
                        
#                         # Display the values being written (show every 5th sample to reduce clutter)
#                         if samples_collected % 5 == 0:
#                             print(f"{timestamp_ms:12d} | {value:9d} | {gesture_idx:10d} | {gesture_name}")
                        
#                         samples_collected += 1
                        
#                         # Show progress more frequently
#                         if samples_collected % 20 == 0:
#                             elapsed = time.time() - start_time
#                             rate = samples_collected / elapsed if elapsed > 0 else 0
#                             remaining = (samples_per_gesture - samples_collected) / rate if rate > 0 else 0
                            
#                             print("-" * 60)
#                             print(f"Progress: {samples_collected}/{samples_per_gesture} samples")
#                             print(f"Sampling rate: {rate:.1f} samples/sec")
#                             print(f"Estimated time remaining: {remaining:.1f} seconds")
#                             if recent_values:
#                                 print(f"Recent values: {recent_values}")
#                             print("-" * 60)
                
#                 except KeyboardInterrupt:
#                     print("\n⚠️ Recording interrupted by user!")
#                     confirm = input(f"Stop recording '{gesture_name}'? (y/n): ").strip().lower()
#                     if confirm in ['y', 'yes']:
#                         print(f"Stopped recording '{gesture_name}' early.")
#                         break
#                     else:
#                         print("Resuming recording...")
#                         continue
#                 except Exception as e:
#                     # Just continue on other errors
#                     continue
            
#             print(f"\n✅ Finished recording '{gesture_name}'")
#             print(f"Total samples collected: {samples_collected}")
            
#             if samples_collected < samples_per_gesture:
#                 print(f"⚠️ Note: Collected {samples_collected} instead of {samples_per_gesture} samples")
            
#             # Ask if user wants to continue
#             if gesture_idx < total_gestures - 1:
#                 next_gesture = GESTURES[gesture_idx + 1]
#                 while True:
#                     choice = input(f"\nContinue with next gesture '{next_gesture}'? (y/n/pause): ").strip().lower()
#                     if choice in ['y', 'yes']:
#                         rest_time = 3
#                         try:
#                             rest_input = input(f"Rest time in seconds (default: {rest_time}): ").strip()
#                             if rest_input:
#                                 rest_time = int(rest_input)
#                         except ValueError:
#                             pass
                        
#                         print(f"Resting for {rest_time} seconds before next gesture...")
#                         time.sleep(rest_time)
#                         break
#                     elif choice in ['n', 'no']:
#                         print("Stopping data collection...")
#                         ser.close()
#                         print_summary_and_preview(filename)
#                         return
#                     elif choice == 'pause':
#                         input("Press Enter to continue...")
#                         break
#                     else:
#                         print("Please enter 'y', 'n', or 'pause'.")
    
#     ser.close()
#     print(f"\n" + "=" * 60)
#     print("🎉 DATA COLLECTION COMPLETE! 🎉")
#     print("=" * 60)
#     print(f"Data saved to: {filename}")
    
#     # Create summary and show first few rows
#     print_summary_and_preview(filename)
    
#     # Ask if user wants to collect more data
#     while True:
#         more = input("\nDo you want to collect data for another session? (y/n): ").strip().lower()
#         if more in ['y', 'yes']:
#             collect_gesture_data()
#             break
#         elif more in ['n', 'no']:
#             print("Thank you for using the EMG Data Collector!")
#             break
#         else:
#             print("Please enter 'y' or 'n'.")

# def print_summary_and_preview(filename):
#     """Print summary and preview of collected data"""
    
#     try:
#         # Read the CSV file
#         with open(filename, 'r') as csvfile:
#             lines = csvfile.readlines()
        
#         # Skip comment lines
#         data_lines = []
#         comments = []
#         for line in lines:
#             if line.startswith('#'):
#                 comments.append(line.strip())
#             else:
#                 data_lines.append(line)
        
#         # Parse header and data
#         header = data_lines[0].strip().split(',')
#         data = [line.strip().split(',') for line in data_lines[1:] if line.strip()]
        
#         print("\n" + "=" * 60)
#         print("SESSION INFORMATION")
#         print("=" * 60)
#         for comment in comments:
#             print(comment)
        
#         print("\n" + "=" * 60)
#         print("DATA COLLECTION SUMMARY")
#         print("=" * 60)
#         print(f"CSV File: {filename}")
#         print(f"Total samples: {len(data)}")
        
#         # Count samples per gesture
#         gesture_counts = {}
#         gesture_names = {}
        
#         for row in data:
#             if len(row) >= 4:
#                 try:
#                     gesture_idx = int(row[2])
#                     gesture_name = row[3]
#                     gesture_counts[gesture_idx] = gesture_counts.get(gesture_idx, 0) + 1
#                     gesture_names[gesture_idx] = gesture_name
#                 except (ValueError, IndexError):
#                     continue
        
#         print("\nSamples per gesture:")
#         print("-" * 40)
#         for gesture_idx in sorted(gesture_counts.keys()):
#             count = gesture_counts[gesture_idx]
#             name = gesture_names[gesture_idx]
#             print(f"{name:20s} (ID:{gesture_idx}): {count:4d} samples")
        
#         # Show sample data
#         if data:
#             print("\n" + "=" * 60)
#             print("SAMPLE DATA (First 5 rows):")
#             print("=" * 60)
#             print(f"{'Timestamp':>12s} | {'EMG Value':>9s} | {'Gesture ID':>10s} | {'Gesture Name':<20s}")
#             print("-" * 60)
            
#             for i, row in enumerate(data[:5]):
#                 if len(row) >= 4:
#                     timestamp, emg_value, gesture_id, gesture_name = row[:4]
#                     print(f"{timestamp:>12s} | {emg_value:>9s} | {gesture_id:>10s} | {gesture_name:<20s}")
        
#         # Show data statistics
#         print("\n" + "=" * 60)
#         print("DATA STATISTICS:")
#         print("=" * 60)
        
#         # Calculate EMG statistics per gesture
#         emg_values_by_gesture = {}
        
#         for row in data:
#             if len(row) >= 4:
#                 gesture_name = row[3]
#                 try:
#                     emg_value = int(row[1])
                    
#                     if gesture_name not in emg_values_by_gesture:
#                         emg_values_by_gesture[gesture_name] = []
                    
#                     emg_values_by_gesture[gesture_name].append(emg_value)
#                 except (ValueError, IndexError):
#                     continue
        
#         for gesture_name, values in emg_values_by_gesture.items():
#             if values and "_skipped" not in gesture_name:
#                 min_val = min(values)
#                 max_val = max(values)
#                 mean_val = np.mean(values)
#                 std_val = np.std(values)
                
#                 print(f"\n{gesture_name}:")
#                 print(f"  Samples: {len(values)}")
#                 print(f"  Min EMG: {min_val}")
#                 print(f"  Max EMG: {max_val}")
#                 print(f"  Mean EMG: {mean_val:.2f}")
#                 print(f"  Std Dev: {std_val:.2f}")
#                 print(f"  Range: {max_val - min_val}")
        
#         print("\n" + "=" * 60)
#         print("✅ DATA COLLECTION SUCCESSFUL!")
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"\n⚠️ Error reading file: {e}")

# def main_menu():
#     """Display main menu"""
#     print("\n" + "=" * 60)
#     print("EMG GESTURE DATA COLLECTOR")
#     print("=" * 60)
#     print("1. Start New Data Collection Session")
#     print("2. View Existing Data Files")
#     print("3. Exit")
#     print("=" * 60)
    
#     while True:
#         choice = input("\nEnter your choice (1-3): ").strip()
        
#         if choice == '1':
#             collect_gesture_data()
#             break
#         elif choice == '2':
#             view_existing_files()
#             break
#         elif choice == '3':
#             print("Goodbye!")
#             return
#         else:
#             print("Invalid choice. Please enter 1, 2, or 3.")

# def view_existing_files():
#     """View existing data files"""
#     data_dir = "emg_data"
    
#     if not os.path.exists(data_dir):
#         print(f"\nNo data directory found at '{data_dir}'")
#         print("Please collect some data first.")
#         return
    
#     csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
#     if not csv_files:
#         print(f"\nNo CSV files found in '{data_dir}'")
#         print("Please collect some data first.")
#         return
    
#     print(f"\nFound {len(csv_files)} CSV file(s) in '{data_dir}':")
#     print("-" * 60)
    
#     for i, filename in enumerate(sorted(csv_files, reverse=True), 1):
#         filepath = os.path.join(data_dir, filename)
#         size = os.path.getsize(filepath)
#         mod_time = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
        
#         # Read first few lines to get session info
#         try:
#             with open(filepath, 'r') as f:
#                 first_lines = [f.readline().strip() for _ in range(5)]
            
#             desc = ""
#             for line in first_lines:
#                 if line.startswith('# Session:'):
#                     desc = line.replace('# Session:', '').strip()
#                     break
#         except:
#             desc = "Could not read description"
        
#         print(f"{i}. {filename}")
#         print(f"   Size: {size:,} bytes")
#         print(f"   Modified: {mod_time}")
#         print(f"   Description: {desc}")
#         print()
    
#     # Ask if user wants to view a specific file
#     while True:
#         choice = input("\nEnter file number to view details, or 'b' to go back: ").strip().lower()
        
#         if choice == 'b':
#             break
        
#         try:
#             file_idx = int(choice) - 1
#             if 0 <= file_idx < len(csv_files):
#                 filepath = os.path.join(data_dir, csv_files[file_idx])
#                 print_summary_and_preview(filepath)
#             else:
#                 print("Invalid file number.")
#         except ValueError:
#             print("Please enter a valid number or 'b'.")
    
#     main_menu()

# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("🎮 EMG GESTURE DATA COLLECTOR 🎮")
#     print("=" * 60)
#     print("This tool helps you collect EMG data for gesture recognition.")
#     print("You'll be prompted to enter gesture names and record data.")
#     print("=" * 60)
    
#     main_menu()



# ----------------------------------------------------------------






import serial
import time
import csv
import os

# ===================== USER SETTINGS =====================
PORT = "COM3"          # Change to your Arduino port
BAUD = 115200
SAMPLES_PER_GESTURE = 200
OUTPUT_DIR = "emg_dataset"
# ==========================================================


def connect_serial():
    """Connect to Arduino"""
    print("Connecting to Arduino...")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)  # wait for Arduino reset
        ser.reset_input_buffer()
        print(f"Connected to {PORT}\n")
        return ser
    except:
        print("ERROR: Could not connect to Arduino")
        exit()


def read_emg_value(ser):
    """Read one integer value from Arduino"""
    try:
        line = ser.readline().decode().strip()
        if line.isdigit():
            return int(line)
        return None
    except:
        return None


def collect_gesture(ser, gesture_name):
    """Collect fixed number of EMG samples"""

    print("\n==============================================")
    print(f"Get ready for gesture: {gesture_name}")
    print("Recording starts in 3...")
    time.sleep(1)
    print("Recording starts in 2...")
    time.sleep(1)
    print("Recording starts in 1...\n")
    time.sleep(1)

    print("START! Hold the gesture...")
    print("Warming up sensor...")

    # ---- Important fix: prevents first-gesture fast sampling issue ----
    ser.reset_input_buffer()
    time.sleep(0.1)

    warmup_samples = 30
    for _ in range(warmup_samples):
        read_emg_value(ser)
        time.sleep(0.01)

    print("Collecting samples...\n")

    samples = []
    sample_no = 1

    while sample_no <= SAMPLES_PER_GESTURE:
        val = read_emg_value(ser)
        if val is not None:
            timestamp_ms = int(time.time() * 1000)  # timestamp in ms

            samples.append((timestamp_ms, sample_no, val))

            if sample_no % 20 == 0:
                print(f"Collected: {sample_no}/{SAMPLES_PER_GESTURE}")

            sample_no += 1

        time.sleep(0.01)  # consistent sampling speed

    print(f"✔ Finished recording {gesture_name}")
    return samples


def save_gesture(gesture, data):
    """Save data to CSV"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    filename = os.path.join(OUTPUT_DIR, f"{gesture}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "sample_no", "emg_value"])

        for row in data:
            writer.writerow(row)

    print(f"Saved: {filename}")


def main():
    ser = connect_serial()

    print("==============================================")
    print(" EMG Gesture Data Collector (Full CSV v3)")
    print("==============================================")
    print("Enter gesture names one by one.")
    print("Type 'done' when finished.\n")

    gestures = []
    while True:
        g = input("Enter gesture name: ").strip()
        if g.lower() == "done":
            break
        if g != "":
            gestures.append(g)

    print("\n==============================================")
    print("GESTURES:", gestures)
    print("==============================================")

    for gesture in gestures:
        data = collect_gesture(ser, gesture)
        save_gesture(gesture, data)

    ser.close()
    print("\nAll gestures recorded successfully!")


if __name__ == "__main__":
    main()
