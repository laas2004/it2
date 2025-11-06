# v3-trying to add options for signtotext, works well with as,bsl recognition and convertion to text!
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import json
from datetime import datetime

class ImprovedDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # Allow up to 2 hands for all languages
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Changed to 2 for all languages
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/backups', exist_ok=True)
        
        # Supported languages
        self.supported_languages = ['asl', 'bsl', 'isl']
        self.current_language = 'asl'
    
    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks - supports single hand"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        # Normalize relative to wrist
        wrist = hand_landmarks.landmark[0]
        normalized = []
        for i in range(0, len(landmarks), 3):
            x = landmarks[i] - wrist.x
            y = landmarks[i + 1] - wrist.y
            z = landmarks[i + 2] - wrist.z
            normalized.extend([x, y, z])
        return np.array(normalized)
    
    def extract_two_hand_landmarks(self, hand_landmarks_list):
        """Extract landmarks for two hands combined"""
        if len(hand_landmarks_list) != 2:
            return None
            
        left_landmarks = self.extract_landmarks(hand_landmarks_list[0])
        right_landmarks = self.extract_landmarks(hand_landmarks_list[1])
        
        # Combine both hand landmarks
        combined_landmarks = np.concatenate([left_landmarks, right_landmarks])
        return combined_landmarks
    
    def get_dataset_path(self, language=None):
        """Get dataset path for specific language"""
        if language is None:
            language = self.current_language
        return f'data/gesture_dataset_{language}.pickle'
    
    def get_metadata_path(self, language=None):
        """Get metadata path for specific language"""
        if language is None:
            language = self.current_language
        return f'data/dataset_metadata_{language}.json'
    
    def show_existing_data(self, language=None):
        """Show what data already exists for a language"""
        if language is None:
            language = self.current_language
            
        dataset_path = self.get_dataset_path(language)
        
        print(f"\n📊 {language.upper()} Dataset:")
        
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            labels = dataset['labels']
            unique_gestures = {}
            for label in labels:
                unique_gestures[label] = unique_gestures.get(label, 0) + 1
            
            total = 0
            for gesture, count in sorted(unique_gestures.items()):
                print(f"   {gesture}: {count} samples")
                total += count
            print(f"   TOTAL: {total} samples across {len(unique_gestures)} gestures")
            
            return dataset
        else:
            print("   No dataset found. Starting fresh!")
            return {'data': [], 'labels': []}
    
    def backup_dataset(self, language):
        """Create backup of dataset"""
        dataset_path = self.get_dataset_path(language)
        
        if os.path.exists(dataset_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/gesture_dataset_{language}_{timestamp}.pickle"
            
            import shutil
            shutil.copy2(dataset_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
    
    def collect_data(self, gesture_name, num_samples=100, auto_mode=False, language='asl'):
        """Collect training data with support for both single and two-handed gestures"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\n📸 Collecting {language.upper()} data for: {gesture_name}")
        print(f"   Target samples: {num_samples}")
        print(f"   Note: Use the number of hands required for this {language.upper()} gesture")
        
        if auto_mode:
            print("   Mode: AUTO (captures when hand detected)")
            print("   Press SPACE to start auto-capture")
        else:
            print("   Mode: MANUAL")
            print("   Press SPACE to capture each sample")
        print("   Press 'q' to finish early")
        
        collecting = False
        sample_count = 0
        collected_samples = []
        last_capture_time = 0
        no_hand_count = 0
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            hand_detected = results.multi_hand_landmarks is not None
            hand_count = len(results.multi_hand_landmarks) if hand_detected else 0
            
            # Draw hand landmarks with different colors
            if hand_detected:
                no_hand_count = 0
                hand_colors = [(0, 255, 0), (0, 0, 255)]  # Green for first hand, Red for second
                
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    color = hand_colors[i] if i < len(hand_colors) else (255, 255, 0)
                    
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Collect in auto mode
                    current_time = time.time()
                    if (collecting and auto_mode and hand_detected and 
                        (current_time - last_capture_time > 0.3)):
                        
                        # Use two-hand landmarks if available, otherwise single-hand
                        if hand_count == 2:
                            landmarks = self.extract_two_hand_landmarks(results.multi_hand_landmarks)
                        else:
                            landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
                            
                        if landmarks is not None:
                            collected_samples.append(landmarks)
                            sample_count += 1
                            last_capture_time = current_time
                            print(f"✅ Auto-captured sample {sample_count}/{num_samples} ({hand_count} hand(s))")
            else:
                no_hand_count += 1
                if no_hand_count > 100:  # ~5 seconds
                    cv2.putText(frame, "NO HAND DETECTED - Check camera", (10, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw guide box
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            box_size = 300
            cv2.rectangle(frame, 
                         (center_x - box_size//2, center_y - box_size//2),
                         (center_x + box_size//2, center_y + box_size//2),
                         (255, 255, 0), 2)
            
            # Display information
            cv2.putText(frame, f"Language: {language.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Hands: {hand_count}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hand status
            if hand_detected:
                cv2.putText(frame, f"Hands: ✅ DETECTED ({hand_count})", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Hands: ❌ NOT DETECTED", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Mode status
            if collecting:
                mode_text = "AUTO CAPTURING..." if auto_mode else "Press SPACE to capture"
                cv2.putText(frame, mode_text, (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to start", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Hand color legend
            if hand_count >= 1:
                cv2.putText(frame, "Green: Hand 1", (10, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if hand_count >= 2:
                cv2.putText(frame, "Red: Hand 2", (10, 230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Progress bar
            progress_width = int((sample_count / num_samples) * 200)
            cv2.rectangle(frame, (10, 250), (210, 270), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 250), (10 + progress_width, 270), (0, 255, 0), -1)
            
            cv2.putText(frame, "Press 'q' to finish", (10, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                if not collecting:
                    collecting = True
                    print("▶️ Started collecting...")
                elif not auto_mode and hand_detected:
                    # Manual capture
                    if hand_count == 2:
                        landmarks = self.extract_two_hand_landmarks(results.multi_hand_landmarks)
                    else:
                        landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
                    
                    if landmarks is not None:
                        collected_samples.append(landmarks)
                        sample_count += 1
                        print(f"✅ Captured sample {sample_count}/{num_samples} ({hand_count} hand(s))")
                    else:
                        print("❌ Failed to extract landmarks")
            elif key == ord('q'):
                if sample_count > 0:
                    print(f"⚠️ Finishing early with {sample_count} samples")
                    break
                else:
                    print("❌ No samples collected")
                    cap.release()
                    cv2.destroyAllWindows()
                    return []
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"🎉 Collected {sample_count} samples for {gesture_name} ({language.upper()})")
        return collected_samples
    
    def save_dataset(self, new_data, new_labels, language):
        """Save or update the dataset for specific language"""
        dataset_path = self.get_dataset_path(language)
        
        # Create backup before modifying
        self.backup_dataset(language)
        
        # Load existing data if it exists
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            # Append new data
            dataset['data'].extend(new_data)
            dataset['labels'].extend(new_labels)
            print(f"\n💾 Updated {language.upper()} dataset")
        else:
            dataset = {'data': new_data, 'labels': new_labels}
            print(f"\n💾 Created new {language.upper()} dataset")
        
        # Save
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Show summary
        unique_gestures = {}
        for label in dataset['labels']:
            unique_gestures[label] = unique_gestures.get(label, 0) + 1
        
        print(f"📊 {language.upper()} dataset now contains:")
        total = 0
        for gesture, count in sorted(unique_gestures.items()):
            print(f"   {gesture}: {count} samples")
            total += count
        print(f"   TOTAL: {total} samples across {len(unique_gestures)} gestures")
    
    def collect_single_gesture(self):
        """Collect data for a single gesture"""
        print(f"\nCurrent language: {self.current_language.upper()}")
        self.show_existing_data()
        
        gesture_name = input("\nEnter gesture name: ").strip().upper()
        if not gesture_name:
            print("❌ Gesture name cannot be empty!")
            return
            
        try:
            num_samples = int(input("Number of samples (50-200 recommended): "))
            if num_samples < 10:
                print("❌ Too few samples! Using minimum of 10.")
                num_samples = 10
        except ValueError:
            print("❌ Invalid number! Using default 50.")
            num_samples = 50
        
        auto = input("Use auto-capture mode? (y/n): ").strip().lower() == 'y'
        
        print(f"\n📝 For {self.current_language.upper()}, use the required number of hands for '{gesture_name}'")
        print("   Green = Hand 1, Red = Hand 2")
        
        samples = self.collect_data(gesture_name, num_samples, auto_mode=auto, language=self.current_language)
        
        if samples:
            labels = [gesture_name] * len(samples)
            self.save_dataset(samples, labels, self.current_language)
        else:
            print("No samples collected!")
    
    def collect_multiple_gestures(self):
        """Collect data for multiple gestures"""
        print(f"\nCurrent language: {self.current_language.upper()}")
        self.show_existing_data()
        
        gestures_input = input("\nEnter gesture names (comma-separated): ").strip()
        if not gestures_input:
            print("❌ No gestures entered!")
            return
            
        gestures = [g.strip().upper() for g in gestures_input.split(',')]
        
        try:
            num_samples = int(input("Samples per gesture (50-100 recommended): "))
            if num_samples < 10:
                num_samples = 10
        except ValueError:
            num_samples = 50
            
        auto = input("Use auto-capture mode? (y/n): ").strip().lower() == 'y'
        
        print(f"\n📝 For {self.current_language.upper()}, use the required number of hands for each gesture")
        print("   Green = Hand 1, Red = Hand 2")
        
        all_samples = []
        all_labels = []
        
        for gesture in gestures:
            input(f"\n▶️ Press Enter to start collecting '{gesture}'...")
            samples = self.collect_data(gesture, num_samples, auto_mode=auto, language=self.current_language)
            
            if samples:
                all_samples.extend(samples)
                all_labels.extend([gesture] * len(samples))
            else:
                print(f"⚠️ No samples collected for {gesture}")
        
        if all_samples:
            self.save_dataset(all_samples, all_labels, self.current_language)
        else:
            print("No samples collected for any gesture!")
    
    def change_language(self):
        """Change the current language for data collection"""
        print("\n🌍 Available Languages:")
        for i, lang in enumerate(self.supported_languages, 1):
            print(f"   {i}. {lang.upper()} (supports 1-2 hands)")
        
        try:
            choice = int(input(f"\nSelect language (1-{len(self.supported_languages)}): "))
            if 1 <= choice <= len(self.supported_languages):
                self.current_language = self.supported_languages[choice - 1]
                print(f"✅ Switched to {self.current_language.upper()} - supports both single and two-handed gestures")
            else:
                print("❌ Invalid choice!")
        except ValueError:
            print("❌ Invalid input!")
    
    def show_all_datasets(self):
        """Show statistics for all languages"""
        print("\n📈 DATASET SUMMARY FOR ALL LANGUAGES")
        print("=" * 50)
        
        for language in self.supported_languages:
            dataset_path = self.get_dataset_path(language)
            if os.path.exists(dataset_path):
                with open(dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
                
                labels = dataset['labels']
                unique_gestures = set(labels)
                print(f"\n{language.upper()}:")
                print(f"   Gestures: {len(unique_gestures)}")
                print(f"   Samples: {len(labels)}")
                if unique_gestures:
                    print(f"   Gestures: {', '.join(sorted(unique_gestures))}")
            else:
                print(f"\n{language.upper()}: No data collected yet")
    
    def export_dataset_info(self):
        """Export dataset information to a text file"""
        export_path = "data/dataset_summary.txt"
        
        with open(export_path, 'w') as f:
            f.write("SIGN LANGUAGE DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for language in self.supported_languages:
                dataset_path = self.get_dataset_path(language)
                
                f.write(f"{language.upper()} DATASET\n")
                f.write("-" * 30 + "\n")
                
                if os.path.exists(dataset_path):
                    with open(dataset_path, 'rb') as df:
                        dataset = pickle.load(df)
                    
                    labels = dataset['labels']
                    unique_gestures = {}
                    for label in labels:
                        unique_gestures[label] = unique_gestures.get(label, 0) + 1
                    
                    f.write(f"Total Samples: {len(labels)}\n")
                    f.write(f"Unique Gestures: {len(unique_gestures)}\n\n")
                    
                    for gesture, count in sorted(unique_gestures.items()):
                        f.write(f"  {gesture}: {count} samples\n")
                else:
                    f.write("No data collected\n")
                
                f.write("\n")
        
        print(f"📄 Dataset summary exported to: {export_path}")
    
    def run(self):
        """Main menu"""
        print("🤖 Multi-Language Gesture Data Collector")
        print("=" * 50)
        print("Note: All languages now support both single and two-handed gestures")
        
        while True:
            print(f"\nCurrent Language: {self.current_language.upper()}")
            
            print("\nOptions:")
            print("1. Collect single gesture")
            print("2. Collect multiple gestures")
            print("3. Change language")
            print("4. Show current dataset")
            print("5. Show all datasets")
            print("6. Export dataset summary")
            print("7. Train model for current language")
            print("8. Clear current dataset")
            print("9. Exit")
            
            choice = input("\nChoice (1-9): ").strip()
            
            if choice == '1':
                self.collect_single_gesture()
            elif choice == '2':
                self.collect_multiple_gestures()
            elif choice == '3':
                self.change_language()
            elif choice == '4':
                self.show_existing_data()
            elif choice == '5':
                self.show_all_datasets()
            elif choice == '6':
                self.export_dataset_info()
            elif choice == '7':
                print(f"\n🚀 Training {self.current_language.upper()} model...")
                try:
                    from train_model import train_from_landmarks
                    train_from_landmarks(self.current_language)
                except ImportError:
                    print("❌ train_model.py not found!")
                except Exception as e:
                    print(f"❌ Training error: {e}")
            elif choice == '8':
                confirm = input(f"⚠️ Delete ALL {self.current_language.upper()} data? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    dataset_path = self.get_dataset_path()
                    if os.path.exists(dataset_path):
                        os.remove(dataset_path)
                    print("✅ Dataset cleared!")
            elif choice == '9':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice!")
            
def main():
    collector = ImprovedDataCollector()
    collector.run()

if __name__ == "__main__":
    main()
