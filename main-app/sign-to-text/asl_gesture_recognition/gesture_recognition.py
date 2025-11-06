# v10 -to fix the output issue for bidirectional and making sure both hands are detected for all languages
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from tensorflow import keras
from collections import deque

class GestureRecognizer:
    def __init__(self, language="asl"):
        self.language = language
        print(f"🔧 Initializing {language.upper()} GestureRecognizer...")
        
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Set max hands based on language
        if language in ["bsl", "isl"]:  # Modified: ISL also uses two hands
            max_hands = 2
        else:
            max_hands = 1  # ASL typically uses one hand
            
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = None
        self.label_encoder = None
        self.is_model_loaded = False
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=5)
        
        self.load_model()
        print(f"✅ {language.upper()} GestureRecognizer initialized")
    
    def load_model(self):
        """Load the trained Keras model and label encoder"""
        try:
            # Try to load language-specific model first
            model_path = f'models/gesture_model_{self.language}.h5'
            label_encoder_path = f'models/label_encoder_{self.language}.pkl'
            
            # Fallback to generic ASL model
            if not os.path.exists(model_path):
                model_path = 'asl_gesture_model.h5'
                label_encoder_path = 'label_encoder.pkl'
            
            if os.path.exists(model_path) and os.path.exists(label_encoder_path):
                # Load Keras model
                self.model = keras.models.load_model(model_path)
                print(f"✅ {self.language.upper()} model loaded successfully")
                
                # Load label encoder
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"✅ Label encoder loaded: {list(self.label_encoder.classes_)}")
                
                self.is_model_loaded = True
            else:
                print(f"⚠️ Model files not found for {self.language.upper()}:")
                print(f"   Looking for: {model_path}")
                print(f"   Looking for: {label_encoder_path}")
                print(f"   Run train_model.py to train a model first!")
                self.is_model_loaded = False
                
        except Exception as e:
            print(f"❌ Error loading {self.language.upper()} model: {e}")
            self.is_model_loaded = False
            import traceback
            traceback.print_exc()
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks - EXACTLY like training"""
        landmarks = []
        
        # Get all landmark coordinates (x, y, z)
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Normalize landmarks relative to wrist position (EXACTLY like training)
        wrist = hand_landmarks.landmark[0]
        normalized_landmarks = []
        
        for i in range(0, len(landmarks), 3):
            x = landmarks[i] - wrist.x
            y = landmarks[i + 1] - wrist.y
            z = landmarks[i + 2] - wrist.z
            normalized_landmarks.extend([x, y, z])
        
        return np.array(normalized_landmarks)
    
    def extract_two_hand_landmarks(self, hand_landmarks_list):
        """Extract landmarks for two hands combined (for BSL and ISL)"""
        if len(hand_landmarks_list) < 2:
            return None
            
        # Extract landmarks for each hand
        hand1_landmarks = self.extract_landmarks(hand_landmarks_list[0])
        hand2_landmarks = self.extract_landmarks(hand_landmarks_list[1])
        
        # Combine both hand landmarks
        combined_landmarks = np.concatenate([hand1_landmarks, hand2_landmarks])
        return combined_landmarks
    
    def extract_pose_landmarks(self, pose_landmarks):
        """Extract pose landmarks for full body context (for BSL and ISL)"""
        if not pose_landmarks:
            return None
            
        landmarks = []
        # Extract key upper body points for sign language
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.NOSE
        ]
        
        for point in key_points:
            landmark = pose_landmarks.landmark[point]
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks using Keras model"""
        if not self.is_model_loaded or self.model is None:
            return "Analyzing..."
        
        try:
            # Ensure landmarks is the right shape
            # Model expects (1, n_features)
            if len(landmarks) < 63:
                # Pad with zeros if needed
                landmarks = np.pad(landmarks, (0, 63 - len(landmarks)), 'constant')
            elif len(landmarks) > 63:
                # For two-hand data, use only first hand for now
                landmarks = landmarks[:63]
            
            # Reshape for prediction
            landmarks = landmarks.reshape(1, -1)
            
            # Make prediction
            predictions = self.model.predict(landmarks, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            # Return prediction if confidence is high enough
            if confidence > 0.7:
                gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
                return gesture_name
            else:
                return "Unknown"
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Analyzing..."
    
    def detect_isl_gestures(self, hand_landmarks_list, pose_landmarks):
        """ISL-specific gesture detection using rule-based approach"""
        if not hand_landmarks_list:
            return "No hands detected"
        
        hand_count = len(hand_landmarks_list)
        
        # Basic ISL gesture detection based on hand configuration
        if hand_count == 2:
            # Two-handed ISL gestures
            left_hand = hand_landmarks_list[0]
            right_hand = hand_landmarks_list[1]
            
            # Simple finger state detection
            def get_finger_states(hand_landmarks):
                # Finger tip indices
                tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
                # Corresponding pip joints
                pips = [3, 6, 10, 14, 18]
                
                states = []
                for tip, pip in zip(tips, pips):
                    tip_y = hand_landmarks.landmark[tip].y
                    pip_y = hand_landmarks.landmark[pip].y
                    # Finger is extended if tip is above pip
                    states.append(tip_y < pip_y)
                return states
            
            left_fingers = get_finger_states(left_hand)
            right_fingers = get_finger_states(right_hand)
            
            # ISL Common gestures
            # NAMASTE: Both hands together (palm to palm)
            if (all(left_fingers[1:]) and all(right_fingers[1:])):  # All fingers extended
                # Check if hands are close together
                left_wrist = left_hand.landmark[0]
                right_wrist = right_hand.landmark[0]
                distance = abs(left_wrist.x - right_wrist.x)
                if distance < 0.1:  # Hands are close
                    return "NAMASTE"
            
            # DHANYAVAD (Thank you): One hand flat moving from chin
            if all(left_fingers[1:]) or all(right_fingers[1:]):
                return "DHANYAVAD"
            
            # KRIPAYA (Please): Both hands in prayer position
            if (left_fingers[1] and right_fingers[1] and  # Index fingers extended
                not any(left_fingers[2:]) and not any(right_fingers[2:])):  # Others folded
                return "KRIPAYA"
            
            # Common ISL Alphabet detection
            # A: Both hands closed fists
            if not any(left_fingers) and not any(right_fingers):
                return "A"
            
            return "Two hands detected"
        
        elif hand_count == 1:
            # Single-hand ISL gestures
            hand = hand_landmarks_list[0]
            
            # Simple finger counting for numbers
            def count_extended_fingers(hand_landmarks):
                tips = [8, 12, 16, 20]  # index, middle, ring, pinky
                pips = [6, 10, 14, 18]
                
                count = 0
                for tip, pip in zip(tips, pips):
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                        count += 1
                return count
            
            finger_count = count_extended_fingers(hand)
            if 1 <= finger_count <= 5:
                return str(finger_count)
            
            # Single hand ISL gestures
            fingers = get_finger_states(hand)
            
            # HAAN (Yes): Thumb up
            if fingers[0] and not any(fingers[1:]):  # Only thumb extended
                return "HAAN"
            
            # NAHI (No): Index finger waving
            if fingers[1] and not any(fingers[2:]):  # Only index finger extended
                return "NAHI"
            
            return "Hand detected"
        
        return "Analyzing..."
    
    def detect_bsl_gestures(self, hand_landmarks_list, pose_landmarks):
        """BSL-specific gesture detection using rule-based approach"""
        if not hand_landmarks_list:
            return "No hands detected"
        
        hand_count = len(hand_landmarks_list)
        
        # Basic BSL gesture detection based on hand configuration
        if hand_count == 2:
            # Two-handed BSL gestures
            left_hand = hand_landmarks_list[0]
            right_hand = hand_landmarks_list[1]
            
            # Simple finger state detection
            def get_finger_states(hand_landmarks):
                # Finger tip indices
                tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
                # Corresponding pip joints
                pips = [3, 6, 10, 14, 18]
                
                states = []
                for tip, pip in zip(tips, pips):
                    tip_y = hand_landmarks.landmark[tip].y
                    pip_y = hand_landmarks.landmark[pip].y
                    # Finger is extended if tip is above pip
                    states.append(tip_y < pip_y)
                return states
            
            left_fingers = get_finger_states(left_hand)
            right_fingers = get_finger_states(right_hand)
            
            # BSL Alphabet detection
            # A: Both hands closed fists
            if not any(left_fingers) and not any(right_fingers):
                return "A"
            
            # B: Both hands flat, palms forward
            if all(left_fingers[1:]) and all(right_fingers[1:]):  # All fingers except thumb extended
                return "B"
            
            # C: Both hands in C shape
            if (left_fingers[0] and left_fingers[1] and left_fingers[2] and 
                right_fingers[0] and right_fingers[1] and right_fingers[2]):
                return "C"
            
            # Common BSL words
            # HELLO: Wave gesture (one hand moving)
            if sum(left_fingers[1:]) >= 3 or sum(right_fingers[1:]) >= 3:
                return "HELLO"
            
            # THANK YOU: Flat hand moving from chin
            if all(left_fingers[1:]) or all(right_fingers[1:]):
                return "THANK YOU"
                
            return "Two hands detected"
        
        elif hand_count == 1:
            # Single-hand BSL gestures
            hand = hand_landmarks_list[0]
            
            # Simple finger counting for numbers
            def count_extended_fingers(hand_landmarks):
                tips = [8, 12, 16, 20]  # index, middle, ring, pinky
                pips = [6, 10, 14, 18]
                
                count = 0
                for tip, pip in zip(tips, pips):
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                        count += 1
                return count
            
            finger_count = count_extended_fingers(hand)
            if 1 <= finger_count <= 5:
                return str(finger_count)
            
            return "Hand detected"
        
        return "Analyzing..."
    
    def recognize_gesture(self, frame):
        """Process frame and recognize gesture based on language"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            hand_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)

            gesture_text = "No hand detected"
            
            if hand_results.multi_hand_landmarks:
                # Draw hand landmarks
                hand_colors = [(0, 255, 0), (0, 0, 255)]  # Green for first hand, Red for second
                
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    color = hand_colors[i] if i < len(hand_colors) else (255, 255, 0)
                    
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2)
                    )
                
                # Use ML model if available and loaded
                if self.is_model_loaded:
                    if self.language in ["bsl", "isl"] and len(hand_results.multi_hand_landmarks) == 2:
                        # For BSL and ISL with two hands, use combined landmarks
                        landmarks = self.extract_two_hand_landmarks(hand_results.multi_hand_landmarks)
                    else:
                        # For ASL or single hand, use first hand only
                        landmarks = self.extract_landmarks(hand_results.multi_hand_landmarks[0])
                    
                    if landmarks is not None:
                        gesture_text = self.predict_gesture(landmarks)
                    else:
                        gesture_text = "Feature extraction failed"
                else:
                    # Fallback to rule-based detection
                    if self.language == "bsl":
                        gesture_text = self.detect_bsl_gestures(
                            hand_results.multi_hand_landmarks, 
                            pose_results.pose_landmarks
                        )
                    elif self.language == "isl":
                        gesture_text = self.detect_isl_gestures(
                            hand_results.multi_hand_landmarks, 
                            pose_results.pose_landmarks
                        )
                    else:
                        # ASL fallback
                        hand_count = len(hand_results.multi_hand_landmarks)
                        if hand_count == 1:
                            gesture_text = "Hand detected"
                        elif hand_count == 2:
                            gesture_text = "Two hands detected"
                        else:
                            gesture_text = f"{hand_count} hands detected"
            
            # Smooth gesture detection using history
            self.gesture_history.append(gesture_text)
            smoothed_gesture = self.smooth_gesture()
            
            return frame, smoothed_gesture
            
        except Exception as e:
            print(f"❌ Error in recognize_gesture: {e}")
            return frame, "Error"
    
    def smooth_gesture(self):
        """Smooth gesture detection using history"""
        if not self.gesture_history:
            return "No gesture detected"
        
        # Count occurrences of each gesture
        gesture_counts = {}
        for gesture in self.gesture_history:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Return the most frequent gesture
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        
        # Only return if it appears at least 2 times in history
        if most_common[1] >= 2:
            return most_common[0]
        else:
            return "Analyzing..."
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'pose'):
                self.pose.close()
        except:
            pass
