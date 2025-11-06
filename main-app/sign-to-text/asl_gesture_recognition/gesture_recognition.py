# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os

# class GestureRecognizer:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.labels = None
#         self.load_model()
    
#     def load_model(self):
#         """Load the trained model and labels"""
#         try:
#             with open('models/gesture_model.pickle', 'rb') as f:
#                 self.model = pickle.load(f)
#             with open('data/labels.pickle', 'rb') as f:
#                 self.labels = pickle.load(f)
#             print("Model loaded successfully")
#         except FileNotFoundError:
#             print("Model not found. Please train the model first.")
    
#     def extract_landmarks(self, hand_landmarks):
#         """Extract hand landmarks and convert to feature vector"""
#         landmarks = []
        
#         # Get all landmark coordinates
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         # Normalize landmarks relative to wrist position
#         wrist = hand_landmarks.landmark[0]
#         normalized_landmarks = []
        
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized_landmarks.extend([x, y, z])
        
#         return np.array(normalized_landmarks)
    
#     def predict_gesture(self, landmarks):
#         """Predict gesture from landmarks"""
#         if self.model is None:
#             return "Model not loaded"
        
#         # Reshape landmarks for prediction
#         landmarks = landmarks.reshape(1, -1)
        
#         # Make prediction
#         prediction = self.model.predict(landmarks)
#         confidence = max(self.model.predict_proba(landmarks)[0])
        
#         # Return prediction if confidence is high enough
#         if confidence > 0.7:
#             #return self.labels[prediction[0]]  trying out the below line
#             return prediction[0]

#         else:
#             return "Unknown"
    
#     def recognize_gesture(self, frame):
#         """Process frame and recognize gesture"""
#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process frame with MediaPipe
#         #results = self.mp_hands.process(rgb_frame) trying out the below line
#         results = self.hands.process(rgb_frame)

#         gesture_text = "No hand detected"
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw hand landmarks
#                 self.mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
#                 )
                
#                 # Extract landmarks and predict gesture
#                 landmarks = self.extract_landmarks(hand_landmarks)
#                 gesture_text = self.predict_gesture(landmarks)
        
#         return frame, gesture_text


#-------------------------------V1
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os
# from collections import deque
# from tensorflow.keras.models import load_model

# class GestureRecognizer:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.label_encoder = None
#         self.prediction_queue = deque(maxlen=5)
#         self.load_model()
    
#     def load_model(self):
#         try:
#             self.model = load_model('models/gesture_model.h5')
#             with open('models/label_encoder.pickle', 'rb') as f:
#                 self.label_encoder = pickle.load(f)
#             print("Model loaded successfully")
#         except Exception as e:
#             print(f"Error loading model: {e}")
    
#     def extract_landmarks(self, hand_landmarks):
#         landmarks = []
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         wrist = hand_landmarks.landmark[0]
#         normalized = []
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized.extend([x, y, z])
        
#         return np.array(normalized)
    
#     def predict_gesture(self, landmarks):
#         if self.model is None:
#             return "Model not loaded"
        
#         landmarks = landmarks.reshape(1, -1)
#         probs = self.model.predict(landmarks, verbose=0)[0]
#         pred_class = np.argmax(probs)
#         pred_label = self.label_encoder.inverse_transform([pred_class])[0]
        
#         self.prediction_queue.append(pred_label)
#         # Majority vote smoothing
#         most_common = max(set(self.prediction_queue), key=self.prediction_queue.count)
#         confidence = np.max(probs)
        
#         return most_common if confidence > 0.7 else "Unknown"
    
#     def recognize_gesture(self, frame):
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(rgb_frame)

#         gesture_text = "No hand detected"
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 self.mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
#                 )
#                 landmarks = self.extract_landmarks(hand_landmarks)
#                 gesture_text = self.predict_gesture(landmarks)
        
#         return frame, gesture_text

# ------------v2
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import pickle
# import time
# from collections import deque

# class EnhancedGestureRecognizer:
#     def __init__(self, model_path='enhanced_gesture_model.h5', tflite_path='gesture_model.tflite'):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5,
#             model_complexity=1
#         )
#         self.mp_draw = mp.solutions.drawing_utils
        
#         # Load model
#         self.use_tflite = True
#         try:
#             # Try to load TensorFlow Lite model first (faster)
#             self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
#             self.interpreter.allocate_tensors()
#             self.input_details = self.interpreter.get_input_details()
#             self.output_details = self.interpreter.get_output_details()
#             print("✅ TensorFlow Lite model loaded successfully")
#         except:
#             # Fall back to Keras model
#             self.use_tflite = False
#             self.model = tf.keras.models.load_model(model_path)
#             print("✅ Keras model loaded successfully")
        
#         # Load label encoder
#         with open('label_encoder.pkl', 'rb') as f:
#             self.label_encoder = pickle.load(f)
        
#         # Gesture smoothing
#         self.gesture_buffer = deque(maxlen=5)  # Keep last 5 predictions
#         self.confidence_threshold = 0.7  # Minimum confidence to accept prediction
        
#         # Performance tracking
#         self.frame_count = 0
#         self.processing_times = deque(maxlen=30)
        
#         print(f"🎯 Loaded {len(self.label_encoder.classes_)} gestures: {list(self.label_encoder.classes_)}")
    
#     def extract_enhanced_features(self, frame):
#         """Extract features with better hand detection"""
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(frame_rgb)
        
#         if not results.multi_hand_landmarks:
#             return None
        
#         features = []
        
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Extract all landmarks
#             landmarks = []
#             for landmark in hand_landmarks.landmark:
#                 landmarks.extend([landmark.x, landmark.y, landmark.z])
            
#             # Enhanced features
#             if len(landmarks) >= 21 * 3:
#                 # Hand geometry features
#                 palm_center_x = np.mean([landmarks[i] for i in range(0, 63, 3)])
#                 palm_center_y = np.mean([landmarks[i+1] for i in range(0, 63, 3)])
                
#                 # Finger distances
#                 thumb_tip = np.array([landmarks[3], landmarks[4]])
#                 index_tip = np.array([landmarks[6], landmarks[7]])
#                 middle_tip = np.array([landmarks[9], landmarks[10]])
                
#                 thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
#                 index_middle_dist = np.linalg.norm(index_tip - middle_tip)
                
#                 # Hand orientation
#                 wrist = np.array([landmarks[0], landmarks[1]])
#                 middle_mcp = np.array([landmarks[18], landmarks[19]])
#                 hand_orientation = np.linalg.norm(wrist - middle_mcp)
                
#                 additional_features = [
#                     palm_center_x, palm_center_y,
#                     thumb_index_dist, index_middle_dist,
#                     hand_orientation
#                 ]
                
#                 features.extend(landmarks)
#                 features.extend(additional_features)
        
#         return np.array(features) if features else None
    
#     def preprocess_features(self, features):
#         """Preprocess features for model input"""
#         if features is None:
#             return None
        
#         # Normalize features
#         features_normalized = (features - np.mean(features)) / np.std(features)
        
#         # Ensure consistent input size (pad if necessary)
#         expected_size = self.input_details[0]['shape'][1] if self.use_tflite else self.model.input_shape[1]
        
#         if len(features_normalized) < expected_size:
#             padded = np.pad(features_normalized, (0, expected_size - len(features_normalized)), 'constant')
#             return padded
#         else:
#             return features_normalized[:expected_size]
    
#     def smooth_prediction(self, current_prediction, confidence):
#         """Apply temporal smoothing to predictions"""
#         if confidence < self.confidence_threshold:
#             return "Unknown"
        
#         self.gesture_buffer.append(current_prediction)
        
#         # Return most frequent prediction in buffer
#         if len(self.gesture_buffer) == self.gesture_buffer.maxlen:
#             counts = np.bincount([self.label_encoder.transform([g])[0] for g in self.gesture_buffer])
#             most_common_idx = np.argmax(counts)
#             return self.label_encoder.classes_[most_common_idx]
        
#         return current_prediction
    
#     def recognize_gesture(self, frame):
#         """Recognize gesture in frame with enhanced processing"""
#         start_time = time.time()
        
#         # Process frame
#         processed_frame = frame.copy()
#         features = self.extract_enhanced_features(frame)
        
#         if features is None:
#             self.processing_times.append(time.time() - start_time)
#             return processed_frame, "No hand detected"
        
#         # Preprocess features
#         processed_features = self.preprocess_features(features)
#         if processed_features is None:
#             self.processing_times.append(time.time() - start_time)
#             return processed_frame, "Unknown"
        
#         # Make prediction
#         try:
#             if self.use_tflite:
#                 # TensorFlow Lite inference
#                 input_data = np.array([processed_features], dtype=np.float32)
#                 self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
#                 self.interpreter.invoke()
#                 predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
#             else:
#                 # Keras model inference
#                 predictions = self.model.predict(np.array([processed_features]), verbose=0)
            
#             # Get prediction
#             predicted_class_idx = np.argmax(predictions[0])
#             confidence = predictions[0][predicted_class_idx]
#             gesture = self.label_encoder.classes_[predicted_class_idx]
            
#             # Apply smoothing
#             smoothed_gesture = self.smooth_prediction(gesture, confidence)
            
#             # Draw hand landmarks
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(frame_rgb)
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     self.mp_draw.draw_landmarks(
#                         processed_frame, 
#                         hand_landmarks, 
#                         self.mp_hands.HAND_CONNECTIONS,
#                         self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                         self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
#                     )
            
#             # Add prediction info to frame
#             cv2.putText(processed_frame, f"Gesture: {smoothed_gesture}", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             cv2.putText(processed_frame, f"Confidence: {confidence:.2f}", 
#                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             # Performance info
#             processing_time = time.time() - start_time
#             self.processing_times.append(processing_time)
#             avg_time = np.mean(self.processing_times) if self.processing_times else 0
#             fps = 1 / avg_time if avg_time > 0 else 0
            
#             cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
#                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
#             self.frame_count += 1
            
#             return processed_frame, smoothed_gesture
            
#         except Exception as e:
#             print(f"❌ Prediction error: {e}")
#             self.processing_times.append(time.time() - start_time)
#             return processed_frame, "Unknown"

# def main():
#     recognizer = EnhancedGestureRecognizer()
    
#     # Test with webcam
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
#     print("🎥 Starting gesture recognition...")
#     print("Press 'q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         processed_frame, gesture = recognizer.recognize_gesture(frame)
        
#         cv2.imshow('Enhanced Gesture Recognition', processed_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

#-----v3-trying to fix the screw up
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import pickle
# import time
# from collections import deque
# import os

# class GestureRecognizer:
#     def __init__(self, model_path='asl_gesture_model.tflite', label_encoder_path='label_encoder.pkl'):
#         print("🔄 Initializing GestureRecognizer...")
        
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_draw = mp.solutions.drawing_utils
        
#         # Initialize variables
#         self.model_loaded = False
#         self.label_encoder = None
#         self.interpreter = None
        
#         # Try to load the model
#         try:
#             if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#                 # Load TensorFlow Lite model
#                 self.interpreter = tf.lite.Interpreter(model_path=model_path)
#                 self.interpreter.allocate_tensors()
#                 self.input_details = self.interpreter.get_input_details()
#                 self.output_details = self.interpreter.get_output_details()
                
#                 # Load label encoder
#                 with open(label_encoder_path, 'rb') as f:
#                     self.label_encoder = pickle.load(f)
                
#                 self.model_loaded = True
#                 print(f"✅ Model loaded successfully! Available gestures: {list(self.label_encoder.classes_)}")
#             else:
#                 print(f"❌ Model files not found: {model_path}, {label_encoder_path}")
#                 print("💡 Please train the model first using collect_training_data.py")
                
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             self.model_loaded = False
        
#         # Gesture processing
#         self.gesture_buffer = deque(maxlen=5)
#         self.confidence_threshold = 0.6
#         self.current_gesture = "No hand detected"
#         self.last_prediction_time = time.time()
#         self.prediction_interval = 0.5
        
#     def extract_landmarks(self, frame):
#         """Extract hand landmarks from frame"""
#         try:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(frame_rgb)
            
#             if not results.multi_hand_landmarks:
#                 return None
            
#             landmarks = []
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Extract x, y coordinates of all 21 landmarks
#                 for landmark in hand_landmarks.landmark:
#                     landmarks.extend([landmark.x, landmark.y])
            
#             return np.array(landmarks, dtype=np.float32)
            
#         except Exception as e:
#             return None
    
#     def predict_gesture(self, landmarks):
#         """Predict gesture from landmarks"""
#         if not self.model_loaded or landmarks is None:
#             return "Unknown", 0.0
        
#         try:
#             # Get expected input size
#             expected_size = self.input_details[0]['shape'][1]
            
#             # Pad or truncate landmarks
#             if len(landmarks) < expected_size:
#                 processed_landmarks = np.pad(landmarks, (0, expected_size - len(landmarks)), 'constant')
#             else:
#                 processed_landmarks = landmarks[:expected_size]
            
#             # Prepare input
#             input_data = np.array([processed_landmarks], dtype=np.float32)
            
#             # Run inference
#             self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
#             self.interpreter.invoke()
#             predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            
#             # Get best prediction
#             predicted_idx = np.argmax(predictions[0])
#             confidence = predictions[0][predicted_idx]
            
#             if predicted_idx < len(self.label_encoder.classes_):
#                 gesture = self.label_encoder.classes_[predicted_idx]
#                 return gesture, confidence
#             else:
#                 return "Unknown", confidence
                
#         except Exception as e:
#             return "Unknown", 0.0
    
#     def recognize_gesture(self, frame):
#         """Main function to recognize gesture"""
#         processed_frame = frame.copy()
        
#         # Extract landmarks
#         landmarks = self.extract_landmarks(frame)
#         hand_detected = landmarks is not None
        
#         # Draw hand landmarks
#         if hand_detected:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(frame_rgb)
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     self.mp_draw.draw_landmarks(
#                         processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
#                         self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                         self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
#                     )
        
#         # Make prediction if model is loaded and it's time
#         current_time = time.time()
#         should_predict = (current_time - self.last_prediction_time) >= self.prediction_interval
        
#         if self.model_loaded and should_predict and hand_detected:
#             gesture, confidence = self.predict_gesture(landmarks)
            
#             if confidence > self.confidence_threshold:
#                 self.gesture_buffer.append(gesture)
#                 self.last_prediction_time = current_time
                
#                 # Get most frequent gesture from buffer
#                 if len(self.gesture_buffer) > 0:
#                     unique, counts = np.unique(list(self.gesture_buffer), return_counts=True)
#                     self.current_gesture = unique[np.argmax(counts)]
        
#         # Update display text
#         if not hand_detected:
#             self.current_gesture = "No hand detected"
        
#         # Add text overlay
#         cv2.putText(processed_frame, f"Gesture: {self.current_gesture}", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
#         status_text = "Hand: ✅ Detected" if hand_detected else "Hand: ❌ Not detected"
#         cv2.putText(processed_frame, status_text, 
#                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
#         # Add model status
#         model_status = "Model: ✅ Loaded" if self.model_loaded else "Model: ❌ Not loaded"
#         cv2.putText(processed_frame, model_status, 
#                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
#                    (0, 255, 0) if self.model_loaded else (0, 0, 255), 2)
        
#         return processed_frame, self.current_gesture
    
#     def is_model_loaded(self):
#         return self.model_loaded
    
#     def get_available_gestures(self):
#         if self.label_encoder is not None:
#             return list(self.label_encoder.classes_)
#         return []

# def main():
#     recognizer = GestureRecognizer()
    
#     if not recognizer.is_model_loaded():
#         print("❌ Please train the model first!")
#         return
    
#     cap = cv2.VideoCapture(0)
#     print("🎥 Press 'q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         processed_frame, gesture = recognizer.recognize_gesture(frame)
        
#         cv2.imshow('Gesture Recognition', processed_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# v4-trying terminal based 
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import pickle
# import time
# from collections import deque
# import os

# class GestureRecognizer:
#     def __init__(self, model_path='asl_gesture_model.h5', label_encoder_path='label_encoder.pkl'):
#         print("🔄 Initializing Gesture Recognizer...")
        
#         # MediaPipe setup (same as your terminal version)
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
        
#         # Load model (same as your terminal version)
#         self.model_loaded = False
#         self.model = None
#         self.label_encoder = None
#         self.prediction_queue = deque(maxlen=5)
        
#         try:
#             if os.path.exists(model_path):
#                 self.model = tf.keras.models.load_model(model_path)
#                 with open(label_encoder_path, 'rb') as f:
#                     self.label_encoder = pickle.load(f)
                
#                 self.model_loaded = True
#                 print(f"✅ Model loaded! Gestures: {list(self.label_encoder.classes_)}")
#             else:
#                 print("❌ Model not found. Please train first.")
                
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
        
#         # Tracking
#         self.current_gesture = "No hand detected"
#         self.last_prediction_time = time.time()
#         self.prediction_interval = 0.3
    
#     def extract_landmarks(self, hand_landmarks):
#         """EXACTLY like your terminal version - x,y,z with wrist normalization"""
#         landmarks = []
        
#         # Get all landmark coordinates (x, y, z)
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         # Normalize landmarks relative to wrist position (like your terminal version)
#         wrist = hand_landmarks.landmark[0]
#         normalized_landmarks = []
        
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized_landmarks.extend([x, y, z])
        
#         return np.array(normalized_landmarks)  # 63 features
    
#     def predict_gesture(self, landmarks):
#         """EXACTLY like your terminal version"""
#         if not self.model_loaded or landmarks is None:
#             return "Unknown", 0.0
        
#         try:
#             landmarks = landmarks.reshape(1, -1)
#             probs = self.model.predict(landmarks, verbose=0)[0]
#             pred_class = np.argmax(probs)
#             pred_label = self.label_encoder.inverse_transform([pred_class])[0]
#             confidence = np.max(probs)
            
#             # Majority vote smoothing (like your terminal version)
#             self.prediction_queue.append(pred_label)
#             most_common = max(set(self.prediction_queue), key=self.prediction_queue.count)
            
#             return most_common if confidence > 0.7 else "Unknown", confidence
            
#         except Exception as e:
#             print(f"❌ Prediction error: {e}")
#             return "Unknown", 0.0
    
#     def recognize_gesture(self, frame):
#         """Process frame and recognize gesture"""
#         processed_frame = frame.copy()
        
#         # Convert to RGB and process
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(frame_rgb)
        
#         hand_detected = results.multi_hand_landmarks is not None
#         gesture_text = "No hand detected"
#         confidence_text = ""
        
#         if hand_detected:
#             # Draw landmarks (like your terminal version)
#             for hand_landmarks in results.multi_hand_landmarks:
#                 self.mp_drawing.draw_landmarks(
#                     processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
#                 )
            
#             # Predict at intervals
#             current_time = time.time()
#             if current_time - self.last_prediction_time >= self.prediction_interval:
#                 landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
#                 gesture, confidence = self.predict_gesture(landmarks)
#                 self.current_gesture = gesture
#                 self.last_prediction_time = current_time
#                 confidence_text = f"Confidence: {confidence:.2f}"
        
#         # Update display
#         if hand_detected and self.model_loaded:
#             gesture_text = self.current_gesture
        
#         # Add overlay text
#         cv2.putText(processed_frame, f"Gesture: {gesture_text}", 
#                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         if confidence_text:
#             cv2.putText(processed_frame, confidence_text, 
#                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
#         status_text = "Hand: ✅ Detected" if hand_detected else "Hand: ❌ Not detected"
#         cv2.putText(processed_frame, status_text, 
#                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
#         return processed_frame, gesture_text
    
#     def is_model_loaded(self):
#         return self.model_loaded

# # Test function
# def main():
#     recognizer = GestureRecognizer()
    
#     if not recognizer.is_model_loaded():
#         print("❌ Model not loaded. Run train_model.py first.")
#         return
    
#     cap = cv2.VideoCapture(0)
#     print("🎥 Press 'q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
#         processed_frame, gesture = recognizer.recognize_gesture(frame)
        
#         cv2.imshow('Gesture Recognition', processed_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# v5-trying old code
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os

# class GestureRecognizer:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.labels = None
#         self.load_model()
    
#     def load_model(self):
#         """Load the trained model and labels"""
#         try:
#             with open('models/gesture_model.pickle', 'rb') as f:
#                 self.model = pickle.load(f)
#             with open('data/labels.pickle', 'rb') as f:
#                 self.labels = pickle.load(f)
#             print("Model loaded successfully")
#         except FileNotFoundError:
#             print("Model not found. Please train the model first.")
    
#     def extract_landmarks(self, hand_landmarks):
#         """Extract hand landmarks and convert to feature vector"""
#         landmarks = []
        
#         # Get all landmark coordinates
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         # Normalize landmarks relative to wrist position
#         wrist = hand_landmarks.landmark[0]
#         normalized_landmarks = []
        
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized_landmarks.extend([x, y, z])
        
#         return np.array(normalized_landmarks)
    
#     def predict_gesture(self, landmarks):
#         """Predict gesture from landmarks"""
#         if self.model is None:
#             return "Model not loaded"
        
#         # Reshape landmarks for prediction
#         landmarks = landmarks.reshape(1, -1)
        
#         # Make prediction
#         prediction = self.model.predict(landmarks)
#         confidence = max(self.model.predict_proba(landmarks)[0])
        
#         # Return prediction if confidence is high enough
#         if confidence > 0.7:
#             #return self.labels[prediction[0]]  trying out the below line
#             return prediction[0]

#         else:
#             return "Unknown"
    
#     def recognize_gesture(self, frame):
#         """Process frame and recognize gesture"""
#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process frame with MediaPipe
#         #results = self.mp_hands.process(rgb_frame) trying out the below line
#         results = self.hands.process(rgb_frame)

#         gesture_text = "No hand detected"
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Draw hand landmarks
#                 self.mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
#                 )
                
#                 # Extract landmarks and predict gesture
#                 landmarks = self.extract_landmarks(hand_landmarks)
#                 gesture_text = self.predict_gesture(landmarks)
        
#         return frame, gesture_text


#-------------------------------V1
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os
# from collections import deque
# from tensorflow.keras.models import load_model

# class GestureRecognizer:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.label_encoder = None
#         self.prediction_queue = deque(maxlen=5)
#         self.load_model()
    
#     def load_model(self):
#         try:
#             self.model = load_model('models/gesture_model.h5')
#             with open('models/label_encoder.pickle', 'rb') as f:
#                 self.label_encoder = pickle.load(f)
#             print("Model loaded successfully")
#         except Exception as e:
#             print(f"Error loading model: {e}")
    
#     def extract_landmarks(self, hand_landmarks):
#         landmarks = []
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         wrist = hand_landmarks.landmark[0]
#         normalized = []
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized.extend([x, y, z])
        
#         return np.array(normalized)
    
#     def predict_gesture(self, landmarks):
#         if self.model is None:
#             return "Model not loaded"
        
#         landmarks = landmarks.reshape(1, -1)
#         probs = self.model.predict(landmarks, verbose=0)[0]
#         pred_class = np.argmax(probs)
#         pred_label = self.label_encoder.inverse_transform([pred_class])[0]
        
#         self.prediction_queue.append(pred_label)
#         # Majority vote smoothing
#         most_common = max(set(self.prediction_queue), key=self.prediction_queue.count)
#         confidence = np.max(probs)
        
#         return most_common if confidence > 0.7 else "Unknown"
    
#     def recognize_gesture(self, frame):
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(rgb_frame)

#         gesture_text = "No hand detected"
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 self.mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
#                 )
#                 landmarks = self.extract_landmarks(hand_landmarks)
#                 gesture_text = self.predict_gesture(landmarks)
        
#         return frame, gesture_text

# v6-trying to make signtotext work(opens camera and places landmarks but can't recognize yet)
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os

# class GestureRecognizer:
#     def __init__(self):
#         print("🔧 Initializing GestureRecognizer...")
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.labels = None
#         self.is_model_loaded = False  # IMPORTANT: This attribute is required
#         self.load_model()
#         print("✅ GestureRecognizer initialized")
    
#     def load_model(self):
#         """Load the trained model and labels"""
#         try:
#             # Try to load the model
#             model_path = 'models/gesture_model.pickle'
#             labels_path = 'data/labels.pickle'
            
#             if os.path.exists(model_path):
#                 with open(model_path, 'rb') as f:
#                     self.model = pickle.load(f)
#                 print("✅ Model loaded successfully")
#             else:
#                 print(f"⚠️ Model not found at {model_path}")
#                 print("   Hand landmarks will be shown, but gesture prediction will be limited")
#                 self.is_model_loaded = False
#                 return
            
#             if os.path.exists(labels_path):
#                 with open(labels_path, 'rb') as f:
#                     self.labels = pickle.load(f)
#                 print("✅ Labels loaded successfully")
#                 self.is_model_loaded = True
#             else:
#                 print(f"⚠️ Labels not found at {labels_path}")
#                 self.is_model_loaded = False
                
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             self.is_model_loaded = False
#             print("⚠️ Gesture recognition will show hand landmarks but predictions may be limited")
    
#     def extract_landmarks(self, hand_landmarks):
#         """Extract hand landmarks and convert to feature vector"""
#         landmarks = []
        
#         # Get all landmark coordinates
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         # Normalize landmarks relative to wrist position
#         wrist = hand_landmarks.landmark[0]
#         normalized_landmarks = []
        
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized_landmarks.extend([x, y, z])
        
#         return np.array(normalized_landmarks)
    
#     def predict_gesture(self, landmarks):
#         """Predict gesture from landmarks"""
#         if not self.is_model_loaded or self.model is None:
#             return "Hand detected"
        
#         try:
#             # Reshape landmarks for prediction
#             landmarks = landmarks.reshape(1, -1)
            
#             # Make prediction
#             prediction = self.model.predict(landmarks)
#             confidence = max(self.model.predict_proba(landmarks)[0])
            
#             # Return prediction if confidence is high enough
#             if confidence > 0.7:
#                 predicted_label = str(prediction[0])
#                 return predicted_label
#             else:
#                 return "Unknown"
#         except Exception as e:
#             print(f"❌ Prediction error: {e}")
#             return "Hand detected"
    
#     def recognize_gesture(self, frame):
#         """Process frame and recognize gesture"""
#         try:
#             # Convert BGR to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Process frame with MediaPipe
#             results = self.hands.process(rgb_frame)

#             gesture_text = "No hand detected"
            
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Draw hand landmarks on the frame
#                     self.mp_drawing.draw_landmarks(
#                         frame, 
#                         hand_landmarks, 
#                         self.mp_hands.HAND_CONNECTIONS,
#                         self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                         self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
#                     )
                    
#                     # Extract landmarks and predict gesture
#                     landmarks = self.extract_landmarks(hand_landmarks)
#                     gesture_text = self.predict_gesture(landmarks)
            
#             return frame, gesture_text
            
#         except Exception as e:
#             print(f"❌ Error in recognize_gesture: {e}")
#             return frame, "Error"
    
#     def __del__(self):
#         """Cleanup when object is destroyed"""
#         try:
#             if hasattr(self, 'hands'):
#                 self.hands.close()
#         except:
#             pass

# v7-trying to make it recognize(works!)
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os
# from tensorflow import keras

# class GestureRecognizer:
#     def __init__(self):
#         print("🔧 Initializing GestureRecognizer...")
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.label_encoder = None
#         self.is_model_loaded = False
#         self.load_model()
#         print("✅ GestureRecognizer initialized")
    
#     def load_model(self):
#         """Load the trained Keras model and label encoder"""
#         try:
#             # Try to load Keras model (.h5 format)
#             model_path = 'asl_gesture_model.h5'
#             label_encoder_path = 'label_encoder.pkl'
            
#             if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#                 # Load Keras model
#                 self.model = keras.models.load_model(model_path)
#                 print("✅ Keras model loaded successfully")
                
#                 # Load label encoder
#                 with open(label_encoder_path, 'rb') as f:
#                     self.label_encoder = pickle.load(f)
#                 print(f"✅ Label encoder loaded: {list(self.label_encoder.classes_)}")
                
#                 self.is_model_loaded = True
#             else:
#                 print(f"⚠️ Model files not found:")
#                 print(f"   Looking for: {model_path}")
#                 print(f"   Looking for: {label_encoder_path}")
#                 print(f"   Run train_model.py to train a model first!")
#                 self.is_model_loaded = False
                
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             self.is_model_loaded = False
#             import traceback
#             traceback.print_exc()
    
#     def extract_landmarks(self, hand_landmarks):
#         """Extract hand landmarks - EXACTLY like training"""
#         landmarks = []
        
#         # Get all landmark coordinates (x, y, z)
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         # Normalize landmarks relative to wrist position (EXACTLY like training)
#         wrist = hand_landmarks.landmark[0]
#         normalized_landmarks = []
        
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized_landmarks.extend([x, y, z])
        
#         return np.array(normalized_landmarks)
    
#     def predict_gesture(self, landmarks):
#         """Predict gesture from landmarks using Keras model"""
#         if not self.is_model_loaded or self.model is None:
#             return "Hand detected"
        
#         try:
#             # Ensure landmarks is the right shape
#             # Model expects (1, n_features) where n_features should be 63
#             if len(landmarks) < 63:
#                 # Pad with zeros if needed
#                 landmarks = np.pad(landmarks, (0, 63 - len(landmarks)), 'constant')
#             elif len(landmarks) > 63:
#                 landmarks = landmarks[:63]
            
#             # Reshape for prediction
#             landmarks = landmarks.reshape(1, -1)
            
#             # Make prediction
#             predictions = self.model.predict(landmarks, verbose=0)[0]
#             predicted_class = np.argmax(predictions)
#             confidence = predictions[predicted_class]
            
#             # Return prediction if confidence is high enough
#             if confidence > 0.7:
#                 gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
#                 return gesture_name
#             else:
#                 return "Unknown"
                
#         except Exception as e:
#             print(f"❌ Prediction error: {e}")
#             return "Hand detected"
    
#     def recognize_gesture(self, frame):
#         """Process frame and recognize gesture"""
#         try:
#             # Convert BGR to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Process frame with MediaPipe
#             results = self.hands.process(rgb_frame)

#             gesture_text = "No hand detected"
            
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Draw hand landmarks on the frame
#                     self.mp_drawing.draw_landmarks(
#                         frame, 
#                         hand_landmarks, 
#                         self.mp_hands.HAND_CONNECTIONS,
#                         self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                         self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
#                     )
                    
#                     # Extract landmarks and predict gesture
#                     landmarks = self.extract_landmarks(hand_landmarks)
#                     gesture_text = self.predict_gesture(landmarks)
            
#             return frame, gesture_text
            
#         except Exception as e:
#             print(f"❌ Error in recognize_gesture: {e}")
#             return frame, "Error"
    
#     def __del__(self):
#         """Cleanup when object is destroyed"""
#         try:
#             if hasattr(self, 'hands'):
#                 self.hands.close()
#         except:
#             pass

# v8-trying to fix few issues with v7
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os
# from tensorflow import keras

# class GestureRecognizer:
#     def __init__(self):
#         print("🔧 Initializing GestureRecognizer...")
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.label_encoder = None
#         self.is_model_loaded = False
#         self.load_model()
#         print("✅ GestureRecognizer initialized")
    
#     def load_model(self):
#         """Load the trained Keras model and label encoder"""
#         try:
#             # Try to load Keras model (.h5 format)
#             model_path = 'asl_gesture_model.h5'
#             label_encoder_path = 'label_encoder.pkl'
            
#             if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#                 # Load Keras model
#                 self.model = keras.models.load_model(model_path)
#                 print("✅ Keras model loaded successfully")
                
#                 # Load label encoder
#                 with open(label_encoder_path, 'rb') as f:
#                     self.label_encoder = pickle.load(f)
#                 print(f"✅ Label encoder loaded: {list(self.label_encoder.classes_)}")
                
#                 self.is_model_loaded = True
#             else:
#                 print(f"⚠️ Model files not found:")
#                 print(f"   Looking for: {model_path}")
#                 print(f"   Looking for: {label_encoder_path}")
#                 print(f"   Run train_model.py to train a model first!")
#                 self.is_model_loaded = False
                
#         except Exception as e:
#             print(f"❌ Error loading model: {e}")
#             self.is_model_loaded = False
#             import traceback
#             traceback.print_exc()
    
#     def extract_landmarks(self, hand_landmarks):
#         """Extract hand landmarks - EXACTLY like training"""
#         landmarks = []
        
#         # Get all landmark coordinates (x, y, z)
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         # Normalize landmarks relative to wrist position (EXACTLY like training)
#         wrist = hand_landmarks.landmark[0]
#         normalized_landmarks = []
        
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized_landmarks.extend([x, y, z])
        
#         return np.array(normalized_landmarks)
    
#     def predict_gesture(self, landmarks):
#         """Predict gesture from landmarks using Keras model"""
#         if not self.is_model_loaded or self.model is None:
#             return "Hand detected"
        
#         try:
#             # Ensure landmarks is the right shape
#             # Model expects (1, n_features) where n_features should be 63
#             if len(landmarks) < 63:
#                 # Pad with zeros if needed
#                 landmarks = np.pad(landmarks, (0, 63 - len(landmarks)), 'constant')
#             elif len(landmarks) > 63:
#                 landmarks = landmarks[:63]
            
#             # Reshape for prediction
#             landmarks = landmarks.reshape(1, -1)
            
#             # Make prediction
#             predictions = self.model.predict(landmarks, verbose=0)[0]
#             predicted_class = np.argmax(predictions)
#             confidence = predictions[predicted_class]
            
#             # Return prediction if confidence is high enough
#             if confidence > 0.8:  # Increased from 0.7 for more certainty
#                 gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
#                 return gesture_name
#             elif confidence > 0.6:  # Medium confidence
#                 gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
#                 return f"{gesture_name}"  # Indicate uncertainty
#             else:
#                 return "Unknown"
                
#         except Exception as e:
#             print(f"❌ Prediction error: {e}")
#             return "Hand detected"
    
#     def recognize_gesture(self, frame):
#         """Process frame and recognize gesture"""
#         try:
#             # Convert BGR to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Process frame with MediaPipe
#             results = self.hands.process(rgb_frame)

#             gesture_text = "No hand detected"
            
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # Draw hand landmarks on the frame
#                     self.mp_drawing.draw_landmarks(
#                         frame, 
#                         hand_landmarks, 
#                         self.mp_hands.HAND_CONNECTIONS,
#                         self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                         self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
#                     )
                    
#                     # Extract landmarks and predict gesture
#                     landmarks = self.extract_landmarks(hand_landmarks)
#                     gesture_text = self.predict_gesture(landmarks)
            
#             return frame, gesture_text
            
#         except Exception as e:
#             print(f"❌ Error in recognize_gesture: {e}")
#             return frame, "Error"
    
#     def __del__(self):
#         """Cleanup when object is destroyed"""
#         try:
#             if hasattr(self, 'hands'):
#                 self.hands.close()
#         except:
#             pass


# v9-trying to add options for signtotext (works well with asl,bsl recognition and convertion to text!)
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os
# from tensorflow import keras
# from collections import deque

# class GestureRecognizer:
#     def __init__(self, language="asl"):
#         self.language = language
#         print(f"🔧 Initializing {language.upper()} GestureRecognizer...")
        
#         self.mp_hands = mp.solutions.hands
#         self.mp_pose = mp.solutions.pose
        
#         # Set max hands based on language
#         if language == "bsl":
#             max_hands = 2  # BSL often uses two hands
#         else:
#             max_hands = 1  # ASL typically uses one hand
            
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=max_hands,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
        
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=False,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
        
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.model = None
#         self.label_encoder = None
#         self.is_model_loaded = False
        
#         # Gesture history for smoothing
#         self.gesture_history = deque(maxlen=5)
        
#         self.load_model()
#         print(f"✅ {language.upper()} GestureRecognizer initialized")
    
#     def load_model(self):
#         """Load the trained Keras model and label encoder"""
#         try:
#             # Try to load language-specific model first
#             model_path = f'models/gesture_model_{self.language}.h5'
#             label_encoder_path = f'models/label_encoder_{self.language}.pkl'
            
#             # Fallback to generic ASL model
#             if not os.path.exists(model_path):
#                 model_path = 'asl_gesture_model.h5'
#                 label_encoder_path = 'label_encoder.pkl'
            
#             if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#                 # Load Keras model
#                 self.model = keras.models.load_model(model_path)
#                 print(f"✅ {self.language.upper()} model loaded successfully")
                
#                 # Load label encoder
#                 with open(label_encoder_path, 'rb') as f:
#                     self.label_encoder = pickle.load(f)
#                 print(f"✅ Label encoder loaded: {list(self.label_encoder.classes_)}")
                
#                 self.is_model_loaded = True
#             else:
#                 print(f"⚠️ Model files not found for {self.language.upper()}:")
#                 print(f"   Looking for: {model_path}")
#                 print(f"   Looking for: {label_encoder_path}")
#                 print(f"   Run train_model.py to train a model first!")
#                 self.is_model_loaded = False
                
#         except Exception as e:
#             print(f"❌ Error loading {self.language.upper()} model: {e}")
#             self.is_model_loaded = False
#             import traceback
#             traceback.print_exc()
    
#     def extract_landmarks(self, hand_landmarks):
#         """Extract hand landmarks - EXACTLY like training"""
#         landmarks = []
        
#         # Get all landmark coordinates (x, y, z)
#         for landmark in hand_landmarks.landmark:
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         # Normalize landmarks relative to wrist position (EXACTLY like training)
#         wrist = hand_landmarks.landmark[0]
#         normalized_landmarks = []
        
#         for i in range(0, len(landmarks), 3):
#             x = landmarks[i] - wrist.x
#             y = landmarks[i + 1] - wrist.y
#             z = landmarks[i + 2] - wrist.z
#             normalized_landmarks.extend([x, y, z])
        
#         return np.array(normalized_landmarks)
    
#     def extract_two_hand_landmarks(self, hand_landmarks_list):
#         """Extract landmarks for two hands combined (for BSL)"""
#         if len(hand_landmarks_list) < 2:
#             return None
            
#         # Extract landmarks for each hand
#         hand1_landmarks = self.extract_landmarks(hand_landmarks_list[0])
#         hand2_landmarks = self.extract_landmarks(hand_landmarks_list[1])
        
#         # Combine both hand landmarks
#         combined_landmarks = np.concatenate([hand1_landmarks, hand2_landmarks])
#         return combined_landmarks
    
#     def extract_pose_landmarks(self, pose_landmarks):
#         """Extract pose landmarks for full body context (for BSL)"""
#         if not pose_landmarks:
#             return None
            
#         landmarks = []
#         # Extract key upper body points for sign language
#         key_points = [
#             self.mp_pose.PoseLandmark.LEFT_SHOULDER,
#             self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
#             self.mp_pose.PoseLandmark.LEFT_ELBOW,
#             self.mp_pose.PoseLandmark.RIGHT_ELBOW,
#             self.mp_pose.PoseLandmark.LEFT_WRIST,
#             self.mp_pose.PoseLandmark.RIGHT_WRIST,
#             self.mp_pose.PoseLandmark.NOSE
#         ]
        
#         for point in key_points:
#             landmark = pose_landmarks.landmark[point]
#             landmarks.extend([landmark.x, landmark.y, landmark.z])
        
#         return np.array(landmarks)
    
#     def predict_gesture(self, landmarks):
#         """Predict gesture from landmarks using Keras model"""
#         if not self.is_model_loaded or self.model is None:
#             return "Analyzing..."
        
#         try:
#             # Ensure landmarks is the right shape
#             # Model expects (1, n_features)
#             if len(landmarks) < 63:
#                 # Pad with zeros if needed
#                 landmarks = np.pad(landmarks, (0, 63 - len(landmarks)), 'constant')
#             elif len(landmarks) > 63:
#                 # For BSL two-hand data, use only first hand for now
#                 landmarks = landmarks[:63]
            
#             # Reshape for prediction
#             landmarks = landmarks.reshape(1, -1)
            
#             # Make prediction
#             predictions = self.model.predict(landmarks, verbose=0)[0]
#             predicted_class = np.argmax(predictions)
#             confidence = predictions[predicted_class]
            
#             # Return prediction if confidence is high enough
#             if confidence > 0.7:
#                 gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
#                 return gesture_name
#             else:
#                 return "Unknown"
                
#         except Exception as e:
#             print(f"❌ Prediction error: {e}")
#             return "Analyzing..."
    
#     def detect_bsl_gestures(self, hand_landmarks_list, pose_landmarks):
#         """BSL-specific gesture detection using rule-based approach"""
#         if not hand_landmarks_list:
#             return "No hands detected"
        
#         hand_count = len(hand_landmarks_list)
        
#         # Basic BSL gesture detection based on hand configuration
#         if hand_count == 2:
#             # Two-handed BSL gestures
#             left_hand = hand_landmarks_list[0]
#             right_hand = hand_landmarks_list[1]
            
#             # Simple finger state detection
#             def get_finger_states(hand_landmarks):
#                 # Finger tip indices
#                 tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
#                 # Corresponding pip joints
#                 pips = [3, 6, 10, 14, 18]
                
#                 states = []
#                 for tip, pip in zip(tips, pips):
#                     tip_y = hand_landmarks.landmark[tip].y
#                     pip_y = hand_landmarks.landmark[pip].y
#                     # Finger is extended if tip is above pip
#                     states.append(tip_y < pip_y)
#                 return states
            
#             left_fingers = get_finger_states(left_hand)
#             right_fingers = get_finger_states(right_hand)
            
#             # BSL Alphabet detection
#             # A: Both hands closed fists
#             if not any(left_fingers) and not any(right_fingers):
#                 return "A"
            
#             # B: Both hands flat, palms forward
#             if all(left_fingers[1:]) and all(right_fingers[1:]):  # All fingers except thumb extended
#                 return "B"
            
#             # C: Both hands in C shape
#             if (left_fingers[0] and left_fingers[1] and left_fingers[2] and 
#                 right_fingers[0] and right_fingers[1] and right_fingers[2]):
#                 return "C"
            
#             # Common BSL words
#             # HELLO: Wave gesture (one hand moving)
#             if sum(left_fingers[1:]) >= 3 or sum(right_fingers[1:]) >= 3:
#                 return "HELLO"
            
#             # THANK YOU: Flat hand moving from chin
#             if all(left_fingers[1:]) or all(right_fingers[1:]):
#                 return "THANK YOU"
                
#             return "Two hands detected"
        
#         elif hand_count == 1:
#             # Single-hand BSL gestures
#             hand = hand_landmarks_list[0]
            
#             # Simple finger counting for numbers
#             def count_extended_fingers(hand_landmarks):
#                 tips = [8, 12, 16, 20]  # index, middle, ring, pinky
#                 pips = [6, 10, 14, 18]
                
#                 count = 0
#                 for tip, pip in zip(tips, pips):
#                     if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
#                         count += 1
#                 return count
            
#             finger_count = count_extended_fingers(hand)
#             if 1 <= finger_count <= 5:
#                 return str(finger_count)
            
#             return "Hand detected"
        
#         return "Analyzing..."
    
#     def recognize_gesture(self, frame):
#         """Process frame and recognize gesture based on language"""
#         try:
#             # Convert BGR to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Process frame with MediaPipe
#             hand_results = self.hands.process(rgb_frame)
#             pose_results = self.pose.process(rgb_frame)

#             gesture_text = "No hand detected"
            
#             if hand_results.multi_hand_landmarks:
#                 # Draw hand landmarks
#                 hand_colors = [(0, 255, 0), (0, 0, 255)]  # Green for first hand, Red for second
                
#                 for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
#                     color = hand_colors[i] if i < len(hand_colors) else (255, 255, 0)
                    
#                     self.mp_drawing.draw_landmarks(
#                         frame, 
#                         hand_landmarks, 
#                         self.mp_hands.HAND_CONNECTIONS,
#                         mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=2),
#                         mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2)
#                     )
                
#                 # Use ML model if available and loaded
#                 if self.is_model_loaded:
#                     if self.language == "bsl" and len(hand_results.multi_hand_landmarks) == 2:
#                         # For BSL with two hands, use combined landmarks
#                         landmarks = self.extract_two_hand_landmarks(hand_results.multi_hand_landmarks)
#                     else:
#                         # For ASL or single hand, use first hand only
#                         landmarks = self.extract_landmarks(hand_results.multi_hand_landmarks[0])
                    
#                     if landmarks is not None:
#                         gesture_text = self.predict_gesture(landmarks)
#                     else:
#                         gesture_text = "Feature extraction failed"
#                 else:
#                     # Fallback to rule-based detection
#                     if self.language == "bsl":
#                         gesture_text = self.detect_bsl_gestures(
#                             hand_results.multi_hand_landmarks, 
#                             pose_results.pose_landmarks
#                         )
#                     else:
#                         # ASL fallback
#                         hand_count = len(hand_results.multi_hand_landmarks)
#                         if hand_count == 1:
#                             gesture_text = "Hand detected"
#                         elif hand_count == 2:
#                             gesture_text = "Two hands detected"
#                         else:
#                             gesture_text = f"{hand_count} hands detected"
            
#             # Smooth gesture detection using history
#             self.gesture_history.append(gesture_text)
#             smoothed_gesture = self.smooth_gesture()
            
#             return frame, smoothed_gesture
            
#         except Exception as e:
#             print(f"❌ Error in recognize_gesture: {e}")
#             return frame, "Error"
    
#     def smooth_gesture(self):
#         """Smooth gesture detection using history"""
#         if not self.gesture_history:
#             return "No gesture detected"
        
#         # Count occurrences of each gesture
#         gesture_counts = {}
#         for gesture in self.gesture_history:
#             gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
#         # Return the most frequent gesture
#         most_common = max(gesture_counts.items(), key=lambda x: x[1])
        
#         # Only return if it appears at least 2 times in history
#         if most_common[1] >= 2:
#             return most_common[0]
#         else:
#             return "Analyzing..."
    
#     def __del__(self):
#         """Cleanup when object is destroyed"""
#         try:
#             if hasattr(self, 'hands'):
#                 self.hands.close()
#             if hasattr(self, 'pose'):
#                 self.pose.close()
#         except:
#             pass


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