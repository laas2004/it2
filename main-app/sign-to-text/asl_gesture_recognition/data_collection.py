#------------------------------V1
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.data = []
        self.labels = []
        
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks and convert to feature vector"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        wrist = hand_landmarks.landmark[0]
        normalized = []
        for i in range(0, len(landmarks), 3):
            x = landmarks[i] - wrist.x
            y = landmarks[i + 1] - wrist.y
            z = landmarks[i + 2] - wrist.z
            normalized.extend([x, y, z])
        return np.array(normalized)
    
    def collect_data(self, gesture_name, num_samples=100):
        cap = cv2.VideoCapture(0)
        print(f"Collecting data for gesture: {gesture_name}")
        print(f"Press SPACE to start collecting {num_samples} samples")
        print("Press 'q' to quit")
        
        collecting = False
        sample_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    if collecting and sample_count < num_samples:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.data.append(landmarks)
                        self.labels.append(gesture_name)
                        sample_count += 1
                        
                        cv2.putText(frame, f"Collecting: {sample_count}/{num_samples}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if sample_count >= num_samples:
                            collecting = False
                            print(f"Collected {num_samples} samples for {gesture_name}")
                            break
            
            if not collecting:
                cv2.putText(frame, f"Gesture: {gesture_name}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press SPACE to start collecting", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not collecting:
                collecting = True
                sample_count = 0
                print("Started collecting...")
            elif key == ord('q'):
                break
            elif sample_count >= num_samples:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_data(self):
        if len(self.data) == 0:
            print("No data to save")
            return
        
        dataset = {"data": self.data, "labels": self.labels}
        with open('data/gesture_dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Saved {len(self.data)} samples with {len(set(self.labels))} unique gestures")
    
    def collect_multiple_gestures(self, gestures, samples_per_gesture=100):
        for gesture in gestures:
            input(f"Press Enter to start collecting data for '{gesture}'...")
            self.collect_data(gesture, samples_per_gesture)
        
        self.save_data()

def main():
    collector = DataCollector()
    gestures = ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','HELLO','CALL','GOOD MORNING'] 
    print("ASL Gesture Data Collection")
    print("Available gestures:", gestures)
    
    collector.collect_multiple_gestures(gestures, samples_per_gesture=50)

if __name__ == "__main__":
    main()
