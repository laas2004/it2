# import cv2
# import numpy as np
# from gesture_recognition import GestureRecognizer
# import time

# class ASLRecognitionApp:
#     def __init__(self):
#         self.recognizer = GestureRecognizer()
#         self.cap = cv2.VideoCapture(0)
#         self.current_gesture = "No gesture detected"
#         self.gesture_history = []
#         self.sentence = ""
#         self.last_gesture_time = time.time()
        
#         # Set camera properties
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
#     def add_to_sentence(self, gesture):
#         """Add recognized gesture to sentence"""
#         current_time = time.time()
        
#         # Only add if gesture is different from last or enough time has passed
#         if (gesture != self.current_gesture or 
#             current_time - self.last_gesture_time > 2.0):
            
#             if gesture not in ["No hand detected", "Unknown"]:
#                 self.sentence += gesture + " "
#                 self.gesture_history.append(gesture)
#                 self.last_gesture_time = current_time
                
#                 # Keep only last 10 gestures in history
#                 if len(self.gesture_history) > 10:
#                     self.gesture_history.pop(0)
    
#     def clear_sentence(self):
#         """Clear the current sentence"""
#         self.sentence = ""
#         self.gesture_history = []
    
#     def display_interface(self, frame):
#         """Display the user interface on the frame"""
#         height, width = frame.shape[:2]
        
#         # Create overlay for text
#         overlay = frame.copy()
        
#         # Draw background rectangles for text
#         cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#         cv2.rectangle(overlay, (10, height - 80), (width - 10, height - 10), (0, 0, 0), -1)
        
#         # Blend overlay with frame
#         frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
#         # Display current gesture
#         cv2.putText(frame, f"Current Gesture: {self.current_gesture}", 
#                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
#         # Display sentence
#         cv2.putText(frame, "Sentence:", 
#                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
#         # Word wrap for long sentences
#         sentence_words = self.sentence.split()
#         line_length = 40
#         lines = []
#         current_line = ""
        
#         for word in sentence_words:
#             if len(current_line + word) <= line_length:
#                 current_line += word + " "
#             else:
#                 lines.append(current_line.strip())
#                 current_line = word + " "
        
#         if current_line:
#             lines.append(current_line.strip())
        
#         for i, line in enumerate(lines[-2:]):  # Show only last 2 lines
#             cv2.putText(frame, line, 
#                        (20, 95 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         # Display instructions
#         cv2.putText(frame, "Controls: 'c' - Clear sentence, 'q' - Quit", 
#                    (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
#         cv2.putText(frame, "Hold gesture for 2 seconds to add to sentence", 
#                    (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
#         return frame
    
#     def run(self):
#         """Main application loop"""
#         print("ASL Gesture Recognition Started")
#         print("Press 'c' to clear sentence")
#         print("Press 'q' to quit")
        
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Failed to capture frame")
#                 break
            
#             # Flip frame horizontally for mirror effect
#             frame = cv2.flip(frame, 1)
            
#             # Recognize gesture
#             processed_frame, detected_gesture = self.recognizer.recognize_gesture(frame)
            
#             # Update current gesture
#             self.current_gesture = detected_gesture
            
#             # Add to sentence if appropriate
#             self.add_to_sentence(detected_gesture)
            
#             # Display interface
#             display_frame = self.display_interface(processed_frame)
            
#             # Show frame
#             cv2.imshow('ASL Gesture Recognition', display_frame)
            
#             # Handle key presses
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('c'):
#                 self.clear_sentence()
#                 print("Sentence cleared")
        
#         # Cleanup
#         self.cap.release()
#         cv2.destroyAllWindows()
        
#         # Print final sentence
#         if self.sentence:
#             print(f"\nFinal sentence: {self.sentence}")

# def main():
#     app = ASLRecognitionApp()
#     app.run()

# if __name__ == "__main__":
#     main()

#-----------------------------V1
import cv2
import time
from gesture_recognition import GestureRecognizer

class ASLRecognitionApp:
    def __init__(self):
        self.recognizer = GestureRecognizer()
        self.cap = cv2.VideoCapture(0)
        self.current_gesture = "No gesture detected"
        self.gesture_history = []
        self.sentence = ""
        self.last_gesture_time = time.time()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def add_to_sentence(self, gesture):
        current_time = time.time()
        if (gesture != self.current_gesture or 
            current_time - self.last_gesture_time > 2.0):
            
            if gesture not in ["No hand detected", "Unknown"]:
                self.sentence += gesture + " "
                self.gesture_history.append(gesture)
                self.last_gesture_time = current_time
                if len(self.gesture_history) > 10:
                    self.gesture_history.pop(0)
    
    def clear_sentence(self):
        self.sentence = ""
        self.gesture_history = []
    
    def display_interface(self, frame):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, height - 80), (width - 10, height - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        cv2.putText(frame, f"Current Gesture: {self.current_gesture}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Sentence:", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        words = self.sentence.split()
        line_length = 40
        lines, current_line = [], ""
        for word in words:
            if len(current_line + word) <= line_length:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line: lines.append(current_line.strip())
        
        for i, line in enumerate(lines[-2:]):
            cv2.putText(frame, line, (20, 95 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, "Controls: 'c' - Clear, 'q' - Quit", 
                   (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return frame
    
    def run(self):
        print("ASL Gesture Recognition Started")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            processed, detected_gesture = self.recognizer.recognize_gesture(frame)
            self.current_gesture = detected_gesture
            self.add_to_sentence(detected_gesture)
            
            display_frame = self.display_interface(processed)
            cv2.imshow('ASL Gesture Recognition', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'): self.clear_sentence()
        
        self.cap.release()
        cv2.destroyAllWindows()
        if self.sentence: print(f"\nFinal sentence: {self.sentence}")

def main():
    app = ASLRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()
