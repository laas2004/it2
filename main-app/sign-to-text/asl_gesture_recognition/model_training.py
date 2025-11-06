# import pickle
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import os

# class ModelTrainer:
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         self.data = None
#         self.labels = None
        
#     def load_data(self):
#         """Load training data from pickle files"""
#         try:
#             with open('data/gesture_data.pickle', 'rb') as f:
#                 self.data = pickle.load(f)
            
#             with open('data/labels.pickle', 'rb') as f:
#                 label_names = pickle.load(f)
            
#             # Create corresponding labels for the data
#             self.labels = []
#             gesture_counts = {}
            
#             # Count samples per gesture (assuming data was collected in order)
#             for gesture in label_names:
#                 gesture_counts[gesture] = 0
            
#             # We need to reconstruct the labels from the data
#             # This assumes data was collected in the order specified
#             print("Data loaded successfully")
#             print(f"Total samples: {len(self.data)}")
#             print(f"Feature dimensions: {len(self.data[0])}")
            
#             return True
            
#         except FileNotFoundError:
#             print("Training data not found. Please collect data first.")
#             return False
    
#     def load_data_with_labels(self):
#         """Load data with proper label mapping"""
#         try:
#             # Load the raw data
#             with open('data/gesture_data.pickle', 'rb') as f:
#                 raw_data = pickle.load(f)
            
#             # For this implementation, we'll assume labels are stored separately
#             # In a real scenario, you'd modify data_collection.py to save labels with data
            
#             # Create dummy labels for demonstration
#             # In practice, these should be saved during data collection
#             gestures = ['A', 'B', 'C', 'D', 'E', 'Hello', 'Thank you', 'Please']
#             samples_per_gesture = len(raw_data) // len(gestures)
            
#             self.data = raw_data
#             self.labels = []
            
#             for i, gesture in enumerate(gestures):
#                 for j in range(samples_per_gesture):
#                     if len(self.labels) < len(raw_data):
#                         self.labels.append(gesture)
            
#             # Handle any remaining samples
#             while len(self.labels) < len(raw_data):
#                 self.labels.append(gestures[-1])
            
#             print(f"Loaded {len(self.data)} samples with {len(set(self.labels))} unique gestures")
#             return True
            
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return False
    
#     def train_model(self):
#         """Train the gesture recognition model"""
#         if not self.load_data_with_labels():
#             return False
        
#         # Convert data to numpy arrays
#         X = np.array(self.data)
#         y = np.array(self.labels)
        
#         # Split data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
        
#         print("Training model...")
        
#         # Train the model
#         self.model.fit(X_train, y_train)
        
#         # Make predictions on test set
#         y_pred = self.model.predict(X_test)
        
#         # Calculate accuracy
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"Model accuracy: {accuracy:.4f}")
        
#         # Print detailed classification report
#         print("\nClassification Report:")
#         print(classification_report(y_test, y_pred))
        
#         return True
    
#     def save_model(self):
#         """Save the trained model"""
#         if self.model is None:
#             print("No model to save")
#             return
        
#         os.makedirs('models', exist_ok=True)
        
#         with open('models/gesture_model.pickle', 'wb') as f:
#             pickle.dump(self.model, f)
        
#         print("Model saved successfully")
    
#     def train_and_save(self):
#         """Complete training pipeline"""
#         if self.train_model():
#             self.save_model()
#             print("Training completed successfully!")
#         else:
#             print("Training failed!")

# def main():
#     trainer = ModelTrainer()
    
#     print("ASL Gesture Model Training")
#     print("Starting training process...")
    
#     trainer.train_and_save()

# if __name__ == "__main__":
#     main()


#-------------------------------V1
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.data = None
        self.labels = None
        self.label_encoder = None
        
    def load_data(self):
        try:
            with open('data/gesture_dataset.pickle', 'rb') as f:
                dataset = pickle.load(f)
            
            self.data = np.array(dataset["data"])
            self.labels = np.array(dataset["labels"])
            
            print(f"Loaded {len(self.data)} samples with {len(set(self.labels))} unique gestures")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def build_model(self, input_dim, num_classes):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_model(self):
        if not self.load_data():
            return False
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.labels)
        y_categorical = to_categorical(y_encoded)

        X_train, X_test, y_train, y_test = train_test_split(
            self.data, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        self.model = self.build_model(input_dim=self.data.shape[1], num_classes=y_categorical.shape[1])
        
        print("Training model...")
        self.model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
        
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {acc:.4f}")
        return True
    
    def save_model(self):
        if self.model is None:
            print("No model to save")
            return
        
        os.makedirs('models', exist_ok=True)
        self.model.save('models/gesture_model.h5')
        
        with open('models/label_encoder.pickle', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("Model and label encoder saved successfully")
    
    def train_and_save(self):
        if self.train_model():
            self.save_model()
            print("Training completed successfully!")
        else:
            print("Training failed!")

def main():
    trainer = ModelTrainer()
    trainer.train_and_save()

if __name__ == "__main__":
    main()
