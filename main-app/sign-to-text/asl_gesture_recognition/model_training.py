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
