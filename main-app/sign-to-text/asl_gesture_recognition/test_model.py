import pickle
import numpy as np
from tensorflow import keras

# Load and test the new model
model = keras.models.load_model('asl_gesture_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("✅ Model loaded successfully!")
print(f"🎯 Gestures in model: {list(label_encoder.classes_)}")
print(f"📊 Model input shape: {model.input_shape}")