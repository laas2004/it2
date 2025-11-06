#v2-trying to add options
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def train_from_landmarks(language='asl'):
    """Train a gesture recognition model from landmark data"""
    print(f"🚀 Training {language.upper()} model...")
    
    # Load dataset
    dataset_path = f'data/gesture_dataset_{language}.pickle'
    
    if not os.path.exists(dataset_path):
        print(f"❌ No dataset found for {language.upper()} at {dataset_path}")
        print("   Please collect data first using collect_data.py")
        return
    
    print(f"📁 Loading dataset from {dataset_path}")
    
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Handle inconsistent feature sizes
        X = dataset['data']
        y = np.array(dataset['labels'])
        
        print(f"📊 Raw dataset loaded:")
        print(f"   Samples: {len(X)}")
        print(f"   Gestures: {len(np.unique(y))}")
        print(f"   Gesture names: {list(np.unique(y))}")
        
        if len(X) == 0:
            print("❌ No data found in dataset!")
            return
        
        # Check feature sizes and standardize
        feature_sizes = [len(x) for x in X]
        print(f"   Feature sizes found: {set(feature_sizes)}")
        
        # Standardize to 63 features (single hand) by truncating or padding
        standardized_X = []
        for sample in X:
            if len(sample) > 63:
                # Truncate two-hand data to first hand (63 features)
                standardized_sample = sample[:63]
            elif len(sample) < 63:
                # Pad with zeros if needed
                standardized_sample = np.pad(sample, (0, 63 - len(sample)), 'constant')
            else:
                standardized_sample = sample
            standardized_X.append(standardized_sample)
        
        X = np.array(standardized_X)
        
        print(f"📊 Standardized dataset:")
        print(f"   Final feature size: {X.shape[1]}")
        
        # Check if we have enough samples per class
        unique, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)
        print(f"   Minimum samples per class: {min_samples}")
        
        if min_samples < 10:
            print(f"⚠️ Warning: Some gestures have very few samples (<10)")
            print("   Consider collecting more data for better accuracy")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"🎯 Training model for {len(label_encoder.classes_)} gestures: {list(label_encoder.classes_)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"📈 Data split: {len(X_train)} training, {len(X_test)} test samples")
        
        # Train the model
        model = train_neural_network(X_train, y_train, X_test, y_test, len(label_encoder.classes_))
        
        # Evaluate model
        train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        if isinstance(train_accuracy, list):
            train_accuracy = train_accuracy[1]  # For categorical accuracy
            test_accuracy = test_accuracy[1]
        
        print(f"📊 Model Performance:")
        print(f"   Training Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        model_path = f'models/gesture_model_{language}.h5'
        encoder_path = f'models/label_encoder_{language}.pkl'
        
        os.makedirs('models', exist_ok=True)
        
        # Save Keras model
        model.save(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save label encoder
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"💾 Label encoder saved to: {encoder_path}")
        
        # Also save as fallback generic model if it's ASL
        if language == 'asl':
            model.save('asl_gesture_model.h5')
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
            print("💾 Also saved as generic ASL model")
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

def train_neural_network(X_train, y_train, X_test, y_test, num_classes):
    """Train a neural network model"""
    print("🧠 Training neural network...")
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return model

if __name__ == "__main__":
    # Train models for all languages that have data
    languages = ['asl', 'bsl', 'isl']
    
    for language in languages:
        dataset_path = f'data/gesture_dataset_{language}.pickle'
        if os.path.exists(dataset_path):
            print(f"\n{'='*50}")
            train_from_landmarks(language)
        else:
            print(f"\n⚠️ No dataset found for {language.upper()}")
