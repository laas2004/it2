# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import cv2
# import mediapipe as mp
# import pickle

# def train_gesture_model():
#     print("🤖 Training Gesture Recognition Model")
#     print("=" * 40)
    
#     # Check data
#     if not os.path.exists('data'):
#         print("❌ No 'data' directory found!")
#         print("💡 Run collect_data.py first to collect training data")
#         return
    
#     gestures = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]
#     if not gestures:
#         print("❌ No gesture data found!")
#         return
    
#     print(f"🎯 Training on {len(gestures)} gestures: {gestures}")
    
#     # Process data with EXACTLY the same preprocessing as your terminal version
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
#     features, labels = [], []
    
#     for gesture in gestures:
#         gesture_path = os.path.join('data', gesture)
#         images = [f for f in os.listdir(gesture_path) if f.endswith(('.jpg', '.png'))]
        
#         print(f"📁 Processing {gesture} ({len(images)} images)...")
        
#         valid_count = 0
#         for img_file in images:
#             img_path = os.path.join(gesture_path, img_file)
#             image = cv2.imread(img_path)
            
#             if image is not None:
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 results = hands.process(image_rgb)
                
#                 if results.multi_hand_landmarks:
#                     # EXACTLY like your terminal version - x,y,z with wrist normalization
#                     landmarks = []
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         for landmark in hand_landmarks.landmark:
#                             landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
#                     # Wrist normalization (like your terminal version)
#                     wrist = results.multi_hand_landmarks[0].landmark[0]
#                     normalized_landmarks = []
#                     for i in range(0, len(landmarks), 3):
#                         x = landmarks[i] - wrist.x
#                         y = landmarks[i + 1] - wrist.y
#                         z = landmarks[i + 2] - wrist.z
#                         normalized_landmarks.extend([x, y, z])
                    
#                     features.append(normalized_landmarks)  # 63 features
#                     labels.append(gesture)
#                     valid_count += 1
        
#         print(f"   ✅ {valid_count} valid samples")
    
#     if not features:
#         print("❌ No valid training data found!")
#         return
    
#     print(f"📊 Total training samples: {len(features)}")
#     print(f"📏 Each sample has {len(features[0])} features (x,y,z normalized)")
    
#     # Pad features to same length
#     max_len = max(len(f) for f in features)
#     X = np.array([np.pad(f, (0, max_len - len(f)), 'constant') if len(f) < max_len else f[:max_len] for f in features])
    
#     # Encode labels
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(labels)
    
#     print(f"🎯 Classes: {list(label_encoder.classes_)}")
#     for cls in label_encoder.classes_:
#         count = np.sum(y == label_encoder.transform([cls])[0])
#         print(f"   {cls}: {count} samples")
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
#     # Build model (similar to your terminal version)
#     model = keras.Sequential([
#         keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(64, activation='relu'),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(32, activation='relu'),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
#     ])
    
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     print("🚀 Training model...")
#     history = model.fit(
#         X_train, y_train,
#         epochs=50,
#         batch_size=16,
#         validation_data=(X_test, y_test),
#         verbose=1
#     )
    
#     # Evaluate
#     test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#     print(f"✅ Final Test Accuracy: {test_accuracy:.4f}")
    
#     # Save model
#     model.save('asl_gesture_model.h5')
    
#     # Save label encoder
#     with open('label_encoder.pkl', 'wb') as f:
#         pickle.dump(label_encoder, f)
    
#     print("💾 Model saved successfully!")
#     print("🎉 Training complete! You can now use gesture recognition.")

# if __name__ == "__main__":
#     train_gesture_model()



# v1(works well like terminal based!)
# import os
# import numpy as np
# import pickle
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# def train_from_landmarks():
#     """Train model from landmark data (not images)"""
#     print("🤖 Training Gesture Recognition Model (Landmark-based)")
#     print("=" * 60)
    
#     # Load dataset
#     dataset_path = 'data/gesture_dataset.pickle'
#     if not os.path.exists(dataset_path):
#         print("❌ No dataset found!")
#         print("💡 Run the data collector first to collect training data")
#         return
    
#     with open(dataset_path, 'rb') as f:
#         dataset = pickle.load(f)
    
#     X = np.array(dataset['data'])
#     y = np.array(dataset['labels'])
    
#     print(f"\n📊 Dataset loaded:")
#     print(f"   Total samples: {len(X)}")
#     print(f"   Feature dimensions: {X.shape[1]}")
#     print(f"   Unique gestures: {len(np.unique(y))}")
    
#     # Show samples per gesture
#     unique_gestures = {}
#     for label in y:
#         unique_gestures[label] = unique_gestures.get(label, 0) + 1
    
#     print(f"\n📈 Samples per gesture:")
#     for gesture, count in sorted(unique_gestures.items()):
#         status = "✅" if count >= 50 else "⚠️" if count >= 20 else "❌"
#         print(f"   {status} {gesture}: {count} samples")
    
#     # Check for low sample counts
#     min_samples = min(unique_gestures.values())
#     if min_samples < 10:
#         print(f"\n⚠️ Warning: Minimum samples per gesture is only {min_samples}")
#         print("   Recommended: 50+ samples per gesture for good accuracy")
#         proceed = input("\nContinue anyway? (y/n): ").strip().lower()
#         if proceed != 'y':
#             return
    
#     # Encode labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
    
#     # Split data
#     test_size = 0.2 if len(X) >= 50 else 0.15
#     try:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y_encoded, 
#             test_size=test_size, 
#             random_state=42, 
#             stratify=y_encoded
#         )
#     except ValueError as e:
#         print(f"\n❌ Cannot split data: {e}")
#         print("💡 Need at least 2 samples per gesture")
#         return
    
#     print(f"\n📊 Data split: {len(X_train)} train, {len(X_test)} test")
    
#     # Build model
#     model = keras.Sequential([
#         keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(64, activation='relu'),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(32, activation='relu'),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
#     ])
    
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     print("\n🚀 Training model...")
#     print(f"   Epochs: 50")
#     print(f"   Batch size: 16")
    
#     # Add early stopping
#     early_stopping = keras.callbacks.EarlyStopping(
#         monitor='val_accuracy',
#         patience=10,
#         restore_best_weights=True
#     )
    
#     history = model.fit(
#         X_train, y_train,
#         epochs=50,
#         batch_size=16,
#         validation_data=(X_test, y_test),
#         callbacks=[early_stopping],
#         verbose=1
#     )
    
#     # Evaluate
#     test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#     print(f"\n✅ Training complete!")
#     print(f"   Final Test Accuracy: {test_accuracy*100:.2f}%")
    
#     if test_accuracy < 0.7:
#         print("\n⚠️ Low accuracy detected! Tips to improve:")
#         print("   - Collect more samples (50-100 per gesture)")
#         print("   - Use consistent hand positions")
#         print("   - Ensure good lighting when collecting")
#         print("   - Make gestures more distinct from each other")
#     elif test_accuracy < 0.85:
#         print("\n👍 Decent accuracy! To improve further:")
#         print("   - Add more varied samples per gesture")
#         print("   - Ensure consistent gesture form")
#     else:
#         print("\n🎉 Excellent accuracy!")
    
#     # Save model
#     model.save('asl_gesture_model.h5')
#     print("\n💾 Saved: asl_gesture_model.h5")
    
#     # Save label encoder
#     with open('label_encoder.pkl', 'wb') as f:
#         pickle.dump(label_encoder, f)
#     print("💾 Saved: label_encoder.pkl")
    
#     print(f"\n📋 Trained gestures: {list(label_encoder.classes_)}")
#     print("\n" + "=" * 60)
#     print("✅ Model ready to use!")
#     print("   Run your Flask app to test gesture recognition")

# def main():
#     train_from_landmarks()

# if __name__ == "__main__":
#     main()

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