# ASL Hand Gesture Recognition System

A real-time American Sign Language (ASL) hand gesture recognition system that converts hand gestures to text using computer vision and machine learning.

## Features

- Real-time hand gesture detection using webcam
- Machine learning-based gesture classification
- Text output of recognized gestures
- Sentence building capability
- Training system for custom gestures
- User-friendly interface

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in requirements.txt

## Installation

1. Clone or download the project files
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Collect Training Data

First, collect training data for the gestures you want to recognize:

```
python data_collection.py
```

- Follow the on-screen instructions
- Hold your hand in the gesture position when prompted
- The system will collect multiple samples for each gesture
- Default gestures: A, B, C, D, E, Hello, Thank you, Please

### 2. Train the Model

Train the machine learning model using the collected data:

```
python model_training.py
```

- The script will train a Random Forest classifier
- Model accuracy will be displayed
- Trained model will be saved automatically

### 3. Run the Recognition System

Start the real-time gesture recognition:

```
python main.py
```

- Point your webcam at your hand
- Make gestures to see them recognized
- Hold gestures for 2 seconds to add to sentence
- Press 'c' to clear sentence
- Press 'q' to quit

## Project Structure

- `main.py` - Main application with GUI
- `gesture_recognition.py` - Core recognition logic
- `data_collection.py` - Data collection utility
- `model_training.py` - Model training script
- `requirements.txt` - Dependencies
- `data/` - Directory for training data
- `models/` - Directory for trained models

## Customization

### Adding New Gestures

1. Modify the gestures list in `data_collection.py`
2. Run data collection for new gestures
3. Retrain the model with updated data

### Improving Accuracy

- Collect more training samples per gesture
- Ensure consistent lighting conditions
- Use clear hand positions
- Train with multiple users

## Troubleshooting

- **Model not found**: Run data collection and training first
- **Low accuracy**: Collect more training data
- **Camera not working**: Check camera permissions
- **Poor recognition**: Ensure good lighting and clear gestures

## Technical Details

- Uses MediaPipe for hand landmark detection
- Extracts 63 features per hand (21 landmarks × 3 coordinates)
- Normalizes landmarks relative to wrist position
- Uses Random Forest classifier for gesture classification
- Confidence threshold of 0.7 for predictions