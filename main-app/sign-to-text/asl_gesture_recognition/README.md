# Sign Language Hand Gesture Recognition System

A real-time ASL, BSL, ISL hand gesture recognition system that converts hand gestures to text.
It can also convert text to the respective gestures.
It also does bidirectional translations between ASL, BSL, ISL.

## Features

- Real-time hand gesture detection using webcam
- Machine learning-based gesture classification
- Text output of recognized gestures
- Bidirectional translation of signs in different sign languages
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
python collect_data.py 
```

- Follow the on-screen instructions
- Hold your hand in the gesture position when prompted
- The system will collect multiple samples for each gesture
- You can change the language you want to collect the data as well


### 2. Train the Model

Train the machine learning model using the collected data:

```
python train_model.py
```
- Model accuracy will be displayed
- Trained model will be saved automatically

### 3. Run the Bidirectional Real-Time Sign Language Translation System

```
python app.py
```

## Start the real-time gesture recognition:

- Point your webcam at your hand
- Make gestures to see them recognized
- Hold gestures for 2 seconds to add to sentence
- Press 'c' to clear sentence
- Press 'q' to quit

## Start the real-time text translation:

-Enter the text in the dialog box
-Make sure the text is present in the dataset

## Start the real-time bidirectional sign translation:

-Select a language you know
-Select a language you want the translation in
-Start gesturing in the sign language you know
-Wait for a littlw while for the translated output to be displayed.

## Project Structure

- `app.py` - Main application with GUI
- `gesture_recognition.py` - Core recognition logic
- `collect_data.py` - Data collection utility
- `train_model.py` - Model training script
- `requirements.txt` - Dependencies
- `data/` - Directory for training data
- `models/` - Directory for trained models

## Customization

### Adding New Gestures

1. Modify the gestures list in `collect_data.py`
2. Run data collection for new gestures
3. Retrain the model with updated data

### Improving Accuracy

- Collect more training samples per gesture
- Ensure consistent lighting conditions
- Use clear hand positions

## Troubleshooting

- **Model not found**: Run data collection and training first
- **Low accuracy**: Collect more training data
- **Camera not working**: Check camera permissions
- **Poor recognition**: Ensure good lighting and clear gestures

## Technical Details

- Uses MediaPipe for hand landmark detection
- Extracts 63 features per hand (21 landmarks × 3 coordinates)
- Normalizes landmarks relative to wrist position
