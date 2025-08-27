# Fitness Tracker

This project is a machine learning-based fitness tracker focused on analyzing push-up exercises using video data. It provides tools for extracting frames, training models, running inference, counting repetitions, and a GUI for user interaction.

## Features
- **Frame Extraction**: Extracts frames from raw videos for both correct and incorrect push-up forms.
- **Model Training**: Trains a MobileNetV2-based model to classify push-up correctness.
- **Inference**: Runs inference on videos to detect and classify push-up form.
- **Repetition Counter**: Counts push-up repetitions using video analysis.
- **Data Splitting**: Splits data into training and validation sets.
- **GUI**: User-friendly interface built with Tkinter.

## Directory Structure
- `data/frames/`: Contains extracted frames for correct and incorrect push-ups.
- `data/raw_videos/`: Raw video files for both classes.
- `data/splits/`: Train/validation splits and class indices.
- `models/`: Trained model files.
- `src/`: Source code for all modules.
    - `frame_extractor.py`: Extracts frames from videos.
    - `gui_tkinter.py`: Tkinter-based GUI.
    - `inference_video.py`: Inference pipeline for video analysis.
    - `make_splits.py`: Script to create train/val splits.
    - `rep_counter.py`: Push-up repetition counter.
    - `train.py`: Model training script.
    - `utils.py`: Utility functions.
- `requirements.txt`: Python dependencies.
- `output.avi`: Example output video.

## Getting Started
1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Prepare data**: Place raw videos in `data/raw_videos/` and run `frame_extractor.py` to generate frames.
3. **Split data**: Use `make_splits.py` to create train/val splits.
4. **Train model**: Run `train.py` to train the push-up correctness classifier.
5. **Inference**: Use `inference_video.py` to analyze new videos.
6. **Count reps**: Run `rep_counter.py` for repetition counting.
7. **Launch GUI**: Start `gui_tkinter.py` for a graphical interface.

## Requirements
- Python 3.8+
- OpenCV
- TensorFlow/Keras
- Tkinter

## License
This project is for educational purposes.
