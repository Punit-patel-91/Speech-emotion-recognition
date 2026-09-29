# Speech Emotion Recognition

A Deep Learning project that recognizes human emotions from speech audio using the RAVDESS dataset.

## Project Objective

The main objective of this project is to classify emotions from speech signals using audio features and a Convolutional Neural Network (CNN).

## Emotions Recognized

The model is trained to recognize different emotions from speech, including:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib
- Scikit-learn

## Methodology

1. Load speech audio files from the RAVDESS dataset.
2. Extract MFCC features from audio signals.
3. Prepare and encode emotion labels.
4. Split the dataset into training and testing sets.
5. Train a Convolutional Neural Network (CNN).
6. Evaluate the model using classification metrics.
7. Generate confusion matrix, accuracy curve and loss curve.
8. Save the trained model for future use.

## Model Performance

The trained model achieved approximately:

- Accuracy: 80%
- Precision: 81%
- Recall: 80%
- F1-Score: 80%

## Project Outputs

The project generates:

- Confusion Matrix
- Accuracy Curve
- Loss Curve
- Classification Report
- Trained Keras Model

## Dataset

This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

The dataset is not included in this repository because of its large size.

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
