# Speech Emotion Recognition (SER) with 2D CNN

This project implements a Speech Emotion Recognition machine learning model that classifies human speech into four emotions: **Neutral**, **Happy**, **Sad**, and **Angry**. It extracts Mel-Frequency Cepstral Coefficients (MFCCs) from raw audio recordings in the RAVDESS dataset and uses a 2D Convolutional Neural Network (CNN) in TensorFlow/Keras for classification.

## Tech Stack
*   **Python 3.13+**
*   **Librosa**: Audio processing and feature extraction (MFCCs)
*   **TensorFlow / Keras**: Custom 2D Convolutional Neural Network implementation
*   **Scikit-Learn**: Dataset splitting, feature scaling (`StandardScaler`), and metrics calculations
*   **Pandas & NumPy**: Data management and array operations
*   **Matplotlib & Seaborn**: Training progress visualization and confusion matrix plotting

---

## File Structure
```
CodeAlpha_voicerecognition/
├── requirements.txt            # Package dependencies
├── download_dataset.py         # Automates downloading & extracting RAVDESS files
├── extract_features.py         # Filters target emotions & extracts 40 MFCC features
├── train.py                    # Splits data, scales features, trains CNN, saves model & plots
├── predict.py                  # Predicts emotion of any given audio (.wav) sample
├── README.md                   # Project documentation (this file)
└── dataset/                    # Directory where RAVDESS zip extracts (auto-created)
```

---

## Workflow & Step-by-Step Instructions

### Step 1: Install Dependencies
First, install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

### Step 2: Download & Extract RAVDESS Dataset
Run the automatic downloader to fetch the speech-only zip file from Zenodo (~215 MB) and extract it to the `dataset/` directory:
```bash
python download_dataset.py
```

### Step 3: Feature Extraction (MFCC)
Run feature extraction. This script:
1. Filters files to only include target emotions (Neutral, Happy, Sad, Angry).
2. Trims silence from the beginning and end of each file.
3. Standardizes the audio length to `3.5 seconds` (padding shorter recordings with silence, truncating longer ones).
4. Extracts `40` MFCC bands per frame, yielding a consistent `(40, 151)` dimension for each file.
5. Saves features to `X.npy`, labels to `y.npy`, and metadata to `dataset_metadata.csv`.
```bash
python extract_features.py
```

### Step 4: Model Training and Evaluation
Train the CNN model:
1. Splits the features into 80% train and 20% test sets using stratified splits.
2. Standardizes the MFCC features using `StandardScaler` and saves the scaler to `scaler.joblib`.
3. Reshapes the input to `(40, 151, 1)` for 2D convolutions.
4. Builds a deep 2D CNN with batch normalization, max pooling, and dropout layers.
5. Trains using Early Stopping and Model Checkpointing, saving the best iteration to `speech_emotion_recognition_model.keras`.
6. Saves validation performance plots to `training_history.png` and a confusion matrix heatmap to `confusion_matrix.png`.
7. Prints accuracy, precision, recall, and F1-score evaluation metrics.
```bash
python train.py
```

### Step 5: Predict Emotion for Custom Audio
Test the model on any audio recording (`.wav` format) using:
```bash
python predict.py <path_to_audio_file.wav>
```
The script will preprocess the file, load the trained model, and print a horizontal bar chart visualizing the prediction probabilities for each emotion.

Example output:
```
=============================================
         SPEECH EMOTION RECOGNITION
=============================================
File: 03-01-03-01-01-01-01.wav
---------------------------------------------
Neutral    | ██████░░░░░░░░░░░░░░░░░░░ |  22.40%
Happy      | ████████████████░░░░░░░░░ |  65.10%
Sad        | █░░░░░░░░░░░░░░░░░░░░░░░░ |   5.20%
Angry      | ██░░░░░░░░░░░░░░░░░░░░░░░ |   7.30%
---------------------------------------------
PREDICTED EMOTION: HAPPY
CONFIDENCE:        65.10%
=============================================
```
