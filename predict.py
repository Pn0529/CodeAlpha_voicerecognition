import os
import sys
import numpy as np
import librosa
import joblib
import tensorflow as tf

# Configuration (must match training parameters)
SR = 22050
DURATION = 3.5  # seconds
TARGET_SAMPLES = int(SR * DURATION)
N_MFCC = 40
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry']

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        return
        
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
        
    # Check if model and scaler files exist
    if not os.path.exists('speech_emotion_recognition_model.keras'):
        print("Error: Trained model 'speech_emotion_recognition_model.keras' not found. Train the model first.")
        return
    if not os.path.exists('scaler.joblib'):
        print("Error: StandardScaler 'scaler.joblib' not found. Train the model first.")
        return
        
    print(f"Loading files and preprocessing: {file_path}...")
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SR)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        
        # Standardize duration (pad or truncate)
        if len(audio) > TARGET_SAMPLES:
            audio = audio[:TARGET_SAMPLES]
        else:
            padding = TARGET_SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
            
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)  # shape (40, 151)
        
    except Exception as e:
        print(f"Error reading/processing audio file: {e}")
        return
        
    try:
        # Load scaler and scale
        scaler = joblib.load('scaler.joblib')
        
        # Reshape to flatten for scaling
        mfccs_flat = mfccs.reshape(1, -1)
        mfccs_scaled = scaler.transform(mfccs_flat)
        
        # Reshape back to CNN format (1, n_mfcc, time_steps, 1)
        mfccs_input = mfccs_scaled.reshape(1, N_MFCC, -1, 1)
        
    except Exception as e:
        print(f"Error processing features with scaler: {e}")
        print("Make sure the model was trained with the same feature dimensions.")
        return
        
    try:
        # Load model and predict
        # Suppress TF logging for cleaner output
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model = tf.keras.models.load_model('speech_emotion_recognition_model.keras')
        
        # Predict
        predictions = model.predict(mfccs_input, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        
        print("\n" + "="*45)
        print("         SPEECH EMOTION RECOGNITION")
        print("="*45)
        print(f"File: {os.path.basename(file_path)}")
        print("-"*45)
        
        # Visual representation of probabilities
        for label, prob in zip(EMOTION_LABELS, predictions):
            bar_length = int(prob * 25)
            bar = "#" * bar_length + "." * (25 - bar_length)
            print(f"{label.capitalize():<10} | {bar} | {prob*100:6.2f}%")
            
        print("-"*45)
        print(f"PREDICTED EMOTION: {EMOTION_LABELS[predicted_class].upper()}")
        print(f"CONFIDENCE:        {predictions[predicted_class]*100:.2f}%")
        print("="*45)
        
    except Exception as e:
        print(f"Error during model loading or prediction: {e}")

if __name__ == "__main__":
    main()
