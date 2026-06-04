import os
import glob
import numpy as np
import pandas as pd
import librosa

# Configuration
SR = 22050
DURATION = 3.5  # seconds
TARGET_SAMPLES = int(SR * DURATION)
N_MFCC = 40

# Emotion mappings
EMOTION_MAP = {
    '01': 0,  # neutral
    '03': 1,  # happy
    '04': 2,  # sad
    '05': 3   # angry
}
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry']

def extract_mfcc_from_file(file_path):
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
        mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found. Please run download_dataset.py first.")
        return
        
    print("Starting feature extraction...")
    
    features = []
    labels = []
    metadata = []
    
    # RAVDESS structure: dataset/Actor_XX/*.wav
    search_path = os.path.join(dataset_dir, "Actor_*", "*.wav")
    file_list = glob.glob(search_path)
    
    if not file_list:
        print("No audio files found. Ensure the dataset is downloaded and extracted correctly.")
        return
        
    print(f"Found {len(file_list)} audio files total.")
    processed_count = 0
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        # Split filename by '-' (e.g. 03-01-06-01-02-01-12.wav)
        parts = filename.split('.')[0].split('-')
        
        if len(parts) < 7:
            continue
            
        emotion_code = parts[2]
        actor_id = int(parts[6])
        
        # Filter for Neutral, Happy, Sad, and Angry
        if emotion_code in EMOTION_MAP:
            label_idx = EMOTION_MAP[emotion_code]
            emotion_name = EMOTION_LABELS[label_idx]
            
            # Extract MFCC
            mfccs = extract_mfcc_from_file(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(label_idx)
                
                # Determine gender (odd actor_id = male, even actor_id = female)
                gender = "male" if actor_id % 2 != 0 else "female"
                
                metadata.append({
                    'filename': filename,
                    'file_path': file_path,
                    'emotion': emotion_name,
                    'label': label_idx,
                    'actor_id': actor_id,
                    'gender': gender
                })
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Processed {processed_count} files...")
                    
    print(f"Successfully processed {processed_count} files for target emotions.")
    
    if len(features) == 0:
        print("No features extracted. Exiting.")
        return
        
    # Convert lists to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Save features and labels
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("Features saved to X.npy and labels saved to y.npy.")
    
    # Save metadata to CSV
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv("dataset_metadata.csv", index=False)
    print("Metadata saved to dataset_metadata.csv.")
    
    print(f"Feature shape: {X.shape}")  # Expecting (N, 40, 151)
    print(f"Label shape: {y.shape}")    # Expecting (N,)

if __name__ == "__main__":
    main()
