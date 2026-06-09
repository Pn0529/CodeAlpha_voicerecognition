import os
import shutil
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "video_assets"

def ensure_out():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

def copy_existing(name):
    if os.path.exists(name):
        shutil.copy(name, os.path.join(OUT_DIR, os.path.basename(name)))

def plot_waveform(file_path, out_path):
    audio, sr = librosa.load(file_path, sr=22050)
    plt.figure(figsize=(10,3))
    times = np.arange(len(audio)) / sr
    plt.fill_between(times, audio, color='skyblue')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_mfcc(file_path, out_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10,3))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ensure_out()

    meta_csv = 'dataset_metadata.csv'
    if not os.path.exists(meta_csv):
        print('No dataset_metadata.csv found; skipping waveform/MFCC generation.')
    else:
        df = pd.read_csv(meta_csv)
        if len(df) > 0:
            sample_path = df.loc[0, 'file_path']
            if os.path.exists(sample_path):
                print('Generating waveform and MFCC from', sample_path)
                plot_waveform(sample_path, os.path.join(OUT_DIR, 'waveform.png'))
                plot_mfcc(sample_path, os.path.join(OUT_DIR, 'mfcc.png'))
            else:
                print('Sample file not found:', sample_path)

    # Copy existing artifacts
    for name in ['training_history.png', 'confusion_matrix.png']:
        copy_existing(name)

    print('Assets written to', OUT_DIR)

if __name__ == '__main__':
    main()
