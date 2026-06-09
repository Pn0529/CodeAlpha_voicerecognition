import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

SR = 22050
DURATION = 3.5
TARGET_SAMPLES = int(SR * DURATION)
N_MFCC = 40
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry']


@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('speech_emotion_recognition_model.keras')
    scaler = joblib.load('scaler.joblib')
    return model, scaler


def extract_mfcc(file_bytes):
    audio, sr = librosa.load(file_bytes, sr=SR)
    audio, _ = librosa.effects.trim(audio)
    if len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        padding = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return mfccs


def main():
    st.title('Speech Emotion Recognition — Streamlit')
    st.markdown('Upload a WAV file and the model will predict the emotion.')

    uploaded_file = st.file_uploader('Choose a WAV file', type=['wav', 'mp3'])
    if uploaded_file is None:
        st.info('Upload a speech audio file to get started.')
        return

    st.audio(uploaded_file)

    with st.spinner('Loading model and preprocessing...'):
        try:
            model, scaler = load_model_and_scaler()
        except Exception as e:
            st.error(f'Error loading model or scaler: {e}')
            return

    # Extract features
    try:
        mfccs = extract_mfcc(uploaded_file)
    except Exception as e:
        st.error(f'Error processing audio: {e}')
        return

    st.write(f'Extracted MFCC shape: {mfccs.shape}')

    # Scale and reshape
    mfccs_flat = mfccs.reshape(1, -1)
    mfccs_scaled = scaler.transform(mfccs_flat)
    mfccs_input = mfccs_scaled.reshape(1, N_MFCC, -1, 1)

    # Predict
    preds = model.predict(mfccs_input)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = EMOTION_LABELS[pred_idx]

    st.subheader('Prediction')
    st.write(f'Predicted emotion: **{pred_label.upper()}**')
    st.write(f'Confidence: **{preds[pred_idx]*100:.2f}%**')

    # Probability bar chart
    probs = {label: float(p) for label, p in zip(EMOTION_LABELS, preds)}
    st.bar_chart(probs)

    # Show MFCC heatmap
    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(mfccs, aspect='auto', origin='lower')
    ax.set_title('MFCC')
    plt.colorbar(img, ax=ax)
    st.pyplot(fig)


if __name__ == '__main__':
    main()
