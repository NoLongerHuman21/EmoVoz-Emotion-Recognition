import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the saved model
loaded_model = load_model('model.h5')

# Load the encoder
emotion_labels = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fear',
    6: 'disgust',
    7: 'surprise'
}

def predict_emotion(audio_path):
    try:
        # Load and preprocess the audio file
        data, sr = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
        feature = np.expand_dims(np.expand_dims(get_features(data), axis=0), axis=2)
        
        # Make prediction
        prediction = loaded_model.predict(feature)
        predicted_emotion_id = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_emotion_id]
        
        return predicted_emotion
    except Exception as e:
        return str(e)

def get_features(data):
    # Calculate Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(data))
    
    # Calculate Mel-frequency cepstral coefficients (MFCC)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=22050, n_mfcc=13).T, axis=0)
    
    # Calculate Zero Crossing Rate (ZCR)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    
    # Calculate Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=22050).T, axis=0)
    
    # Calculate Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=22050).T, axis=0)
    
    # Calculate Spectral Rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=stft, sr=22050).T, axis=0)
    
    # Concatenate all features into a single feature vector
    features = np.hstack((mfcc, zcr, spectral_centroid, spectral_contrast, spectral_rolloff))
    
    return features

# Streamlit app
def main():
    st.title('EmoVoz:An Emotion Recognition System')
    st.write('Upload an audio file to recognize the emotion')

    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Save the uploaded file
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Predict emotion
        emotion = predict_emotion("temp_audio.wav")
        st.success(f"Predicted emotion: {emotion}")

if __name__ == "__main__":
    main()
