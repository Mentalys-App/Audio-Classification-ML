from flask import Flask, request, jsonify, render_template
import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
import sounddevice as sd
import wave
import time

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and scalers
with open('CNN_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("best_model.weights.h5")

with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

# Feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_result = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc_result.T) if not flatten else np.ravel(mfcc_result.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((
        result,
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    return result

def get_predict_feat(path, expected_shape=(1, 2376)):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)

    # Ensure res is reshaped or padded to match the expected shape
    if res.shape != expected_shape:
        flat_size = np.prod(expected_shape)
        if res.size < flat_size:
            pad_width = (0, flat_size - res.size)
            res = np.pad(res, pad_width=pad_width, mode='constant')
        else:
            res = np.resize(res, expected_shape)

    i_result = scaler.transform(res.reshape(1, -1))
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

# Record audio from microphone
def record_audio(filename, duration=10, sr=22050):
    print("Recording...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio_data = (audio_data * 32767).astype('int16')  # Convert to int16
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bit
        wf.setframerate(sr)
        wf.writeframes(audio_data.tobytes())
    print(f"Recording saved as {filename}")

# Process long audio
def process_long_audio(file_path, frame_duration=2.5, step_duration=1.0):
    features = []
    for offset in np.arange(0, librosa.get_duration(filename=file_path), step_duration):
        try:
            d, sr = librosa.load(file_path, sr=None, duration=frame_duration, offset=offset)
            res = extract_features(d, sr)
            features.append(res)
        except Exception as e:
            break
    return np.array(features)

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'audio_file' not in request.files:
        return render_template('index.html', error='No audio file uploaded')

    audio_file = request.files['audio_file']
    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    try:
        # Ekstraksi fitur dan prediksi
        features = get_predict_feat(file_path)
        predictions = loaded_model.predict(features)
        label_names = list(encoder.categories_[0])
        predicted_label_index = np.argmax(predictions)
        predicted_label = label_names[predicted_label_index]

        # Confidence scores (dalam persentase dan dibulatkan ke 8 angka desimal)
        confidence_scores = {label: round(float(predictions[0][idx]) * 100, 8) for idx, label in enumerate(label_names)}

        # Kategorisasi emosi
        non_depression_labels = ['neutral', 'calm', 'happy', 'surprise']
        depression_labels = ['sad', 'angry', 'fear', 'disgust']

        if predicted_label in depression_labels:
            category = "Depression"
            support_percentage = round(sum(confidence_scores[label] for label in depression_labels), 8)
        else:
            category = "Non-Depression"
            support_percentage = round(sum(confidence_scores[label] for label in non_depression_labels), 8)

        os.remove(file_path)

        # Kirim hasil ke template
        return render_template('result.html', 
                               prediction=predicted_label,
                               confidence_scores=confidence_scores,
                               category=category,
                               support_percentage=support_percentage)

    except Exception as e:
        return render_template('index.html', error=str(e))



@app.route('/record', methods=['GET'])
def record_from_mic():
    file_path = 'uploads/recorded_audio.wav'
    record_audio(file_path, duration=5)
    return render_template('index.html')

@app.route('/predict_recorded', methods=['GET'])
def predict_recorded():
    # Path ke file rekaman
    recorded_file_path = 'uploads/recorded_audio.wav'
    
    if not os.path.exists(recorded_file_path):
        return render_template('index.html', error='No recorded file found. Please record audio first.')

    try:
        # Proses prediksi menggunakan file rekaman
        features = get_predict_feat(recorded_file_path)
        predictions = loaded_model.predict(features)
        label_names = list(encoder.categories_[0])
        predicted_label_index = np.argmax(predictions)
        predicted_label = label_names[predicted_label_index]

        # Confidence scores
        confidence_scores = {label: round(float(predictions[0][idx]) * 100, 8) for idx, label in enumerate(label_names)}

        # Kategorisasi emosi
        non_depression_labels = ['neutral', 'calm', 'happy', 'surprise']
        depression_labels = ['sad', 'angry', 'fear', 'disgust']

        if predicted_label in depression_labels:
            category = "Depression"
            support_percentage = round(sum(confidence_scores[label] for label in depression_labels), 8)
        else:
            category = "Non-Depression"
            support_percentage = round(sum(confidence_scores[label] for label in non_depression_labels), 8)

        # Tampilkan hasil di template
        return render_template('result.html', 
                               prediction=predicted_label,
                               confidence_scores=confidence_scores,
                               category=category,
                               support_percentage=support_percentage)

    except Exception as e:
        return render_template('index.html', error=f"Error processing recorded file: {str(e)}")


@app.route('/')
def home():
    return render_template('index.html')  # Create an `index.html` for file upload interface

# Run the app
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
