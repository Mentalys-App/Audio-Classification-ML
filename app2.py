from flask import Flask, request, jsonify, render_template
import os
import pickle
import librosa
import numpy as np
import tensorflow as tf


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

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio_file']
    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    try:
        # Feature extraction and prediction
        features = get_predict_feat(file_path)
        predictions = loaded_model.predict(features)
        label_names = list(encoder.categories_[0])
        predicted_label_index = np.argmax(predictions)
        predicted_label = label_names[predicted_label_index]

        # Confidence scores
        confidence_scores = {label: float(predictions[0][idx]) for idx, label in enumerate(label_names)}

        # Category of emotion
        non_depression_labels = ['neutral', 'calm', 'happy', 'surprise']
        depression_labels = ['sad', 'angry', 'fear', 'disgust']

        if predicted_label in depression_labels:
            category = "Depression"
            support_percentage = sum(confidence_scores[label] for label in depression_labels) * 100
        else:
            category = "NonDepression"
            support_percentage = sum(confidence_scores[label] for label in non_depression_labels) * 100

        os.remove(file_path)  

        return jsonify({
            'predicted_emotion': predicted_label,
            'category': category,
            'support_percentage': round(support_percentage, 2),
            'confidence_scores': confidence_scores
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Home route
@app.route('/')
def home():
    return render_template('index_copy.html') 

# Run the app
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
