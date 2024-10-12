from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import os
import pickle

app = Flask(__name__)

# Ensure temp directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

# Function to load and split a single audio file into 10-second segments
def load_and_split_test_audio(file_path, segment_length=10):
    audio_segments = []
    audio, sr = librosa.load(file_path, sr=22050)
    total_length = len(audio) / sr
    num_segments = int(total_length // segment_length)
    for i in range(num_segments):
        start_sample = int(i * segment_length * sr)
        end_sample = int((i + 1) * segment_length * sr)
        segment = audio[start_sample:end_sample]
        audio_segments.append(segment)
    return audio_segments

# Function to extract 40 features from the audio segments
def extract_features_from_segments(segments, sr=22050):
    features = []
    for segment in segments:
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        rms = librosa.feature.rms(y=segment)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, roll_percent=0.85)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

        segment_features = [
            np.mean(chroma), np.var(chroma),
            np.mean(rms), np.var(rms),
            np.mean(spectral_centroid), np.var(spectral_centroid),
            np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
            tempo, tempo ** 2
        ]
        segment_features.extend([np.mean(mfcc) for mfcc in mfccs])
        segment_features.extend([np.var(mfcc) for mfcc in mfccs])

        features.append(segment_features)

    return np.array(features)

# Define route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle audio file upload and genre classification
@app.route('/classify', methods=['POST'])
def classify_genre():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get selected model from dropdown
    selected_model = request.form.get('model')

    # Save the uploaded file temporarily
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    # Process and predict
    segments = load_and_split_test_audio(file_path)
    features = extract_features_from_segments(segments)

    # Load the corresponding scaler
    with open('scaler.pkl', 'rb') as scaler_file:
        scaled = pickle.load(scaler_file)

    # Scale the features using the loaded scaler
    scaled_features = scaled.transform(features)

    # Load the corresponding model based on user selection
    model_mapping = {
        'svm_model': 'svm_model.pkl',
        'dt_model': 'DT_model.pkl',
        'rf_model': 'rf_model.pkl',
        'knn_model':'knn_model.pkl',
        'logistic_model': 'logistic_model.pkl',
    }

    model_metrics = {
        'svm_model': {'accuracy': 0.98, 'precision': 0.98, 'f1': 0.98, 'Recall':0.98, 'confusion_matrix': (66,1,1,52)},
        'dt_model': {'accuracy': 0.94, 'precision': 0.94, 'f1': 0.94, 'confusion_matrix': (63, 4, 3, 50)},
        'rf_model': {'accuracy': 0.97, 'precision': 0.97, 'f1': 0.97, 'confusion_matrix': (66,1,2,51)},
        'knn_model': {'accuracy': 0.98, 'precision': 0.98, 'f1': 0.98, 'confusion_matrix': (67,0,2,51)},
        'logistic_model': {'accuracy': 0.98, 'precision': 0.98, 'f1': 0.98, 'confusion_matrix': (66,1,1,52)},
    }

    model_path = model_mapping.get(selected_model)
    if model_path:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Predict the genre for each segment
        predicted_genres = model.predict(scaled_features)
        os.remove(file_path)  # Clean up the temp file after processing

        # Return the majority predicted genre
        majority_genre = max(set(predicted_genres), key=list(predicted_genres).count)
        genre_label = 'Classical' if majority_genre == 0 else 'Rock'

        # Pass the model's metrics to the result.html
        metrics = model_metrics[selected_model]
        return render_template('result.html', predicted_genre=genre_label, metrics=metrics)

    return jsonify({'error': 'Invalid model selection'}), 400

if __name__ == '__main__':
    app.run(debug=True)
