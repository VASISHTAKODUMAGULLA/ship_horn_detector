"""
Inference Script for Ship Horn Detection ( 4-Class Model)
Classifies a single audio file
"""

import sys
import os
import numpy as np
import librosa
import tensorflow_hub as hub
import pickle
import json

# Configuration
SAMPLE_RATE = 16000

# Try multiple possible model paths
MODEL_PATHS = [
    'models/ship_horn_classifier.pkl',
    'new_perspective/models/ship_horn_classifier.pkl',
    'ship_horn_classifier.pkl'
]

METADATA_PATHS = [
    'models/metadata.json',
    'new_perspective/models/metadata.json',
    'metadata.json'
]

def find_file(paths):
    """Find the first existing file from a list of paths"""
    for path in paths:
        if os.path.exists(path):
            return path
    return None

MODEL_PATH = find_file(MODEL_PATHS)
METADATA_PATH = find_file(METADATA_PATHS)

def extract_yamnet_embedding(wav_path, max_duration=15):
    """Extract YAMNet embeddings from audio file"""
    waveform, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    
    # Limit duration
    max_samples = SAMPLE_RATE * max_duration
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    
    # Get YAMNet embeddings
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Aggregate embeddings
    embedding_mean = np.mean(embeddings.numpy(), axis=0)
    embedding_max = np.max(embeddings.numpy(), axis=0)
    embedding_std = np.std(embeddings.numpy(), axis=0)
    
    combined_embedding = np.concatenate([
        embedding_mean,
        embedding_max,
        embedding_std
    ])
    
    return combined_embedding

def extract_custom_features(wav_path):
    """Extract custom acoustic features"""
    try:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        
        features = []
        
        # 1. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.min(spectral_centroids),
            np.max(spectral_centroids)
        ])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # 2. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # 3. MFCCs (10 for model)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        for mfcc in mfccs:
            features.extend([np.mean(mfcc), np.std(mfcc)])
        
        # 4. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms), np.max(rms)])
        
        # 5. Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True,
            delta=0.5
        )
        features.append(len(onsets))
        
        # 6. Tempogram
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features.extend([np.mean(tempogram), np.std(tempogram)])
        
        # 7. Duration
        features.append(len(y) / sr)
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting custom features: {e}")
        return None

def predict_horn_signal(audio_path):
    """Main prediction function"""
    
    # Load model and metadata
    if MODEL_PATH is None:
        print("Error: Model file not found!")
        print("Searched in:")
        for path in MODEL_PATHS:
            print(f"  - {path}")
        print("\nPlease train the model first: python train_model.py")
        return None
    
    if METADATA_PATH is None:
        print(" Warning: Metadata file not found, using default class names")
        metadata = {'class_names': {
            '0': 'alter_starboard (1 short)',
            '1': 'doubt_intentions (5 short)',
            '2': 'approaching_bend (1 long)',
            '3': 'agreement (long-short-long-short)'
        }}
    else:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print(f"\n{'='*70}")
    print("SHIP HORN SIGNAL CLASSIFIER (4-Class Model)")
    print(f"{'='*70}")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    if 'test_accuracy' in metadata:
        print(f"Trained Accuracy: {metadata['test_accuracy']:.1%}")
    print(f"Audio file: {audio_path}")
    print(f"{'='*70}\n")
    
    # Extract features
    print("Extracting features...")
    yamnet_emb = extract_yamnet_embedding(audio_path)
    custom_feat = extract_custom_features(audio_path)
    
    if yamnet_emb is None or custom_feat is None:
        print("Error: Failed to extract features")
        return None
    
    # Combine features
    features = np.concatenate([yamnet_emb, custom_feat]).reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        
        print("PREDICTION RESULTS:")
        print("-" * 70)
        print(f"Predicted Class: {prediction}")
        print(f"Signal Type: {metadata['class_names'][str(prediction)]}")
        print("\nConfidence Scores:")
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            class_name = metadata['class_names'][str(idx)]
            prob = probabilities[idx]
            bar = '█' * int(prob * 50)
            print(f"  Class {idx}: {prob:6.1%} {bar} {class_name}")
    else:
        print("PREDICTION RESULTS:")
        print("-" * 70)
        print(f"Predicted Class: {prediction}")
        print(f"Signal Type: {metadata['class_names'][str(prediction)]}")
    
    print(f"\n{'='*70}\n")
    
    return prediction

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <audio_file.wav>")
        print("\nExample:")
        print("  python inference.py dataset/0/0.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    # Load YAMNet
    print("Loading YAMNet model...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("✓ YAMNet model loaded\n")
    
    # Run prediction
    prediction = predict_horn_signal(audio_file)
    
    if prediction is None:
        sys.exit(1)

