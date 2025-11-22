"""
Inference Script for Improved 8-Class Ship Horn Detection Model
Uses enhanced features for better accuracy on real audio data
"""

import sys
import os
import numpy as np
import librosa
import tensorflow_hub as hub
import pickle
import json
from scipy import signal

# Configuration
SAMPLE_RATE = 16000

# Model paths - try multiple locations
MODEL_PATHS = [
    'models/ship_horn_classifier_8class_improved.pkl',
    '../models/ship_horn_classifier_8class_improved.pkl',
    'ship_horn_classifier_8class_improved.pkl'
]

METADATA_PATHS = [
    'models/metadata_8class_improved.json',
    '../models/metadata_8class_improved.json',
    'metadata_8class_improved.json'
]

def find_file(paths):
    """Find the first existing file from a list of paths"""
    for path in paths:
        if os.path.exists(path):
            return path
    return None

MODEL_PATH = find_file(MODEL_PATHS)
METADATA_PATH = find_file(METADATA_PATHS)

# Load YAMNet model (global)
yamnet_model = None

def extract_yamnet_embedding(wav_path, max_duration=15):
    """Extract YAMNet embeddings from audio file"""
    global yamnet_model
    if yamnet_model is None:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    waveform, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    
    max_samples = SAMPLE_RATE * max_duration
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    embedding_mean = np.mean(embeddings.numpy(), axis=0)
    embedding_max = np.max(embeddings.numpy(), axis=0)
    embedding_std = np.std(embeddings.numpy(), axis=0)
    
    combined_embedding = np.concatenate([
        embedding_mean,
        embedding_max,
        embedding_std
    ])
    
    return combined_embedding

def detect_blast_pattern(y, sr):
    """
    IMPROVED: Detect and classify blast pattern
    Returns features that help distinguish long+short patterns
    """
    # RMS energy envelope
    rms = librosa.feature.rms(y=y)[0]
    rms_threshold = np.percentile(rms, 30)  # Adaptive threshold
    
    # Find energy peaks (blasts)
    peaks, properties = signal.find_peaks(rms, height=rms_threshold, distance=int(sr * 0.5))
    
    if len(peaks) == 0:
        return {
            'num_blasts': 0,
            'blast_durations': [],
            'blast_intervals': [],
            'long_blast_count': 0,
            'short_blast_count': 0,
            'avg_blast_duration': 0,
            'pattern_ratio': 0,
            'max_blast_duration': 0,
            'min_blast_duration': 0,
            'duration_std': 0
        }
    
    # CRITICAL FIX: RMS is at frame rate, not sample rate!
    # librosa.feature.rms uses hop_length=512 by default
    hop_length = 512  # librosa default
    frame_rate = sr / hop_length  # ~31.25 frames/second
    
    # Estimate blast durations by looking at energy above threshold
    blast_durations = []
    blast_intervals = []
    
    for i, peak in enumerate(peaks):
        # Find start and end of this blast (in RMS frame indices)
        start_frame = peak
        end_frame = peak
        
        # Go backwards to find start
        while start_frame > 0 and rms[start_frame] > rms_threshold:
            start_frame -= 1
        
        # Go forwards to find end
        while end_frame < len(rms) - 1 and rms[end_frame] > rms_threshold:
            end_frame += 1
        
        # FIXED: Convert frame indices to time properly
        duration_seconds = (end_frame - start_frame) / frame_rate
        duration_ms = duration_seconds * 1000  # Convert to milliseconds
        blast_durations.append(duration_ms)
        
        # Interval to next blast (also in seconds)
        if i < len(peaks) - 1:
            interval_frames = peaks[i+1] - peak
            interval = interval_frames / frame_rate  # Convert to seconds
            blast_intervals.append(interval)
    
    # Classify blasts as long or short based on COLREG specs
    # Short blast: ~1 second, Long blast: 4-6 seconds
    # Use 2.5 seconds as the dividing line
    long_threshold = 2.5  # seconds
    short_threshold = 2.5  # seconds
    
    # Convert durations from ms to seconds for comparison
    blast_durations_seconds = [d / 1000 for d in blast_durations]
    
    long_blast_count = sum(1 for d in blast_durations_seconds if d > long_threshold)
    short_blast_count = sum(1 for d in blast_durations_seconds if d <= short_threshold)
    
    avg_blast_duration = np.mean(blast_durations) if blast_durations else 0  # Still in ms for return value
    avg_interval = np.mean(blast_intervals) if blast_intervals else 0
    
    # Pattern ratio: helps distinguish long+short vs all short
    pattern_ratio = long_blast_count / max(len(peaks), 1)
    
    return {
        'num_blasts': len(peaks),
        'blast_durations': blast_durations,
        'blast_intervals': blast_intervals,
        'long_blast_count': long_blast_count,
        'short_blast_count': short_blast_count,
        'avg_blast_duration': avg_blast_duration,
        'avg_interval': avg_interval,
        'pattern_ratio': pattern_ratio,
        'max_blast_duration': max(blast_durations) if blast_durations else 0,
        'min_blast_duration': min(blast_durations) if blast_durations else 0,
        'duration_std': np.std(blast_durations) if len(blast_durations) > 1 else 0
    }

def extract_enhanced_features(wav_path):
    """
    IMPROVED: Extract enhanced features with better temporal analysis
    Same as training script
    """
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
        
        # 3. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for mfcc in mfccs:
            features.extend([np.mean(mfcc), np.std(mfcc)])
        
        # 4. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms), np.max(rms)])
        
        # 5. IMPROVED: Enhanced onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True,
            delta=0.3  # More sensitive
        )
        features.append(len(onsets))
        
        # 6. NEW: Blast pattern analysis (critical for long+short patterns)
        pattern_info = detect_blast_pattern(y, sr)
        features.extend([
            pattern_info['num_blasts'],
            pattern_info['long_blast_count'],
            pattern_info['short_blast_count'],
            pattern_info['avg_blast_duration'] / 1000,  # Convert to seconds
            pattern_info['avg_interval'],
            pattern_info['pattern_ratio'],
            pattern_info['max_blast_duration'] / 1000,
            pattern_info['min_blast_duration'] / 1000,
            pattern_info['duration_std'] / 1000
        ])
        
        # 7. NEW: Sequence features (helps distinguish 2 vs 3 short)
        if len(pattern_info['blast_durations']) > 0:
            if pattern_info['min_blast_duration'] > 0:
                duration_ratio = pattern_info['max_blast_duration'] / pattern_info['min_blast_duration']
            else:
                duration_ratio = 0
            features.append(duration_ratio)
        else:
            features.append(0)
        
        # 8. Tempo/rhythm
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features.extend([np.mean(tempogram), np.std(tempogram)])
        
        # 9. REMOVED: Total audio duration (includes padding - misleading!)
        # Instead, we use avg_blast_duration from pattern_info which is accurate
        
        # 10. NEW: Energy distribution (helps distinguish patterns)
        mid_point = len(rms) // 2
        energy_first_half = np.sum(rms[:mid_point])
        energy_second_half = np.sum(rms[mid_point:])
        total_energy = energy_first_half + energy_second_half
        if total_energy > 0:
            energy_ratio = energy_first_half / total_energy
        else:
            energy_ratio = 0.5
        features.append(energy_ratio)
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features from {wav_path}: {e}")
        return None

def predict_horn_signal(audio_path):
    """Main prediction function"""
    
    # Load model and metadata
    if MODEL_PATH is None:
        print("❌ Error: Model file not found!")
        print("Searched in:")
        for path in MODEL_PATHS:
            print(f"  - {path}")
        print("\nPlease ensure the model file exists.")
        return None
    
    if METADATA_PATH is None:
        print("⚠️  Warning: Metadata file not found, using default class names")
        metadata = {'class_names': {
            '0': 'alter_starboard (1 short)',
            '1': 'alter_port (2 short)',
            '2': 'astern_propulsion (3 short)',
            '3': 'doubt_intentions (5 short)',
            '4': 'approaching_bend (1 long)',
            '5': 'overtake_starboard (1 long + 1 short)',
            '6': 'overtake_port (1 long + 2 short)',
            '7': 'agreement (long-short-long-short)'
        }, 'test_accuracy': 0.955}
    else:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print(f"\n{'='*70}")
    print("SHIP HORN SIGNAL CLASSIFIER (8-Class Improved Model)")
    print(f"{'='*70}")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print(f"Model Type: {metadata.get('model_type', 'Unknown')}")
    print(f"Trained Accuracy: {metadata.get('test_accuracy', 0):.1%}")
    print(f"Audio file: {audio_path}")
    print(f"{'='*70}\n")
    
    # Extract features
    print("Extracting features...")
    print("  - Loading YAMNet model...")
    yamnet_emb = extract_yamnet_embedding(audio_path)
    print("  - Extracting enhanced acoustic features...")
    custom_feat = extract_enhanced_features(audio_path)
    
    if yamnet_emb is None or custom_feat is None:
        print("❌ Error: Failed to extract features")
        return None
    
    # Combine features
    features = np.concatenate([yamnet_emb, custom_feat]).reshape(1, -1)
    print(f"  ✓ Feature vector shape: {features.shape}")
    
    # Predict
    print("\nRunning prediction...")
    prediction = model.predict(features)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        
        print("\nPREDICTION RESULTS:")
        print("-" * 70)
        print(f"Predicted Class: {prediction}")
        print(f"Signal Type: {metadata['class_names'][str(prediction)]}")
        print("\nConfidence Scores (All 8 Classes):")
        print("-" * 70)
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            class_name = metadata['class_names'][str(idx)]
            prob = probabilities[idx]
            bar = '█' * int(prob * 50)
            marker = "→" if idx == prediction else " "
            print(f"{marker} Class {idx}: {prob:6.1%} {bar} {class_name}")
    else:
        print("\nPREDICTION RESULTS:")
        print("-" * 70)
        print(f"Predicted Class: {prediction}")
        print(f"Signal Type: {metadata['class_names'][str(prediction)]}")
    
    # Show detected pattern info
    print("\n" + "-" * 70)
    print("DETECTED PATTERN ANALYSIS:")
    print("-" * 70)
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        pattern_info = detect_blast_pattern(y, sr)
        print(f"  Number of blasts detected: {pattern_info['num_blasts']}")
        print(f"  Long blasts (>3s): {pattern_info['long_blast_count']}")
        print(f"  Short blasts (<2s): {pattern_info['short_blast_count']}")
        if pattern_info['avg_blast_duration'] > 0:
            print(f"  Average blast duration: {pattern_info['avg_blast_duration']/1000:.2f} seconds")
        if pattern_info['avg_interval'] > 0:
            print(f"  Average interval: {pattern_info['avg_interval']:.2f} seconds")
    except:
        print("  (Pattern analysis unavailable)")
    
    print(f"\n{'='*70}\n")
    
    return prediction

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_8class_improved.py <audio_file.wav>")
        print("\nExample:")
        print("  python inference_8class_improved.py 6.wav")
        print("  python inference_8class_improved.py dataset_v2_improved/0/0.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: File not found: {audio_file}")
        sys.exit(1)
    
    # Run prediction
    prediction = predict_horn_signal(audio_file)
    
    if prediction is None:
        sys.exit(1)

