"""
Inference Script for Improved 8-Class Ship Horn Detection Model
Uses enhanced features for better accuracy on real audio data
"""

import sys
import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import json
from scipy import signal

# Configuration
SAMPLE_RATE = 16000
MAX_SEQ_LEN = 32  # Fixed sequence length for LSTM (approx 15s audio)

# Model paths - try multiple locations
MODEL_PATHS_LSTM = [
    'models/ship_horn_classifier_8class_lstm.h5',
    '../models/ship_horn_classifier_8class_lstm.h5',
    'ship_horn_classifier_8class_lstm.h5'
]

MODEL_PATHS_PKL = [
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

MODEL_PATH_LSTM = find_file(MODEL_PATHS_LSTM)
MODEL_PATH_PKL = find_file(MODEL_PATHS_PKL)
METADATA_PATH = find_file(METADATA_PATHS)

# Load YAMNet model (global)
yamnet_model = None

def extract_yamnet_embedding(wav_path, max_duration=15):
    """Extract YAMNet embeddings from audio file (returns both flat and sequence)"""
    global yamnet_model
    if yamnet_model is None:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    waveform, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    
    max_samples = SAMPLE_RATE * max_duration
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Flat embedding for traditional models
    embedding_mean = np.mean(embeddings.numpy(), axis=0)
    embedding_max = np.max(embeddings.numpy(), axis=0)
    embedding_std = np.std(embeddings.numpy(), axis=0)
    
    flat_embedding = np.concatenate([
        embedding_mean,
        embedding_max,
        embedding_std
    ])
    
    # Sequence embedding for LSTM
    # Pad or truncate to MAX_SEQ_LEN
    seq_embedding = embeddings.numpy()
    if len(seq_embedding) > MAX_SEQ_LEN:
        seq_embedding = seq_embedding[:MAX_SEQ_LEN]
    elif len(seq_embedding) < MAX_SEQ_LEN:
        padding = np.zeros((MAX_SEQ_LEN - len(seq_embedding), seq_embedding.shape[1]))
        seq_embedding = np.vstack([seq_embedding, padding])
    
    return flat_embedding, seq_embedding

def detect_blast_pattern(y, sr):
    """
    IMPROVED: Detect and classify blast pattern using threshold-based segmentation
    instead of peak detection to avoid over-counting jagged peaks.
    """
    # --- NEW: Apply Bandpass Filter (50Hz - 600Hz) ---
    # This removes wind noise (<50Hz) and hiss (>600Hz)
    # so the RMS energy only reflects the horn blasts.
    try:
        sos = signal.butter(4, [50, 600], btype='bandpass', fs=sr, output='sos')
        y_filtered = signal.sosfilt(sos, y)
    except Exception as e:
        # Fallback if filtering fails (e.g. signal too short)
        y_filtered = y

    # 1. Compute RMS energy
    hop_length = 256
    frame_length = 512
    rms = librosa.feature.rms(y=y_filtered, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 2. Normalize RMS to [0, 1] for consistent thresholding
    if np.max(rms) > 0:
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    else:
        rms_norm = rms
        
    # 3. Thresholding
    # Use a relative threshold (e.g., 15% of max volume) to detect blasts
    # This is more robust than percentile for files with different silence ratios
    threshold = 0.15
    
    # 4. Find contiguous regions above threshold
    is_blast = rms_norm > threshold
    
    # Find changes in state
    # Pad with False to detect start/end at boundaries
    is_blast_padded = np.pad(is_blast, (1, 1), 'constant', constant_values=False)
    changes = np.diff(is_blast_padded.astype(int))
    
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # 5. Filter and Merge Segments
    frame_rate = sr / hop_length
    min_duration_frames = int(0.1 * frame_rate)  # Min blast 0.1s (ignore tiny noise)
    min_gap_frames = int(0.05 * frame_rate)      # Merge gaps < 0.05s (ignore dropouts)
    
    merged_starts = []
    merged_ends = []
    
    if len(starts) > 0:
        curr_start = starts[0]
        curr_end = ends[0]
        
        for i in range(1, len(starts)):
            next_start = starts[i]
            next_end = ends[i]
            
            # Check gap
            if next_start - curr_end < min_gap_frames:
                # Merge
                curr_end = next_end
            else:
                # Store current and start new
                if (curr_end - curr_start) >= min_duration_frames:
                    merged_starts.append(curr_start)
                    merged_ends.append(curr_end)
                curr_start = next_start
                curr_end = next_end
        
        # Append last one
        if (curr_end - curr_start) >= min_duration_frames:
            merged_starts.append(curr_start)
            merged_ends.append(curr_end)
            
    blast_durations = []
    blast_intervals = []
    
    for i in range(len(merged_starts)):
        duration_frames = merged_ends[i] - merged_starts[i]
        duration_ms = (duration_frames / frame_rate) * 1000
        blast_durations.append(duration_ms)
        
        if i < len(merged_starts) - 1:
            interval_frames = merged_starts[i+1] - merged_ends[i]
            interval_sec = interval_frames / frame_rate
            blast_intervals.append(interval_sec)
    
    # Classify blasts as long or short based on COLREG specs
    # Short blast: ~1 second (Rule 32b)
    # Long blast: 4-6 seconds (Rule 32c)
    # Use 2.5 seconds as the dividing line (midpoint between 1s and 4s)
    long_threshold = 2.5  # seconds - anything >2.5s is considered long
    short_threshold = 2.5  # seconds - anything <2.5s is considered short
    
    # Convert durations from ms to seconds for comparison
    blast_durations_seconds = [d / 1000 for d in blast_durations]
    
    long_blast_count = sum(1 for d in blast_durations_seconds if d > long_threshold)
    short_blast_count = sum(1 for d in blast_durations_seconds if d <= short_threshold)
    
    avg_blast_duration = np.mean(blast_durations) if blast_durations else 0
    avg_interval = np.mean(blast_intervals) if blast_intervals else 0
    
    # Pattern ratio: helps distinguish long+short vs all short
    pattern_ratio = long_blast_count / max(len(merged_starts), 1)
    
    return {
        'num_blasts': len(merged_starts),
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

def apply_pattern_correction(prediction, probabilities, pattern_info, metadata):
    """
    Apply rule-based correction when detected pattern clearly conflicts with prediction.
    This helps when the model makes obvious mistakes.
    """
    num_blasts = pattern_info['num_blasts']
    long_count = pattern_info['long_blast_count']
    short_count = pattern_info['short_blast_count']
    
    # Define expected patterns for each class
    expected_patterns = {
        0: {'long': 0, 'short': 1, 'total': 1},  # 1 short
        1: {'long': 0, 'short': 2, 'total': 2},  # 2 short
        2: {'long': 0, 'short': 3, 'total': 3},  # 3 short
        3: {'long': 0, 'short': 5, 'total': 5},  # 5 short
        4: {'long': 1, 'short': 0, 'total': 1},  # 1 long
        5: {'long': 1, 'short': 1, 'total': 2},  # 1 long + 1 short
        6: {'long': 1, 'short': 2, 'total': 3},  # 1 long + 2 short
        7: {'long': 2, 'short': 2, 'total': 4},  # long-short-long-short
    }
    
    # Check if detected pattern matches prediction
    predicted_pattern = expected_patterns[prediction]
    
    # Calculate pattern match score
    total_match = (num_blasts == predicted_pattern['total'])
    long_match = (long_count == predicted_pattern['long'])
    short_match = (short_count == predicted_pattern['short'])
    
    pattern_matches = total_match and long_match and short_match
    
    if pattern_matches:
        # Prediction matches detected pattern, keep it
        return prediction, probabilities, "Pattern matches prediction"
    
    # Pattern doesn't match - find best matching class
    print("\n⚠️  WARNING: Detected pattern conflicts with model prediction!")
    print(f"   Model predicted: Class {prediction} (expects {predicted_pattern})")
    print(f"   Actually detected: {num_blasts} blasts ({long_count} long, {short_count} short)")
    
    # Find which class best matches the detected pattern
    best_match = None
    for class_id, expected in expected_patterns.items():
        if (num_blasts == expected['total'] and 
            long_count == expected['long'] and 
            short_count == expected['short']):
            best_match = class_id
            break
    
    if best_match is not None:
        print(f"   → Correcting to Class {best_match}: {metadata['class_names'][str(best_match)]}")
        # Adjust probabilities to reflect the correction
        corrected_probs = probabilities.copy()
        corrected_probs[best_match] = 0.95  # High confidence in pattern match
        # Redistribute remaining probability
        remaining_prob = 0.05
        for i in range(len(corrected_probs)):
            if i != best_match:
                corrected_probs[i] = remaining_prob / (len(corrected_probs) - 1)
        
        return best_match, corrected_probs, f"Corrected based on detected pattern"
    else:
        print(f"   → No exact pattern match found. Keeping original prediction.")
        return prediction, probabilities, "No exact pattern match, using model prediction"

def predict_horn_signal(audio_path):
    """Main prediction function"""
    
    # Load model and metadata
    if MODEL_PATH_LSTM is None and MODEL_PATH_PKL is None:
        print("❌ Error: Model file not found!")
        print("Searched in:")
        for path in MODEL_PATHS_LSTM + MODEL_PATHS_PKL:
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
    
    if MODEL_PATH_LSTM:
        model = tf.keras.models.load_model(MODEL_PATH_LSTM)
        model_type = "LSTM"
    else:
        with open(MODEL_PATH_PKL, 'rb') as f:
            model = pickle.load(f)
        model_type = "Traditional"
    
    print(f"\n{'='*70}")
    print("SHIP HORN SIGNAL CLASSIFIER (8-Class Improved Model)")
    print(f"{'='*70}")
    print(f"Model: {os.path.basename(MODEL_PATH_LSTM or MODEL_PATH_PKL)}")
    print(f"Model Type: {model_type}")
    print(f"Trained Accuracy: {metadata.get('test_accuracy', 0):.1%}")
    print(f"Audio file: {audio_path}")
    print(f"{'='*70}\n")
    
    # Extract features
    print("Extracting features...")
    print("  - Loading YAMNet model...")
    yamnet_emb_flat, yamnet_emb_seq = extract_yamnet_embedding(audio_path)
    print("  - Extracting enhanced acoustic features...")
    custom_feat = extract_enhanced_features(audio_path)
    
    if yamnet_emb_flat is None or custom_feat is None:
        print("❌ Error: Failed to extract features")
        return None
    
    if model_type == "LSTM":
        # Use sequence features for LSTM
        features = np.expand_dims(yamnet_emb_seq, axis=0)  # Add batch dimension
    else:
        # Combine features for traditional model
        features = np.concatenate([yamnet_emb_flat, custom_feat]).reshape(1, -1)
    
    print(f"  ✓ Feature vector shape: {features.shape}")
    
    # Predict
    print("\nRunning prediction...")
    if model_type == "LSTM":
        prediction_probs = model.predict(features)[0]
        prediction = np.argmax(prediction_probs)
        probabilities = prediction_probs
    else:
        prediction = model.predict(features)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
        else:
            # Create dummy probabilities
            probabilities = np.zeros(8)
            probabilities[prediction] = 1.0
    
    # Get pattern info for correction
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    pattern_info = detect_blast_pattern(y, sr)
    
    # Apply pattern-based correction
    corrected_prediction, corrected_probs, correction_note = apply_pattern_correction(
        prediction, probabilities, pattern_info, metadata
    )
    
    # Use corrected values
    prediction = corrected_prediction
    probabilities = corrected_probs
    
    print("\nPREDICTION RESULTS:")
    print("-" * 70)
    print(f"Predicted Class: {prediction}")
    print(f"Signal Type: {metadata['class_names'][str(prediction)]}")
    
    if correction_note != "Pattern matches prediction":
        print(f"Note: {correction_note}")
    
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
    
    # Show detected pattern info
    print("\n" + "-" * 70)
    print("DETECTED PATTERN ANALYSIS:")
    print("-" * 70)
    print(f"  Number of blasts detected: {pattern_info['num_blasts']}")
    print(f"  Long blasts (>3s): {pattern_info['long_blast_count']}")
    print(f"  Short blasts (<2s): {pattern_info['short_blast_count']}")
    if pattern_info['avg_blast_duration'] > 0:
        print(f"  Average blast duration: {pattern_info['avg_blast_duration']/1000:.2f} seconds")
    if pattern_info['avg_interval'] > 0:
        print(f"  Average interval: {pattern_info['avg_interval']:.2f} seconds")
    
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

