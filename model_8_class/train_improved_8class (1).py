"""
Improved 8-Class Training Script
- Enhanced temporal features for long+short pattern detection
- Better blast counting and sequence analysis
- Optimized for 8 COLREG classes
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from tqdm import tqdm
from scipy import signal
import time

# Configuration
DATASET_PATH = 'dataset_v3_improved/'  # Use the full 8-class dataset
SAMPLE_RATE = 16000
MODEL_OUTPUT = 'models/'
MAX_SEQ_LEN = 32  # Fixed sequence length for LSTM (approx 15s audio)

# All 8 COLREG Classes
CLASS_NAMES = {
    0: "alter_starboard (1 short)",
    1: "alter_port (2 short)",
    2: "astern_propulsion (3 short)",
    3: "doubt_intentions (5 short)",
    4: "approaching_bend (1 long)",
    5: "overtake_starboard (1 long + 1 short)",
    6: "overtake_port (1 long + 2 short)",
    7: "agreement (long-short-long-short)",
}

print("=" * 70)
print("SHIP HORN DETECTION - IMPROVED 8-CLASS TRAINING")
print("Enhanced features for better pattern recognition")
print("=" * 70)

# Load YAMNet model
print("\n[1/6] Loading YAMNet model from TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("✓ YAMNet model loaded successfully")

def extract_yamnet_embedding(wav_path, max_duration=15):
    """Extract YAMNet embeddings from audio file (returns both flat and sequence)"""
    try:
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
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None, None

def detect_blast_pattern(y, sr):
    """
    IMPROVED: Detect and classify blast pattern using threshold-based segmentation
    instead of peak detection to avoid over-counting jagged peaks.
    """
    # --- NEW: Apply Bandpass Filter (70Hz - 2000Hz) ---
    # This removes wind noise (<70Hz) and hiss (>2000Hz)
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
            # Ratio of longest to shortest blast
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
        # How much energy is in first half vs second half
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

# Load dataset
print("\n[2/6] Loading and processing dataset...")
X_yamnet_flat = []
X_yamnet_seq = []
X_custom = []
y = []

if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset not found at {DATASET_PATH}")
    print("Please run: python generate_realistic_dataset.py")
    exit(1)

class_dirs = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])

print(f"Found {len(class_dirs)} classes: {class_dirs}")

for label in tqdm(class_dirs, desc="Processing classes"):
    class_dir = os.path.join(DATASET_PATH, label)
    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    
    for fname in tqdm(wav_files, desc=f"  Class {label}", leave=False):
        fpath = os.path.join(class_dir, fname)
        
        yamnet_flat, yamnet_seq = extract_yamnet_embedding(fpath)
        if yamnet_flat is None:
            continue
        
        custom_feat = extract_enhanced_features(fpath)
        if custom_feat is None:
            continue
        
        X_yamnet_flat.append(yamnet_flat)
        X_yamnet_seq.append(yamnet_seq)
        X_custom.append(custom_feat)
        y.append(int(label))

X_yamnet_flat = np.array(X_yamnet_flat)
X_yamnet_seq = np.array(X_yamnet_seq)
X_custom = np.array(X_custom)
y = np.array(y)

# Combine features for traditional models
X_combined = np.concatenate([X_yamnet_flat, X_custom], axis=1)

print(f"\n✓ Dataset loaded successfully")
print(f"  YAMNet flat features shape: {X_yamnet_flat.shape}")
print(f"  YAMNet sequence features shape: {X_yamnet_seq.shape}")
print(f"  Enhanced custom features shape: {X_custom.shape}")
print(f"  Combined flat features shape: {X_combined.shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Split dataset
print("\n[3/6] Splitting dataset...")
# Split indices first to ensure same split for all data types
indices = np.arange(len(y))
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    indices, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Create splits for different feature sets
X_train_flat = X_combined[X_train_idx]
X_test_flat = X_combined[X_test_idx]

X_train_seq = X_yamnet_seq[X_train_idx]
X_test_seq = X_yamnet_seq[X_test_idx]

print(f"  Training samples: {len(X_train_flat)}")
print(f"  Test samples: {len(X_test_flat)}")

# Train multiple models with scaling
print("\n[4/6] Training models...")

results = {}

# --- 1. Traditional Models with GridSearchCV ---
print("\n--- Training Traditional Models (Random Forest, Gradient Boosting) ---")

# Define pipelines
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

gb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define parameter grids
param_grid_rf = {
    'classifier__n_estimators': [200, 500],
    'classifier__max_depth': [20, 30, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__max_features': ['sqrt', 'log2']
}

param_grid_gb = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5],
    'classifier__subsample': [0.8, 1.0]
}

models_to_train = [
    ('Random Forest', rf_pipeline, param_grid_rf),
    # ('Gradient Boosting', gb_pipeline, param_grid_gb)
]

for name, pipeline, param_grid in models_to_train:
    print(f"\n  Training {name} with GridSearchCV...")
    print("    " + "="*60)
    
    try:
        start_time = time.time()
        
        # Use GridSearchCV
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_flat, y_train)
        
        train_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"    ✓ Best parameters: {best_params}")
        print(f"    ✓ Best CV accuracy: {grid_search.best_score_:.3f}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_flat)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"    ✓ Test accuracy: {test_accuracy:.3f}")
        print(f"    ⏱️  Total time: {train_time:.1f} seconds")
        
        results[name] = {
            'model': best_model,
            'cv_accuracy': grid_search.best_score_,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'best_params': best_params
        }
        
    except Exception as e:
        print(f"    ❌ Error training {name}: {e}")

# --- 2. LSTM Model ---
print("\n--- Training LSTM Model ---")
print("    " + "="*60)

try:
    # Prepare data for LSTM
    # X_train_seq shape: (samples, time_steps, features)
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    num_classes = len(np.unique(y))
    
    # Define LSTM model - SIMPLIFIED & REGULARIZED
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.GaussianNoise(0.1),  # Add noise to embeddings to prevent overfitting
        tf.keras.layers.LSTM(64, return_sequences=False),  # Reduced from 128 to 64
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),  # Reduced from 64 to 32
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    lstm_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    lstm_model.summary()
    
    # Train LSTM
    print("\n    Training LSTM...")
    history = lstm_model.fit(
        X_train_seq, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Evaluate LSTM
    print("\n    Evaluating LSTM...")
    loss, test_accuracy = lstm_model.evaluate(X_test_seq, y_test, verbose=0)
    y_pred_prob = lstm_model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print(f"    ✓ Test accuracy: {test_accuracy:.3f}")
    
    results['LSTM'] = {
        'model': lstm_model,
        'cv_accuracy': max(history.history['val_accuracy']), # Approx CV score
        'test_accuracy': test_accuracy,
        'predictions': y_pred,
        'history': history.history
    }
    
except Exception as e:
    print(f"    ❌ Error training LSTM: {e}")


# Select best model
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_model_info = results[best_model_name]
best_predictions = best_model_info['predictions']

print(f"\n✓ Best model: {best_model_name}")
print(f"  Test accuracy: {best_model_info['test_accuracy']:.3f}")

# Detailed evaluation
print("\n[5/6] Detailed Evaluation:")
print("\nClassification Report:")
print("=" * 70)
target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
print(classification_report(y_test, best_predictions, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"Class {i}" for i in sorted(CLASS_NAMES.keys())],
            yticklabels=[f"Class {i}" for i in sorted(CLASS_NAMES.keys())])
plt.title(f'Confusion Matrix - {best_model_name} (Improved 8-Class)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/confusion_matrix_improved_8class.png', dpi=150)
print("\n✓ Confusion matrix saved to results/confusion_matrix_improved_8class.png")

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-" * 70)
for i in sorted(CLASS_NAMES.keys()):
    if np.sum(y_test == i) > 0:
        class_accuracy = np.sum((y_test == i) & (best_predictions == i)) / np.sum(y_test == i)
        print(f"  Class {i} ({CLASS_NAMES[i]}): {class_accuracy:.1%}")

# Save model
print("\n[6/6] Saving model...")
os.makedirs(MODEL_OUTPUT, exist_ok=True)

# Save the best model (handle Keras vs Sklearn)
if best_model_name == 'LSTM':
    model_path = os.path.join(MODEL_OUTPUT, 'ship_horn_classifier_8class_lstm.h5')
    best_model_info['model'].save(model_path)
    print(f"✓ LSTM Model saved to {model_path}")
else:
    model_path = os.path.join(MODEL_OUTPUT, 'ship_horn_classifier_8class_improved.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_info['model'], f)
    print(f"✓ Sklearn Model saved to {model_path}")

# Save metadata
metadata = {
    'model_type': best_model_name,
    'test_accuracy': float(best_model_info['test_accuracy']),
    'cv_accuracy': float(best_model_info['cv_accuracy']),
    'feature_dims': {
        'yamnet_flat': X_yamnet_flat.shape[1],
        'yamnet_seq': list(X_yamnet_seq.shape[1:]),
        'custom': X_custom.shape[1],
        'total_flat': X_combined.shape[1]
    },
    'class_names': CLASS_NAMES,
    'num_classes': len(CLASS_NAMES),
    'num_train_samples': len(X_train_flat),
    'num_test_samples': len(X_test_flat),
    'improvements': [
        'GridSearchCV for hyperparameter tuning',
        'Added LSTM model for sequence analysis',
        'Enhanced blast pattern detection',
        'Long/short blast classification',
        'Sequence analysis features',
        'Energy distribution features',
        'Balanced class weights'
    ]
}

if 'best_params' in best_model_info:
    metadata['best_params'] = best_model_info['best_params']

metadata_path = os.path.join(MODEL_OUTPUT, 'metadata_8class_improved.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to {metadata_path}")

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {best_model_info['test_accuracy']:.1%}")
print("=" * 70)

