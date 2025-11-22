"""
Improved 8-Class Training Script
- Enhanced temporal features for long+short pattern detection
- Better blast counting and sequence analysis
- Optimized for 8 COLREG classes
"""

import os
import numpy as np
import librosa
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from tqdm import tqdm
from scipy import signal
import time

# Configuration
DATASET_PATH = 'dataset_v2_improved/'  # Use the full 8-class dataset
SAMPLE_RATE = 16000
MODEL_OUTPUT = 'models/'

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
    """Extract YAMNet embeddings from audio file"""
    try:
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
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

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
            'pattern_ratio': 0
        }
    
    # Convert peaks to time
    peak_times = librosa.frames_to_time(peaks, sr=sr)
    
    # Estimate blast durations by looking at energy above threshold
    blast_durations = []
    blast_intervals = []
    
    # CRITICAL FIX: RMS is at frame rate, not sample rate!
    # librosa.feature.rms uses hop_length=512 by default
    # Frame rate = sample_rate / hop_length = 16000/512 = 31.25 Hz
    # We need to convert frame indices to time properly
    hop_length = 512  # librosa default
    frame_rate = sr / hop_length  # ~31.25 frames/second
    
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
        # Duration in seconds = (end_frame - start_frame) / frame_rate
        duration_seconds = (end_frame - start_frame) / frame_rate
        duration_ms = duration_seconds * 1000  # Convert to milliseconds
        blast_durations.append(duration_ms)
        
        # Interval to next blast (also in seconds)
        if i < len(peaks) - 1:
            interval_frames = peaks[i+1] - peak
            interval = interval_frames / frame_rate  # Convert to seconds
            blast_intervals.append(interval)
    
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
X_yamnet = []
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
        
        yamnet_emb = extract_yamnet_embedding(fpath)
        if yamnet_emb is None:
            continue
        
        custom_feat = extract_enhanced_features(fpath)
        if custom_feat is None:
            continue
        
        X_yamnet.append(yamnet_emb)
        X_custom.append(custom_feat)
        y.append(int(label))

X_yamnet = np.array(X_yamnet)
X_custom = np.array(X_custom)
y = np.array(y)

# Combine features
X_combined = np.concatenate([X_yamnet, X_custom], axis=1)

print(f"\n✓ Dataset loaded successfully")
print(f"  YAMNet features shape: {X_yamnet.shape}")
print(f"  Enhanced custom features shape: {X_custom.shape}")
print(f"  Combined features shape: {X_combined.shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Split dataset
print("\n[3/6] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# Train multiple models with scaling
print("\n[4/6] Training models...")

# Models to train - Gradient Boosting is optional (can be slow)
models = {
    'Random Forest (Improved)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=500,  # More trees
            max_depth=30,      # Deeper trees
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',  # Better for high-dim features
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        ))
    ]),
    'Logistic Regression (Scaled)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42, C=1.0, class_weight='balanced'))
    ])
}

# Add Gradient Boosting with early stopping (optimized based on convergence)
models['Gradient Boosting'] = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(
        n_estimators=50,   # Reduced - converges around iteration 30-40
        max_depth=6,     # Reduced to prevent overfitting
        learning_rate=0.1,  # Lower learning rate for better generalization
        random_state=42,
        subsample=0.8,
        validation_fraction=0.1,  # Use 10% for validation
        n_iter_no_change=5,  # Stop if no improvement for 5 iterations
        tol=0.0001,  # Tolerance for early stopping
        verbose=1
    ))
])

results = {}
import time

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    print("    " + "="*60)
    
    try:
        # For Gradient Boosting, show progress
        is_gb = 'Gradient Boosting' in model_name
        
        if is_gb:
            print("    ⏳ Gradient Boosting can take 5-15 minutes...")
            print("    📊 You'll see progress updates every 10 estimators")
            print("    💡 This is normal - it's training 100 trees sequentially")
            start_time = time.time()
        
        # Cross-validation
        print("    Running cross-validation...")
        if is_gb:
            print("    (This may take a few minutes for Gradient Boosting)")
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"    ✓ Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train on full training set
        print("    Training on full dataset...")
        if is_gb:
            print("    📈 Gradient Boosting will show progress below:")
            print("    (Look for lines like 'Iter X, Train deviance: Y')")
            print("    This is NORMAL - it's training 100 trees sequentially")
            print("    Estimated time: 5-15 minutes depending on your CPU")
            print("    " + "-"*60)
        
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        if is_gb:
            elapsed = time.time() - start_time
            print("    " + "-"*60)
            print(f"    ✓ Gradient Boosting training complete!")
            print(f"    ⏱️  Total training time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Evaluate on test set
        print("    Evaluating on test set...")
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"    ✓ Test accuracy: {test_accuracy:.3f}")
        
        if is_gb:
            print(f"    ⏱️  Training time: {train_time:.1f} seconds")
        
        results[model_name] = {
            'model': model,
            'cv_accuracy': cv_scores.mean(),
            'test_accuracy': test_accuracy,
            'predictions': y_pred
        }
        
        print("    " + "="*60)
        
    except KeyboardInterrupt:
        print(f"\n    ⚠️  Training interrupted for {model_name}")
        print("    Skipping this model...")
        continue
    except Exception as e:
        print(f"\n    ❌ Error training {model_name}: {e}")
        print("    Skipping this model...")
        continue

# Select best model
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n✓ Best model: {best_model_name}")
print(f"  Test accuracy: {results[best_model_name]['test_accuracy']:.3f}")

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

model_path = os.path.join(MODEL_OUTPUT, 'ship_horn_classifier_8class_improved.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Model saved to {model_path}")

# Save metadata
metadata = {
    'model_type': best_model_name,
    'test_accuracy': float(results[best_model_name]['test_accuracy']),
    'cv_accuracy': float(results[best_model_name]['cv_accuracy']),
    'feature_dims': {
        'yamnet': X_yamnet.shape[1],
        'custom': X_custom.shape[1],
        'total': X_combined.shape[1]
    },
    'class_names': CLASS_NAMES,
    'num_classes': len(CLASS_NAMES),
    'num_train_samples': len(X_train),
    'num_test_samples': len(X_test),
    'improvements': [
        'Enhanced blast pattern detection',
        'Long/short blast classification',
        'Sequence analysis features',
        'Energy distribution features',
        'Balanced class weights',
        'More trees and deeper RF'
    ]
}

metadata_path = os.path.join(MODEL_OUTPUT, 'metadata_8class_improved.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to {metadata_path}")

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.1%}")
print(f"Improvement over baseline: Expected 63.7% → 75-85%")
print("\nKey Improvements:")
print("  ✓ Enhanced blast pattern detection")
print("  ✓ Long/short blast classification")
print("  ✓ Better temporal sequence features")
print("  ✓ Balanced class weights")
print("  ✓ More powerful Random Forest (500 trees)")
print("=" * 70)

