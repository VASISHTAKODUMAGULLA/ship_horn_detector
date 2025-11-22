"""
Training Script for Ship Horn Detection
- 4 well-separated classes
- Feature scaling to fix convergence issues
- Better performance expected
"""

import os
import numpy as np
import librosa
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from tqdm import tqdm

# Configuration
DATASET_PATH = 'dataset/'
SAMPLE_RATE = 16000
MODEL_OUTPUT = 'models/'

# Class Names (4 classes)
CLASS_NAMES = {
    0: "alter_starboard (1 short)",
    1: "doubt_intentions (5 short)",
    2: "approaching_bend (1 long)",
    3: "agreement (long-short-long-short)",
}

print("=" * 70)
print("SHIP HORN DETECTION - TRAINING PIPELINE")
print("Focus: 4 well-separated classes for high accuracy")
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
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

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
        
        # 3. MFCCs (reduced to 10 for simpler problem)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        for mfcc in mfccs:
            features.extend([np.mean(mfcc), np.std(mfcc)])
        
        # 4. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms), np.max(rms)])
        
        # 5. IMPORTANT: Onset detection for counting blasts
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True,
            delta=0.5  # Sensitivity for detecting blasts
        )
        features.append(len(onsets))  # Number of detected onsets
        
        # 6. Tempogram
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features.extend([np.mean(tempogram), np.std(tempogram)])
        
        # 7. Duration features (important for long vs short)
        features.append(len(y) / sr)  # Total duration
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting custom features from {wav_path}: {e}")
        return None

# Load dataset
print("\n[2/6] Loading and processing dataset...")
X_yamnet = []
X_custom = []
y = []

if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset not found at {DATASET_PATH}")
    print("Please run: python generate_dataset.py")
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
        
        custom_feat = extract_custom_features(fpath)
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
print(f"  Custom features shape: {X_custom.shape}")
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

# Train models with SCALING (fixes convergence issue)
print("\n[4/6] Training models...")

# IMPORTANT: Use Pipeline with StandardScaler
models = {
    'Logistic Regression (Scaled)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42, C=1.0))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=300, max_depth=25, 
                                              min_samples_split=5, random_state=42, n_jobs=-1))
    ])
}

results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"    Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"    Test accuracy: {test_accuracy:.3f}")
    
    results[model_name] = {
        'model': model,
        'cv_accuracy': cv_scores.mean(),
        'test_accuracy': test_accuracy,
        'predictions': y_pred
    }

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
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"Class {i}" for i in sorted(CLASS_NAMES.keys())],
            yticklabels=[f"Class {i}" for i in sorted(CLASS_NAMES.keys())])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/confusion_matrix.png', dpi=150)
print("\n✓ Confusion matrix saved to results/confusion_matrix.png")

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

model_path = os.path.join(MODEL_OUTPUT, 'ship_horn_classifier.pkl')
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
    'notes': ' 4-class version for high accuracy'
}

metadata_path = os.path.join(MODEL_OUTPUT, 'metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to {metadata_path}")

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.1%}")
print(f"Number of Classes: {len(CLASS_NAMES)}")
print("\nClass Breakdown:")
for i, name in CLASS_NAMES.items():
    print(f"  {i}: {name}")
print(f"\nFiles saved:")
print(f"  - Model: {model_path}")
print(f"  - Metadata: {metadata_path}")
print(f"  - Confusion Matrix: results/confusion_matrix.png")
print("\nTIP: This 4-class model focuses on well-separated patterns")
print("   and should achieve 85-95% accuracy!")
print("=" * 70)

