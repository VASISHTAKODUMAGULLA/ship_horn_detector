# 🚢 Ship Horn Signal Classifier

A machine learning system for classifying ship horn signals according to COLREG (International Regulations for Preventing Collisions at Sea) standards. This project implements complete 8-class classification using advanced audio processing techniques and hybrid deep learning models.

---

## 📋 Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [COLREG Classes](#colreg-classes-8-classes)
- [Key Improvements](#key-improvements)
- [Technical Details](#technical-details)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Known Issues & Solutions](#known-issues--solutions)

---

## ✨ Features

- **Complete COLREG Coverage**: All 8 ship horn signal classes from Rule 34
- **Hybrid Models**: Combines LSTM neural networks with Random Forest
- **Advanced Audio Processing**:
  - YAMNet embeddings for deep audio representation
  - Band-pass filtering (50-600 Hz) to remove wind noise and hiss
  - Enhanced blast pattern detection using threshold-based segmentation
  - Temporal sequence analysis for long+short pattern recognition
- **Pattern-Based Correction**: Rule-based validation to catch obvious model errors
- **High Accuracy**: 96.5% test accuracy with intelligent error correction

---

## 🚀 Getting Started

### 1. Create Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 COLREG Classes (8 Classes)

### Complete Classification System

| Class | Description                   | Pattern | COLREG Rule |
|-------|-------------------------------|---------|-------------|
| 0     | Alter course to starboard    | 1 short | Rule 34(a)(i) |
| 1     | Alter course to port         | 2 short | Rule 34(a)(ii) |
| 2     | Operating astern propulsion  | 3 short | Rule 34(a)(iii) |
| 3     | Doubt/warning signal         | 5 short | Rule 34(d) |
| 4     | Approaching blind bend       | 1 long  | Rule 34(e) |
| 5     | Overtaking on starboard side | 1 long + 1 short | Rule 34(c)(i) |
| 6     | Overtaking on port side      | 1 long + 2 short | Rule 34(c)(ii) |
| 7     | Agreement signal             | long-short-long-short | Rule 34(c) |

**COLREG Blast Specifications:**
- **Short blast**: approximately 1 second duration
- **Long blast**: 4 to 6 seconds duration

---

## 🔧 Key Improvements

### 1. **Band-Pass Filtering (50-600 Hz)**
- Removes low-frequency wind noise (<50 Hz)
- Filters out high-frequency hiss (>600 Hz)
- Focuses on ship horn frequency range
- Implemented using Butterworth filter (4th order)

### 2. **Enhanced Blast Detection**
- **Threshold-based segmentation** instead of peak detection
- Avoids over-counting on jagged/noisy signals
- Automatic blast merging (gaps < 0.05s)
- Minimum blast duration filtering (> 0.1s)

### 3. **Long vs Short Classification**
- Long blast: > 2.5 seconds (COLREG: 4-6 seconds)
- Short blast: < 2.5 seconds (COLREG: ~1 second)
- Accurate detection for complex patterns (e.g., 1 long + 2 short)

### 4. **Hybrid Model Architecture**
- **LSTM Model**: Uses YAMNet sequence embeddings for temporal patterns
- **Random Forest**: Uses combined YAMNet + custom acoustic features
- **Pattern Correction**: Rule-based validation layer that catches obvious errors

### 5. **Rich Feature Set**
- YAMNet embeddings (1024D x 3 = 3072D)
- Spectral features (centroids, rolloff, ZCR)
- MFCCs (13 coefficients with mean/std)
- Blast pattern features (count, duration, intervals)
- Energy distribution analysis
- Temporal rhythm features

---

## 📊 Technical Details

### Audio Processing Pipeline

1. **Load Audio**: Librosa (16kHz mono)
2. **Band-Pass Filter**: 50-600 Hz (removes noise)
3. **Feature Extraction**:
   - YAMNet deep embeddings
   - Custom acoustic features (spectral, temporal, blast patterns)
4. **Model Prediction**: LSTM or Random Forest
5. **Pattern Validation**: Rule-based correction
6. **Output**: Class prediction with confidence scores

### Blast Detection Algorithm

```python
# Pseudocode
1. Apply band-pass filter (50-600 Hz)
2. Compute RMS energy (frame_length=512, hop_length=256)
3. Normalize RMS to [0, 1]
4. Threshold at 15% of max energy
5. Find contiguous regions above threshold
6. Merge gaps < 0.05s
7. Filter blasts < 0.1s duration
8. Classify as long (>2.5s) or short (<2.5s)
9. Extract durations and intervals
```

### Model Performance

| Model | Test Accuracy | CV Accuracy | Notes |
|-------|--------------|-------------|-------|
| Random Forest | ~95% | ~93% | Uses all features including blast patterns |
| LSTM | 96.5% | ~94% | Better at sequence patterns |
| With Pattern Correction | ~98%* | N/A | Rule-based override for obvious errors |

*Estimated based on pattern matching logic

---

## 💻 Usage

### Step 1: Generate Dataset (Optional)
```bash
cd model_8_class
python generate_improved_8class.py
```
> Generates realistic synthetic ship horn signals with background noise
> Default: 100 samples per class = 800 total samples

### Step 2: Train Model
```bash
cd model_8_class
python train_improved_8class_new.py
```
> Trains both Random Forest and LSTM models with GridSearchCV
> Expected training time: 5-15 minutes depending on hardware
> Best model is automatically selected and saved

### Step 3: Run Inference
```bash
cd model_8_class
python inference_8class_improved.py <audio_file.wav>
```

**Examples:**
```bash
# Test with real recordings
python inference_8class_improved.py 5_2.wav
python inference_8class_improved.py 1_1.wav

# Test with generated samples
python inference_8class_improved.py dataset_v2_improved/0/0.wav
python inference_8class_improved.py dataset_v2_improved/3/5.wav
```

### Quick Test Multiple Files
```bash
cd model_8_class

# Test 1 short blast
python inference_8class_improved.py 1_1.wav

# Test 2 short blasts
python inference_8class_improved.py 2_1.wav

# Test 5 short blasts
python inference_8class_improved.py 5_2.wav

# Test long-short-long-short pattern
python inference_8class_improved.py 7.wav
```

### Batch Processing
```bash
cd model_8_class

# Test all samples in a class folder
for file in dataset_v2_improved/3/*.wav; do
    echo "Testing: $file"
    python inference_8class_improved.py "$file"
    echo "---"
done
```

---

## 📤 Sample Output

```
======================================================================
SHIP HORN SIGNAL CLASSIFIER (8-Class Improved Model)
======================================================================
Model: ship_horn_classifier_8class_lstm.h5
Model Type: LSTM
Trained Accuracy: 96.5%
Audio file: 5_2.wav
======================================================================

⚠️  WARNING: Detected pattern conflicts with model prediction!
   Model predicted: Class 2 (expects 3 short blasts)
   Actually detected: 5 blasts (0 long, 5 short)
   → Correcting to Class 3: doubt_intentions (5 short)

PREDICTION RESULTS:
----------------------------------------------------------------------
Predicted Class: 3
Signal Type: doubt_intentions (5 short)
Note: Corrected based on detected pattern

Confidence Scores (All 8 Classes):
----------------------------------------------------------------------
→ Class 3:  95.0% ████████████████████████████████████████████████ doubt_intentions (5 short)

----------------------------------------------------------------------
DETECTED PATTERN ANALYSIS:
----------------------------------------------------------------------
  Number of blasts detected: 5
  Long blasts (>2.5s): 0
  Short blasts (<2.5s): 5
  Average blast duration: 0.93 seconds
  Average interval: 0.35 seconds
======================================================================
```

---

## 📁 Project Structure

```
ship_horn_detector/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
└── model_8_class/                     # 8-class model (main)
    ├── generate_improved_8class.py    # Dataset generator
    ├── train_improved_8class_new.py   # Training script (latest)
    ├── inference_8class_improved.py   # Inference with pattern correction
    │
    ├── models/
    │   ├── ship_horn_classifier_8class_lstm.h5        # LSTM model
    │   ├── ship_horn_classifier_8class_improved.pkl   # Random Forest
    │   └── metadata_8class_improved.json              # Model metadata
    │
    ├── dataset_v2_improved/           # Generated samples (8 classes)
    │   ├── 0/                         # 1 short blast
    │   ├── 1/                         # 2 short blasts
    │   ├── 2/                         # 3 short blasts
    │   ├── 3/                         # 5 short blasts
    │   ├── 4/                         # 1 long blast
    │   ├── 5/                         # 1 long + 1 short
    │   ├── 6/                         # 1 long + 2 short
    │   └── 7/                         # long-short-long-short
    │
    ├── results/                       # Training results, plots
    │   └── confusion_matrix_improved_8class.png
    │
    └── Test samples:                  # Real recordings for testing
        ├── 1_1.wav                    # 1 short blast sample
        ├── 2_1.wav, 2_2.wav          # 2 short blasts samples
        ├── 3_2.wav                    # 3 short blasts sample
        ├── 5_1.wav, 5_2.wav          # 5 short blasts samples
        └── 7.wav                      # Agreement signal sample
```

---

## 🔍 Known Issues & Solutions

### Issue 1: LSTM Predicts Wrong Class Despite Correct Pattern Detection
**Problem**: LSTM model predicts Class 2 (3 short) when pattern clearly shows 5 short blasts

**Root Cause**: LSTM only uses YAMNet embeddings, doesn't directly see blast pattern features

**Solution**: ✅ Pattern-based correction layer now automatically fixes obvious mismatches

### Issue 2: CUDA/GPU Warnings
**Problem**: TensorFlow shows GPU-related warnings on CPU-only systems

**Solution**: These are harmless warnings. Model runs fine on CPU. To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

### Issue 3: Model File Not Found
**Problem**: Inference script can't find model files

**Solution**: Run inference from `model_8_class/` directory:
```bash
cd model_8_class
python inference_8class_improved.py <audio_file>
```

---

## 🎓 COLREG Reference

**Rule 34 - Maneuvering and Warning Signals**

International regulations governing sound signals for vessels:

- **(a)(i)**: **1 short blast** = I am altering my course to starboard
- **(a)(ii)**: **2 short blasts** = I am altering my course to port  
- **(a)(iii)**: **3 short blasts** = I am operating astern propulsion
- **(c)**: **Overtaking signals**:
  - 1 long + 1 short = Intending to overtake on your starboard side
  - 1 long + 2 short = Intending to overtake on your port side
  - Agreement: long-short-long-short pattern
- **(d)**: **5 or more short blasts** = Doubt/warning signal (danger)
- **(e)**: **1 long blast** = Approaching blind bend or restricted visibility area

---

## 📊 Future Improvements

- [ ] Real-world audio data collection for fine-tuning
- [ ] Multi-modal LSTM that combines embeddings + blast features
- [ ] Real-time streaming inference
- [ ] Web interface for easy testing
- [ ] Distance estimation from signal amplitude
- [ ] Background noise robustness testing
- [ ] Mobile app deployment
- [ ] Integration with AIS (Automatic Identification System)

---

## 🙏 Acknowledgments

- **YAMNet**: Google's pretrained audio event detection model
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing library
- **scikit-learn**: Machine learning algorithms
- **COLREG**: International Maritime Organization regulations

---

## 📄 License

This project is for educational and research purposes.

---

## 👤 Author

Created as part of ship horn signal classification research.

---

**Last Updated**: November 23, 2025
