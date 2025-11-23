# 🚢 Ship Horn Signal Classifier

A machine learning system for classifying ship horn signals according to COLREG (International Regulations for Preventing Collisions at Sea) standards. This project implements both 4-class and 8-class classification models using advanced audio processing techniques.

---

## 📋 Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [Model 4-Class](#model-4-class-4-classes)
- [Model 8-Class (Improved)](#model-8-class-improved-8-classes)
- [Technical Details](#technical-details)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)

---

## ✨ Features

- **Dual Classification Systems**: 4-class (basic) and 8-class (complete COLREG coverage)
- **Hybrid Models**: Combines LSTM neural networks with traditional ML (Random Forest)
- **Advanced Audio Processing**:
  - YAMNet embeddings for deep audio representation
  - Band-pass filtering (50-600 Hz) to remove wind noise and hiss
  - Enhanced blast pattern detection using threshold-based segmentation
  - Temporal sequence analysis for long+short pattern recognition
- **Pattern-Based Correction**: Rule-based validation to catch obvious model errors
- **High Accuracy**: 96.5% test accuracy on 8-class model

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

## 📊 Model 4-Class (4 Classes)

### Location
The complete trained model is saved in the `Model_4_class/` folder.

### Classes
| Class | Description                   | COLREG Rule |
|-------|-------------------------------|-------------|
| 0     | 1 short blast                | Rule 34(a) - Altering course to starboard |
| 3     | 5 short blasts               | Rule 34(d) - Doubt/warning signal |
| 4     | 1 long blast                 | Rule 34(e) - Approaching blind bend |
| 7     | long-short-long-short pattern | Rule 34(c) - Agreement signal |

### Generate Dataset (Optional)
If you'd like to generate the dataset again:
```bash
cd Model_4_class
python generate_dataset.py
```
> You can customize the number of samples per class by changing the variables in the script (default: 250 each).

### Train Model
```bash
cd Model_4_class
python train_model.py
```

### Run Inference
```bash
cd Model_4_class
python inference.py 6.wav
```

---

## 🎯 Model 8-Class Improved (8 Classes)

### Location
All files are in the `model_8_class/` folder.

### Complete COLREG Classes
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

### Key Improvements

#### 1. **Band-Pass Filtering (50-600 Hz)**
- Removes low-frequency wind noise (<50 Hz)
- Filters out high-frequency hiss (>600 Hz)
- Focuses on ship horn frequency range
- Implemented using Butterworth filter (4th order)

#### 2. **Enhanced Blast Detection**
- **Threshold-based segmentation** instead of peak detection
- Avoids over-counting on jagged/noisy signals
- Automatic blast merging (gaps < 0.05s)
- Minimum blast duration filtering (> 0.1s)

#### 3. **Long vs Short Classification**
- Long blast: > 2.5 seconds (COLREG: 4-6 seconds)
- Short blast: < 2.5 seconds (COLREG: ~1 second)
- Accurate for complex patterns (e.g., 1 long + 2 short)

#### 4. **Hybrid Model Architecture**
- **LSTM Model**: Uses YAMNet sequence embeddings for temporal patterns
- **Random Forest**: Uses combined YAMNet + custom acoustic features
- **Pattern Correction**: Rule-based validation layer

#### 5. **Rich Feature Set**
- YAMNet embeddings (1024D x 3 = 3072D)
- Spectral features (centroids, rolloff, ZCR)
- MFCCs (13 coefficients with mean/std)
- Blast pattern features (count, duration, intervals)
- Energy distribution analysis
- Temporal rhythm features

### Generate Dataset
```bash
cd model_8_class
python generate_improved_8class.py
```
> Generates realistic synthetic ship horn signals with background noise
> Default: 100 samples per class = 800 total samples

### Train Model
```bash
cd model_8_class
python train_improved_8class_new.py
```
> Trains both Random Forest and LSTM models with GridSearchCV
> Expected training time: 5-15 minutes depending on hardware
> Best model is automatically selected and saved

### Run Inference
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

### Sample Output
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

## 🔧 Technical Details

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

## 📝 Usage Examples

### Quick Test with Sample Files
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

### Process Multiple Files
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

## 📁 Project Structure

```
ship_horn_detector/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Model_4_class/                     # 4-class model (basic)
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── inference.py
│   ├── models/                        # Trained models
│   ├── dataset/                       # Generated audio samples
│   └── results/                       # Confusion matrices, plots
│
└── model_8_class/                     # 8-class model (improved)
    ├── generate_improved_8class.py    # Dataset generator
    ├── train_improved_8class_new.py   # Training script (latest)
    ├── inference_8class_improved.py   # Inference with pattern correction
    ├── models/
    │   ├── ship_horn_classifier_8class_lstm.h5        # LSTM model
    │   ├── ship_horn_classifier_8class_improved.pkl   # Random Forest
    │   └── metadata_8class_improved.json              # Model metadata
    ├── dataset_v2_improved/           # Generated samples (8 classes)
    │   ├── 0/                         # 1 short blast
    │   ├── 1/                         # 2 short blasts
    │   ├── 2/                         # 3 short blasts
    │   ├── 3/                         # 5 short blasts
    │   ├── 4/                         # 1 long blast
    │   ├── 5/                         # 1 long + 1 short
    │   ├── 6/                         # 1 long + 2 short
    │   └── 7/                         # long-short-long-short
    └── results/                       # Training results, plots
```

---

## 🔍 Known Issues & Solutions

### Issue 1: LSTM Predicts Wrong Class Despite Correct Pattern Detection
**Problem**: LSTM model predicts Class 2 (3 short) when pattern clearly shows 5 short blasts

**Root Cause**: LSTM only uses YAMNet embeddings, doesn't see blast pattern features

**Solution**: Pattern-based correction layer now automatically fixes obvious mismatches

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

- **(a)(i)**: 1 short = Altering course to starboard
- **(a)(ii)**: 2 short = Altering course to port  
- **(a)(iii)**: 3 short = Operating astern propulsion
- **(c)**: Overtaking signals (1 long + 1-2 short, or agreement pattern)
- **(d)**: 5 short = Doubt/warning signal
- **(e)**: 1 long = Approaching blind bend

**Blast Specifications:**
- Short blast: approximately 1 second
- Long blast: 4 to 6 seconds

---

## 📊 Future Improvements

- [ ] Real-world audio data collection for fine-tuning
- [ ] Multi-modal LSTM that combines embeddings + blast features
- [ ] Real-time streaming inference
- [ ] Web interface for easy testing
- [ ] Distance estimation from signal amplitude
- [ ] Background noise robustness testing
- [ ] Mobile app deployment

---

## 🙏 Acknowledgments

- **YAMNet**: Google's pretrained audio event detection model
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing library
- **scikit-learn**: Machine learning algorithms

---

## 📄 License

This project is for educational and research purposes.

---

## 👤 Author

Created as part of ship horn signal classification research.

---

**Last Updated**: November 23, 2025
