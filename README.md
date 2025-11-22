## 🚀 Getting Started with the Horn Signal Classifier

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Model 4-Classifier (4 Classes)

The complete trained model is saved in the `model_4class/` folder.

If you'd like to generate the dataset again:
```bash
py generate_dataset.py
```
> You can customize the number of samples per class by changing the variables in the script (default: 250 each).

Then retrain the model:
```bash
py train_model.py
```

---

## Running Inference on Real Data

To test the model with a real horn sample, run:
```bash
py inference.py 6.wav
```
This will predict the class of the audio file `6.wav` (your only real horn recording).

---

## Classes (4-Class Model)
| Class | Description                   |
|-------|-------------------------------|
| 0     | 1 short blast                |
| 3     | 5 short blasts               |
| 4     | 1 long blast                 |
| 7     | long-short-long-short pattern |

---

## 8th Class (In Progress)
The dataset for the 8th class has already been generated.

> You can regenerate it if needed:
```bash
py generate_dataset.py
```
As before, change the class sample size inside the script (default: 250 each).
