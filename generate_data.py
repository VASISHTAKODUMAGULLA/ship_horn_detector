import numpy as np
import scipy.io.wavfile as wav
import os
import random

# Configuration
SAMPLE_RATE = 16000
OUTPUT_DIR = "./data/train"
NUM_SAMPLES_PER_CLASS = 5   # Generate 100 files per class

# COLREGs Definitions (based on Source 57-59)
SHORT_DURATION = 1.0  
LONG_DURATION = 5.5   
INTERVAL_DURATION = 1.0 

# Class Map based on signals
CLASSES = {
    "alter_starboard": {"pattern": ["short"], "label": 0},
    "alter_port": {"pattern": ["short", "short"], "label": 1},
    "astern_propulsion": {"pattern": ["short", "short", "short"], "label": 2},
    # "turn_starboard": {"pattern": ["long", "short"], "label": 3},
    # "turn_port": {"pattern": ["long", "short", "short"], "label": 4},
    # "keep_clear": {"pattern": ["short", "short", "short", "short", "short"], "label": 5}, # 5+ shorts
    # "underway": {"pattern": ["long"], "label": 6},
    # "overtake_starboard": {"pattern": ["long", "long", "short"], "label": 7},
    # "overtake_port": {"pattern": ["long", "long", "short", "short"], "label": 8},
    # "agree_overtake": {"pattern": ["long", "short", "long", "short"], "label": 9}
}

def generate_tone(duration, frequency=150, harmonics=True):
    """Generates a ship-horn like sound (fundamental + harmonics)."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    # Add amplitude variation for realism
    amplitude_variation = random.uniform(0.5, 0.7)
    
    # Fundamental frequency (Low pitch for large ships)
    audio = amplitude_variation * np.sin(2 * np.pi * frequency * t)
    if harmonics:
        # Add harmonics for realism with varied amplitudes
        audio += random.uniform(0.2, 0.4) * np.sin(2 * np.pi * (frequency * 2) * t)
        audio += random.uniform(0.05, 0.15) * np.sin(2 * np.pi * (frequency * 3) * t)
        
        # Optional: add a fourth harmonic for more texture
        if random.random() > 0.5:
            audio += random.uniform(0.02, 0.08) * np.sin(2 * np.pi * (frequency * 4) * t)
    
    return audio

def generate_silence(duration):
    return np.zeros(int(SAMPLE_RATE * duration))

def add_noise(audio_signal):
    """Adds background noise (wind/water simulation)"""
    noise_level = random.uniform(0.01, 0.1)  # Reduced from 0.05-0.3 to 0.01-0.1
    noise = np.random.normal(0, 1, len(audio_signal))
    return audio_signal + (noise * noise_level)

def build_sequence(pattern):
    sequence = []
    # Randomize frequency slightly per ship (100Hz to 350Hz)
    freq = random.randint(100, 350)
    
    for i, note in enumerate(pattern):
        if note == "short":
            # Increased variation from ±0.2s to ±0.3s
            dur = SHORT_DURATION + random.uniform(-0.3, 0.3)
            sequence.append(generate_tone(dur, freq))
        elif note == "long":
            # Increased variation from ±0.5s to ±0.8s
            dur = LONG_DURATION + random.uniform(-0.8, 0.8)
            sequence.append(generate_tone(dur, freq))
        
        # Add silence after blast (unless it's the last one)
        if i < len(pattern) - 1:
            # Randomize interval duration too
            interval = INTERVAL_DURATION + random.uniform(-0.2, 0.2)
            sequence.append(generate_silence(interval))
            
    return np.concatenate(sequence)

# Main Generation Loop
if __name__ == "__main__":
    for _, class_info in CLASSES.items():
        class_name = _
        pattern = class_info["pattern"]
        label = class_info["label"]
        class_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Generating data for: {class_name}...")
        
        for i in range(NUM_SAMPLES_PER_CLASS):
            # 1. Build the pure signal
            raw_signal = build_sequence(pattern)
            
            # 2. Add Environmental Noise (Source 89, 115)
            noisy_signal = add_noise(raw_signal)
            
            # 3. Normalize to 16-bit PCM
            noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
            noisy_signal = (noisy_signal * 32767).astype(np.int16)
            
            # 4. Save
            filename = f"{i}.wav"
            wav.write(os.path.join(class_dir, filename), SAMPLE_RATE, noisy_signal)

    print("Dataset generation complete.")