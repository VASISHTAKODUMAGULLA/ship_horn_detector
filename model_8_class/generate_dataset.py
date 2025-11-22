"""
Improved 8-Class Dataset Generator
- Better separation for long+short patterns
- More consistent timing for better classification
- Optimized for 8 COLREG classes
"""

import numpy as np
import scipy.io.wavfile as wav
import os
import random
from scipy.signal import butter, lfilter, resample
from scipy import signal

# Configuration
SAMPLE_RATE = 16000
OUTPUT_DIR = "./dataset_v2_improved"
NUM_SAMPLES_PER_CLASS = 250  # More samples

# COLREGs Definitions - TIGHTER for better separation
SHORT_DURATION = 1.0  
LONG_DURATION = 5.0   
INTERVAL_DURATION = 1.0 

# All 8 COLREG Signal Classes
CLASSES = {
    "alter_starboard": {"pattern": ["short"], "label": 0},
    "alter_port": {"pattern": ["short", "short"], "label": 1},
    "astern_propulsion": {"pattern": ["short", "short", "short"], "label": 2},
    "doubt_intentions": {"pattern": ["short", "short", "short", "short", "short"], "label": 3},
    "approaching_bend": {"pattern": ["long"], "label": 4},
    "overtake_starboard": {"pattern": ["long", "short"], "label": 5},
    "overtake_port": {"pattern": ["long", "short", "short"], "label": 6},
    "agreement": {"pattern": ["long", "short", "long", "short"], "label": 7},
}

def generate_realistic_horn(duration, base_freq=150, ship_size='large'):
    """Generate realistic ship horn"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    if ship_size == 'large':
        base_freq = random.uniform(100, 200)
    elif ship_size == 'medium':
        base_freq = random.uniform(200, 300)
    else:
        base_freq = random.uniform(300, 500)
    
    audio = np.sin(2 * np.pi * base_freq * t)
    
    harmonics = [(2, 0.6), (3, 0.3), (4, 0.15), (5, 0.08), (6, 0.04)]
    
    for harmonic_num, amplitude in harmonics:
        freq_mod = base_freq * harmonic_num * (1 + 0.002 * np.sin(2 * np.pi * 5 * t))
        audio += amplitude * np.sin(2 * np.pi * freq_mod * t)
    
    envelope = create_adsr_envelope(len(audio), 0.05, 0.1, 0.8, 0.15)
    audio = audio * envelope
    
    am_freq = random.uniform(3, 8)
    am_depth = random.uniform(0.05, 0.15)
    audio = audio * (1 + am_depth * np.sin(2 * np.pi * am_freq * t))
    
    audio = np.tanh(audio * random.uniform(1.0, 1.5))
    
    return audio * random.uniform(0.6, 0.9)

def create_adsr_envelope(length, attack_time=0.05, decay_time=0.1, 
                        sustain_level=0.8, release_time=0.15):
    """Create ADSR envelope"""
    envelope = np.zeros(length)
    
    attack_samples = int(attack_time * SAMPLE_RATE)
    decay_samples = int(decay_time * SAMPLE_RATE)
    release_samples = int(release_time * SAMPLE_RATE)
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    if decay_samples > 0:
        envelope[attack_samples:attack_samples + decay_samples] = \
            np.linspace(1, sustain_level, decay_samples)
    
    sustain_end = length - release_samples
    envelope[attack_samples + decay_samples:sustain_end] = sustain_level
    
    if release_samples > 0 and sustain_end < length:
        envelope[sustain_end:] = np.linspace(sustain_level, 0, length - sustain_end)
    
    return envelope

def generate_maritime_noise(duration, noise_type='mixed'):
    """Generate realistic maritime background noise"""
    samples = int(SAMPLE_RATE * duration)
    noise = np.zeros(samples)
    
    if noise_type in ['mixed', 'machinery']:
        rumble = np.random.normal(0, 1, samples)
        b, a = butter(4, 200 / (SAMPLE_RATE / 2), btype='low')
        rumble = lfilter(b, a, rumble)
        noise += rumble * random.uniform(0.05, 0.15)
    
    if noise_type in ['mixed', 'wind']:
        wind = np.random.normal(0, 1, samples)
        b, a = butter(3, [500 / (SAMPLE_RATE / 2), 3000 / (SAMPLE_RATE / 2)], btype='band')
        wind = lfilter(b, a, wind)
        noise += wind * random.uniform(0.03, 0.1)
    
    if noise_type in ['mixed', 'water']:
        water = np.random.normal(0, 1, samples)
        noise += water * random.uniform(0.02, 0.08)
    
    return noise

def generate_silence(duration):
    """Generate silence with minimal noise"""
    samples = int(SAMPLE_RATE * duration)
    return np.random.normal(0, 0.001, samples)

def apply_distance_effect(audio, distance='near'):
    """Simulate different distances"""
    if distance == 'near':
        return audio
    elif distance == 'medium':
        b, a = butter(2, 4000 / (SAMPLE_RATE / 2), btype='low')
        return lfilter(b, a, audio) * random.uniform(0.6, 0.8)
    else:
        b, a = butter(3, 2000 / (SAMPLE_RATE / 2), btype='low')
        return lfilter(b, a, audio) * random.uniform(0.3, 0.5)

def build_sequence_improved(pattern, class_label):
    """
    IMPROVED: Build sequence with better separation for problematic classes
    """
    sequence = []
    
    ship_size = random.choice(['small', 'medium', 'large'])
    distance = random.choice(['near', 'near', 'medium', 'far'])
    
    # IMPROVED: For classes with long+short patterns (5, 6, 7), use clearer separation
    is_mixed_pattern = class_label in [5, 6, 7]
    
    for i, note in enumerate(pattern):
        if note == "short":
            # TIGHTER variation for better counting
            if is_mixed_pattern:
                # For long+short patterns, make shorts very clear (0.8-1.2s)
                dur = SHORT_DURATION + random.uniform(-0.2, 0.2)
            else:
                # For all-short patterns, slightly more variation
                dur = SHORT_DURATION + random.uniform(-0.15, 0.15)
            horn = generate_realistic_horn(dur, ship_size=ship_size)
            
        elif note == "long":
            # Make long blasts VERY clear (4.5-6.0s) for mixed patterns
            if is_mixed_pattern:
                dur = LONG_DURATION + random.uniform(-0.5, 1.0)  # 4.5-6.0s
            else:
                dur = LONG_DURATION + random.uniform(-0.5, 0.5)  # 4.5-5.5s
            horn = generate_realistic_horn(dur, ship_size=ship_size)
        
        horn = apply_distance_effect(horn, distance)
        sequence.append(horn)
        
        # IMPROVED: More consistent intervals for better pattern recognition
        if i < len(pattern) - 1:
            if is_mixed_pattern:
                # Clear intervals for mixed patterns
                interval = INTERVAL_DURATION + random.uniform(-0.1, 0.1)
            else:
                interval = INTERVAL_DURATION + random.uniform(-0.15, 0.15)
            sequence.append(generate_silence(interval))
    
    signal = np.concatenate(sequence)
    
    # IMPROVED: Lower noise for problematic classes (better SNR)
    noise_type = random.choice(['mixed', 'machinery', 'wind', 'water'])
    background = generate_maritime_noise(len(signal) / SAMPLE_RATE, noise_type)
    
    if len(background) > len(signal):
        background = background[:len(signal)]
    elif len(background) < len(signal):
        background = np.pad(background, (0, len(signal) - len(background)), mode='constant')
    
    # IMPROVED: Higher SNR for problematic classes
    if class_label in [1, 2, 5, 6]:  # Classes that get confused
        snr_db = random.uniform(12, 25)  # Higher SNR
    else:
        snr_db = random.uniform(8, 20)  # Normal SNR
    
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(background ** 2)
    if noise_power > 0:
        desired_noise_power = signal_power / (10 ** (snr_db / 10))
        background = background * np.sqrt(desired_noise_power / noise_power)
    
    mixed = signal + background
    
    # Add padding
    padding_start = generate_maritime_noise(random.uniform(0.5, 2.0), noise_type)
    padding_end = generate_maritime_noise(random.uniform(0.5, 2.0), noise_type)
    
    mixed = np.concatenate([padding_start, mixed, padding_end])
    
    return mixed

def normalize_audio(audio):
    """Normalize to 16-bit PCM"""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.7
    return (audio * 32767).astype(np.int16)

# Main Generation Loop
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("IMPROVED 8-CLASS COLREG SHIP HORN DATASET GENERATOR")
    print("Better separation for long+short patterns")
    print("=" * 70)
    
    for class_name, class_info in CLASSES.items():
        pattern = class_info["pattern"]
        label = class_info["label"]
        class_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"\n[Class {label}] Generating {NUM_SAMPLES_PER_CLASS} samples for: {class_name}")
        print(f"  Pattern: {' + '.join(pattern)}")
        
        for i in range(NUM_SAMPLES_PER_CLASS):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{NUM_SAMPLES_PER_CLASS}")
            
            # Use improved sequence builder
            signal = build_sequence_improved(pattern, label)
            signal = normalize_audio(signal)
            
            filename = f"{i}.wav"
            wav.write(os.path.join(class_dir, filename), SAMPLE_RATE, signal)
        
        print(f"  ✓ Complete: {NUM_SAMPLES_PER_CLASS} files saved")
    
    print("\n" + "=" * 70)
    print("✓ IMPROVED DATASET GENERATION COMPLETE")
    print(f"Total classes: {len(CLASSES)}")
    print(f"Samples per class: {NUM_SAMPLES_PER_CLASS}")
    print(f"Total samples: {len(CLASSES) * NUM_SAMPLES_PER_CLASS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nImprovements:")
    print("  ✓ Clearer long/short separation for mixed patterns")
    print("  ✓ Higher SNR for problematic classes")
    print("  ✓ More consistent timing")
    print("=" * 70)

