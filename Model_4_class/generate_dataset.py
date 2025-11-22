"""
Ship Horn Dataset Generator - Focus on What Works
Uses only 4 well-separated classes for higher accuracy
"""

import numpy as np
import scipy.io.wavfile as wav
import os
import random
from scipy.signal import butter, lfilter

# Configuration
SAMPLE_RATE = 16000
OUTPUT_DIR = "./dataset"
NUM_SAMPLES_PER_CLASS = 250  # More samples for fewer classes

# COLREGs Definitions
SHORT_DURATION = 1.0  
LONG_DURATION = 5.0   
INTERVAL_DURATION = 1.0 

#: Only 4 well-separated classes
CLASSES = {
    "alter_starboard": {"pattern": ["short"], "label": 0},
    "doubt_intentions": {"pattern": ["short", "short", "short", "short", "short"], "label": 1},
    "approaching_bend": {"pattern": ["long"], "label": 2},
    "agreement": {"pattern": ["long", "short", "long", "short"], "label": 3},
}

def generate_realistic_horn(duration, base_freq=150, ship_size='large'):
    """Generate realistic ship horn with proper envelope and harmonics"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    # Ship size affects frequency range
    if ship_size == 'large':
        base_freq = random.uniform(100, 200)
    elif ship_size == 'medium':
        base_freq = random.uniform(200, 300)
    else:  # small
        base_freq = random.uniform(300, 500)
    
    # Generate fundamental
    audio = np.sin(2 * np.pi * base_freq * t)
    
    # Add rich harmonics
    harmonics = [(2, 0.6), (3, 0.3), (4, 0.15), (5, 0.08), (6, 0.04)]
    
    for harmonic_num, amplitude in harmonics:
        freq_mod = base_freq * harmonic_num * (1 + 0.002 * np.sin(2 * np.pi * 5 * t))
        audio += amplitude * np.sin(2 * np.pi * freq_mod * t)
    
    # Create ADSR envelope
    envelope = create_adsr_envelope(len(audio), 
                                    attack_time=0.05,
                                    decay_time=0.1,
                                    sustain_level=0.8,
                                    release_time=0.15)
    
    audio = audio * envelope
    
    # Add amplitude modulation
    am_freq = random.uniform(3, 8)
    am_depth = random.uniform(0.05, 0.15)
    audio = audio * (1 + am_depth * np.sin(2 * np.pi * am_freq * t))
    
    # Apply slight distortion
    audio = np.tanh(audio * random.uniform(1.0, 1.5))
    
    return audio * random.uniform(0.6, 0.9)

def create_adsr_envelope(length, attack_time=0.05, decay_time=0.1, 
                        sustain_level=0.8, release_time=0.15):
    """Create ADSR envelope for realistic horn sound"""
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
    
    # Low frequency rumble
    if noise_type in ['mixed', 'machinery']:
        rumble = np.random.normal(0, 1, samples)
        b, a = butter(4, 200 / (SAMPLE_RATE / 2), btype='low')
        rumble = lfilter(b, a, rumble)
        noise += rumble * random.uniform(0.05, 0.15)
    
    # Wind noise
    if noise_type in ['mixed', 'wind']:
        wind = np.random.normal(0, 1, samples)
        b, a = butter(3, [500 / (SAMPLE_RATE / 2), 3000 / (SAMPLE_RATE / 2)], btype='band')
        wind = lfilter(b, a, wind)
        noise += wind * random.uniform(0.03, 0.1)
    
    # Water sounds
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
    else:  # far
        b, a = butter(3, 2000 / (SAMPLE_RATE / 2), btype='low')
        return lfilter(b, a, audio) * random.uniform(0.3, 0.5)

def build_sequence(pattern):
    """Build complete signal sequence with variations"""
    sequence = []
    
    ship_size = random.choice(['small', 'medium', 'large'])
    distance = random.choice(['near', 'near', 'medium', 'far'])
    
    for i, note in enumerate(pattern):
        if note == "short":
            # IMPORTANT: Keep short blasts more consistent for better counting
            dur = SHORT_DURATION + random.uniform(-0.15, 0.15)  # Reduced variation
            horn = generate_realistic_horn(dur, ship_size=ship_size)
        elif note == "long":
            dur = LONG_DURATION + random.uniform(-0.4, 0.4)
            horn = generate_realistic_horn(dur, ship_size=ship_size)
        
        horn = apply_distance_effect(horn, distance)
        sequence.append(horn)
        
        # IMPORTANT: Keep intervals more consistent
        if i < len(pattern) - 1:
            interval = INTERVAL_DURATION + random.uniform(-0.1, 0.1)  # Reduced variation
            sequence.append(generate_silence(interval))
    
    signal = np.concatenate(sequence)
    
    # Add maritime background noise
    noise_type = random.choice(['mixed', 'machinery', 'wind', 'water'])
    background = generate_maritime_noise(len(signal) / SAMPLE_RATE, noise_type)
    
    # Fix: Ensure exact length match
    if len(background) > len(signal):
        background = background[:len(signal)]
    elif len(background) < len(signal):
        background = np.pad(background, (0, len(signal) - len(background)), mode='constant')
    
    # Higher SNR for better learning (10-25 dB instead of 5-20)
    snr_db = random.uniform(10, 25)
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
    """Normalize to 16-bit PCM with headroom"""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.7
    return (audio * 32767).astype(np.int16)

# Main Generation Loop
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("COLREG SHIP HORN DATASET GENERATOR")
    print("Focus: 4 well-separated classes for high accuracy")
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
            
            signal = build_sequence(pattern)
            signal = normalize_audio(signal)
            
            filename = f"{i}.wav"
            wav.write(os.path.join(class_dir, filename), SAMPLE_RATE, signal)
        
        print(f"  ✓ Complete: {NUM_SAMPLES_PER_CLASS} files saved")
    
    print("\n" + "=" * 70)
    print("✓ DATASET GENERATION COMPLETE")
    print(f"Total classes: {len(CLASSES)}")
    print(f"Samples per class: {NUM_SAMPLES_PER_CLASS}")
    print(f"Total samples: {len(CLASSES) * NUM_SAMPLES_PER_CLASS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nSelected classes (well-separated for high accuracy):")
    print("  0: 1 short blast (alter starboard)")
    print("  1: 5 short blasts (doubt/warning)")
    print("  2: 1 long blast (restricted visibility)")
    print("  3: Long-short-long-short (agreement)")
    print("=" * 70)

