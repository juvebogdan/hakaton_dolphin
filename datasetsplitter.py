import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from scipy.signal import butter, filtfilt, firwin
import soundfile as sf
import csv
import shutil
import random
from pathlib import Path

SAMPLE_RATE = 96000

# Define paths
base_dir = Path('../hakaton')
train_source_dir = base_dir
train_dest_dir = base_dir / 'audio_train'
test_dest_dir = base_dir / 'audio_test'
csv_path = base_dir / 'train.csv'

# Create destination directories if they don't exist
train_dest_dir.mkdir(exist_ok=True)
test_dest_dir.mkdir(exist_ok=True)

# Count original dataset files
whistle_files = list((train_source_dir / 'whistles').glob('*.wav'))
noise_files = list((train_source_dir / 'noise').glob('*.wav'))

# Function to preprocess audio - with reduced printing
def read_and_preprocess_segment(file_path, start_time=0, end_time=None):
    # First get file info to get sample rate
    info = sf.info(file_path)
    sr = info.samplerate
    duration = info.duration

    # If end_time is not specified, use the entire file
    if end_time is None:
        end_time = duration

    # Calculate sample positions
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Read the audio segment
    data, sr = sf.read(file_path, start=start_sample, stop=end_sample)

    # Average channels if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Design bandpass filter (5-20 kHz)
    nyquist = sr / 2
    low = 5000 / nyquist
    high = 15000 / nyquist
    b, a = butter(4, [low, high], btype='band')

    # Apply bandpass filter
    filtered_data = filtfilt(b, a, data)

    # Whitening filter
    whitening_filter = firwin(101, [low, high], pass_zero=False)
    whitened_data = filtfilt(whitening_filter, [1], filtered_data)

    return whitened_data, sr

def process_files(class_name, train_ratio=0.8):
    files = list((train_source_dir / class_name).glob('*.wav'))
    random.shuffle(files)

    # Determine split index
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    train_file_info = []
    test_file_info = []

    # Copy training files and collect info for CSV
    for file in train_files:
        shutil.copy2(file, train_dest_dir / file.name)
        train_file_info.append({'fname': file.name, 'label': class_name})

    # Copy test files and collect info for CSV
    for file in test_files:
        shutil.copy2(file, test_dest_dir / file.name)
        test_file_info.append({'fname': file.name, 'label': class_name})

    return train_file_info, test_file_info, len(train_files), len(test_files)

# Process both classes
train_file_info = []
test_file_info = []

# Process both classes
whistle_train_info, whistle_test_info, whistle_train_count, whistle_test_count = process_files('whistles')
noise_train_info, noise_test_info, noise_train_count, noise_test_count = process_files('noise')

train_file_info.extend(whistle_train_info)
train_file_info.extend(noise_train_info)
test_file_info.extend(whistle_test_info)
test_file_info.extend(noise_test_info)

# Create CSV file for training files
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['fname', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for file_info in train_file_info:
        writer.writerow(file_info)

# Create CSV file for test files
test_csv_path = base_dir / 'test.csv'
with open(test_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['fname', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for file_info in test_file_info:
        writer.writerow(file_info)
