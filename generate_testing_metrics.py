import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import wave
from scipy import signal
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import librosa
from scipy.stats import skew
import warnings


from dolphin_detector.metricsDolphin import (
    computeMetrics, highFreqTemplate, buildHeader
)
from dolphin_detector.config import MAX_FREQUENCY, MAX_TIME
from dolphin_detector.generate_training_metrics2 import (
    TemplateManager, create_bar_templates
)

# Suppress warnings
warnings.filterwarnings('ignore')

class AudioDataTest:
    """Simple class to handle test audio files without labels"""
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        if not self.files:
            raise ValueError(f"No WAV files found in {test_dir}")
        
        # Read sample rate from first file
        with wave.open(os.path.join(test_dir, self.files[0]), 'rb') as wav:
            self.sample_rate = wav.getframerate()
        
        self.num_files = len(self.files)
        print(f"Found {self.num_files} WAV files in {test_dir}")
    
    def get_sample(self, index):
        """Get spectrogram for a test file"""
        audio_file = os.path.join(self.test_dir, self.files[index])
        with wave.open(audio_file, 'rb') as wav:
            frames = wav.getnframes()
            audio_bytes = wav.readframes(frames)
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Generate spectrogram
        params = {'NFFT': 2048, 'Fs': self.sample_rate, 'noverlap': 1536}
        freqs, times, Sxx = signal.spectrogram(audio,
                                             fs=params['Fs'],
                                             nperseg=params['NFFT'],
                                             noverlap=params['noverlap'],
                                             scaling='density')
        
        Sxx = 10 * np.log10(Sxx + 1e-10)
        return Sxx, freqs, times
    
    def get_raw(self, index):
        """Get raw audio data for a test file"""
        audio_file = os.path.join(self.test_dir, self.files[index])
        with wave.open(audio_file, 'rb') as wav:
            frames = wav.getnframes()
            audio_bytes = wav.readframes(frames)
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            # Normalize to float32 in range [-1, 1]
            return audio.astype(np.float32) / 32768.0

def process_test_file(args):
    """Process a single test audio file and compute metrics"""
    file_index, audio_data, tmpl, bar_templates = args
    try:
        # Get spectrogram and raw audio
        P, freqs, bins = audio_data.get_sample(file_index)
        audio_data_raw = audio_data.get_raw(file_index)
        sr = audio_data.sample_rate
        
        # Compute template metrics first to ensure they match training
        out = computeMetrics(P, tmpl, bins, MAX_FREQUENCY, MAX_TIME)
        
        # Add high frequency template metrics
        bar_, bar1_, bar2_ = bar_templates
        out += highFreqTemplate(P, bar_)
        out += highFreqTemplate(P, bar1_)
        out += highFreqTemplate(P, bar2_)
        
        # Initialize features dictionary for additional metrics
        features = {}
        
        # Compute additional features only if audio is not empty/silent
        if len(audio_data_raw) > 0 and not np.all(np.abs(audio_data_raw) < 1e-6):
            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data_raw))) > 0)
            features['zero_crossing_rate'] = zero_crossings / len(audio_data_raw)

            # Spectral features
            stft = np.abs(librosa.stft(audio_data_raw))
            
            if np.any(stft):
                # Spectral centroid
                centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
                features['spectral_centroid'] = np.mean(centroid)
                features['spectral_centroid_std'] = np.std(centroid)
                features['spectral_centroid_skew'] = skew(centroid)

                # Spectral bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
                features['spectral_bandwidth'] = np.mean(bandwidth)
                features['spectral_bandwidth_std'] = np.std(bandwidth)
                features['spectral_bandwidth_skew'] = skew(bandwidth)

                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
                features['spectral_rolloff'] = np.mean(rolloff)
                features['spectral_rolloff_std'] = np.std(rolloff)
                features['spectral_rolloff_skew'] = skew(rolloff)

                # Spectral contrast
                contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)[0]
                features['spectral_contrast'] = np.mean(contrast)
                features['spectral_contrast_std'] = np.std(contrast)
                features['spectral_contrast_skew'] = skew(contrast)

                # MFCCs
                mfccs = librosa.feature.mfcc(y=audio_data_raw, sr=sr, n_mfcc=20)
                for i, mfcc in enumerate(mfccs):
                    features[f'mfcc_{i+1}_mean'] = np.mean(mfcc)
                    features[f'mfcc_{i+1}_std'] = np.std(mfcc)
                    features[f'mfcc_{i+1}_skew'] = skew(mfcc)
                    features[f'mfcc_{i+1}_max'] = np.max(mfcc)
                    features[f'mfcc_{i+1}_min'] = np.min(mfcc)

                # Chroma
                chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
                features['chroma_mean'] = np.mean(chroma)
                features['chroma_std'] = np.std(chroma)
            else:
                # Set default values for spectral features if STFT is empty
                for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast']:
                    features[key] = 0.0
                    features[f'{key}_std'] = 0.0
                    features[f'{key}_skew'] = 0.0
                
                # Set default values for MFCCs
                for i in range(1, 21):
                    features[f'mfcc_{i}_mean'] = 0.0
                    features[f'mfcc_{i}_std'] = 0.0
                    features[f'mfcc_{i}_skew'] = 0.0
                    features[f'mfcc_{i}_max'] = 0.0
                    features[f'mfcc_{i}_min'] = 0.0
                
                features['chroma_mean'] = 0.0
                features['chroma_std'] = 0.0

            # Energy
            features['energy'] = np.sum(audio_data_raw**2) / len(audio_data_raw)
        else:
            # Set all features to 0 for empty/silent audio
            features = {
                'zero_crossing_rate': 0.0,
                'spectral_centroid': 0.0, 'spectral_centroid_std': 0.0, 'spectral_centroid_skew': 0.0,
                'spectral_bandwidth': 0.0, 'spectral_bandwidth_std': 0.0, 'spectral_bandwidth_skew': 0.0,
                'spectral_rolloff': 0.0, 'spectral_rolloff_std': 0.0, 'spectral_rolloff_skew': 0.0,
                'spectral_contrast': 0.0, 'spectral_contrast_std': 0.0, 'spectral_contrast_skew': 0.0,
                'chroma_mean': 0.0, 'chroma_std': 0.0,
                'energy': 0.0
            }
            for i in range(1, 21):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
                features[f'mfcc_{i}_skew'] = 0.0
                features[f'mfcc_{i}_max'] = 0.0
                features[f'mfcc_{i}_min'] = 0.0
        
        # Add additional features to output
        out += list(features.values())
        
        # Return metrics with file index and placeholder for truth (0)
        return [0, file_index] + out
        
    except Exception as e:
        print(f"Error processing file {file_index}: {e}")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate testing metrics for dolphin whistle detection')
    parser.add_argument('--template-audio-dir', type=str, required=True,
                       help='Directory containing template audio files')
    parser.add_argument('--template-file', type=str, required=True,
                       help='Path to template definitions CSV file')
    parser.add_argument('--test-audio-dir', type=str, required=True,
                       help='Directory containing test audio files to process')
    parser.add_argument('--output-file', type=str, default='test_metrics.csv',
                       help='Name of the output CSV file (default: test_metrics.csv)')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of processes to use (default: number of CPU cores)')
    return parser.parse_args()

def main():
    """Main function to process test audio files and generate metrics"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load templates
    print("\nLoading templates...")
    print(f"Template file: {args.template_file}")
    print(f"Template audio directory: {args.template_audio_dir}")
    tmpl = TemplateManager(args.template_file, args.template_audio_dir)
    
    # Check if we have templates
    if tmpl.size == 0:
        print("No templates found. Please check the template file and template audio directory.")
        return
    
    # Load test audio data
    print("\nLoading test audio data...")
    audio_data = AudioDataTest(args.test_audio_dir)
    
    # Create vertical bar templates
    bar_templates = create_bar_templates()
    
    # Create header using buildHeader from training script
    out_hdr = buildHeader(tmpl, MAX_FREQUENCY, MAX_TIME)
    
    # Add feature headers
    additional_headers = [
        'zero_crossing_rate',
        'spectral_centroid', 'spectral_centroid_std', 'spectral_centroid_skew',
        'spectral_bandwidth', 'spectral_bandwidth_std', 'spectral_bandwidth_skew',
        'spectral_rolloff', 'spectral_rolloff_std', 'spectral_rolloff_skew',
        'spectral_contrast', 'spectral_contrast_std', 'spectral_contrast_skew'
    ]
    
    # Add MFCC headers
    for i in range(1, 21):
        additional_headers.extend([
            f'mfcc_{i}_mean',
            f'mfcc_{i}_std',
            f'mfcc_{i}_skew',
            f'mfcc_{i}_max',
            f'mfcc_{i}_min'
        ])
    
    # Add chroma and energy headers
    additional_headers.extend([
        'chroma_mean',
        'chroma_std',
        'energy'
    ])
    
    out_hdr = out_hdr + ',' + ','.join(additional_headers)
    
    # Prepare for processing
    print("\nPreparing to process files...")
    print(f"Found {audio_data.num_files} test files")
    print(f"Using {tmpl.size} templates")
    
    # Print feature count for verification
    feature_count = len(out_hdr.split(','))
    print(f"\nGenerating {feature_count} features per file")
    
    # Prepare arguments for parallel processing
    all_args = [(i, audio_data, tmpl, bar_templates) for i in range(audio_data.num_files)]
    
    # Set up multiprocessing
    num_processes = args.num_processes or cpu_count()
    print(f"\nUsing {num_processes} processes for parallel processing")
    
    # Process files in parallel with progress bar
    print("\nProcessing files...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_test_file, all_args),
            total=len(all_args),
            desc="Processing audio files",
            unit="file"
        ))
    
    # Filter out None results (failed processing)
    h_list = [result for result in results if result is not None]
    h_list = np.array(h_list)
    
    # Print processing summary
    num_failed = len(all_args) - len(h_list)
    if num_failed > 0:
        print(f"\nWarning: {num_failed} files failed to process")
    
    # Verify feature count in results
    if len(h_list) > 0:
        features_per_file = len(h_list[0]) - 2  # Subtract truth and index columns
        if features_per_file != feature_count:
            print(f"\nWarning: Feature count mismatch!")
            print(f"Expected {feature_count} features but got {features_per_file}")
    
    # Save metrics to CSV
    print("\nSaving metrics to CSV...")
    output_file = os.path.join(output_dir, os.path.basename(args.output_file))
    with open(output_file, 'w') as file:
        file.write("Truth,Index," + out_hdr + "\n")
        np.savetxt(file, h_list, delimiter=',')
    
    # Save file mapping
    mapping_file = os.path.join(output_dir, "test_file_mapping.csv")
    with open(mapping_file, 'w') as f:
        f.write("Index,Filename\n")
        for i, fname in enumerate(audio_data.files):
            f.write(f"{i},{fname}\n")
    
    print(f"Generated {len(h_list)} samples")
    print(f"Saved metrics to {output_file}")
    print(f"Saved file mapping to {mapping_file}")
    
    # Print first few column names for verification
    print("\nFirst few feature columns:")
    columns = ["Truth", "Index"] + out_hdr.split(',')
    for i, col in enumerate(columns[:10]):
        print(f"{i}: {col}")

if __name__ == "__main__":
    main() 
