import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd
import wave
from scipy import signal
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import librosa
from scipy.stats import skew
from metricsDolphin import computeMetrics, highFreqTemplate, slidingWindowV, buildHeader
from fileio import AudioData  # Import AudioData from fileio
from config import MAX_FREQUENCY, MAX_TIME  # Import from config module
import warnings

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Suppress librosa warnings
warnings.filterwarnings('ignore', module='librosa')

class TemplateManager:
    """Template Manager for dolphin whistle detection
    
    Loads and manages templates for matching against spectrograms
    """
    def __init__(self, template_file, audio_dir):
        self.template_file = template_file
        self.audio_dir = audio_dir
        self.templates = []
        self.info = []
        self.size = 0
        
        # Load templates if file exists
        if os.path.exists(template_file):
            self.load_templates()
    
    def load_templates(self):
        """Load templates from CSV file"""
        # Read template definitions
        template_df = pd.read_csv(self.template_file)
        
        # Clear existing templates
        self.templates = []
        self.info = []
        
        # Load each template
        for _, row in template_df.iterrows():
            # Store template info
            template_info = {
                'file': row['fname'],
                'file_type': row['file_type'],
                'time_start': row['time_start'],
                'time_end': row['time_end'],
                'freq_start': row['freq_start'],
                'freq_end': row['freq_end']
            }
            
            # Extract template from audio file
            audio_file = os.path.join(self.audio_dir, row['fname'])
            if os.path.exists(audio_file):
                # Read audio file
                with wave.open(audio_file, 'rb') as wav:
                    sample_rate = wav.getframerate()
                    frames = wav.getnframes()
                    audio_bytes = wav.readframes(frames)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Generate spectrogram
                params = {'NFFT': 2048, 'Fs': sample_rate, 'noverlap': 1536}
                freqs, times, Sxx = signal.spectrogram(audio,
                                                     fs=params['Fs'],
                                                     nperseg=params['NFFT'],
                                                     noverlap=params['noverlap'],
                                                     scaling='density')
                
                Sxx = 10 * np.log10(Sxx + 1e-10)
                
                # Find indices for template extraction
                time_start_idx = np.argmin(np.abs(times - row['time_start']))
                time_end_idx = np.argmin(np.abs(times - row['time_end']))
                freq_start_idx = np.argmin(np.abs(freqs - row['freq_start']))
                freq_end_idx = np.argmin(np.abs(freqs - row['freq_end']))
                
                # Apply sliding window normalization
                Sxx_norm = slidingWindowV(Sxx)
                
                # Extract template region
                template = Sxx_norm[freq_start_idx:freq_end_idx, time_start_idx:time_end_idx]
                
                # Convert to binary mask using mean thresholding
                mean = np.mean(template)
                std = np.std(template)
                min_val = template.min()
                template[template < mean + 0.5*std] = min_val
                template[template > min_val] = 1
                template[template < 0] = 0
                
                self.templates.append(template.astype('float32'))
                self.info.append(template_info)
        
        # Update size
        self.size = len(self.info)
        print(f"Loaded {self.size} templates")

def create_bar_templates():
    """Create the vertical bar templates used in highFreqTemplate"""
    bar_ = np.zeros((24, 18), dtype='float32')
    bar1_ = np.zeros((24, 24), dtype='float32')
    bar2_ = np.zeros((24, 12), dtype='float32')
    
    bar_[:, 6:12] = 1.
    bar1_[:, 8:16] = 1.
    bar2_[:, 4:8] = 1.
    
    return bar_, bar1_, bar2_

def process_file(args):
    """Process a single audio file and compute metrics
    
    Args:
        args: Tuple containing (file_index, is_whistle, audio_data, templates, bar_templates)
        
    Returns:
        List of metrics for the file
    """
    file_index, is_whistle, audio_data, templates, bar_templates = args
    try:
        # Get spectrogram based on file type
        if is_whistle:
            P, freqs, bins = audio_data.get_whistle_sample(file_index)
            audio_data_raw = audio_data.get_whistle_raw(file_index)
            sr = audio_data.sample_rate
        else:
            P, freqs, bins = audio_data.get_noise_sample(file_index)
            audio_data_raw = audio_data.get_noise_raw(file_index)
            sr = audio_data.sample_rate
        
        # Initialize features dictionary
        features = {}
        
        # Check if audio is empty or silent
        if len(audio_data_raw) == 0 or np.all(np.abs(audio_data_raw) < 1e-6):
            
            features['zero_crossing_rate'] = 0.0
            features['spectral_centroid'] = 0.0
            features['spectral_centroid_std'] = 0.0
            features['spectral_centroid_skew'] = 0.0
            features['spectral_bandwidth'] = 0.0
            features['spectral_bandwidth_std'] = 0.0
            features['spectral_bandwidth_skew'] = 0.0
            features['spectral_rolloff'] = 0.0
            features['spectral_rolloff_std'] = 0.0
            features['spectral_rolloff_skew'] = 0.0
            features['spectral_contrast'] = 0.0
            features['spectral_contrast_std'] = 0.0
            features['spectral_contrast_skew'] = 0.0
            features['chroma_mean'] = 0.0
            features['chroma_std'] = 0.0
            features['energy'] = 0.0
            
           
            for i in range(1, 21):
                features[f'mfcc_{i}_mean'] = 0.0
                features[f'mfcc_{i}_std'] = 0.0
                features[f'mfcc_{i}_skew'] = 0.0
                features[f'mfcc_{i}_max'] = 0.0
                features[f'mfcc_{i}_min'] = 0.0
        else:
            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data_raw))) > 0)
            features['zero_crossing_rate'] = zero_crossings / len(audio_data_raw)

            # Spectral features
            stft = np.abs(librosa.stft(audio_data_raw))
            
            if np.any(stft):  
                # Spectral centroid
                centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
                features['spectral_centroid'] = np.mean(centroid) if len(centroid) > 0 else 0.0
                features['spectral_centroid_std'] = np.std(centroid) if len(centroid) > 0 else 0.0
                features['spectral_centroid_skew'] = skew(centroid) if len(centroid) > 0 else 0.0

                # Spectral bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
                features['spectral_bandwidth'] = np.mean(bandwidth) if len(bandwidth) > 0 else 0.0
                features['spectral_bandwidth_std'] = np.std(bandwidth) if len(bandwidth) > 0 else 0.0
                features['spectral_bandwidth_skew'] = skew(bandwidth) if len(bandwidth) > 0 else 0.0

                # Spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
                features['spectral_rolloff'] = np.mean(rolloff) if len(rolloff) > 0 else 0.0
                features['spectral_rolloff_std'] = np.std(rolloff) if len(rolloff) > 0 else 0.0
                features['spectral_rolloff_skew'] = skew(rolloff) if len(rolloff) > 0 else 0.0

                # Spectral contrast
                contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)[0]
                features['spectral_contrast'] = np.mean(contrast) if len(contrast) > 0 else 0.0
                features['spectral_contrast_std'] = np.std(contrast) if len(contrast) > 0 else 0.0
                features['spectral_contrast_skew'] = skew(contrast) if len(contrast) > 0 else 0.0

                # MFCCs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mfccs = librosa.feature.mfcc(y=audio_data_raw, sr=sr, n_mfcc=20)
                for i, mfcc in enumerate(mfccs):
                    features[f'mfcc_{i+1}_mean'] = np.mean(mfcc)
                    features[f'mfcc_{i+1}_std'] = np.std(mfcc)
                    features[f'mfcc_{i+1}_skew'] = skew(mfcc)
                    features[f'mfcc_{i+1}_max'] = np.max(mfcc)
                    features[f'mfcc_{i+1}_min'] = np.min(mfcc)

                # Chroma
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
                        if np.any(chroma):
                            features['chroma_mean'] = np.mean(chroma)
                            features['chroma_std'] = np.std(chroma)
                        else:
                            features['chroma_mean'] = 0.0
                            features['chroma_std'] = 0.0
                    except:
                        features['chroma_mean'] = 0.0
                        features['chroma_std'] = 0.0

            # Energy
            features['energy'] = np.sum(audio_data_raw**2) / len(audio_data_raw)
        
        # Compute original metrics
        out = computeMetrics(P, templates, bins, MAX_FREQUENCY, MAX_TIME)
        
        # Add high frequency template metrics
        bar_, bar1_, bar2_ = bar_templates
        out += highFreqTemplate(P, bar_)
        out += highFreqTemplate(P, bar1_)
        out += highFreqTemplate(P, bar2_)
        
        # Add new features to output
        out += list(features.values())
        
        # Return metrics with label and index
        return [1 if is_whistle else 0, file_index] + out
        
    except Exception as e:
        print(f"Error processing {'whistle' if is_whistle else 'noise'} file {file_index}: {e}")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate training metrics for dolphin whistle detection')
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced dataset')
    parser.add_argument('--test-size', type=int, default=5, help='Number of files to process in test mode (default: 5)')
    parser.add_argument('--num-processes', type=int, default=None, 
                       help='Number of processes to use (default: number of CPU cores)')
    parser.add_argument('--audio-dir', type=str, default=None,
                       help='Path to the directory containing audio files')
    parser.add_argument('--csv-path', type=str, default=None,
                       help='Path to the CSV file with labels')
    parser.add_argument('--template-audio-dir', type=str, default=None,
                       help='Path to the directory containing template audio files')
    parser.add_argument('--output-file', type=str, default='dolphin_train_metrics.csv',
                       help='Name of the output CSV file (default: dolphin_train_metrics.csv)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Base directories - use current directory instead of parent
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(base_dir, 'output'), exist_ok=True)
    
    # Set up paths using command line arguments or defaults
    audio_dir = args.audio_dir or os.path.join(base_dir, 'hakaton', 'audio_train')
    csv_path = args.csv_path or os.path.join(base_dir, 'hakaton', 'train.csv')
    template_audio_dir = args.template_audio_dir or os.path.join(base_dir, 'templates', 'audio')
    
    # Print the paths being used
    print("\nUsing the following paths:")
    print(f"Audio directory: {audio_dir}")
    print(f"CSV path: {csv_path}")
    print(f"Template audio directory: {template_audio_dir}")
    
    # Load audio data
    print("Loading audio data...")
    audio_data = AudioData(csv_path, audio_dir)
    
    # Load templates
    print("Loading templates...")
    # Update template file path to use current directory
    template_file = os.path.join(base_dir, 'templates', 'template_definitions.csv')
    tmpl = TemplateManager(template_file, template_audio_dir)
    
    # Check if we have templates
    if tmpl.size == 0:
        print("No templates found. Please check the template file.")
        return
    
    # Create vertical bar templates
    bar_templates = create_bar_templates()
    
    # Create header
    out_hdr = buildHeader(tmpl, MAX_FREQUENCY, MAX_TIME)
    
    # Add new feature headers
    additional_headers = [
        'zero_crossing_rate',
        'spectral_centroid', 'spectral_centroid_std', 'spectral_centroid_skew',
        'spectral_bandwidth', 'spectral_bandwidth_std', 'spectral_bandwidth_skew',
        'spectral_rolloff', 'spectral_rolloff_std', 'spectral_rolloff_skew',
        'spectral_contrast', 'spectral_contrast_std', 'spectral_contrast_skew'
    ]
    
    # Add MFCC headers
    for i in range(1, 21):  # 20 MFCCs
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
    
    # Determine number of files to process based on test mode
    test_mode = args.test
    test_size = args.test_size
    
    # If in test mode, limit the number of files
    num_whistles = min(test_size, audio_data.num_whistles) if test_mode else audio_data.num_whistles
    num_noise = min(test_size, audio_data.num_noise) if test_mode else audio_data.num_noise
    
    if test_mode:
        print(f"Running in TEST mode with {num_whistles} whistle files and {num_noise} noise files")
    
    print("\nPreparing to process files...")
    print(f"- Whistle files: {num_whistles}")
    print(f"- Noise files: {num_noise}")
    
    # Prepare arguments for parallel processing
    whistle_args = [(i, True, audio_data, tmpl, bar_templates) for i in range(num_whistles)]
    noise_args = [(i, False, audio_data, tmpl, bar_templates) for i in range(num_noise)]
    all_args = whistle_args + noise_args
    
    # Set up multiprocessing
    num_processes = args.num_processes or cpu_count()
    print(f"Using {num_processes} processes for parallel processing")
    
    # Process files in parallel with progress bar
    print("\nProcessing files...")
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file, all_args),
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
    
    # Save metrics to CSV
    print("\nSaving metrics to CSV...")
    output_file = os.path.join(base_dir, 'output', args.output_file)
    with open(output_file, 'w') as file:
        file.write("Truth,Index," + out_hdr + "\n")
        np.savetxt(file, h_list, delimiter=',')
    
    print(f"Generated {len(h_list)} samples")
    print(f"Saved metrics to {output_file}")
    
    # Visualize feature distributions
    print("\nGenerating feature distribution visualization...")
    visualize_feature_distributions(h_list, tmpl, base_dir=base_dir)

def visualize_feature_distributions(h_list, tmpl, test_mode=False, base_dir=None):
    """
    Create a visualization of the feature distributions.
    
    Args:
        h_list: Array of metrics with class labels
        tmpl: Template manager object
        test_mode: Whether running in test mode
        base_dir: Base directory for saving the visualization
    """
    # Optional: Create a visualization of the feature distribution
    plt.figure(figsize=(12, 6))
    
    # Calculate mean feature values for each class
    whistle_features = h_list[h_list[:, 0] == 1, 2:]
    noise_features = h_list[h_list[:, 0] == 0, 2:]
    
    whistle_mean = np.mean(whistle_features, axis=0)
    noise_mean = np.mean(noise_features, axis=0)
    
    # Plot the first 50 features (template matching results)
    num_templates = tmpl.size
    feature_indices = np.arange(num_templates * 3 * 2)  # 3 values per template (max, xLoc, yLoc) * 2 enhancement methods
    
    plt.subplot(1, 2, 1)
    plt.bar(feature_indices - 0.2, 
            whistle_mean[feature_indices], 
            width=0.4, 
            label='Whistles', 
            color='blue', 
            alpha=0.7)
    plt.bar(feature_indices + 0.2, 
            noise_mean[feature_indices], 
            width=0.4, 
            label='Noise', 
            color='red', 
            alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.title('Template Matching Features')
    plt.legend()
    
    # Plot high frequency template features
    hf_indices = np.array([-3, -2, -1])
    
    plt.subplot(1, 2, 2)
    plt.bar(['bar_', 'bar1_', 'bar2_'], 
            whistle_mean[hf_indices], 
            width=0.35, 
            label='Whistles', 
            color='blue', 
            alpha=0.7)
    plt.bar(['bar_', 'bar1_', 'bar2_'], 
            noise_mean[hf_indices], 
            width=0.35, 
            label='Noise', 
            color='red', 
            alpha=0.7, 
            align='edge')
    plt.xlabel('High Frequency Template')
    plt.ylabel('Mean Value')
    plt.title('High Frequency Template Features')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the visualization
    vis_file = os.path.join(base_dir, 'output', 'feature_distribution.png')
    plt.savefig(vis_file)
    print(f"Feature distribution visualization saved to {vis_file}")

if __name__ == "__main__":
    main() 
