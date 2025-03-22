"""
generate_training_metrics.py

This file generates the training metrics for dolphin whistle detection
based on the original whale detection approach but adapted for higher frequency dolphin sounds.
"""

import numpy as np
import os
import pandas as pd
import wave
from scipy import signal
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from dolphin_detector.metricsDolphin import computeMetrics, highFreqTemplate, slidingWindowV
from dolphin_detector.fileio import AudioData  # Import AudioData from fileio
from dolphin_detector.config import MAX_FREQUENCY  # Import from config module

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
    # Scaled up for dolphin audio (2x size of original whale templates)
    bar_ = np.zeros((24, 18), dtype='float32')
    bar1_ = np.zeros((24, 24), dtype='float32')
    bar2_ = np.zeros((24, 12), dtype='float32')
    
    # Keep the same pattern but scaled up
    bar_[:, 6:12] = 1.  # Center vertical bar (was 3:6 in original)
    bar1_[:, 8:16] = 1.  # Center vertical bar (was 4:8 in original)
    bar2_[:, 4:8] = 1.  # Center vertical bar (was 2:4 in original)
    
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
        else:
            P, freqs, bins = audio_data.get_noise_sample(file_index)
        
        # Compute metrics
        out = computeMetrics(P, templates, bins, MAX_FREQUENCY)
        
        # Add high frequency template metrics
        bar_, bar1_, bar2_ = bar_templates
        out += highFreqTemplate(P, bar_)
        out += highFreqTemplate(P, bar1_)
        out += highFreqTemplate(P, bar2_)
        
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
    return parser.parse_args()

def main():
    """
    Main function to process audio files and generate training metrics.
    """
    # Parse command line arguments
    args = parse_args()

    # Base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)
    
    # Parameters
    data_dir = os.path.join(base_dir, 'hakaton')  # Data directory
    audio_dir = os.path.join(data_dir, 'audio_train')  # Audio files directory
    csv_path = os.path.join(data_dir, 'train.csv')  # CSV file with labels
    
    # Load audio data
    print("Loading audio data...")
    audio_data = AudioData(csv_path, audio_dir)
    
    # Load templates
    print("Loading templates...")
    template_file = os.path.join(base_dir, 'dolphin_detector/templates/template_definitions.csv')
    tmpl = TemplateManager(template_file, audio_dir)
    
    # Check if we have templates
    if tmpl.size == 0:
        print("No templates found. Please check the template file.")
        return
    
    # Create vertical bar templates
    bar_templates = create_bar_templates()
    
    # Create header
    from dolphin_detector.metricsDolphin import buildHeader
    out_hdr = buildHeader(tmpl, MAX_FREQUENCY)
    
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
    output_file = os.path.join(os.path.dirname(__file__), 'output', 'dolphin_train_metrics.csv')
    with open(output_file, 'w') as file:
        file.write("Truth,Index," + out_hdr + "\n")
        np.savetxt(file, h_list, delimiter=',')
    
    print(f"Generated {len(h_list)} samples")
    print(f"Saved metrics to {output_file}")
    
    # Visualize feature distributions
    print("\nGenerating feature distribution visualization...")
    visualize_feature_distributions(h_list, tmpl)

def visualize_feature_distributions(h_list, tmpl, test_mode=False):
    """
    Create a visualization of the feature distributions.
    
    Args:
        h_list: Array of metrics with class labels
        tmpl: Template manager object
        test_mode: Whether running in test mode
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
    vis_file = os.path.join(os.path.dirname(__file__), 'output', 'feature_distribution.png')
    plt.savefig(vis_file)
    print(f"Feature distribution visualization saved to {vis_file}")

if __name__ == "__main__":
    main() 
