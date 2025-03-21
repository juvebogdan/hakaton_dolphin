import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal
import wave
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow both package and script usage
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dolphin_detector.metricsDolphin import slidingWindowV

def read_wav(filepath):
    """Read WAV file and return signal"""
    with wave.open(filepath, 'rb') as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        audio_bytes = wav.readframes(frames)
        return np.frombuffer(audio_bytes, dtype=np.int16), sample_rate

def get_spectrogram(audio, sample_rate, params=None):
    """Compute spectrogram with parameters suited for dolphin whistles"""
    if params is None:
        params = {
            'NFFT': 2048,
            'Fs': sample_rate,
            'noverlap': 1536
        }
        
    freqs, times, Sxx = signal.spectrogram(audio,
                                         fs=params['Fs'],
                                         nperseg=params['NFFT'],
                                         noverlap=params['noverlap'],
                                         scaling='density')
    
    Sxx = 10 * np.log10(Sxx + 1e-10)
    
    return Sxx, freqs, times

def main():
    # Base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    template_file = os.path.join(base_dir, 'dolphin_detector/templates/template_definitions.csv')
    audio_dir = os.path.join(base_dir, 'hakaton/audio_train')
    
    # Check if files exist
    if not os.path.exists(template_file):
        print(f"Error: Template file not found: {template_file}")
        return
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        return
    
    # Load template definitions
    template_df = pd.read_csv(template_file)
    
    # Get the first template definition
    if len(template_df) == 0:
        print("Error: No templates found in the template definitions file")
        return
    
    template = template_df.iloc[0]
    print(f"Visualizing template from file: {template['fname']}")
    print(f"Time range: {template['time_start']} - {template['time_end']} seconds")
    print(f"Frequency range: {template['freq_start']} - {template['freq_end']} Hz")
    
    # Load the audio file
    audio_file = os.path.join(audio_dir, template['fname'])
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Read the audio file
    audio, sample_rate = read_wav(audio_file)
    
    # Generate spectrogram
    params = {'NFFT': 2048, 'Fs': sample_rate, 'noverlap': 1536}
    Sxx, freqs, times = get_spectrogram(audio, sample_rate, params)
    
    # Find indices for template extraction
    time_start_idx = np.argmin(np.abs(times - template['time_start']))
    time_end_idx = np.argmin(np.abs(times - template['time_end']))
    freq_start_idx = np.argmin(np.abs(freqs - template['freq_start']))
    freq_end_idx = np.argmin(np.abs(freqs - template['freq_end']))
    
    # Apply sliding window normalization
    Sxx_enhanced = slidingWindowV(Sxx)
    
    # Extract template region
    template_region = Sxx_enhanced[freq_start_idx:freq_end_idx, time_start_idx:time_end_idx]
    
    # Create binary template
    template_binary = template_region.copy()
    mean = np.mean(template_binary)
    std = np.std(template_binary)
    min_val = template_binary.min()
    template_binary[template_binary < mean + 0.5*std] = min_val
    template_binary[template_binary > min_val] = 1
    template_binary[template_binary < 0] = 0
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Full spectrogram
    im1 = axes[0, 0].pcolormesh(times, freqs/1000, Sxx, shading='gouraud')
    axes[0, 0].set_ylabel('Frequency (kHz)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_title('Original Spectrogram')
    plt.colorbar(im1, ax=axes[0, 0], label='Intensity (dB)')
    
    # Highlight template region
    axes[0, 0].add_patch(plt.Rectangle(
        (template['time_start'], template['freq_start']/1000),
        template['time_end'] - template['time_start'],
        (template['freq_end'] - template['freq_start'])/1000,
        fill=False, edgecolor='r', linewidth=2
    ))
    
    # Plot 2: Enhanced spectrogram
    im2 = axes[0, 1].pcolormesh(times, freqs/1000, Sxx_enhanced, shading='gouraud')
    axes[0, 1].set_ylabel('Frequency (kHz)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_title('Enhanced Spectrogram')
    plt.colorbar(im2, ax=axes[0, 1], label='Intensity (dB)')
    
    # Highlight template region
    axes[0, 1].add_patch(plt.Rectangle(
        (template['time_start'], template['freq_start']/1000),
        template['time_end'] - template['time_start'],
        (template['freq_end'] - template['freq_start'])/1000,
        fill=False, edgecolor='r', linewidth=2
    ))
    
    # Plot 3: Extracted template region
    template_times = times[time_start_idx:time_end_idx]
    template_freqs = freqs[freq_start_idx:freq_end_idx]
    im3 = axes[1, 0].pcolormesh(
        template_times, 
        template_freqs/1000, 
        template_region, 
        shading='gouraud'
    )
    axes[1, 0].set_ylabel('Frequency (kHz)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Extracted Template Region')
    plt.colorbar(im3, ax=axes[1, 0], label='Intensity (dB)')
    
    # Plot 4: Binary template
    im4 = axes[1, 1].pcolormesh(
        template_times, 
        template_freqs/1000, 
        template_binary, 
        shading='gouraud', 
        cmap='binary'
    )
    axes[1, 1].set_ylabel('Frequency (kHz)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Binary Template')
    plt.colorbar(im4, ax=axes[1, 1], label='Value')
    
    # Add template info as text
    template_info = (
        f"File: {template['fname']}\n"
        f"Type: {template['file_type']}\n"
        f"Time: {template['time_start']:.2f} - {template['time_end']:.2f} s\n"
        f"Freq: {template['freq_start']/1000:.1f} - {template['freq_end']/1000:.1f} kHz\n"
        f"Template shape: {template_region.shape}"
    )
    fig.text(0.5, 0.01, template_info, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('Dolphin Whistle Template Visualization', fontsize=16)
    
    # Save figure
    output_dir = os.path.join(base_dir, 'dolphin_detector/plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'template_visualization.png'))
    
    print(f"Visualization saved to {os.path.join(output_dir, 'template_visualization.png')}")
    plt.show()

if __name__ == "__main__":
    main() 
