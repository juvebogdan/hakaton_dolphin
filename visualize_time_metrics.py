import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal
import wave
import cv2
from dolphin_detector.metricsDolphin import slidingWindowV, slidingWindowH, matchTemplate, highFreqTemplate, timeMetrics
from scipy.stats import skew
from dolphin_detector.config import MAX_FREQUENCY

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

def calculate_time_metrics(P, bins, maxT=MAX_FREQUENCY):
    """Calculate statistics for a range of frequency slices
    
    Args:
        P: 2-d numpy array image
        bins: time bins
        maxT: maximum frequency slice for time stats
        
    Returns:
        Dictionary of metrics by frequency slice
    """
    # Calculate metrics using numpy broadcasting
    m, n = P.shape
    maxT = min(maxT, m)
    
    # Extract slices of interest
    P_sliced = P[:maxT, :]
    
    # Calculate metrics
    # Add a small epsilon to prevent division by zero
    epsilon = 1e-10
    
    # Sum of each slice
    P_sum = np.sum(P_sliced, axis=1, keepdims=True) + epsilon
    
    # Centroid
    centroid = np.sum(P_sliced * bins, axis=1) / P_sum.squeeze()
    
    # Bandwidth (spread around centroid)
    bw = np.sqrt(np.sum(P_sliced * (bins - centroid[:, np.newaxis])**2, axis=1) / P_sum.squeeze())
    
    # Skewness
    sk = np.array([skew(P_sliced[i, :]) for i in range(maxT)])
    
    # Total variation
    tv = np.sum(np.abs(P_sliced[:, 1:] - P_sliced[:, :-1]), axis=1)
    
    return {
        'centroid': centroid,
        'bandwidth': bw,
        'skewness': sk,
        'total_variation': tv
    }

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
    
    # Filter to only whistles
    whistle_templates = template_df[template_df['file_type'] == 'whistles']
    
    # Get the first template definition
    if len(whistle_templates) == 0:
        print("Error: No whistle templates found in the template definitions file")
        return
    
    template_def = whistle_templates.iloc[0]
    print(f"Using template from file: {template_def['fname']}")
    print(f"Time range: {template_def['time_start']:.2f} - {template_def['time_end']:.2f} seconds")
    print(f"Frequency range: {template_def['freq_start']:.1f} - {template_def['freq_end']:.1f} Hz")
    
    # Load the audio file
    audio_file = os.path.join(audio_dir, template_def['fname'])
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Read the audio file
    audio, sample_rate = read_wav(audio_file)
    
    # Generate spectrogram
    params = {'NFFT': 2048, 'Fs': sample_rate, 'noverlap': 1536}
    Sxx, freqs, times = get_spectrogram(audio, sample_rate, params)
    
    # Calculate time metrics
    maxT = MAX_FREQUENCY
    time_metrics = calculate_time_metrics(Sxx, times, maxT)
    
    # Create bar templates for highFreqTemplate
    bar_, bar1_, bar2_ = create_bar_templates()
    
    # Apply highFreqTemplate matching
    hf_max = highFreqTemplate(Sxx, bar_)[0]
    hf_max1 = highFreqTemplate(Sxx, bar1_)[0]
    hf_max2 = highFreqTemplate(Sxx, bar2_)[0]
    
    print(f"High frequency template matching results:")
    print(f"  bar_: max={hf_max:.4f}")
    print(f"  bar1_: max={hf_max1:.4f}")
    print(f"  bar2_: max={hf_max2:.4f}")
    
    # Create output directory for plots
    output_dir = os.path.join(base_dir, 'dolphin_detector/plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Original spectrogram
    ax1 = fig.add_subplot(3, 2, 1)
    im1 = ax1.pcolormesh(times, freqs/1000, Sxx, shading='gouraud')
    ax1.set_ylabel('Frequency (kHz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Original Spectrogram')
    plt.colorbar(im1, ax=ax1, label='Intensity (dB)')
    
    # Highlight template region
    time_start_idx = np.argmin(np.abs(times - template_def['time_start']))
    time_end_idx = np.argmin(np.abs(times - template_def['time_end']))
    freq_start_idx = np.argmin(np.abs(freqs - template_def['freq_start']))
    freq_end_idx = np.argmin(np.abs(freqs - template_def['freq_end']))
    
    rect = plt.Rectangle(
        (template_def['time_start'], template_def['freq_start']/1000),
        template_def['time_end'] - template_def['time_start'],
        (template_def['freq_end'] - template_def['freq_start'])/1000,
        fill=False, edgecolor='r', linewidth=2
    )
    ax1.add_patch(rect)
    
    # 2. Centroid visualization
    ax2 = fig.add_subplot(3, 2, 2)
    # Create a 2D representation of centroids
    centroid_image = np.zeros_like(Sxx[:maxT])
    for i in range(maxT):
        if time_metrics['centroid'][i] < len(times):
            centroid_idx = int(np.argmin(np.abs(times - time_metrics['centroid'][i])))
            centroid_image[i, centroid_idx] = 1
    
    # Plot the spectrogram with centroids overlaid
    im2 = ax2.pcolormesh(times, freqs[:maxT]/1000, Sxx[:maxT], shading='gouraud', alpha=0.7)
    ax2.scatter(time_metrics['centroid'], freqs[:maxT]/1000, color='r', s=5, alpha=0.5)
    ax2.set_ylabel('Frequency (kHz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Centroids by Frequency')
    plt.colorbar(im2, ax=ax2, label='Intensity (dB)')
    
    # 3. Bandwidth visualization
    ax3 = fig.add_subplot(3, 2, 3)
    # Plot bandwidth as a function of frequency
    ax3.plot(freqs[:maxT]/1000, np.sqrt(time_metrics['bandwidth']), 'b-')
    ax3.set_ylabel('Bandwidth (s)')
    ax3.set_xlabel('Frequency (kHz)')
    ax3.set_title('Bandwidth by Frequency')
    ax3.grid(True)
    
    # 4. Skewness visualization
    ax4 = fig.add_subplot(3, 2, 4)
    # Plot skewness as a function of frequency
    ax4.plot(freqs[:maxT]/1000, time_metrics['skewness'], 'g-')
    ax4.set_ylabel('Skewness')
    ax4.set_xlabel('Frequency (kHz)')
    ax4.set_title('Skewness by Frequency')
    ax4.grid(True)
    
    # 5. Total variation visualization
    ax5 = fig.add_subplot(3, 2, 5)
    # Plot total variation as a function of frequency
    ax5.plot(freqs[:maxT]/1000, time_metrics['total_variation'], 'r-')
    ax5.set_ylabel('Total Variation')
    ax5.set_xlabel('Frequency (kHz)')
    ax5.set_title('Total Variation by Frequency')
    ax5.grid(True)
    
    # 6. High frequency template results
    ax6 = fig.add_subplot(3, 2, 6)
    # Create a bar chart of high frequency template results
    bar_names = ['bar_', 'bar1_', 'bar2_']
    bar_results = [hf_max, hf_max1, hf_max2]
    
    ax6.bar(bar_names, bar_results, color=['blue', 'green', 'red'])
    ax6.set_ylabel('Match Score')
    ax6.set_title('High Frequency Template Results')
    ax6.set_ylim([0, 1.0])
    
    # Add the values on top of the bars
    for i, v in enumerate(bar_results):
        ax6.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Add summary info
    summary_info = (
        f"File: {template_def['fname']}\n"
        f"Time: {template_def['time_start']:.2f} - {template_def['time_end']:.2f} s\n"
        f"Frequency: {template_def['freq_start']/1000:.1f} - {template_def['freq_end']/1000:.1f} kHz\n"
        f"Analyzed frequency slices: {maxT}"
    )
    fig.text(0.5, 0.01, summary_info, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('Time Metrics Visualization', fontsize=16)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'time_metrics_visualization.png'))
    
    print(f"Visualization saved to {os.path.join(output_dir, 'time_metrics_visualization.png')}")
    plt.show()
    
    # Create a more detailed visualization of time metrics for specific frequency ranges
    fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define frequency ranges of interest (in kHz)
    freq_ranges = [
        (5, 10),   # Lower dolphin whistle range
        (10, 15),  # Middle dolphin whistle range
        (15, 20),  # Upper dolphin whistle range
        (5, 20)    # Full dolphin whistle range
    ]
    
    # Plot time metrics for each frequency range
    for i, (low, high) in enumerate(freq_ranges):
        # Convert kHz to Hz
        low_hz, high_hz = low * 1000, high * 1000
        
        # Find indices for this frequency range
        low_idx = np.argmin(np.abs(freqs - low_hz))
        high_idx = np.argmin(np.abs(freqs - high_hz))
        
        # Ensure indices are within bounds
        low_idx = max(0, min(low_idx, maxT-1))
        high_idx = max(0, min(high_idx, maxT-1))
        
        # Calculate average metrics for this range
        avg_centroid = np.mean(time_metrics['centroid'][low_idx:high_idx+1])
        avg_bandwidth = np.mean(np.sqrt(time_metrics['bandwidth'][low_idx:high_idx+1]))
        avg_skewness = np.mean(time_metrics['skewness'][low_idx:high_idx+1])
        avg_total_var = np.mean(time_metrics['total_variation'][low_idx:high_idx+1])
        
        # Plot on appropriate subplot
        ax = axes[i//2, i%2]
        
        # Create bar chart of average metrics
        metric_names = ['Centroid (s)', 'Bandwidth (s)', 'Skewness', 'Total Var']
        # Normalize values for better visualization
        metric_values = [
            avg_centroid,
            avg_bandwidth,
            avg_skewness / 10 if abs(avg_skewness) > 10 else avg_skewness,  # Scale skewness if large
            avg_total_var / 100  # Scale total variation
        ]
        
        ax.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'purple'])
        ax.set_title(f'Frequency Range: {low}-{high} kHz')
        ax.set_ylabel('Value')
        
        # Add the values on top of the bars
        for j, v in enumerate(metric_values):
            if j == 2:  # Skewness
                original = avg_skewness
                ax.text(j, v + 0.02, f"{original:.2f}", ha='center')
            elif j == 3:  # Total variation
                original = avg_total_var
                ax.text(j, v + 0.02, f"{original:.2f}", ha='center')
            else:
                ax.text(j, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('Time Metrics by Frequency Range', fontsize=16)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'time_metrics_by_range.png'))
    
    print(f"Frequency range visualization saved to {os.path.join(output_dir, 'time_metrics_by_range.png')}")
    plt.show()

if __name__ == "__main__":
    main() 
