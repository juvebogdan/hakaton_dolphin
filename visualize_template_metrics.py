import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wave
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter
import sys
from pathlib import Path
import argparse

# Add the parent directory to sys.path to allow both package and script usage
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from metricsDolphin import slidingWindowV, slidingWindowH, matchTemplate, highFreqTemplate
from config import MAX_FREQUENCY

def read_wav(filepath):
    """Read WAV file and return signal"""
    with wave.open(filepath, 'rb') as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        audio_bytes = wav.readframes(frames)
        return np.frombuffer(audio_bytes, dtype=np.int16), sample_rate

def apply_gaussian_filter(spectrogram, sigma=(1, 1)):
    """Apply 2D Gaussian filter to spectrogram
    
    Args:
        spectrogram: 2D numpy array of spectrogram values
        sigma: Tuple of (sigma_freq, sigma_time) for Gaussian kernel size
               Higher values = more smoothing
    
    Returns:
        Filtered spectrogram
    """
    return gaussian_filter(spectrogram, sigma=sigma)

def get_spectrogram(audio, sample_rate, params=None, apply_gaussian=True, gaussian_sigma=(1, 1)):
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
    
    if apply_gaussian:
        Sxx = apply_gaussian_filter(Sxx, gaussian_sigma)
    
    return Sxx, freqs, times

class SimpleTemplateManager:
    """Simple template manager for visualization purposes"""
    def __init__(self):
        self.templates = []
        self.info = []
        self.size = 0
    
    def add_template(self, template, info):
        self.templates.append(template)
        self.info.append(info)
        self.size += 1

def templateMetrics(P, tmpl):
    """Template matching function from metricsDolphin.py"""
    maxs, xs, ys = [], [], []
    for k in range(tmpl.size):
        mf, y, x = matchTemplate(P, tmpl.templates[k])
        maxs.append(mf)
        xs.append(x)
        ys.append(y)
    return maxs, xs, ys

def create_bar_templates():
    """Create the vertical bar templates used in highFreqTemplate"""
    # Scaled up for dolphin audio (2x size of original whale templates)
    bar_ = np.zeros((24, 18), dtype='float32')
    bar1_ = np.zeros((24, 24), dtype='float32')
    bar2_ = np.zeros((24, 12), dtype='float32')
    
    # Keep the same pattern but scaled up
    bar_[:, 3:6] = 1.  # Center vertical bar (was 3:6 in original)
    bar1_[:, 4:8] = 1.  # Center vertical bar (was 4:8 in original)
    bar2_[:, 2:4] = 1.  # Center vertical bar (was 2:4 in original)
    
    return bar_, bar1_, bar2_

def visualize_template_matching(template_def, test_file, audio_dir, output_dir, **spectrogram_params):
    """Create and save visualization for a single template-test file pair"""
    
    # Load the template audio file
    template_audio_file = os.path.join(audio_dir, template_def['fname'])
    if not os.path.exists(template_audio_file):
        print(f"Error: Template audio file not found: {template_audio_file}")
        return
    
    # Read the template audio file
    template_audio, sample_rate = read_wav(template_audio_file)
    
    # Generate spectrogram for template
    params = {'NFFT': 2048, 'Fs': sample_rate, 'noverlap': 1536}
    template_Sxx, template_freqs, template_times = get_spectrogram(template_audio, sample_rate, params, **spectrogram_params)
    
    # Find indices for template extraction
    time_start_idx = np.argmin(np.abs(template_times - template_def['time_start']))
    time_end_idx = np.argmin(np.abs(template_times - template_def['time_end']))
    freq_start_idx = np.argmin(np.abs(template_freqs - template_def['freq_start']))
    freq_end_idx = np.argmin(np.abs(template_freqs - template_def['freq_end']))
    
    # Apply sliding window normalization
    template_Sxx_enhanced = slidingWindowV(template_Sxx)
    
    # Extract template region
    template_region = template_Sxx_enhanced[freq_start_idx:freq_end_idx, time_start_idx:time_end_idx]
    
    # Create binary template
    template_binary = template_region.copy()
    mean = np.mean(template_binary)
    std = np.std(template_binary)
    min_val = template_binary.min()
    template_binary[template_binary < mean + 0.5*std] = min_val
    template_binary[template_binary > min_val] = 1
    template_binary[template_binary < 0] = 0
    
    # Create a simple template manager
    tmpl = SimpleTemplateManager()
    tmpl.add_template(template_binary.astype('float32'), {
        'file': template_def['fname'],
        'time_start': template_def['time_start'],
        'time_end': template_def['time_end'],
        'freq_start': template_def['freq_start'],
        'freq_end': template_def['freq_end']
    })
    
    # Load the test audio file
    test_audio_file = os.path.join(audio_dir, test_file)
    if not os.path.exists(test_audio_file):
        print(f"Error: Test audio file not found: {test_audio_file}")
        return
    
    # Read the test audio file
    test_audio, test_sample_rate = read_wav(test_audio_file)
    
    # Generate spectrogram for test file
    test_Sxx, test_freqs, test_times = get_spectrogram(test_audio, test_sample_rate, params)
    
    # Apply sliding window normalization (vertical and horizontal)
    test_Sxx_V = slidingWindowV(test_Sxx)
    test_Sxx_H = slidingWindowH(test_Sxx)
    
    # Perform template matching
    maxs_V, xs_V, ys_V = templateMetrics(test_Sxx_V, tmpl)
    maxs_H, xs_H, ys_H = templateMetrics(test_Sxx_H, tmpl)
    
    print(f"Template {template_def['fname']} vs {test_file}:")
    print(f"  Vertical match: max={maxs_V[0]:.4f}, x={xs_V[0]}, y={ys_V[0]}")
    print(f"  Horizontal match: max={maxs_H[0]:.4f}, x={xs_H[0]}, y={ys_H[0]}")
    
    # Create bar templates for highFreqTemplate
    bar_, bar1_, bar2_ = create_bar_templates()
    
    # Apply highFreqTemplate matching
    hf_max = highFreqTemplate(test_Sxx, bar_)[0]
    hf_max1 = highFreqTemplate(test_Sxx, bar1_)[0]
    hf_max2 = highFreqTemplate(test_Sxx, bar2_)[0]
    
    print(f"  High frequency matches: bar_={hf_max:.4f}, bar1_={hf_max1:.4f}, bar2_={hf_max2:.4f}")
    
    # Create visualization
    fig = plt.figure(figsize=(15, 18))
    plt.suptitle(f'Template Matching: {template_def["fname"]} vs {test_file}', fontsize=16)
    
    # 1. Original template
    ax1 = fig.add_subplot(4, 2, 1)
    im1 = ax1.pcolormesh(
        template_times[time_start_idx:time_end_idx], 
        template_freqs[freq_start_idx:freq_end_idx]/1000, 
        template_region, 
        shading='gouraud'
    )
    ax1.set_ylabel('Frequency (kHz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Original Template Region')
    plt.colorbar(im1, ax=ax1, label='Intensity (dB)')
    
    # 2. Binary template
    ax2 = fig.add_subplot(4, 2, 2)
    im2 = ax2.pcolormesh(
        template_times[time_start_idx:time_end_idx], 
        template_freqs[freq_start_idx:freq_end_idx]/1000, 
        template_binary, 
        shading='gouraud', 
        cmap='binary'
    )
    ax2.set_ylabel('Frequency (kHz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Binary Template')
    plt.colorbar(im2, ax=ax2, label='Value')
    
    # 3. Test spectrogram with vertical enhancement
    ax3 = fig.add_subplot(4, 2, 3)
    im3 = ax3.pcolormesh(test_times, test_freqs/1000, test_Sxx_V, shading='gouraud')
    ax3.set_ylabel('Frequency (kHz)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Test Spectrogram (Vertical Enhancement)')
    plt.colorbar(im3, ax=ax3, label='Intensity (dB)')
    
    # Mark the best match location
    template_height = template_binary.shape[0]
    template_width = template_binary.shape[1]
    
    # Vertical match rectangle - with bounds checking
    if xs_V[0] < len(test_times) and ys_V[0] < len(test_freqs):
        rect_v = plt.Rectangle(
            (test_times[xs_V[0]], test_freqs[ys_V[0]]/1000),
            test_times[min(xs_V[0] + template_width, len(test_times)-1)] - test_times[xs_V[0]],
            test_freqs[min(ys_V[0] + template_height, len(test_freqs)-1)]/1000 - test_freqs[ys_V[0]]/1000,
            fill=False, edgecolor='r', linewidth=2
        )
        ax3.add_patch(rect_v)
        ax3.text(test_times[xs_V[0]], test_freqs[ys_V[0]]/1000, f"Match: {maxs_V[0]:.2f}", 
                color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    else:
        print(f"Warning: Vertical match position ({xs_V[0]}, {ys_V[0]}) out of bounds")
    
    # 4. Test spectrogram with horizontal enhancement
    ax4 = fig.add_subplot(4, 2, 4)
    im4 = ax4.pcolormesh(test_times, test_freqs/1000, test_Sxx_H, shading='gouraud')
    ax4.set_ylabel('Frequency (kHz)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Test Spectrogram (Horizontal Enhancement)')
    plt.colorbar(im4, ax=ax4, label='Intensity (dB)')
    
    # Mark the best match location
    # Horizontal match rectangle - with bounds checking
    if xs_H[0] < len(test_times) and ys_H[0] < len(test_freqs):
        rect_h = plt.Rectangle(
            (test_times[xs_H[0]], test_freqs[ys_H[0]]/1000),
            test_times[min(xs_H[0] + template_width, len(test_times)-1)] - test_times[xs_H[0]],
            test_freqs[min(ys_H[0] + template_height, len(test_freqs)-1)]/1000 - test_freqs[ys_H[0]]/1000,
            fill=False, edgecolor='r', linewidth=2
        )
        ax4.add_patch(rect_h)
        ax4.text(test_times[xs_H[0]], test_freqs[ys_H[0]]/1000, f"Match: {maxs_H[0]:.2f}", 
                color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    else:
        print(f"Warning: Horizontal match position ({xs_H[0]}, {ys_H[0]}) out of bounds")
    
    # 5. High frequency region for bar template matching
    # Get the high frequency region used in highFreqTemplate
    test_Sxx_HF = slidingWindowH(test_Sxx, inner=7, maxM=MAX_FREQUENCY, norm=True)[200:,:]
    
    ax5 = fig.add_subplot(4, 2, 5)
    im5 = ax5.pcolormesh(
        test_times, 
        test_freqs[200:]/1000, 
        test_Sxx_HF, 
        shading='gouraud'
    )
    ax5.set_ylabel('Frequency (kHz)')
    ax5.set_xlabel('Time (s)')
    ax5.set_title('High Frequency Region for Template Matching')
    plt.colorbar(im5, ax=ax5, label='Intensity (dB)')
    
    # 6. Bar templates
    ax6 = fig.add_subplot(4, 2, 6)
    # Create a figure with all three bar templates side by side
    # Calculate total width needed
    total_width = bar_.shape[1] + bar1_.shape[1] + bar2_.shape[1] + 4  # 4 for spacing
    bar_combined = np.zeros((24, total_width), dtype='float32')
    
    # Place each template with spacing
    bar_combined[:, 0:bar_.shape[1]] = bar_
    bar_combined[:, bar_.shape[1]+2:bar_.shape[1]+2+bar1_.shape[1]] = bar1_
    bar_combined[:, bar_.shape[1]+2+bar1_.shape[1]+2:] = bar2_
    
    im6 = ax6.pcolormesh(
        np.arange(bar_combined.shape[1]), 
        np.arange(bar_combined.shape[0]), 
        bar_combined, 
        shading='gouraud',
        cmap='binary'
    )
    ax6.set_ylabel('Frequency bin')
    ax6.set_xlabel('Time bin')
    ax6.set_title('Bar Templates (bar_, bar1_, bar2_)')
    plt.colorbar(im6, ax=ax6, label='Value')
    
    # Add labels for each template
    ax6.text(bar_.shape[1]//2, -2, 'bar_', ha='center')
    ax6.text(bar_.shape[1]+2+bar1_.shape[1]//2, -2, 'bar1_', ha='center')
    ax6.text(bar_.shape[1]+2+bar1_.shape[1]+2+bar2_.shape[1]//2, -2, 'bar2_', ha='center')
    
    # 7. Template matching results visualization
    # Create a correlation map for one of the bar templates
    mf = cv2.matchTemplate(test_Sxx_HF.astype('float32'), bar_, cv2.TM_CCOEFF_NORMED)
    
    ax7 = fig.add_subplot(4, 2, 7)
    im7 = ax7.pcolormesh(
        test_times[:mf.shape[1]], 
        test_freqs[200:200+mf.shape[0]]/1000, 
        mf, 
        shading='gouraud',
        cmap='hot'
    )
    ax7.set_ylabel('Frequency (kHz)')
    ax7.set_xlabel('Time (s)')
    ax7.set_title(f'Correlation Map for bar_ (max={hf_max:.4f})')
    plt.colorbar(im7, ax=ax7, label='Correlation')
    
    # Mark the maximum correlation
    max_loc = np.unravel_index(np.argmax(mf), mf.shape)
    ax7.plot(test_times[max_loc[1]], test_freqs[200+max_loc[0]]/1000, 'go', markersize=10)
    
    # 8. Bar template matching results
    ax8 = fig.add_subplot(4, 2, 8)
    # Create a bar chart of the results
    bar_results = [hf_max, hf_max1, hf_max2]
    bar_names = ['bar_', 'bar1_', 'bar2_']
    
    ax8.bar(bar_names, bar_results, color=['blue', 'green', 'red'])
    ax8.set_ylabel('Maximum Correlation')
    ax8.set_title('High Frequency Template Matching Results')
    ax8.set_ylim([0, max(1.0, max(bar_results) * 1.1)])
    
    # Add the values on top of the bars
    for i, v in enumerate(bar_results):
        ax8.text(i, v + 0.02, f"{v:.4f}", ha='center')
    
    # Add summary info
    summary_info = (
        f"Template: {template_def['fname']}\n"
        f"Template size: {template_binary.shape[0]}×{template_binary.shape[1]}\n"
        f"Test file: {test_file}\n"
        f"Vertical match: max={maxs_V[0]:.4f}, position=({xs_V[0]}, {ys_V[0]})\n"
        f"Horizontal match: max={maxs_H[0]:.4f}, position=({xs_H[0]}, {ys_H[0]})\n"
        f"High frequency matches: bar_={hf_max:.4f}, bar1_={hf_max1:.4f}, bar2_={hf_max2:.4f}"
    )
    fig.text(0.5, 0.01, summary_info, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    output_filename = f'template_{os.path.splitext(template_def["fname"])[0]}_vs_{os.path.splitext(test_file)[0]}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to: {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize template matching for dolphin whistle detection')
    parser.add_argument('--template-file', type=str, help='Name of the template audio file (must be in audio_train directory)')
    parser.add_argument('--test-file', type=str, help='Name of the test audio file to match against (must be in audio_train directory)')
    parser.add_argument('--list-files', action='store_true', help='List available audio files and exit')
    parser.add_argument('--all-templates', action='store_true', help='Process all templates')
    parser.add_argument('--template-type', type=str, choices=['whistles', 'clicks', 'all'], 
                      default='whistles', help='Type of templates to process (default: whistles)')
    # Add Gaussian filter controls
    parser.add_argument('--no-gaussian', action='store_true', help='Disable Gaussian filtering')
    parser.add_argument('--sigma-freq', type=float, default=1., help='Gaussian sigma for frequency axis (default: 1.0)')
    parser.add_argument('--sigma-time', type=float, default=1., help='Gaussian sigma for time axis (default: 1.0)')
    args = parser.parse_args()

    # Base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_file = os.path.join(base_dir, 'dolphin_detector/templates/template_definitions.csv')
    audio_dir = os.path.join(base_dir, 'hakaton/audio_train')
    output_dir = os.path.join(base_dir, 'dolphin_detector/plots/templates')
    os.makedirs(output_dir, exist_ok=True)

    # Check if files exist
    if not os.path.exists(template_file):
        print(f"Error: Template file not found: {template_file}")
        return
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        return

    # Load template definitions
    template_df = pd.read_csv(template_file)
    
    # Print template type statistics
    template_types = template_df['file_type'].value_counts()
    print("\nTemplate statistics:")
    for type_name, count in template_types.items():
        print(f"  {type_name}: {count} templates")
    print(f"  Total: {len(template_df)} templates")
    
    # Print Gaussian filter settings
    print("\nGaussian filter settings:")
    if args.no_gaussian:
        print("  Gaussian filtering: Disabled")
    else:
        print(f"  Gaussian filtering: Enabled")
        print(f"  Sigma (freq, time): ({args.sigma_freq}, {args.sigma_time})")
    
    # If --list-files flag is used, show available files and exit
    if args.list_files:
        print("\nAvailable audio files by type:")
        for file_type in sorted(template_df['file_type'].unique()):
            print(f"\n{file_type.upper()}:")
            type_files = template_df[template_df['file_type'] == file_type]
            for idx, row in type_files.iterrows():
                print(f"{idx + 1}. {row['fname']}")
        return

    # Create a dictionary of spectrogram parameters
    spectrogram_params = {
        'apply_gaussian': not args.no_gaussian,
        'gaussian_sigma': (args.sigma_freq, args.sigma_time)
    }

    if args.all_templates:
        # Process templates based on type
        if args.template_type == 'all':
            templates_to_process = template_df
        else:
            templates_to_process = template_df[template_df['file_type'] == args.template_type]
        
        total_templates = len(templates_to_process)
        print(f"\nProcessing {total_templates} templates...")
        
        for idx, template_def in templates_to_process.iterrows():
            print(f"\nTemplate {idx + 1}/{total_templates}")
            
            # Use the same filename as both template and test file
            test_file = template_def['fname']
            visualize_template_matching(template_def, test_file, audio_dir, output_dir, **spectrogram_params)
            
    else:
        # Process single template
        if args.template_file:
            template_def = template_df[template_df['fname'] == args.template_file]
            if len(template_def) == 0:
                print(f"Error: Template file '{args.template_file}' not found in definitions")
                print("\nAvailable template files:")
                for idx, row in template_df.iterrows():
                    print(f"{idx + 1}. {row['fname']} ({row['file_type']})")
                return
            template_def = template_def.iloc[0]
        else:
            template_def = template_df.iloc[0]
            print("No template file specified, using default:", template_def['fname'])

        # For test file, if not specified, use a different file of the same type
        if args.test_file:
            if args.test_file not in template_df['fname'].values:
                print(f"Error: Test file '{args.test_file}' not found in available files")
                print("\nAvailable test files:")
                for idx, row in template_df.iterrows():
                    print(f"{idx + 1}. {row['fname']} ({row['file_type']})")
                return
            test_file = args.test_file
        else:
            same_type_files = template_df[
                (template_df['file_type'] == template_def['file_type']) & 
                (template_df['fname'] != template_def['fname'])
            ]['fname']
            test_file = next(iter(same_type_files), None)
            if test_file is None:
                print(f"Error: Could not find a different {template_def['file_type']} file for testing")
                return
            print("No test file specified, using:", test_file)

        visualize_template_matching(template_def, test_file, audio_dir, output_dir, **spectrogram_params)

if __name__ == "__main__":
    main() 
