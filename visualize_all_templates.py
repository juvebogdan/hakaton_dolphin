import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal
import wave
import cv2
from dolphin_detector.metricsDolphin import slidingWindowV, slidingWindowH, matchTemplate, highFreqTemplate
import random

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
    bar_[:, 6:12] = 1.  # Center vertical bar (was 3:6 in original)
    bar1_[:, 8:16] = 1.  # Center vertical bar (was 4:8 in original)
    bar2_[:, 4:8] = 1.  # Center vertical bar (was 2:4 in original)
    
    return bar_, bar1_, bar2_

def extract_template(audio_file, template_def, params):
    """Extract a template from an audio file based on template definition"""
    # Read the audio file
    audio, sample_rate = read_wav(audio_file)
    
    # Generate spectrogram
    Sxx, freqs, times = get_spectrogram(audio, sample_rate, params)
    
    # Find indices for template extraction
    time_start_idx = np.argmin(np.abs(times - template_def['time_start']))
    time_end_idx = np.argmin(np.abs(times - template_def['time_end']))
    freq_start_idx = np.argmin(np.abs(freqs - template_def['freq_start']))
    freq_end_idx = np.argmin(np.abs(freqs - template_def['freq_end']))
    
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
    
    return template_binary.astype('float32'), freqs, times, time_start_idx, time_end_idx, freq_start_idx, freq_end_idx

def test_template(template, test_file, params):
    """Test a template against a test file"""
    # Read the test audio file
    test_audio, test_sample_rate = read_wav(test_file)
    
    # Generate spectrogram for test file
    test_Sxx, test_freqs, test_times = get_spectrogram(test_audio, test_sample_rate, params)
    
    # Apply sliding window normalization (vertical and horizontal)
    test_Sxx_V = slidingWindowV(test_Sxx)
    test_Sxx_H = slidingWindowH(test_Sxx)
    
    # Create a simple template manager with just this template
    tmpl = SimpleTemplateManager()
    tmpl.add_template(template, {})
    
    # Perform template matching
    maxs_V, xs_V, ys_V = templateMetrics(test_Sxx_V, tmpl)
    maxs_H, xs_H, ys_H = templateMetrics(test_Sxx_H, tmpl)
    
    # Create bar templates for highFreqTemplate
    bar_, bar1_, bar2_ = create_bar_templates()
    
    # Apply highFreqTemplate matching
    hf_max = highFreqTemplate(test_Sxx, bar_)[0]
    hf_max1 = highFreqTemplate(test_Sxx, bar1_)[0]
    hf_max2 = highFreqTemplate(test_Sxx, bar2_)[0]
    
    return {
        'test_Sxx': test_Sxx,
        'test_Sxx_V': test_Sxx_V,
        'test_Sxx_H': test_Sxx_H,
        'test_freqs': test_freqs,
        'test_times': test_times,
        'maxs_V': maxs_V[0],
        'xs_V': xs_V[0],
        'ys_V': ys_V[0],
        'maxs_H': maxs_H[0],
        'xs_H': xs_H[0],
        'ys_H': ys_H[0],
        'hf_max': hf_max,
        'hf_max1': hf_max1,
        'hf_max2': hf_max2
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
    
    # Get unique files for testing
    unique_files = whistle_templates['fname'].unique()
    
    # Spectrogram parameters
    params = {'NFFT': 2048, 'Fs': 96000, 'noverlap': 1536}
    
    # Create output directory for plots
    output_dir = os.path.join(base_dir, 'dolphin_detector/plots/templates')
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to max 10 templates for visualization
    max_templates = min(10, len(whistle_templates))
    template_indices = random.sample(range(len(whistle_templates)), max_templates)
    
    # Store results for all templates
    all_results = []
    
    # Process each template
    for i, idx in enumerate(template_indices):
        template_def = whistle_templates.iloc[idx]
        template_file = os.path.join(audio_dir, template_def['fname'])
        
        print(f"\nProcessing template {i+1}/{max_templates}:")
        print(f"  File: {template_def['fname']}")
        print(f"  Time: {template_def['time_start']:.2f} - {template_def['time_end']:.2f} s")
        print(f"  Freq: {template_def['freq_start']/1000:.1f} - {template_def['freq_end']/1000:.1f} kHz")
        
        # Extract template
        template, freqs, times, t_start_idx, t_end_idx, f_start_idx, f_end_idx = extract_template(
            template_file, template_def, params
        )
        
        # Select a different file for testing
        test_files = []
        for file in unique_files:
            if file != template_def['fname']:
                test_files.append(file)
        
        # Select up to 3 test files
        num_test_files = min(3, len(test_files))
        selected_test_files = random.sample(test_files, num_test_files)
        
        # Test against each selected file
        template_results = []
        for test_file_name in selected_test_files:
            test_file_path = os.path.join(audio_dir, test_file_name)
            result = test_template(template, test_file_path, params)
            result['test_file'] = test_file_name
            template_results.append(result)
            
            print(f"  Tested against {test_file_name}:")
            print(f"    Vertical match: {result['maxs_V']:.4f}")
            print(f"    Horizontal match: {result['maxs_H']:.4f}")
            print(f"    High freq matches: bar_={result['hf_max']:.4f}, bar1_={result['hf_max1']:.4f}, bar2_={result['hf_max2']:.4f}")
        
        # Create visualization for this template
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Original template
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.pcolormesh(
            times[t_start_idx:t_end_idx], 
            freqs[f_start_idx:f_end_idx]/1000, 
            template, 
            shading='gouraud',
            cmap='binary'
        )
        ax1.set_ylabel('Frequency (kHz)')
        ax1.set_xlabel('Time (s)')
        ax1.set_title(f'Template from {os.path.basename(template_def["fname"])}')
        plt.colorbar(im1, ax=ax1, label='Value')
        
        # 2-4. Test results for each file
        for j, result in enumerate(template_results):
            ax = fig.add_subplot(2, 3, j+2)
            im = ax.pcolormesh(
                result['test_times'], 
                result['test_freqs']/1000, 
                result['test_Sxx_V'], 
                shading='gouraud'
            )
            
            # Mark the best match location
            template_height, template_width = template.shape
            
            # Check if match position is valid
            if (result['xs_V'] < len(result['test_times']) and 
                result['ys_V'] < len(result['test_freqs'])):
                
                # Calculate end points safely
                end_x = min(result['xs_V'] + template_width, len(result['test_times'])-1)
                end_y = min(result['ys_V'] + template_height, len(result['test_freqs'])-1)
                
                # Calculate width and height for rectangle
                width = result['test_times'][end_x] - result['test_times'][result['xs_V']]
                height = result['test_freqs'][end_y]/1000 - result['test_freqs'][result['ys_V']]/1000
                
                # Draw rectangle
                rect = plt.Rectangle(
                    (result['test_times'][result['xs_V']], result['test_freqs'][result['ys_V']]/1000),
                    width, height,
                    fill=False, edgecolor='r', linewidth=2
                )
                ax.add_patch(rect)
                
                # Add match info
                match_info = f"V: {result['maxs_V']:.3f}, H: {result['maxs_H']:.3f}"
                ax.text(0.05, 0.05, match_info, transform=ax.transAxes, 
                       color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))
            else:
                # Add error message if match position is invalid
                ax.text(0.5, 0.5, "Invalid match position", 
                       transform=ax.transAxes, ha='center',
                       color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            # Add labels
            ax.set_ylabel('Frequency (kHz)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Match in {os.path.basename(result["test_file"])}')
            plt.colorbar(im, ax=ax, label='Intensity (dB)')
        
        # 5. Bar chart of results
        ax5 = fig.add_subplot(2, 3, 5)
        
        # Prepare data for bar chart
        test_files = [os.path.basename(r['test_file']) for r in template_results]
        v_matches = [r['maxs_V'] for r in template_results]
        h_matches = [r['maxs_H'] for r in template_results]
        
        # Set width of bars
        barWidth = 0.35
        r1 = np.arange(len(test_files))
        r2 = [x + barWidth for x in r1]
        
        # Create bars
        ax5.bar(r1, v_matches, width=barWidth, label='Vertical', color='blue')
        ax5.bar(r2, h_matches, width=barWidth, label='Horizontal', color='green')
        
        # Add labels and legend
        ax5.set_ylabel('Match Score')
        ax5.set_title('Template Matching Results')
        ax5.set_xticks([r + barWidth/2 for r in range(len(test_files))])
        ax5.set_xticklabels(test_files, rotation=45, ha='right')
        ax5.legend()
        
        # 6. High frequency template results
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Prepare data for bar chart
        bar_names = ['bar_', 'bar1_', 'bar2_']
        bar_results_by_file = []
        
        for result in template_results:
            bar_results_by_file.append([result['hf_max'], result['hf_max1'], result['hf_max2']])
        
        # Set width of bars
        barWidth = 0.25
        r1 = np.arange(len(bar_names))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Create bars for each test file
        for j, result in enumerate(bar_results_by_file):
            if j == 0:
                ax6.bar([r + j*barWidth for r in r1], result, width=barWidth, 
                       label=os.path.basename(template_results[j]['test_file']))
            else:
                ax6.bar([r + j*barWidth for r in r1], result, width=barWidth, 
                       label=os.path.basename(template_results[j]['test_file']))
        
        # Add labels and legend
        ax6.set_ylabel('Match Score')
        ax6.set_title('High Frequency Template Results')
        ax6.set_xticks([r + barWidth for r in range(len(bar_names))])
        ax6.set_xticklabels(bar_names)
        ax6.legend()
        
        # Add summary info
        summary_info = (
            f"Template: {template_def['fname']}\n"
            f"Time: {template_def['time_start']:.2f} - {template_def['time_end']:.2f} s\n"
            f"Frequency: {template_def['freq_start']/1000:.1f} - {template_def['freq_end']/1000:.1f} kHz\n"
            f"Template size: {template.shape[0]}×{template.shape[1]}"
        )
        fig.text(0.5, 0.01, summary_info, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.suptitle(f'Template {i+1}/{max_templates} Matching Results', fontsize=16)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'template_{i+1}_results.png'))
        plt.close(fig)
        
        # Store results for summary
        all_results.append({
            'template_def': template_def,
            'template': template,
            'results': template_results
        })
    
    # Create summary visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for summary chart
    template_names = [f"T{i+1}" for i in range(len(all_results))]
    avg_v_matches = []
    avg_h_matches = []
    avg_hf_matches = []
    
    for result in all_results:
        v_matches = [r['maxs_V'] for r in result['results']]
        h_matches = [r['maxs_H'] for r in result['results']]
        hf_matches = [(r['hf_max'] + r['hf_max1'] + r['hf_max2'])/3 for r in result['results']]
        
        avg_v_matches.append(np.mean(v_matches))
        avg_h_matches.append(np.mean(h_matches))
        avg_hf_matches.append(np.mean(hf_matches))
    
    # Set width of bars
    barWidth = 0.25
    r1 = np.arange(len(template_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    ax.bar(r1, avg_v_matches, width=barWidth, label='Vertical Match', color='blue')
    ax.bar(r2, avg_h_matches, width=barWidth, label='Horizontal Match', color='green')
    ax.bar(r3, avg_hf_matches, width=barWidth, label='High Freq Match (avg)', color='red')
    
    # Add labels and legend
    ax.set_ylabel('Average Match Score')
    ax.set_title('Summary of Template Matching Results')
    ax.set_xticks([r + barWidth for r in range(len(template_names))])
    ax.set_xticklabels(template_names)
    ax.legend()
    
    # Add template info as text
    template_info = ""
    for i, result in enumerate(all_results):
        template_def = result['template_def']
        template_info += f"T{i+1}: {os.path.basename(template_def['fname'])} " + \
                        f"({template_def['time_start']:.1f}-{template_def['time_end']:.1f}s, " + \
                        f"{template_def['freq_start']/1000:.1f}-{template_def['freq_end']/1000:.1f}kHz)\n"
    
    fig.text(0.5, 0.01, template_info, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'template_summary.png'))
    
    print(f"\nAll template visualizations saved to {output_dir}")
    print(f"Summary visualization saved to {os.path.join(output_dir, 'template_summary.png')}")

if __name__ == "__main__":
    main() 
