import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, firwin
import soundfile as sf
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from dolphin_detector.metricsDolphin import slidingWindowV
except ImportError:
    # Try relative import if module import fails
    from metricsDolphin import slidingWindowV

def apply_bandpass_filter(data, sr):
    """Apply the same bandpass filter as in datasplitter"""
    nyquist = sr / 2
    low = 5000 / nyquist
    high = 15000 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    # Whitening filter
    whitening_filter = firwin(101, [low, high], pass_zero=False)
    whitened_data = filtfilt(whitening_filter, [1], filtered_data)
    return whitened_data

def create_synthetic_whistle(duration=1.0, sample_rate=96000):
    """Create a synthetic dolphin whistle for visualization"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    frequency = 8000 + 4000 * np.sin(2 * np.pi * 5 * t)  # Frequency modulation
    signal = np.sin(2 * np.pi * frequency * t)
    return signal, sample_rate

def plot_enhancement_parameters(data, sr, title="Enhancement Parameter Comparison"):
    """Plot spectrograms with different enhancement parameters"""
    # Apply bandpass filter
    filtered_data = apply_bandpass_filter(data, sr)
    
    # Create spectrograms for both original and filtered data
    f, t, Sxx_orig = signal.spectrogram(data,
                                       fs=sr,
                                       nperseg=2048,
                                       noverlap=1536,
                                       scaling='density')
    f, t, Sxx = signal.spectrogram(filtered_data,
                                  fs=sr,
                                  nperseg=2048,
                                  noverlap=1536,
                                  scaling='density')
    
    Sxx_orig = 10 * np.log10(Sxx_orig + 1e-10)
    Sxx = 10 * np.log10(Sxx + 1e-10)
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot original and filtered spectrograms
    im1 = axes[0][0].imshow(Sxx_orig, aspect='auto', origin='lower',
                           extent=[t[0], t[-1], f[0]/1000, f[-1]/1000])
    axes[0][0].set_title('Original Spectrogram')
    axes[0][0].set_xlabel('Time (s)')
    axes[0][0].set_ylabel('Frequency (kHz)')
    axes[0][0].axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Filter range')
    axes[0][0].axhline(y=15, color='r', linestyle='--', alpha=0.5)
    plt.colorbar(im1, ax=axes[0][0])
    
    im2 = axes[0][1].imshow(Sxx, aspect='auto', origin='lower',
                           extent=[t[0], t[-1], f[0]/1000, f[-1]/1000])
    axes[0][1].set_title('After Bandpass Filter (5-15 kHz)')
    axes[0][1].set_xlabel('Time (s)')
    axes[0][1].set_ylabel('Frequency (kHz)')
    axes[0][1].axhline(y=5, color='r', linestyle='--', alpha=0.5)
    axes[0][1].axhline(y=15, color='r', linestyle='--', alpha=0.5)
    plt.colorbar(im2, ax=axes[0][1])
    
    # Define enhancement parameters for remaining plots
    params = [
        {'inner': 3, 'outer': 16, 'label': 'Small Windows (inner=3, outer=16)'},
        {'inner': 3, 'outer': 64, 'label': 'Default Windows (inner=3, outer=64)'},
        {'inner': 3, 'outer': 128, 'label': 'Large Outer (inner=3, outer=128)'},
        {'inner': 10, 'outer': 64, 'label': 'Large Inner (inner=10, outer=64)'}
    ]
    
    # Plot enhanced versions
    for idx, p in enumerate(params):
        ax = axes[(idx+2)//2][(idx+2)%2]
        enhanced = slidingWindowV(Sxx, inner=p['inner'], outer=p['outer'])
        im = ax.imshow(enhanced, aspect='auto', origin='lower',
                      extent=[t[0], t[-1], f[0]/1000, f[-1]/1000])
        ax.set_title(p['label'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=15, color='r', linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_maxM_comparison(data, sr, title="MaxM Parameter Comparison"):
    """Plot spectrograms with different maxM values"""
    # Apply bandpass filter
    filtered_data = apply_bandpass_filter(data, sr)
    
    # Create spectrogram
    f, t, Sxx = signal.spectrogram(filtered_data,
                                  fs=sr,
                                  nperseg=2048,
                                  noverlap=1536,
                                  scaling='density')
    Sxx = 10 * np.log10(Sxx + 1e-10)
    
    # Find frequency bin indices for 5kHz and 15kHz
    freq_5k_idx = np.argmin(np.abs(f - 5000))
    freq_15k_idx = np.argmin(np.abs(f - 15000))
    
    # Define maxM values around the bandpass range
    maxM_values = [
        freq_5k_idx,                    # Up to 5 kHz
        freq_15k_idx,                   # Up to 15 kHz (bandpass upper limit)
        int(freq_15k_idx * 1.5),        # 1.5x the bandpass range
        Sxx.shape[0]                    # Full range
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for ax, maxM in zip(axes, maxM_values):
        enhanced = slidingWindowV(Sxx, inner=3, outer=64, maxM=maxM)
        im = ax.imshow(enhanced, aspect='auto', origin='lower', 
                      extent=[t[0], t[-1], f[0]/1000, f[-1]/1000])
        ax.set_title(f'maxM = {maxM}\n(max freq = {f[maxM-1]/1000:.1f} kHz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        # Show bandpass filter range
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Filter range')
        ax.axhline(y=15, color='r', linestyle='--', alpha=0.5)
        # Show maxM cutoff
        ax.axhline(y=f[maxM-1]/1000, color='g', linestyle='--', alpha=0.5, label='maxM cutoff')
        ax.legend()
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_multiple_whistles(whistle_files, whistle_dir, num_examples=4):
    """Plot multiple whistle examples with their enhancements"""
    fig, axes = plt.subplots(num_examples, 2, figsize=(15, 5*num_examples))
    plt.suptitle('Multiple Whistle Examples with Enhancement')
    
    for i, whistle_file in enumerate(whistle_files[:num_examples]):
        whistle_path = os.path.join(whistle_dir, whistle_file)
        data, sr = sf.read(whistle_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Apply bandpass filter
        filtered_data = apply_bandpass_filter(data, sr)
            
        # Create spectrogram
        f, t, Sxx = signal.spectrogram(filtered_data,
                                     fs=sr,
                                     nperseg=2048,
                                     noverlap=1536,
                                     scaling='density')
        Sxx = 10 * np.log10(Sxx + 1e-10)
        enhanced = slidingWindowV(Sxx, inner=3, outer=64)
        
        # Plot original
        im1 = axes[i][0].imshow(Sxx, aspect='auto', origin='lower',
                               extent=[t[0], t[-1], f[0]/1000, f[-1]/1000])
        axes[i][0].set_title(f'Filtered - {whistle_file}')
        axes[i][0].set_xlabel('Time (s)')
        axes[i][0].set_ylabel('Frequency (kHz)')
        axes[i][0].axhline(y=5, color='r', linestyle='--', alpha=0.5)
        axes[i][0].axhline(y=15, color='r', linestyle='--', alpha=0.5)
        plt.colorbar(im1, ax=axes[i][0])
        
        # Plot enhanced
        im2 = axes[i][1].imshow(enhanced, aspect='auto', origin='lower',
                               extent=[t[0], t[-1], f[0]/1000, f[-1]/1000])
        axes[i][1].set_title(f'Enhanced - {whistle_file}')
        axes[i][1].set_xlabel('Time (s)')
        axes[i][1].set_ylabel('Frequency (kHz)')
        axes[i][1].axhline(y=5, color='r', linestyle='--', alpha=0.5)
        axes[i][1].axhline(y=15, color='r', linestyle='--', alpha=0.5)
        plt.colorbar(im2, ax=axes[i][1])
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate all visualizations"""
    try:
        # Setup output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Process specific file
        print("Processing SM_190805_021932_093...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, 'hakaton', 'whistles', 'SM_190805_021932_093.wav')
        
        if os.path.exists(file_path):
            # Load and process the file
            data, sr = sf.read(file_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Plot enhancement parameter comparison
            fig1 = plot_enhancement_parameters(data, sr, "SM_190805_021932_093 Enhancement Parameters")
            fig1.savefig(os.path.join(output_dir, 'SM_190805_021932_093_enhancement.png'))
            plt.close(fig1)
            
            # Plot maxM parameter comparison
            fig2 = plot_maxM_comparison(data, sr, "SM_190805_021932_093 MaxM Parameter Effects")
            fig2.savefig(os.path.join(output_dir, 'SM_190805_021932_093_maxM.png'))
            plt.close(fig2)
            
            print("\nVisualizations have been saved to the output directory:")
            print(f"- {os.path.join(output_dir, 'SM_190805_021932_093_enhancement.png')}")
            print(f"- {os.path.join(output_dir, 'SM_190805_021932_093_maxM.png')}")
        else:
            print(f"File not found: {file_path}")
            print("Please make sure the file exists in the hakaton/whistles directory")
            
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main() 
