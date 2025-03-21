import numpy as np
import matplotlib.pyplot as plt
from fileio import AudioData
import os

def plot_spectrogram(Sxx, freqs, times, title, ax):
    """Plot a spectrogram with proper frequency range for dolphins
    
    Args:
        Sxx: Spectrogram data
        freqs: Frequency array
        times: Time array
        title: Plot title
        ax: Matplotlib axis to plot on
    """
    # Focus on dolphin frequency range (5-15 kHz)
    mask = (freqs >= 5000) & (freqs <= 15000)
    
    im = ax.pcolormesh(times, freqs[mask]/1000, Sxx[mask], shading='gouraud')
    plt.colorbar(im, ax=ax, label='Intensity (dB)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim([5, 15])  # Set y-axis to dolphin frequency range
    
def main():
    # Create output directory for plots
    os.makedirs('./plots', exist_ok=True)
    
    # Initialize AudioData with CSV file and audio directory
    csv_path = "../hakaton/train.csv"  # Path to your CSV file
    audio_dir = "../hakaton/audio_train"              # Directory containing the audio files
    
    # Check if files and directories exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found")
        return
        
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found")
        return
        
    print(f"Loading data from:")
    print(f"CSV file: {csv_path}")
    print(f"Audio directory: {audio_dir}")
    
    audio_data = AudioData(csv_path, audio_dir)
    print(f"Found {audio_data.num_whistles} whistle files and {audio_data.num_noise} noise files")
    
    # Plot whistle samples
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Randomly select 4 whistle indices
    whistle_indices = np.random.choice(audio_data.num_whistles, size=min(4, audio_data.num_whistles), replace=False)
    
    for i, idx in enumerate(whistle_indices):
        Sxx, freqs, times = audio_data.get_whistle_sample(idx)
        #plot_spectrogram(Sxx, freqs, times, f'Whistle Sample {idx+1}', axes[i])
    
    plt.tight_layout()
    #plt.savefig('./plots/whistle_samples.png')
    plt.close()
    
    # Plot noise samples
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Randomly select 4 noise indices
    noise_indices = np.random.choice(audio_data.num_noise, size=min(4, audio_data.num_noise), replace=False)
    
    for i, idx in enumerate(noise_indices):
        Sxx, freqs, times = audio_data.get_noise_sample(idx)
        #plot_spectrogram(Sxx, freqs, times, f'Noise Sample {idx+1}', axes[i])
    
    plt.tight_layout()
    #plt.savefig('./plots/noise_samples.png')
    plt.close()
    
    print("\nPlots have been saved to:")
    print("- /plots/whistle_samples.png")
    print("- /plots/noise_samples.png")

if __name__ == "__main__":
    main() 
