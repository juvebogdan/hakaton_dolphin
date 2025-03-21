import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from fileio import AudioData
from metrics import compute_metrics, compute_spectrogram, build_feature_header
from classifier import DolphinClassifier

def generate_training_data(whistle_dir, noise_dir, output_file):
    """Generate training data from audio files
    
    Args:
        whistle_dir: Directory containing whistle WAV files
        noise_dir: Directory containing noise WAV files
        output_file: Where to save the metrics CSV
    """
    # Load audio data
    audio_data = AudioData(whistle_dir, noise_dir)
    
    # Initialize output arrays
    metrics_array = []
    truth_array = []
    
    # Process whistle files
    print("Processing whistle files...")
    for i, fname in enumerate(audio_data.whistles):
        print(f"File {i+1}/{len(audio_data.whistles)}: {fname}")
        signal = audio_data.read_wav(os.path.join(whistle_dir, fname))
        Sxx, freqs, times = compute_spectrogram(signal, sr=audio_data.sample_rate)
        metrics = compute_metrics(Sxx, freqs, times)
        metrics_array.append(metrics)
        truth_array.append(1)
        
    # Process noise files
    print("\nProcessing noise files...")
    for i, fname in enumerate(audio_data.noise):
        print(f"File {i+1}/{len(audio_data.noise)}: {fname}")
        signal = audio_data.read_wav(os.path.join(noise_dir, fname))
        Sxx, freqs, times = compute_spectrogram(signal, sr=audio_data.sample_rate)
        metrics = compute_metrics(Sxx, freqs, times)
        metrics_array.append(metrics)
        truth_array.append(0)
        
    # Convert to numpy arrays
    metrics_array = np.array(metrics_array)
    truth_array = np.array(truth_array)
    
    # Create header
    header = build_feature_header()
    
    # Save with header
    print(f"\nSaving metrics to {output_file}")
    with open(output_file, 'w') as f:
        f.write('Index,Truth,' + header + '\n')
        for i in range(len(truth_array)):
            line = f"{i},{truth_array[i]}," + ','.join(map(str, metrics_array[i])) + '\n'
            f.write(line)
            
def main():
    # Paths
    whistle_dir = "hakaton/whistles"
    noise_dir = "hakaton/noise"
    train_file = "dolphin_detector/train_metrics.csv"
    
    # Generate training data if needed
    if not os.path.exists(train_file):
        print("Generating training data...")
        generate_training_data(whistle_dir, noise_dir, train_file)
    
    # Initialize classifier with whale-like parameters
    print("\nInitializing classifier...")
    clf = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.5,
        min_samples_split=20,
        min_samples_leaf=20,
        max_features=30,
        random_state=42
    )
    
    # Train and validate
    print("Training and validating...")
    dc = DolphinClassifier(train_file)
    dc.validate(clf, n_folds=4, plot_roc=True)
    
if __name__ == "__main__":
    main() 
