import os
import numpy as np
from fileio import AudioData
from metrics import compute_metrics, compute_spectrogram, build_feature_header
from template_detector import TemplateDetector

def generate_test_metrics(test_dir, template_file, output_file):
    """Generate metrics for test data
    
    Args:
        test_dir: Directory containing test WAV files
        template_file: Path to template definitions CSV
        output_file: Where to save the metrics CSV
    """
    # Load audio data
    audio_data = AudioData()
    
    # Get list of test files
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory '{test_dir}' not found")
        
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    if not test_files:
        raise ValueError(f"No WAV files found in {test_dir}")
    
    print(f"Found {len(test_files)} test files")
    
    # Initialize template detector
    template_detector = TemplateDetector(template_file)
    
    # Initialize output array
    metrics_array = []
    
    # Process test files
    print("Processing test files...")
    for i, fname in enumerate(test_files):
        print(f"File {i+1}/{len(test_files)}: {fname}")
        signal = audio_data.read_wav(os.path.join(test_dir, fname))
        Sxx, freqs, times = compute_spectrogram(signal, sr=audio_data.sample_rate)
        
        # Compute metrics including template correlations
        try:
            metrics_list = compute_metrics(Sxx, freqs, times)
            
            # Add template correlation metrics
            detections, _ = template_detector.detect(os.path.join(test_dir, fname))
            if detections:
                max_scores = [0] * len(template_detector.templates)
                for det in detections:
                    template_idx = template_detector.template_info.index(det['template_info'])
                    max_scores[template_idx] = max(max_scores[template_idx], det['score'])
                metrics_list.extend(max_scores)
            else:
                metrics_list.extend([0] * len(template_detector.templates))
                
            # Replace any NaN values with 0
            metrics_list = [0 if np.isnan(x) else x for x in metrics_list]
            
            metrics_array.append(metrics_list)
        except Exception as e:
            print(f"Error processing file {fname}: {str(e)}")
            # Create a placeholder with zeros
            total_features = 13 + 50 + len(template_detector.templates)  # Basic + oops + templates
            metrics_array.append([0] * total_features)
        
    # Convert to numpy array
    metrics_array = np.array(metrics_array)
    
    # Create header
    base_features = [
        'centroid_mean', 'centroid_std',
        'bandwidth_mean', 'bandwidth_std',
        'energy_mean', 'energy_std', 'energy_skew',
        'energy_max', 'energy_variation',
        'bar_template1', 'bar_template2', 'bar_template3'
    ]
    base_features.extend([f'oops_centroid_{i}' for i in range(50)])
    template_features = [f'template_{i}' for i in range(len(template_detector.templates))]
    header = ','.join(base_features + template_features)
    
    # Save with header
    print(f"\nSaving metrics to {output_file}")
    with open(output_file, 'w') as f:
        f.write('Index,Truth,' + header + '\n')  # Truth will be ignored for test data
        for i in range(len(metrics_array)):
            line = f"{i},0," + ','.join(map(str, metrics_array[i])) + '\n'
            f.write(line)
    
    print(f"Test metrics saved to {output_file}")

def main():
    # Paths
    test_dir = "hakaton/test"  # Update with your test directory
    template_file = "dolphin_detector/templates/template_definitions.csv"
    test_file = "dolphin_detector/test_metrics.csv"
    
    # Generate test metrics
    print("Generating test metrics...")
    generate_test_metrics(test_dir, template_file, test_file)

if __name__ == "__main__":
    main() 
