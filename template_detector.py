import os
import numpy as np
import cv2
from metrics import compute_spectrogram, sliding_window_normalize
import wave

class TemplateDetector:
    """Template-based detector for dolphin whistles"""
    
    def __init__(self, template_file, whistle_dir="hakaton/whistles"):
        """Initialize detector with template definitions
        
        Args:
            template_file: Path to CSV file with template definitions
            whistle_dir: Directory containing whistle files
        """
        self.templates = []
        self.template_info = []
        self.whistle_dir = whistle_dir
        self.load_templates(template_file)
        
    def load_templates(self, template_file):
        """Load template definitions and extract templates
        
        Args:
            template_file: Path to template definitions CSV
        """
        with open(template_file, 'r') as f:
            # Skip header
            header = f.readline().strip().split(',')
            
            for line in f:
                # Parse template definition
                tokens = line.strip().split(',')
                info = {
                    'filename': tokens[0],
                    'file_type': tokens[1],
                    'time_start': float(tokens[2]),
                    'time_end': float(tokens[3]),
                    'freq_start': float(tokens[4]),
                    'freq_end': float(tokens[5])
                }
                
                # Extract template from audio file
                signal = self.read_audio(os.path.join(self.whistle_dir, info['filename']))
                Sxx, freqs, times = compute_spectrogram(signal)
                
                # Find time and frequency indices
                time_start_idx = np.argmin(np.abs(times - info['time_start']))
                time_end_idx = np.argmin(np.abs(times - info['time_end']))
                freq_start_idx = np.argmin(np.abs(freqs - info['freq_start']))
                freq_end_idx = np.argmin(np.abs(freqs - info['freq_end']))
                
                # Apply sliding window normalization
                Sxx_norm = sliding_window_normalize(Sxx)
                
                # Extract template region
                template = Sxx_norm[freq_start_idx:freq_end_idx, time_start_idx:time_end_idx]
                
                # Convert to binary mask using mean thresholding (like whale code)
                mean = np.mean(template)
                std = np.std(template)
                min_val = template.min()
                template[template < mean + 0.5*std] = min_val
                template[template > min_val] = 1
                template[template < 0] = 0
                
                self.templates.append(template.astype(np.float32))
                self.template_info.append(info)
                
        print(f"Loaded {len(self.templates)} templates")
    
    def detect(self, audio_file, threshold=0.5, plot=False):
        """Detect whistles using template matching
        
        Args:
            audio_file: Path to audio file
            threshold: Detection threshold (0-1)
            plot: Whether to save detection plot
            
        Returns:
            List of detections (dicts with score and location)
            Path to plot if plot=True, else None
        """
        # Load and process audio
        signal = self.read_audio(audio_file)
        Sxx, freqs, times = compute_spectrogram(signal)
        
        # Normalize spectrogram
        Sxx_norm = sliding_window_normalize(Sxx)
        
        # Initialize detections list
        detections = []
        
        # Apply each template
        for template, info in zip(self.templates, self.template_info):
            # Template matching
            result = cv2.matchTemplate(
                Sxx_norm.astype(np.float32),
                template,
                cv2.TM_CCOEFF_NORMED
            )
            
            # Find peaks above threshold
            peaks = np.where(result >= threshold)
            scores = result[peaks]
            
            # Add detections
            for score, (y, x) in zip(scores, zip(*peaks)):
                detection = {
                    'score': float(score),
                    'location': (int(x), int(y)),
                    'template_info': info
                }
                detections.append(detection)
        
        # Sort by score
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Generate plot if requested
        plot_path = None
        if plot and detections:
            plot_path = self._generate_plot(Sxx, freqs, times, detections)
            
        return detections, plot_path
    
    def read_audio(self, filepath):
        """Read audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            numpy array of audio samples
        """
        with wave.open(filepath, 'rb') as wav:
            frames = wav.getnframes()
            audio_bytes = wav.readframes(frames)
            return np.frombuffer(audio_bytes, dtype=np.int16)
            
    def _generate_plot(self, Sxx, freqs, times, detections):
        """Generate detection plot
        
        Args:
            Sxx: Spectrogram array
            freqs: Frequency bins
            times: Time bins
            detections: List of detections
            
        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-10))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        # Plot detection boxes
        for det in detections[:3]:  # Plot top 3 detections
            x, y = det['location']
            info = det['template_info']
            w = info['time_end'] - info['time_start']
            h = info['freq_end'] - info['freq_start']
            plt.gca().add_patch(plt.Rectangle(
                (times[x], freqs[y]),
                w,
                h,
                fill=False,
                color='r'
            ))
            
        # Save plot
        plot_path = 'dolphin_detector/detections/latest_detection.png'
        os.makedirs('dolphin_detector/detections', exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

def main():
    # Create detections directory
    os.makedirs('dolphin_detector/detections', exist_ok=True)
    
    # Initialize detector
    detector = TemplateDetector()
    
    # Get random test file
    whistle_dir = "hakaton/whistles"
    test_file = "SM_190916_092137_612.wav"  # Different random file
    audio_file = os.path.join(whistle_dir, test_file)
    
    print(f"Testing file: {test_file}")
    
    # Test with different thresholds
    thresholds = [0.5, 0.6, 0.7]  # Added 0.6 for finer granularity
    for threshold in thresholds:
        print(f"\nTesting with threshold = {threshold}")
        detector.process_file(audio_file, threshold=threshold, plot=True)
        # Move plot file
        src = 'dolphin_detector/detections/latest_detection.png'
        dst = f'dolphin_detector/detections/detection_thresh_{threshold:.1f}.png'
        if os.path.exists(dst):
            os.remove(dst)
        if os.path.exists(src):
            os.rename(src, dst)

if __name__ == "__main__":
    main() 
