import numpy as np
import cv2
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

try:
    from dolphin_detector.metricsDolphin import slidingWindowV
except ImportError:
    from metricsDolphin import slidingWindowV

class AudioData:
    """Audio data loader for dolphin whistles classification with single folder
    
    Args:
        csv_path: Path to CSV file containing fname,label pairs (optional)
        audio_dir: Directory containing all audio files
    """
    def __init__(self, audio_dir, csv_path=None):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.data = None
        self.sample_rate = 96000  # Default sample rate for dolphin recordings
        self.files = []
        self.labels = []
        self.num_files = 0
        
        # Load file lists
        self.load_files()
        
    def load_files(self):
        """Load file lists from CSV file or directory"""
        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError("Audio directory not found")
            
        if self.csv_path and os.path.exists(self.csv_path):
            # Load from CSV if provided
            self.data = pd.read_csv(self.csv_path)
            
            # Debug: Print CSV columns
            print(f"CSV file: {self.csv_path}")
            print(f"CSV columns: {self.data.columns.tolist()}")
            
            # Check if CSV has the required columns
            if 'fname' not in self.data.columns or 'label' not in self.data.columns:
                raise ValueError("CSV must contain 'fname' and 'label' columns")
                
            self.files = self.data['fname'].tolist()
            self.labels = self.data['label'].tolist()
        else:
            # Load all WAV files from directory
            self.files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
            # Assume all files are whistles when no CSV is provided
            self.labels = ['whistle'] * len(self.files)
            
        self.num_files = len(self.files)
        
        # Count by label
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            
        print(f"Loaded {self.num_files} files:")
        for label, count in label_counts.items():
            print(f"  - {label}: {count} files")
        
    def read_wav(self, filepath):
        """Read WAV file and return signal"""
        with wave.open(filepath, 'rb') as wav:
            self.sample_rate = wav.getframerate()
            frames = wav.getnframes()
            audio_bytes = wav.readframes(frames)
            return np.frombuffer(audio_bytes, dtype=np.int16)
            
    def get_spectrogram(self, audio, params=None):
        """Compute spectrogram with parameters suited for dolphin whistles"""
        if params is None:
            params = {
                'NFFT': 2048,
                'Fs': self.sample_rate,
                'noverlap': 1536
            }
            
        freqs, times, Sxx = signal.spectrogram(audio,
                                             fs=params['Fs'],
                                             nperseg=params['NFFT'],
                                             noverlap=params['noverlap'],
                                             scaling='density')
        
        Sxx = 10 * np.log10(Sxx + 1e-10)
        
        return Sxx, freqs, times
            
    def get_sample(self, index, params=None):
        """Get a sample spectrogram by index"""
        if self.audio_dir is None:
            raise ValueError("Audio directory not set")
            
        filepath = os.path.join(self.audio_dir, self.files[index])
        signal = self.read_wav(filepath)
        return self.get_spectrogram(signal, params), self.files[index], self.labels[index]


class TemplateSelector:
    def __init__(self, audio_data):
        self.audio_data = audio_data
        self.templates = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_image = None
        self.window_name = "Template Selector"
        self.scale_factor = 1.0  # For converting display coordinates to actual coordinates
        self.current_selections = []  # Store multiple selections for current image
        self.enhance_contrast = False  # Toggle for contrast enhancement
        self.original_spectrogram = None  # Store original spectrogram
        self.enhanced_spectrogram = None  # Store enhanced spectrogram
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for template selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self.current_image.copy()
            self.end_point = (x, y)
            # Draw all previous selections
            for sel in self.current_selections:
                cv2.rectangle(img_copy, sel[0], sel[1], (0, 255, 255), 2)  # Yellow for accepted
            # Draw current selection
            cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 255, 0), 2)  # Green for current
            cv2.imshow(self.window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
    def normalize_spectrogram(self, Sxx):
        """Normalize spectrogram for display"""
        # Scale to 0-255 range
        Sxx_norm = ((Sxx - Sxx.min()) * (255/(Sxx.max() - Sxx.min()))).astype(np.uint8)
        # Convert to RGB
        Sxx_rgb = cv2.cvtColor(Sxx_norm, cv2.COLOR_GRAY2BGR)
        return Sxx_rgb
        
    def process_file(self, file_idx):
        """Process a single file for template selection"""
        # Get spectrogram
        (Sxx, freqs, times), filename, label = self.audio_data.get_sample(file_idx)
            
        # Focus on dolphin frequency range (5-20 kHz)
        mask = (freqs >= 5000) & (freqs <= 20000)
        Sxx = Sxx[mask]
        freqs = freqs[mask]
        
        # Store original spectrogram
        self.original_spectrogram = Sxx.copy()
        
        # Create enhanced version
        self.enhanced_spectrogram = slidingWindowV(Sxx, maxM=Sxx.shape[0])
        
        # Use original or enhanced based on toggle
        display_spectrogram = self.enhanced_spectrogram if self.enhance_contrast else self.original_spectrogram
        
        # Normalize and prepare for display
        display_height = 800
        self.scale_factor = display_height / display_spectrogram.shape[0]
        display_width = int(display_spectrogram.shape[1] * self.scale_factor)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Normalize and resize spectrogram for display
        self.current_image = cv2.resize(self.normalize_spectrogram(display_spectrogram), 
                                      (display_width, display_height))
        
        # Reset selections for new image
        self.current_selections = []
        
        print(f"\nProcessing: {filename} (Label: {label})")
        print("Instructions:")
        print("- Left click and drag to select a whistle region")
        print("- Press 'a' to accept current selection (can select multiple whistles)")
        print("- Press 'r' to reset/clear all selections in current image")
        print("- Press 'e' to toggle enhanced contrast")
        print("- Press 'n' for next file")
        print("- Press 'q' to quit")
        
        while True:
            # Show image with all selections
            display_img = self.current_image.copy()
            
            # Draw all accepted selections in yellow
            for sel in self.current_selections:
                cv2.rectangle(display_img, sel[0], sel[1], (0, 255, 255), 2)
                
            # Draw current selection in green
            if self.start_point and self.end_point:
                cv2.rectangle(display_img, self.start_point, self.end_point, (0, 255, 0), 2)
                
            # Add file info to display
            contrast_mode = "Enhanced" if self.enhance_contrast else "Original"
            info_text = f"{filename} | Label: {label} | Mode: {contrast_mode} | Selections: {len(self.current_selections)}"
            cv2.putText(display_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            cv2.imshow(self.window_name, display_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False  # Signal to stop processing
            elif key == ord('n'):
                # Save all selections from current image
                for sel in self.current_selections:
                    self.save_selection(sel[0], sel[1], filename, label, times, freqs)
                self.current_selections = []
                self.start_point = None
                self.end_point = None
                break
            elif key == ord('a') and self.start_point and self.end_point:
                # Add current selection to list
                self.current_selections.append((self.start_point, self.end_point))
                self.start_point = None
                self.end_point = None
                print(f"Selection added. Total selections for this image: {len(self.current_selections)}")
            elif key == ord('r'):
                # Reset all selections for current image
                self.current_selections = []
                self.start_point = None
                self.end_point = None
                print("Selections reset for current image")
            elif key == ord('e'):
                # Toggle enhanced contrast
                self.enhance_contrast = not self.enhance_contrast
                display_spectrogram = self.enhanced_spectrogram if self.enhance_contrast else self.original_spectrogram
                self.current_image = cv2.resize(self.normalize_spectrogram(display_spectrogram), 
                                             (display_width, display_height))
                print(f"Contrast enhancement: {'ON' if self.enhance_contrast else 'OFF'}")
                
        return True  # Continue processing
        
    def save_selection(self, start_point, end_point, filename, label, times, freqs):
        """Convert and save a selection to template"""
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Convert to time and frequency
        time_start = times[int(min(x1, x2) / self.scale_factor)]
        time_end = times[int(max(x1, x2) / self.scale_factor)]
        freq_start = freqs[int(min(y1, y2) / self.scale_factor)] / 1000  # Convert to kHz
        freq_end = freqs[int(max(y1, y2) / self.scale_factor)] / 1000
        
        # Add template
        self.add_template(filename, label, time_start, time_end, freq_start, freq_end)
                
    def add_template(self, filename, label, time_start, time_end, freq_start, freq_end):
        """Add a template definition"""
        template = {
            'fname': filename,
            'file_type': label,
            'time_start': float(time_start),
            'time_end': float(time_end),
            'freq_start': float(freq_start) * 1000,  # Convert kHz to Hz
            'freq_end': float(freq_end) * 1000
        }
        self.templates.append(template)
        print(f"Template added: {template}")
        
    def save_templates(self):
        """Save selected templates to CSV file"""
        if not self.templates:
            print("No templates to save!")
            return
            
        os.makedirs('templates', exist_ok=True)
        output_file = os.path.join('templates', 'template_definitions.csv')
        
        # Save to CSV
        with open(output_file, 'w') as f:
            # Write header
            f.write('fname,file_type,time_start,time_end,freq_start,freq_end\n')
            
            # Write templates
            for t in self.templates:
                f.write(f"{t['fname']},{t['file_type']},{t['time_start']:.4f},"
                       f"{t['time_end']:.4f},{t['freq_start']:.1f},{t['freq_end']:.1f}\n")
                       
        print(f"\nTemplates saved to {output_file}")


def main():
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Select templates from audio files')
    parser.add_argument('--audio_dir', required=True, help='Directory containing audio files')
    parser.add_argument('--csv', required=False, help='Optional: Path to CSV file with fname,label columns')
    parser.add_argument('--enhance', action='store_true', help='Start with contrast enhancement enabled')
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory '{args.audio_dir}' not found")
        return
        
    print(f"Loading data from:")
    print(f"Audio directory: {args.audio_dir}")
    if args.csv:
        print(f"CSV file: {args.csv}")
    print(f"Contrast enhancement: {'Enabled' if args.enhance else 'Disabled'} (toggle with 'e' key)")
    
    # Initialize AudioData with audio directory and optional CSV
    try:
        audio_data = AudioData(args.audio_dir, args.csv)
        selector = TemplateSelector(audio_data)
        
        # Set initial contrast enhancement state
        selector.enhance_contrast = args.enhance
        
        # Process all files
        print("\nProcessing files...")
        for i in range(audio_data.num_files):
            if not selector.process_file(i):
                break
        
        # Save templates
        selector.save_templates()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 
