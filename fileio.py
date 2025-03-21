import numpy as np
import wave
import os
import pandas as pd
from scipy import signal

class AudioData:
    """Audio data loader for dolphin whistles classification
    
    Args:
        csv_path: Path to CSV file containing filename,class pairs
        audio_dir: Base directory containing audio files
    """
    def __init__(self, csv_path=None, audio_dir=None):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.data = None
        self.sample_rate = 96000  # Default sample rate for dolphin recordings
        
        # Only load file lists if paths are provided
        if csv_path is not None and audio_dir is not None:
            self.load_files()
        
    def load_files(self):
        """Load file lists from CSV file"""
        if not os.path.exists(self.csv_path) or not os.path.exists(self.audio_dir):
            raise FileNotFoundError("CSV file or audio directory not found")
            
        self.data = pd.read_csv(self.csv_path, header=None, names=['filename', 'class'])
        self.whistles = self.data[self.data['class'] == 'whistles']['filename'].tolist()
        self.noise = self.data[self.data['class'] == 'noise']['filename'].tolist()
        self.num_whistles = len(self.whistles)
        self.num_noise = len(self.noise)
        
    def read_wav(self, filepath):
        """Read WAV file and return signal
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            numpy array containing audio data
        """
        with wave.open(filepath, 'rb') as wav:
            self.sample_rate = wav.getframerate()
            frames = wav.getnframes()
            audio_bytes = wav.readframes(frames)
            return np.frombuffer(audio_bytes, dtype=np.int16)
            
    def get_spectrogram(self, audio, params=None):
        """Compute spectrogram with parameters suited for dolphin whistles
        
        Args:
            audio: Audio signal array
            params: Dictionary containing spectrogram parameters
                   Default: {'NFFT': 2048, 'Fs': 96000, 'noverlap': 1536}
            
        Returns:
            tuple: (spectrogram array, frequency bins, time bins)
        """
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
            
    def get_whistle_sample(self, index, params=None):
        """Get a whistle sample spectrogram by index
        
        Args:
            index: Index of file to read
            params: Spectrogram parameters
            
        Returns:
            tuple: (spectrogram array, frequency bins, time bins)
        """
        if self.audio_dir is None:
            raise ValueError("Audio directory not set")
            
        filepath = os.path.join(self.audio_dir, self.whistles[index])
        signal = self.read_wav(filepath)
        return self.get_spectrogram(signal, params)
        
    def get_noise_sample(self, index, params=None):
        """Get a noise sample spectrogram by index
        
        Args:
            index: Index of file to read
            params: Spectrogram parameters
            
        Returns:
            tuple: (spectrogram array, frequency bins, time bins)
        """
        if self.audio_dir is None:
            raise ValueError("Audio directory not set")
            
        filepath = os.path.join(self.audio_dir, self.noise[index])
        signal = self.read_wav(filepath)
        return self.get_spectrogram(signal, params) 
