import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

class AudioFile:
    """
    A class to handle audio files and provide utilities for analysis and visualization.

    Attributes:
        file_path (str): Path to the audio file.
        file_name (str): Name of the audio file (extracted from the path).
        label (str): Label of the audio file (derived from the parent directory name).
        audio (np.ndarray): Loaded audio data.
        sample_rate (int): Sampling rate of the audio file.
        duration (float): Duration of the audio file in seconds.
    """

    def __init__(self, file_path):
        """
        Initialize the AudioFile instance by loading the audio file and extracting metadata.

        Args:   
            file_path (str): Path to the audio file.
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.label = os.path.basename(os.path.dirname(self.file_path))
        self.audio, self.sample_rate = librosa.load(file_path)
        self.audio = librosa.util.normalize(self.audio)   # normalize audio
        self.duration = librosa.get_duration(y=self.audio, sr=self.sample_rate)

    def display_waveform(self):
        """
        Display the waveform of the audio file.
        """
        librosa.display.waveshow(self.audio, sr=self.sample_rate)
        plt.show()
        plt.close()

    def play(self):
        """
        Play the audio file.

        Returns:
            IPython.display.Audio: audio player widget.
        """
        return ipd.display(ipd.Audio(self.audio, rate=self.sample_rate))

    def trim(self, top_db=50):
        """
        Trim silent parts of the audio based on a decibel threshold.

        Args:
            top_db (int, optional): Decibel threshold below which audio is considered silent.
        """
        self.audio, _ = librosa.effects.trim(self.audio, top_db=top_db)

    def create_spectrogram(self):
        """
        Create a mel spectrogram of the audio file.

        Returns:
            np.ndarray: The mel spectrogram in decibel units.
        """
        mel_scale_sgram = librosa.feature.melspectrogram(
            y=self.audio,
            sr=self.sample_rate,
            power=1)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        return mel_sgram

    def display_spectrogram(self):
        """
        Display the spectrogram of the audio file.
        """
        _spectrogram = self.create_spectrogram()

        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            _spectrogram,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='mel',
            ax=ax)
        plt.colorbar(img, format='%+2.0f dB')

        # remove whitespace around image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.show()
        plt.close(fig)

    def save_spectrogram(self, output_dir=None, skip_existing=True):
        """
        Save the spectrogram as a PNG file.

        Args:
            output_dir (str, optional): Directory to save the spectrogram. Defaults to the directory of the audio file.
            skip_existing (bool, optional): Whether to skip saving if the file already exists. Defaults to True.
        """
        if not output_dir:
            output_dir = os.path.dirname(self.file_path)
        else:
            output_dir = os.path.join(output_dir, self.label)

        base, _ = os.path.splitext(self.file_name)
        output_file = os.path.join(output_dir, base + ".png")

        if skip_existing and os.path.exists(output_file):
            return

        spectrogram = self.create_spectrogram()
        librosa.display.specshow(spectrogram, sr=self.sample_rate)

        os.makedirs(output_dir, exist_ok=True)
        # save, removing whitespace
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
