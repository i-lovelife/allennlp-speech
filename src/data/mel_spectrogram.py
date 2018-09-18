from overrides import overrides
import librosa
import numpy as np

from src.data.audio_transformer import AudioTransformer

@AudioTransformer.register('spec')
class MelSpectrogram(AudioTransformer):
    def __init__(self,
                 sr: int = 16000,
                 frame_shift: float = 0.0125,
                 frame_length: float = 0.05,
                 preemphasis: float = 0.97,
                 n_fft: int = 1024,
                 trim: bool = True,
                 ref_db: int = 20,
                 max_db: int = 100
                ) -> None:
        self.sr = sr # pylint: disable-msg=C0103
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.preemphasis = preemphasis
        self.n_fft = n_fft
        self.trim = trim
        self.ref_db = ref_db
        self.max_db = max_db

    @overrides
    def transform(self, fpath):
        """Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
        Args:
        sound_file: A string. The full path of a sound file.
        Returns:
        mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
        """

        # Loading sound file
        audio, _ = librosa.load(fpath, sr=self.sr)

        hop_length = int(self.sr * self.frame_shift)
        win_length = int(self.sr * self.frame_length)
        # Trimming
        if self.trim:
            audio, _ = librosa.effects.trim(audio)

        # Preemphasis
        audio = np.append(audio[0], audio[1:] - self.preemphasis * audio[:-1])

        # stft
        linear = librosa.stft(y=audio,
                              n_fft=self.n_fft,
                              hop_length=hop_length,
                              win_length=win_length)

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)

        # to decibel
        mag = 20 * np.log10(np.maximum(1e-5, mag))

        # normalize
        mag = np.clip((mag - self.ref_db + self.max_db) / self.max_db, 1e-8, 1)

        # Transpose
        mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

        return mag
