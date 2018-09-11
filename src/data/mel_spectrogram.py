from overrides import overrides
import librosa
from numpy import np

from src.data.audio_transformer import AudioTransformer

@AudioTransformer.register('mel_spec')
class MelSpectrogram(AudioTransformer):
    def __init__(self,
                sr: int = None,
                frame_shift: float = 0.0125,
                frame_length: float = 0.05,
                preemphasis: float = 0.97,
                n_fft: int = 1024,
                trim: bool = True,
                n_mels: int = 80,
                ref_db: int = 20,
                max_db: int = 100
                ) -> None:
        self.sr = sr
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.preemphasis = preemphasis
        self.n_fft = n_fft
        self.trim = trim
        self.n_mels = n_mels
        self.ref_db = ref_db
        self.max_db = max_db

    @overrides
    def transform(self, fpath):
        """Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
        Args:
        sound_file: A string. The full path of a sound file.
        Returns:
        mel: A 2d array of shape (T, n_mels) <- Transposed
        mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
        """

        # Loading sound file
        y, sr = librosa.load(fpath, sr=self.sr)

        hop_length = int(sr * self.frame_shift)
        win_length = int(sr * self.frame_length)
        # Trimming
        if self.trim:
            y, _ = librosa.effects.trim(y)

        # Preemphasis
        y = np.append(y[0], y[1:] - self.preemphasis * y[:-1])

        # stft
        linear = librosa.stft(y=y,
                            n_fft=self.n_fft,
                            hop_length=hop_length,
                            win_length=win_length)

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)

        # mel spectrogram
        mel_basis = librosa.filters.mel(self.sr, self.n_fft, self.n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, t)

        # to decibel
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mag = 20 * np.log10(np.maximum(1e-5, mag))

        # normalize
        mel = np.clip((mel - self.ref_db + self.max_db) / self.max_db, 1e-8, 1)# ref_db=20, max_db=100
        mag = np.clip((mag - self.ref_db + self.max_db) / self.max_db, 1e-8, 1)

        # Transpose
        mel = mel.T.astype(np.float32)  # (T, n_mels)
        mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

        return mel