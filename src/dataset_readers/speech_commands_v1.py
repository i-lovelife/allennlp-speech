from typing import Dict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, ArrayField, Field

from scipy.io import wavfile
from scipy import signal
import numpy as np
from overrides import overrides

@DatasetReader.register('speech_commands_v1')
class SpeechCommandsV1(DatasetReader):
    """
    A dataset reader for google speech commands dataset v0.01
    """
    def __init__(self, lazy: bool = False) -> None:
        super(SpeechCommandsV1, self).__init__(lazy)

    @overrides
    def _read(self, file_path):
        with open(file_path) as fh:
            file_list = fh.readlines()
        file_list = [s.strip() for s in file_list]
        for file_name in file_list:
            sample_rate, sample = wavfile.read(file_name)
            feature = SpeechCommandsV1.transform(sample, sample_rate)
            fields: Dict[str, Field] = {}
            lable = file_name.split('/')[0]
            fields['lable'] = LabelField(lable)
            fields['features'] = ArrayField(feature)
            yield Instance(fields)
    
    @staticmethod
    def transform(audio, sample_rate, window_size=20,
                  step_size=10, eps=1e-10) -> np.ndarray:
        """
        Transform audio to spectrogram
        """
        nperseg = int(round(window_size * sample_rate / 1e3))# nperseg = 320
        noverlap = int(round(step_size * sample_rate / 1e3))# noverlap = 160
        _, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        return np.log(spec.T.astype(np.float32) + eps)
