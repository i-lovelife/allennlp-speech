from typing import Dict
from pathlib import Path
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, ArrayField, Field

from overrides import overrides

from src.data.audio_transformer import AudioTransformer
@DatasetReader.register('speech_commands_v1')
class SpeechCommandsV1(DatasetReader):
    """
    A dataset reader for google speech commands dataset v0.01
    """
    def __init__(self, 
                 data_dir: str,
                 transformer: AudioTransformer,
                 lazy: bool = False) -> None:
        """
        Parameters
        ----------
        data_dir: a root dir for all data. In case the path is relative path, this will be append
            for get data's absolute path
        """
        super(SpeechCommandsV1, self).__init__(lazy)
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"{data_dir} is not a dir")
        self.transformer = transformer

    @overrides
    def _read(self, file_path: str):
        """
        Read data list from file_path.
        Note `.train` should for train data, `.dev` for dev data and `.test` for test data
        Parameters
        ----------
        file_path: a file contains list of audio file
        """
        with open(file_path) as fh:
            file_list = fh.readlines()
        file_list = [s.strip() for s in file_list]
        
        for file_name in file_list:
            yield self.text_to_instance(file_name)

    @overrides
    def text_to_instance(self, file_name: str):
        feature = self.transformer.transform(str(self.data_dir / file_name))
        fields: Dict[str, Field] = {}
        label = file_name.split('/')[0]
        fields['label'] = LabelField(label)
        fields['features'] = ArrayField(feature)
        return Instance(fields)
