from allennlp.data import DatasetReader


from overrides import overrides

class SpeechDatasetReader(DatasetReader):
    """
    A SpeechDatasetReader Base
    """
    def audio_to_instance(self, *inputs):
        raise NotImplementedError

    @overrides
    def text_to_instance(self, *inputs):
        return self.audio_to_instance(*inputs)
