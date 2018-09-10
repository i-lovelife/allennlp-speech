# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list

from src.dataset_readers.speech_commands_v1 import SpeechCommandsV1
from src import TEST_FIXTURES_ROOT, DATA_ROOT


class TestSpeechCommandsV1:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = SpeechCommandsV1(data_root=DATA_ROOT / 'speech_commands_v1', lazy=lazy)
        instances = ensure_list(reader.read(TEST_FIXTURES_ROOT / 'speech_commands_v1' / 'train.txt'))
        assert len(instances) == 5

        assert [instance.fields['label'].label for instance in instances] == ['nine', 'nine', 'down', 'nine', 'left']

        for instance in instances:
            assert instance.fields['features'].array.shape == (99, 161)
