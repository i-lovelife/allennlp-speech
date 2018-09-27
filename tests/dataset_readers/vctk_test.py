import json
from pathlib import Path
from src.common.speech_test_case import SpeechTestCase
from src.dataset_readers.vctk import VCTK

VCTK_ROOT = Path('/home/lnan6257/work/dataset/speech/VCTK-Corpus')
class TestVCTK(SpeechTestCase):
    def test_generate_dataset(self):
        train_path, dev_path, _ = VCTK.generate_dataset(VCTK_ROOT)
        def check_num(file_path, num):
            assert file_path.exists()
            with open(file_path) as file:
                examples = json.load(file)
            assert len(examples) == num
            for example in examples:
                audio_file = Path(example['audio_file'])
                assert audio_file.exists()
                meta_data = example['meta_data']
                assert 'speaker_id' in meta_data
                assert 'chapter_id' in meta_data
                assert 'sex' in meta_data
                assert 'txt' in meta_data
        check_num(train_path, 34293)
        check_num(dev_path, 4172)
