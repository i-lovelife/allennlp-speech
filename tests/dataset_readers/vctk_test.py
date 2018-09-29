import json
from pathlib import Path
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.common.util import ensure_list
from src.common.speech_test_case import SpeechTestCase
from src.dataset_readers.vctk import VCTK

VCTK_ROOT = Path('/home/lnan6257/work/dataset/speech/VCTK-Corpus')
class TestVCTK(SpeechTestCase):
    def get_instance(self):
        dataset_path = self.FIXTURES_ROOT / 'vctk' / 'dataset_list.json'
        cache_path = self.TEST_DIR / 'data-small.hdf5'
        dataset_reader = VCTK(cache_path=cache_path)
        instances = ensure_list(dataset_reader.read(dataset_path))
        return instances
        
    def test_read_from_file(self):
        # pylint: disable-msg=c0330
        instances = self.get_instance()
        assert len(instances) == 4
        assert instances[0].fields['feature'].array.shape[1] == (513)
        assert instances[0].fields['feature_length'].label == instances[0].fields['feature'].array.shape[0]
        assert ''.join(list(map(lambda x: x.text, instances[0].fields['txt_label'].tokens))) == \
               "i haven't enjoyed the last couple of years "
        assert instances[0].fields['sex_label'].label == 0
        assert instances[0].fields['txt_length'].label == 43
        assert (instances[0].fields['meta_data'].metadata == \
                {
                    "speaker_id": "345",
                    "chapter_id": "046",
                    'txt_label': "i haven't enjoyed the last couple of years ",
                    "sex_label": 0,
                    "txt": "I haven't enjoyed the last couple of years.",
                    "audio_file": "/home/lnan6257/work/dataset/speech/VCTK-Corpus/wav48/p345/p345_046.wav"
                }
               )

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
    