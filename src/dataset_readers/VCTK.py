import json
import re
from pathlib import Path
from typing import Dict, Iterable, Union, Tuple
import glob
import numpy as np

import h5py
from allennlp.data import Instance
from allennlp.data.fields import (ArrayField, Field, LabelField, MetadataField,
                                  TextField)
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from overrides import overrides

from src.data.audio_transformer import AudioTransformer
from src.dataset_readers.speech_dataset_reader import SpeechDatasetReader


class VCTK(SpeechDatasetReader):
    def __init__(self,
                 cache_path: str,
                 audio_transformer: AudioTransformer,
                 feature_name: str = 'mag',
                 lazy: bool = False) -> None:
        super(VCTK, self).__init__(lazy)
        self.cache_path = cache_path
        self.audio_transformer = audio_transformer
        self.feature_name = feature_name
        self.tokenizer = CharacterTokenizer(lowercase_characters=True)
        self.token_indexers = {'character': SingleIdTokenIndexer(namespace='character')}

    @overrides
    def audio_to_instance(self, *inputs):
        pass

    @overrides
    def _read(self, file_path: Union[str, Path]) -> Iterable[Instance]:
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)

        cache_file = h5py.File(self.cache_path, 'a')
        for example in dataset:
            fields: Dict[str, Field] = {}
            audio_file = example['audio_file']
            meta_data = example['metadata']
            sex_label = meta_data['sex']
            txt_label = meta_data['txt']
            speaker_id = meta_data['speaker_id']
            chapter_id = meta_data['chapter_id']

            location = f'{speaker_id}/{chapter_id}/{self.feature_name}'
            if location in cache_file:
                feature = cache_file[location].value
            else:
                feature = self.audio_transformer(audio_file)
                cache_file.create_dataset(location, data=feature, dtype=np.float32)
            fields['feature'] = ArrayField(feature)
            filtered_txt = [c if re.match(r"[a-zA-Z']", c) else ' ' for c in txt_label]
            fields['txt_label'] = TextField(self.tokenizer.tokenize(filtered_txt), self.token_indexers)
            fields['sex_label'] = LabelField(sex_label, label_namespace='sex_labels', skip_indexing=True)
            fields['meta_data'] = MetadataField(meta_data)
            yield Instance(fields)
        cache_file.close()

    @staticmethod
    def generate_dataset(dataset_root: Union[str, Path],
                         force: bool = False) -> Tuple[Path, Path, Path]:
        """
        Generate data for train/dev/test
        [{
            'audio_file':
            'meta_data':{
                'speaker_id':
                'chapter_id':
                'sex':
                'txt':
            }
        }]
        """
        dataset_root = Path(dataset_root)
        generate_dir = dataset_root / 'generated'
        generate_dir.mkdir(exist_ok=True)

        dataset_train = generate_dir / 'dataset_list.train'
        dataset_dev = generate_dir / 'dataset_list.dev'
        dataset_test = generate_dir / 'dataset_list.test'

        generated_sign = generate_dir / 'generate.done'
        # Check generated before
        if not force and generated_sign.exists():
            return (dataset_train, dataset_dev, dataset_test)

        speaker_path = dataset_root / 'speaker-info.txt'
        with open(speaker_path) as speaker_file:
            speakers = speaker_file.readlines()

        def reform(lst):
            lst = [x for x in lst if x != '']
            ret = {'id': int(lst[0]),
                   'age': int(lst[1]),
                   'sex': 1 if lst[2] == 'F' else 0,
                   'accents': lst[3]
                  }
            return ret

        speakers = [reform(x.strip().split(' ')) for x in speakers[1:]] # type: ignore
        speakers = [speaker for speaker in speakers if speaker['id'] != 315] # type: ignore

        girls = [x for x in speakers if x['sex'] == 1] # type: ignore
        boys = [x for x in speakers if x['sex'] == 0] # type: ignore

        def split_data(data):
            train_len = int(len(data) * 0.8)
            dev_len = int(len(data) * 0.1)
            return (data[:train_len], data[train_len: train_len + dev_len], data[train_len + dev_len:])

        girls_train, girls_dev, girls_test = split_data(girls)
        boys_train, boys_dev, boys_test = split_data(boys)

        def add_example(examples, speakers):
            # pylint: disable-msg=c0330
            for speaker in speakers:
                idx = speaker['id']
                files = glob.glob(str(dataset_root) + '/wav48/p' + str(idx) + '/*.wav')
                for file in files:
                    txt_file_path = file.replace('wav48', 'txt').replace('wav', 'txt')
                    with open(txt_file_path) as txt_file:
                        txt = txt_file.readline().strip().replace('\n', '')
                    speaker_id = str(idx)
                    speaker_chapter = file.split('.')[0].split('_')[1]
                    example = {}
                    example['audio_file'] = file
                    example['meta_data'] = {
                        'speaker_id': speaker_id,
                        'chapter_id': speaker_chapter,
                        'sex': speaker['sex'],
                        'txt': txt,
                    }

                    examples.append(example)

        example_train, example_dev, example_test = [], [], [] # type: ignore
        add_example(example_train, girls_train)
        add_example(example_train, boys_train)
        add_example(example_dev, girls_dev)
        add_example(example_dev, boys_dev)
        add_example(example_test, girls_test)
        add_example(example_test, boys_test)

        def dump_to_file(obj, file_path):
            with open(file_path) as file:
                json.dump(obj, file)
        dump_to_file(example_train, dataset_train)
        dump_to_file(example_dev, dataset_dev)
        dump_to_file(example_test, dataset_test)
        generated_sign.touch()
        return (dataset_train, dataset_dev, dataset_test)
