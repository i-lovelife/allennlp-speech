from pathlib import Path
import shutil
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
PROJECT_ROOT = Path('/home/lnan6257/work/allennlp-speech')
SERIALIZATION_DIR = Path('/tmp/allennlp-speech/vctk-small/')
if __name__ == '__main__':
    import_submodules('src')
    parameter_filename = PROJECT_ROOT / 'experiments/vctk_asr/test.json'
    serialization_dir = SERIALIZATION_DIR
    if serialization_dir.exists():
        shutil.rmtree(serialization_dir)
    recover = False
    train_model_from_file(str(parameter_filename), str(serialization_dir), recover=recover)
