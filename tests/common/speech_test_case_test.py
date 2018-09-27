from pathlib import Path
from src.common.speech_test_case import SpeechTestCase

class TestSpeechTestCase:
    def test_path(self):
        assert SpeechTestCase.PROJECT_ROOT == (Path(__file__).parent / '..' / '..').resolve()
