"""
Export some data root
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FIXTURES_ROOT = PROJECT_ROOT / 'tests' / 'fixtures'
DATA_ROOT = PROJECT_ROOT / 'datasets'
