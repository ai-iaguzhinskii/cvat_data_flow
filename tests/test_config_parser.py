import sys
from pathlib import Path

import pytest


@pytest.fixture
def config(tmp_path):
    config_content = """
[TEST]
LIST = [1, 2, 3]
DICT = {'a': 1, 'b': 2}
BOOL = True
"""
    config_file = tmp_path / "temp.ini"
    config_file.write_text(config_content)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cvat_data_flow" / "src"))
    from utils.config_parser import Config
    cfg = Config(str(config_file))
    yield cfg
    sys.path.pop(0)


def test_get_list(config):
    assert config.get('TEST', 'LIST', list) == ['1', '2', '3']


def test_get_dict(config):
    assert config.get('TEST', 'DICT', dict) == {'a': 1, 'b': 2}


def test_get_bool(config):
    assert config.get('TEST', 'BOOL', bool) is True
