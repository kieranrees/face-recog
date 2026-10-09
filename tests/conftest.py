from collections.abc import Callable
from pathlib import Path

import pytest

from face_recognition import api

TEST_IMAGES = Path(__file__).parent / "test_images"


@pytest.fixture
def load() -> Callable[[str], api.Image]:
    """Load a test image by file name."""

    def _load(name: str) -> api.Image:
        return api.load_image_file(str(TEST_IMAGES / name))

    return _load
