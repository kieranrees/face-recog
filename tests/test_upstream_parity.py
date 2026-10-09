"""Encodings must match upstream face_recognition 1.3.0 exactly.

reference_encodings_1_3_0.npz was generated with upstream 1.3.0 (and dlib 20.0.1). If these
drift, encodings stored by the upstream version would no longer match new ones.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from face_recognition import api

REFERENCE = np.load(Path(__file__).parent / "reference_encodings_1_3_0.npz")


@pytest.mark.parametrize("name", ["obama", "obama2", "obama3", "biden"])
@pytest.mark.parametrize("model", ["small", "large"])
def test_encoding_matches_upstream(load: Callable[[str], api.Image], name: str, model: api.LandmarkModel) -> None:
    (encoding,) = api.face_encodings(load(f"{name}.jpg"), model=model)

    np.testing.assert_allclose(encoding, REFERENCE[f"{name}_{model}"], rtol=0, atol=1e-6)
