"""The bundled model files are present and unchanged."""

import hashlib
from pathlib import Path

import pytest

import face_recognition_models

EXPECTED_SHA256 = {
    face_recognition_models.pose_predictor_model_location: (
        "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"
    ),
    face_recognition_models.pose_predictor_five_point_model_location: (
        "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888"
    ),
    face_recognition_models.face_recognition_model_location: (
        "55533b28a95800a551ba546ba62fe69625c7e95a7061c338adffead08719da30"
    ),
    face_recognition_models.cnn_face_detector_model_location: (
        "4cb19393e2fbaf2b1609a9319ad5386618c886a6234ec1b971f3e87c85d87fe6"
    ),
}


@pytest.mark.parametrize("location", EXPECTED_SHA256, ids=lambda fn: fn.__name__)
def test_model_file_is_unchanged(location: object) -> None:
    assert callable(location)
    path = Path(location())

    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256[location]
