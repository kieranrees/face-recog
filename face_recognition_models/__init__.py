"""Paths to the pre-trained dlib model files that face_recognition loads."""

# importlib.resources instead of pkg_resources, which setuptools 82 removed.
# The package is installed unzipped, so these are real filesystem paths.
from importlib.resources import files

__author__ = """Adam Geitgey"""
__email__ = "ageitgey@gmail.com"
__version__ = "0.4.0"


def _model_path(filename: str) -> str:
    return str(files(__name__) / "models" / filename)


def pose_predictor_model_location() -> str:
    """Path to the 68-point landmark predictor."""
    return _model_path("shape_predictor_68_face_landmarks.dat")


def pose_predictor_five_point_model_location() -> str:
    """Path to the 5-point landmark predictor."""
    return _model_path("shape_predictor_5_face_landmarks.dat")


def face_recognition_model_location() -> str:
    """Path to the ResNet face recognition (encoding) model."""
    return _model_path("dlib_face_recognition_resnet_model_v1.dat")


def cnn_face_detector_model_location() -> str:
    """Path to the CNN (MMOD) face detector."""
    return _model_path("mmod_human_face_detector.dat")
