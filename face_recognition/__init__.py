"""Face recognition on top of dlib: a maintained fork of ageitgey/face_recognition."""

from .api import (
    batch_face_locations,
    compare_faces,
    face_distance,
    face_encodings,
    face_landmarks,
    face_locations,
    load_image_file,
)

__author__ = """Adam Geitgey"""
__email__ = "ageitgey@gmail.com"
__version__ = "0.4.0"

__all__ = [
    "batch_face_locations",
    "compare_faces",
    "face_distance",
    "face_encodings",
    "face_landmarks",
    "face_locations",
    "load_image_file",
]
