# -*- coding: utf-8 -*-

__author__ = """Adam Geitgey"""
__email__ = 'ageitgey@gmail.com'
__version__ = '0.3.1'

# importlib.resources instead of pkg_resources, which setuptools 82 removed.
# The package is installed unzipped (zip_safe=False), so these are real paths.
from importlib.resources import files


def _model_path(filename):
    return str(files(__name__) / "models" / filename)


def pose_predictor_model_location():
    return _model_path("shape_predictor_68_face_landmarks.dat")

def pose_predictor_five_point_model_location():
    return _model_path("shape_predictor_5_face_landmarks.dat")

def face_recognition_model_location():
    return _model_path("dlib_face_recognition_resnet_model_v1.dat")

def cnn_face_detector_model_location():
    return _model_path("mmod_human_face_detector.dat")
