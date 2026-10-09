"""Face detection, landmarks and 128-d encodings on top of dlib.

The algorithms and results are those of upstream face_recognition 1.3.0. The
difference is that each dlib model loads the first time a function needs it,
rather than all four at import time (~1 s and ~110 MB for models a caller may
never use: the default encoding path needs only the 5-point predictor and the
ResNet encoder).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import IO, Any, Literal

import dlib
import numpy as np
import PIL.Image
from numpy.typing import NDArray
from PIL import ImageFile

import face_recognition_models

ImageFile.LOAD_TRUNCATED_IMAGES = True

# An image as returned by load_image_file: (height, width, 3) for RGB, (height, width) for L
Image = NDArray[np.uint8]
# A face bounding box in css order: (top, right, bottom, left)
FaceLocation = tuple[int, int, int, int]
# A 128-dimensional face encoding
FaceEncoding = NDArray[np.float64]
Point = tuple[int, int]

DetectionModel = Literal["hog", "cnn"]
LandmarkModel = Literal["large", "small"]


@cache
def get_face_detector() -> Any:
    """The HOG frontal face detector (built into dlib, no model file)."""
    return dlib.get_frontal_face_detector()


@cache
def get_cnn_face_detector() -> Any:
    """The CNN (MMOD) face detector."""
    return dlib.cnn_face_detection_model_v1(face_recognition_models.cnn_face_detector_model_location())


@cache
def get_pose_predictor_68_point() -> Any:
    """The 68-point landmark predictor (~100 MB)."""
    return dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())


@cache
def get_pose_predictor_5_point() -> Any:
    """The 5-point landmark predictor."""
    return dlib.shape_predictor(face_recognition_models.pose_predictor_five_point_model_location())


@cache
def get_face_encoder() -> Any:
    """The ResNet face recognition model that produces 128-d encodings."""
    return dlib.face_recognition_model_v1(face_recognition_models.face_recognition_model_location())


# Upstream exposed the loaded models as module attributes. Keep those names
# working for any caller that used them, without loading anything at import.
_LEGACY_ATTRIBUTES = {
    "face_detector": get_face_detector,
    "cnn_face_detector": get_cnn_face_detector,
    "pose_predictor_68_point": get_pose_predictor_68_point,
    "pose_predictor_5_point": get_pose_predictor_5_point,
    "face_encoder": get_face_encoder,
}


def __getattr__(name: str) -> Any:
    if name in _LEGACY_ATTRIBUTES:
        return _LEGACY_ATTRIBUTES[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _rect_to_css(rect: Any) -> FaceLocation:
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css: FaceLocation) -> Any:
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css: FaceLocation, image_shape: tuple[int, ...]) -> FaceLocation:
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def face_distance(
    face_encodings: Sequence[FaceEncoding] | NDArray[np.float64], face_to_compare: FaceEncoding
) -> NDArray[np.float64]:
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty(0)

    return np.linalg.norm(np.asarray(face_encodings) - face_to_compare, axis=1)


def load_image_file(file: str | IO[bytes], mode: str = "RGB") -> Image:
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im: PIL.Image.Image = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _raw_face_locations(img: Image, number_of_times_to_upsample: int = 1, model: DetectionModel = "hog") -> Any:
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return get_cnn_face_detector()(img, number_of_times_to_upsample)
    return get_face_detector()(img, number_of_times_to_upsample)


def face_locations(
    img: Image, number_of_times_to_upsample: int = 1, model: DetectionModel = "hog"
) -> list[FaceLocation]:
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if model == "cnn":
        return [
            _trim_css_to_bounds(_rect_to_css(face.rect), img.shape)
            for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")
        ]
    return [
        _trim_css_to_bounds(_rect_to_css(face), img.shape)
        for face in _raw_face_locations(img, number_of_times_to_upsample, model)
    ]


def _raw_face_locations_batched(
    images: Sequence[Image], number_of_times_to_upsample: int = 1, batch_size: int = 128
) -> Any:
    """
    Returns an 2d array of dlib rects of human faces in a image using the cnn face detector

    :param images: A list of images (each as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return get_cnn_face_detector()(images, number_of_times_to_upsample, batch_size=batch_size)


def batch_face_locations(
    images: Sequence[Image], number_of_times_to_upsample: int = 1, batch_size: int = 128
) -> list[list[FaceLocation]]:
    """
    Returns an 2d array of bounding boxes of human faces in a image using the cnn face detector
    If you are using a GPU, this can give you much faster results since the GPU
    can process batches of images at once. If you aren't using a GPU, you don't need this function.

    :param images: A list of images (each as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param batch_size: How many images to include in each GPU processing batch.
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """

    def convert_cnn_detections_to_css(detections: Any) -> list[FaceLocation]:
        return [_trim_css_to_bounds(_rect_to_css(face.rect), images[0].shape) for face in detections]

    raw_detections_batched = _raw_face_locations_batched(images, number_of_times_to_upsample, batch_size)

    return list(map(convert_cnn_detections_to_css, raw_detections_batched))


def _raw_face_landmarks(
    face_image: Image, face_locations: Sequence[FaceLocation] | None = None, model: LandmarkModel = "large"
) -> list[Any]:
    if face_locations is None:
        rects = _raw_face_locations(face_image)
    else:
        rects = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = get_pose_predictor_5_point() if model == "small" else get_pose_predictor_68_point()

    return [pose_predictor(face_image, rect) for rect in rects]


def face_landmarks(
    face_image: Image, face_locations: Sequence[FaceLocation] | None = None, model: LandmarkModel = "large"
) -> list[dict[str, list[Point]]]:
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    if model not in ("large", "small"):
        raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")

    landmarks = _raw_face_landmarks(face_image, face_locations, model)
    landmarks_as_tuples: list[list[Point]] = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    if model == "large":
        return [
            {
                "chin": points[0:17],
                "left_eyebrow": points[17:22],
                "right_eyebrow": points[22:27],
                "nose_bridge": points[27:31],
                "nose_tip": points[31:36],
                "left_eye": points[36:42],
                "right_eye": points[42:48],
                "top_lip": [*points[48:55], points[64], points[63], points[62], points[61], points[60]],
                "bottom_lip": [*points[54:60], points[48], points[60], points[67], points[66], points[65], points[64]],
            }
            for points in landmarks_as_tuples
        ]
    return [
        {
            "nose_tip": [points[4]],
            "left_eye": points[2:4],
            "right_eye": points[0:2],
        }
        for points in landmarks_as_tuples
    ]


def face_encodings(
    face_image: Image,
    known_face_locations: Sequence[FaceLocation] | None = None,
    num_jitters: int = 1,
    model: LandmarkModel = "small",
) -> list[FaceEncoding]:
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" or "small" (default) which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    face_encoder = get_face_encoder()
    return [
        np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
        for raw_landmark_set in raw_landmarks
    ]


def compare_faces(
    known_face_encodings: Sequence[FaceEncoding] | NDArray[np.float64],
    face_encoding_to_check: FaceEncoding,
    tolerance: float = 0.6,
) -> list[bool]:
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return [bool(match) for match in face_distance(known_face_encodings, face_encoding_to_check) <= tolerance]
