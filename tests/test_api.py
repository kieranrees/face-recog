"""The upstream face_recognition test suite (minus the removed CLI), ported to pytest."""

from collections.abc import Callable

import numpy as np
import pytest

from face_recognition import api

Load = Callable[[str], api.Image]


def test_load_image_file(load: Load) -> None:
    assert load("obama.jpg").shape == (1137, 910, 3)


def test_load_image_file_32bit(load: Load) -> None:
    assert load("32bit.png").shape == (1200, 626, 3)


def test_raw_face_locations(load: Load) -> None:
    detected_faces = api._raw_face_locations(load("obama.jpg"))

    assert len(detected_faces) == 1
    assert detected_faces[0].top() == 142
    assert detected_faces[0].bottom() == 409


def test_cnn_raw_face_locations(load: Load) -> None:
    detected_faces = api._raw_face_locations(load("obama.jpg"), model="cnn")

    assert len(detected_faces) == 1
    assert detected_faces[0].rect.top() == pytest.approx(144, abs=25)
    assert detected_faces[0].rect.bottom() == pytest.approx(389, abs=25)


def test_raw_face_locations_32bit_image(load: Load) -> None:
    detected_faces = api._raw_face_locations(load("32bit.png"))

    assert len(detected_faces) == 1
    assert detected_faces[0].top() == 290
    assert detected_faces[0].bottom() == 558


def test_cnn_raw_face_locations_32bit_image(load: Load) -> None:
    detected_faces = api._raw_face_locations(load("32bit.png"), model="cnn")

    assert len(detected_faces) == 1
    assert detected_faces[0].rect.top() == pytest.approx(259, abs=25)
    assert detected_faces[0].rect.bottom() == pytest.approx(552, abs=25)


def test_face_locations(load: Load) -> None:
    assert api.face_locations(load("obama.jpg")) == [(142, 617, 409, 349)]


def test_cnn_face_locations(load: Load) -> None:
    detected_faces = api.face_locations(load("obama.jpg"), model="cnn")

    assert len(detected_faces) == 1
    assert detected_faces[0] == pytest.approx((144, 608, 389, 363), abs=25)


def test_partial_face_locations(load: Load) -> None:
    assert api.face_locations(load("obama_partial_face.jpg")) == [(142, 191, 365, 0)]
    assert api.face_locations(load("obama_partial_face2.jpg")) == [(142, 551, 409, 349)]


def test_raw_face_locations_batched(load: Load) -> None:
    img = load("obama.jpg")
    batched_detected_faces = api._raw_face_locations_batched([img, img, img], number_of_times_to_upsample=0)

    for detected_faces in batched_detected_faces:
        assert len(detected_faces) == 1
        assert detected_faces[0].rect.top() == 154
        assert detected_faces[0].rect.bottom() == 390


def test_batched_face_locations(load: Load) -> None:
    img = load("obama.jpg")
    batched_detected_faces = api.batch_face_locations([img, img, img], number_of_times_to_upsample=0)

    assert batched_detected_faces == [[(154, 611, 390, 375)]] * 3


def test_raw_face_landmarks(load: Load) -> None:
    face_landmarks = api._raw_face_landmarks(load("obama.jpg"))
    example_landmark = face_landmarks[0].parts()[10]

    assert len(face_landmarks) == 1
    assert face_landmarks[0].num_parts == 68
    assert (example_landmark.x, example_landmark.y) == (552, 399)


def test_face_landmarks(load: Load) -> None:
    face_landmarks = api.face_landmarks(load("obama.jpg"))

    assert set(face_landmarks[0]) == {
        "chin",
        "left_eyebrow",
        "right_eyebrow",
        "nose_bridge",
        "nose_tip",
        "left_eye",
        "right_eye",
        "top_lip",
        "bottom_lip",
    }
    assert face_landmarks[0]["chin"] == [
        (369, 220),
        (372, 254),
        (378, 289),
        (384, 322),
        (395, 353),
        (414, 382),
        (437, 407),
        (464, 424),
        (495, 428),
        (527, 420),
        (552, 399),
        (576, 372),
        (594, 344),
        (604, 314),
        (610, 282),
        (613, 250),
        (615, 219),
    ]


def test_face_landmarks_small_model(load: Load) -> None:
    face_landmarks = api.face_landmarks(load("obama.jpg"), model="small")

    assert set(face_landmarks[0]) == {"nose_tip", "left_eye", "right_eye"}
    assert face_landmarks[0]["nose_tip"] == [(496, 295)]


def test_face_landmarks_rejects_unknown_model(load: Load) -> None:
    with pytest.raises(ValueError, match="Invalid landmarks model type"):
        api.face_landmarks(load("obama.jpg"), model="medium")  # type: ignore[arg-type]


def test_face_encodings(load: Load) -> None:
    encodings = api.face_encodings(load("obama.jpg"))

    assert len(encodings) == 1
    assert len(encodings[0]) == 128


def test_face_encodings_large_model(load: Load) -> None:
    encodings = api.face_encodings(load("obama.jpg"), model="large")

    assert len(encodings) == 1
    assert len(encodings[0]) == 128


def _obama_and_biden_encodings(load: Load) -> tuple[api.FaceEncoding, list[api.FaceEncoding]]:
    """obama.jpg's encoding, and those of obama2.jpg, obama3.jpg and biden.jpg to compare it to."""
    target = api.face_encodings(load("obama.jpg"))[0]
    others = [api.face_encodings(load(name))[0] for name in ("obama2.jpg", "obama3.jpg", "biden.jpg")]
    return target, others


def test_face_distance(load: Load) -> None:
    target, others = _obama_and_biden_encodings(load)

    distance_results = api.face_distance(others, target)

    # 0.6 is the default match threshold: the two other Obama photos fall under it, Biden over it
    assert isinstance(distance_results, np.ndarray)
    assert distance_results[0] <= 0.6
    assert distance_results[1] <= 0.6
    assert distance_results[2] > 0.6


@pytest.mark.parametrize("empty", [[], np.array([])])
def test_face_distance_empty_lists(load: Load, empty: list[api.FaceEncoding] | np.ndarray) -> None:
    face_encoding = api.face_encodings(load("biden.jpg"))[0]

    distance_results = api.face_distance(empty, face_encoding)

    assert isinstance(distance_results, np.ndarray)
    assert len(distance_results) == 0


def test_compare_faces(load: Load) -> None:
    target, others = _obama_and_biden_encodings(load)

    assert api.compare_faces(others, target) == [True, True, False]


@pytest.mark.parametrize("empty", [[], np.array([])])
def test_compare_faces_empty_lists(load: Load, empty: list[api.FaceEncoding] | np.ndarray) -> None:
    face_encoding = api.face_encodings(load("biden.jpg"))[0]

    assert api.compare_faces(empty, face_encoding) == []
