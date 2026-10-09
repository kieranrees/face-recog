"""Models load on first use, and only the ones a call needs."""

import subprocess
import sys
import textwrap

from face_recognition import api

LOADERS = {
    "detector": "get_face_detector",
    "cnn": "get_cnn_face_detector",
    "68": "get_pose_predictor_68_point",
    "5": "get_pose_predictor_5_point",
    "encoder": "get_face_encoder",
}


def _loaded_after(code: str) -> set[str]:
    """Run `code` in a fresh interpreter and return which models it loaded."""
    script = textwrap.dedent(f"""
        from pathlib import Path
        import face_recognition
        from face_recognition import api
        IMAGES = Path({str(api.__file__)!r}).parent.parent / "tests" / "test_images"
        {code}
        loaders = {LOADERS!r}
        print(",".join(k for k, fn in loaders.items() if getattr(api, fn).cache_info().currsize))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    return set(output.split(",")) if output else set()


def test_import_loads_no_models() -> None:
    assert _loaded_after("pass") == set()


def test_default_encoding_loads_only_what_it_needs() -> None:
    loaded = _loaded_after(
        'face_recognition.face_encodings(face_recognition.load_image_file(str(IMAGES / "obama.jpg")))'
    )

    assert loaded == {"detector", "5", "encoder"}


def test_large_landmarks_load_the_68_point_predictor() -> None:
    loaded = _loaded_after(
        'face_recognition.face_landmarks(face_recognition.load_image_file(str(IMAGES / "obama.jpg")))'
    )

    assert loaded == {"detector", "68"}


def test_models_are_loaded_once() -> None:
    assert api.get_face_encoder() is api.get_face_encoder()


def test_legacy_module_attributes_still_work() -> None:
    assert api.face_encoder is api.get_face_encoder()
    assert api.pose_predictor_68_point is api.get_pose_predictor_68_point()
