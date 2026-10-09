# face-recog

A maintained fork of Adam Geitgey's [face_recognition](https://github.com/ageitgey/face_recognition)
and [face_recognition_models](https://github.com/ageitgey/face_recognition_models), both
unmaintained since 2020 and 2017. One install provides both import packages, so code written
for upstream works unchanged:

```python
import face_recognition

image = face_recognition.load_image_file("photo.jpg")
encodings = face_recognition.face_encodings(image)
```

## What's different from upstream

- **Works on current setuptools.** Model paths come from `importlib.resources`; upstream used
  `pkg_resources`, which setuptools 82 removed, so `import face_recognition` crashed.
- **Models load on first use.** Upstream loaded all four dlib models at import time (about 1 s
  and 110 MB). Now each loads when a function first needs it, and is then reused. The default
  `face_encodings` call only ever loads the 5-point predictor and the encoder (about 31 MB);
  the 68-point predictor loads only for `face_landmarks()` or `model="large"`, and the CNN
  detector only for `model="cnn"` or `batch_face_locations()`.
- **Typed.** Full type hints and a `py.typed` marker, so mypy checks calls into it.
- **No CLI.** The `face_recognition` and `face_detection` commands (and the Click dependency) are
  gone; this is a library only.
- **No `quit()`.** Upstream called `quit()` if the models package failed to import, killing the
  whole process. A failure now raises an ordinary `ImportError`.
- `compare_faces` returns plain Python `bool`s instead of `numpy.bool_`.

The algorithms, defaults and results are unchanged: the test suite checks that encodings are
identical to those from upstream 1.3.0, so encodings stored by the upstream version still match.

The old module attributes (`face_recognition.api.face_encoder`, `pose_predictor_68_point`, ...)
still work; reading one loads that model.

## Install

```bash
uv add "face-recog @ git+https://github.com/kieranrees/face-recog@v0.4.0"
```

dlib has no wheels on PyPI, so installing builds it from source: you need CMake and a C++
compiler.

## Development

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check . && uv run ruff format --check .
```

## Model files

`face_recognition_models/models/` holds the four pre-trained dlib models. The 68-point predictor
is 99.7 MB, just under GitHub's 100 MB file limit; the files never change, so they are kept in
plain git rather than Git LFS. `tests/test_models.py` checks their SHA-256 hashes.

## Licences

- `face_recognition`: MIT, see `LICENSE-face_recognition`.
- `face_recognition_models` and the model files: CC0 1.0, see `LICENSE`. The models come from
  [dlib](http://dlib.net/); see dlib's notes on the training data of the 68-point predictor
  (iBUG 300-W) before using it commercially.
