"""Regression tests for the FastAPI app (form field names, temp-file safety)."""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from vindr.app import app

    return TestClient(app)


@pytest.fixture()
def xray(tmp_path) -> Path:
    """Synthetic chest X-ray-like grayscale image."""
    rng = np.random.default_rng(0)
    img = np.full((256, 256), 30, np.uint8)
    img[60:196, 90:166] = 180  # bright mediastinum
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    path = tmp_path / "cxr.jpg"
    Image.fromarray(img).save(path)
    return path


def test_checkbox_field_name_matches_endpoint_signature(client):
    """The HTML checkbox must send the name the endpoint actually reads.

    Regression: the form posted ``name=detect`` while the endpoint expected
    ``detect_enabled``, so ticking the box silently did nothing.
    """
    import inspect

    from vindr.app import predict

    html = client.get("/").text
    assert "type=checkbox" in html
    param = inspect.signature(predict).parameters
    assert "detect_enabled" in param, "endpoint must read detect_enabled"
    assert "name=detect_enabled" in html, "checkbox name must match the endpoint parameter"


def test_predict_without_detection_does_not_load_detector(client, xray, monkeypatch):
    """Without the checkbox the detector must not be initialised."""
    import vindr.app as app_mod

    called = {"n": 0}

    def boom():
        called["n"] += 1
        raise RuntimeError("detector must not load when the checkbox is unticked")

    monkeypatch.setattr(app_mod, "load_default_detector", boom)
    with open(xray, "rb") as fh:
        r = client.post("/predict", files={"file": fh}, data={"lang": "ru"})
    assert r.status_code == 200
    assert called["n"] == 0


def test_predict_with_detection_uses_unique_temp_file(client, xray, monkeypatch, tmp_path):
    """Detector input must go to a unique temp file, not a shared path.

    Regression: a fixed ``_det_input.png`` was shared by all requests, so
    concurrent uploads could read each other's bytes.
    """
    import vindr.app as app_mod
    import vindr.detect as detect_mod

    monkeypatch.setattr(detect_mod, "_default_weights", None)
    monkeypatch.setattr(
        app_mod, "load_default_detector", lambda *a, **k: object()
    )
    seen: list[Path] = []

    def fake_detect(_model, path):
        seen.append(Path(path))
        return []

    def fake_overlay(path, _finds):
        arr = np.zeros((64, 64, 3), np.uint8)
        Image.fromarray(arr).save(path)
        return arr

    monkeypatch.setattr(app_mod, "detect", fake_detect)
    monkeypatch.setattr(app_mod, "render_overlay", fake_overlay)
    monkeypatch.setattr(app_mod, "read_image", lambda p: np.zeros((64, 64), np.uint8))

    with open(xray, "rb") as fh:
        r = client.post(
            "/predict", files={"file": fh}, data={"lang": "ru", "detect_enabled": "on"}
        )
    assert r.status_code == 200
    assert len(seen) == 1
    assert seen[0].name != "_det_input.png"
    assert not seen[0].exists(), "temp file must be cleaned up"
    assert not list(Path(app_mod.__file__).parent.glob("_det_input*"))


def test_concurrent_detect_requests_do_not_collide(client, xray, monkeypatch, tmp_path):
    """Parallel uploads must each get their own detector input file."""
    import vindr.app as app_mod

    monkeypatch.setattr(app_mod, "load_default_detector", lambda *a, **k: object())

    active: list[Path] = []
    lock_collisions = {"n": 0}

    def fake_detect(_model, path):
        p = Path(path)
        # simulate slow inference so overlapping requests are likely
        import time

        active.append(p)
        if len(set(active)) != len(active):
            lock_collisions["n"] += 1
        time.sleep(0.05)
        active.remove(p)
        return []

    monkeypatch.setattr(app_mod, "detect", fake_detect)
    monkeypatch.setattr(app_mod, "render_overlay", lambda path, _f: np.zeros((64, 64, 3), np.uint8))
    monkeypatch.setattr(app_mod, "read_image", lambda p: np.zeros((64, 64), np.uint8))

    def hit(_i):
        with open(xray, "rb") as fh:
            return client.post(
                "/predict", files={"file": fh}, data={"lang": "ru", "detect_enabled": "on"}
            ).status_code

    with cf.ThreadPoolExecutor(6) as ex:
        codes = list(ex.map(hit, range(6)))

    assert all(c == 200 for c in codes)
    assert lock_collisions["n"] == 0, "two requests shared the same temp file"
