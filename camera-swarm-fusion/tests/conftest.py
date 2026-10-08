import sys
import types


def _register_facade_sdk_stub():
    """Stub out the facade_sdk client so CLI/system tests can import it."""
    if "facade_sdk" in sys.modules:
        return

    module = types.ModuleType("facade_sdk")

    class _DummyClient:
        def __init__(self, **_):
            self.published = []

        def publish(self, line):
            self.published.append(line)

        def close(self):
            pass

    module.Client = _DummyClient
    sys.modules["facade_sdk"] = module


# Geometry and image tests require real OpenCV; never replace it globally.
import cv2
import pytest

@pytest.fixture(autouse=True)
def forbid_unmocked_camera(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must inject a frame source or explicitly mock VideoCapture")
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)

_register_facade_sdk_stub()
