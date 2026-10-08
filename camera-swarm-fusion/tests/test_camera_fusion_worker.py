from pathlib import Path

from camera_fusion.config import CameraConfig
from camera_fusion.worker import CameraWorker

import cv2
import csv
import time
import numpy as np

def test_worker_dry_run_creates_session(tmp_path: Path, monkeypatch):
    def forbid_camera(*args, **kwargs):
        raise AssertionError("dry_run attempted to open a real camera")

    monkeypatch.setattr(cv2, "VideoCapture", forbid_camera)

    cfg = CameraConfig(
        camera_name="drycam",
        session_root=str(tmp_path),
        dry_run=True,
        max_frames=2,
        save_frames=False,
        save_annotated=False,
        no_detect=True,
    )

    worker = CameraWorker(cfg)
    summary = worker.run()

    assert summary.frames_processed == 2
    assert summary.errors == 0
    assert Path(summary.session_path).exists()
    assert Path(summary.csv_path).exists()
    assert Path(summary.log_path).exists()


def test_worker_logs_failed_read(tmp_path: Path, monkeypatch):

    class FakeFrameSource:
        def __init__(self):
            self.calls = 0

        def start(self):
            pass

        def read(self):
            self.calls += 1

            if self.calls == 1:
                return None

            frame_id = self.calls - 1
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            return image, time.time_ns(), frame_id

        def stop(self):
            pass

    cfg = CameraConfig(
        camera_name="failure_test",
        session_root=str(tmp_path),
        dry_run=True,
        max_frames=2,
        save_frames=False,
        save_annotated=False,
        no_detect=True,
    )

    worker = CameraWorker(cfg)
    fake_source = FakeFrameSource()

    monkeypatch.setattr(
        worker, "_build_frame_source", lambda: fake_source
    )

    summary = worker.run()

    with open(
        Path(summary.session_path) / "attempts.csv",
        newline=""
    ) as file:
        records = list(csv.DictReader(file))

    assert summary.frames_processed == 2
    assert summary.errors == 1
    assert fake_source.calls == 3

    assert [row["capture_status"] for row in records] == [
        "capture_failed",
        "captured",
        "captured",
    ]

    assert records[0]["frame_id"] == ""



import pytest
from unittest.mock import Mock


@pytest.mark.parametrize('failure', [RuntimeError('read failed'), KeyboardInterrupt()])
def test_read_exception_is_logged_and_propagated(tmp_path, monkeypatch, failure):
    import camera_fusion.worker as module
    source = Mock()
    source.read.side_effect = failure
    output = Mock()
    loggers = []
    real_logger = module.AttemptLogger

    def make_logger(directory):
        logger = real_logger(directory)
        loggers.append(logger)
        return logger

    monkeypatch.setattr(module, 'AttemptLogger', make_logger)
    ticks = iter([100, 200])
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: next(ticks))
    worker = CameraWorker(CameraConfig(session_root=str(tmp_path), dry_run=True),
                          frame_source=source, outputs=[output])
    with pytest.raises(type(failure)):
        worker.run()
    source.stop.assert_called_once()
    output.close.assert_called_once()
    assert loggers[0]._file.closed
    with loggers[0].csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['capture_status'] == 'capture_failed'
    assert rows[0]['frame_id'] == ''
    assert (rows[0]['read_start_ns'], rows[0]['read_end_ns']) == ('100', '200')


@pytest.mark.parametrize('stage', ['output', 'calibration', 'start'])
def test_setup_failure_closes_resources(tmp_path, monkeypatch, stage):
    import camera_fusion.worker as module
    source, output = Mock(), Mock()
    logger = module.AttemptLogger(tmp_path)
    monkeypatch.setattr(module, 'AttemptLogger', lambda directory: logger)
    if stage == 'output':
        output.open.side_effect = RuntimeError('output')
    elif stage == 'calibration':
        monkeypatch.setattr(module, 'load_calib', Mock(side_effect=RuntimeError('calibration')))
    else:
        source.start.side_effect = RuntimeError('start')
    worker = CameraWorker(CameraConfig(session_root=str(tmp_path), dry_run=stage != 'calibration'),
                          frame_source=source, outputs=[output])
    with pytest.raises(RuntimeError, match=stage):
        worker.run()
    assert logger._file.closed
    output.close.assert_called_once()
    source.read.assert_not_called()
    if stage != 'output':
        source.stop.assert_called_once()


def test_wall_clock_rollback_does_not_invalidate_read_duration(tmp_path, monkeypatch):
    import camera_fusion.worker as module
    source = Mock()
    def read():
        # A wall-clock adjustment must not reverse acquisition interval endpoints.
        monkeypatch.setattr(module.time, 'time_ns', lambda: 1)
        return np.zeros((10, 10, 3), dtype=np.uint8), 1, 1
    source.read.side_effect = read
    monkeypatch.setattr(module.time, 'time_ns', lambda: 999999)
    ticks = iter([100, 200])
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: next(ticks))
    cfg = CameraConfig(session_root=str(tmp_path), dry_run=True, max_frames=1,
                       save_frames=False, save_annotated=False)
    summary = CameraWorker(cfg, frame_source=source).run()
    with (Path(summary.session_path) / 'attempts.csv').open(newline='') as stream:
        row = next(csv.DictReader(stream))
    assert int(row['read_end_ns']) - int(row['read_start_ns']) == 100
    assert row['capture_status'] == 'captured'
    import json
    manifest = json.loads((Path(summary.session_path) / 'config.json').read_text())
    assert manifest['attempt_timing']['clock'] == 'time.monotonic_ns'
