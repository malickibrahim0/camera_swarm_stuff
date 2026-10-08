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


@pytest.mark.parametrize('visible', [True, False])
def test_detection_only_uses_real_detector_without_calibration(tmp_path, monkeypatch, visible):
    import json
    import camera_fusion.worker as module
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    if visible:
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)
        image[50:250, 50:250] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    source = Mock()
    source.read.return_value = (image, time.time_ns(), 1)
    def forbidden(*args, **kwargs):
        raise AssertionError('Detection-only must not load calibration or estimate pose')
    monkeypatch.setattr(module, 'load_calib', forbidden)
    monkeypatch.setattr(module, 'PnPLocalize', forbidden)
    monkeypatch.setattr(module, 'SimpleUndistort', forbidden)
    cfg = CameraConfig(session_root=str(tmp_path), detection_only=True, max_frames=1,
                       target_ids=[0], marker_lengths_m={0: 0.035})
    summary = CameraWorker(cfg, frame_source=source).run()
    assert summary.frames_processed == 1
    source.start.assert_called_once()
    source.stop.assert_called_once()
    session = Path(summary.session_path)
    with Path(summary.csv_path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == int(visible)
    if visible:
        assert rows[0]['marker_id'] == '0'
        assert all(rows[0][field] == 'nan' for field in ('rvec_x', 'tvec_x', 'tvec_y', 'tvec_z'))
    with (session / 'metrics_frames.csv').open(newline='') as f:
        metrics = next(csv.DictReader(f))
    assert metrics['pose_recovered'] == '0'
    assert metrics['aruco_success'] == str(int(visible))
    assert json.loads((session / 'config.json').read_text())['detection_only'] is True
    assert len(list((session / 'frames').glob('*.jpg'))) == 1


def test_detection_only_config_and_cli(tmp_path):
    import json
    from camera_fusion.config import load_config
    from camera_fusion.run import _build_parser, _apply_args
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'detection_only': True}))
    assert load_config(path).detection_only
    args = _build_parser().parse_args(['--config', str(path), '--detection-only'])
    assert _apply_args(CameraConfig(), args).detection_only
