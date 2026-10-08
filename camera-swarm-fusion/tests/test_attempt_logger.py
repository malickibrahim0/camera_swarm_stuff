import csv

from camera_fusion.attempt_logger import AttemptLogger


def test_attempt_logger(tmp_path):
    logger = AttemptLogger(tmp_path)

    logger.log_attempt(
        attempt_id=1,
        frame_id=1,
        read_start_ns=1000,
        read_end_ns=2000,
        capture_status="captured",
    )

    logger.log_attempt(
        attempt_id=2,
        frame_id=None,
        read_start_ns=3000,
        read_end_ns=4000,
        capture_status="capture_failed",
    )

    logger.close()

    with open(tmp_path / "attempts.csv", newline="") as file:
        records = list(csv.DictReader(file))

    assert len(records) == 2
    assert records[0]["capture_status"] == "captured"
    assert records[1]["capture_status"] == "capture_failed"
    assert records[1]["frame_id"] == ""

