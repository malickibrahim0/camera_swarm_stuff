
import csv
from pathlib import Path


class AttemptLogger:
    """Record acquisition attempts. Read timestamps use host monotonic nanoseconds.

    They measure host read duration, not wall-clock time or sensor exposure."""

    FIELDNAMES = [
        "run_id",
        "cycle_id",
        "pose_id",
        "attempt_id",
        "frame_id",
        "read_start_ns",
        "read_end_ns",
        "capture_status",
    ]

    def __init__(self, session_dir: str | Path):
        session_dir = Path(session_dir)
        self.run_id = session_dir.name
        self.csv_path = session_dir / "attempts.csv"

        # 'x' prevents overwriting an existing log.
        self._file = self.csv_path.open(
            "x", newline="", encoding="utf-8"
        )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.FIELDNAMES,
        )
        self._writer.writeheader()
        self._file.flush()

    def log_attempt(
        self,
        attempt_id: int,
        frame_id: int | None,
        read_start_ns: int,
        read_end_ns: int,
        capture_status: str,
        cycle_id: int | None = None,
        pose_id: str | None = None,
    ) -> None:
        if capture_status not in ("captured", "capture_failed"):
            raise ValueError("Invalid capture status")

        if read_end_ns < read_start_ns:
            raise ValueError("Read end precedes read start")

        record = {
            "run_id": self.run_id,
            "cycle_id": cycle_id,
            "pose_id": pose_id,
            "attempt_id": attempt_id,
            "frame_id": frame_id,
            "read_start_ns": read_start_ns,
            "read_end_ns": read_end_ns,
            "capture_status": capture_status,
        }

        self._writer.writerow(record)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

