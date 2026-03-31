"""
Latency Profiler — mesure et journalise la latence par frame
=============================================================
Enregistre le temps de traitement de chaque étape du pipeline
et calcule les percentiles P50/P95/P99 pour la soutenance.
"""

import time
import json
import csv
from collections import deque
from pathlib import Path


WINDOW = 300   # 10 secondes à 30 fps


class LatencyProfiler:

    def __init__(self, log_dir: str = "logs"):
        self._stages: dict[str, deque] = {}
        self._frame_times: deque = deque(maxlen=WINDOW)
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True)
        self._csv_path = self._log_dir / "latency.csv"
        self._frame_idx = 0
        self._start_time = time.time()

        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "timestamp_s",
                             "bandwidth_ms", "privacy_ms", "ar_ms",
                             "total_ms", "fps_instant"])

    def record(self, bandwidth_ms: float, privacy_ms: float,
               ar_ms: float, total_ms: float):
        self._frame_idx += 1
        ts = round(time.time() - self._start_time, 3)
        self._frame_times.append(total_ms)

        fps = self._instant_fps()

        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self._frame_idx, ts,
                round(bandwidth_ms, 2),
                round(privacy_ms, 2),
                round(ar_ms, 2),
                round(total_ms, 2),
                round(fps, 1),
            ])

    def get_summary(self) -> dict:
        if not self._frame_times:
            return {}
        times = sorted(self._frame_times)
        n = len(times)
        return {
            "frame_count":  self._frame_idx,
            "p50_ms":       round(times[int(n * 0.50)], 2),
            "p95_ms":       round(times[int(n * 0.95)], 2),
            "p99_ms":       round(times[min(int(n * 0.99), n-1)], 2),
            "mean_ms":      round(sum(times) / n, 2),
            "max_ms":       round(times[-1], 2),
            "under_33ms_pct": round(
                sum(1 for t in times if t < 33) / n * 100, 1
            ),
            "avg_fps":      round(self._instant_fps(), 1),
        }

    def _instant_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = len(self._frame_times) / 30   # estimation
        return min(30.0, len(self._frame_times) / max(elapsed, 0.001))
