"""
Tests unitaires VisionEdge
===========================
Testent chaque module de façon isolée avec des frames synthétiques.
Pas besoin de caméra ni de MediaPipe installé.

Lancer : python -m pytest tests/test_modules.py -v
"""

import sys
import time
import numpy as np
import pytest

sys.path.insert(0, ".")
from modules.bandwidth_optimizer import BandwidthOptimizer
from modules.latency_profiler    import LatencyProfiler


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_frame(w=640, h=480, noise=0):
    """Frame BGR synthétique avec bruit optionnel."""
    base = np.zeros((h, w, 3), dtype=np.uint8)
    if noise:
        base += np.random.randint(0, noise, (h, w, 3), dtype=np.uint8)
    return base


# ─── Module A ────────────────────────────────────────────────────────────────

class TestBandwidthOptimizer:

    def setup_method(self):
        self.opt = BandwidthOptimizer()

    def test_initial_call_returns_normal(self):
        frame = make_frame()
        result = self.opt.update(frame)
        assert result["bandwidth_mode"] in ("ECO", "NORMAL", "HIGH")
        assert "motion_score" in result
        assert "target_fps"   in result
        assert "scale_factor" in result

    def test_static_scene_goes_eco(self):
        """8 frames identiques → doit passer en ECO."""
        frame = make_frame(noise=1)
        for _ in range(10):
            result = self.opt.update(frame.copy())
        assert result["bandwidth_mode"] == "ECO"
        assert result["target_fps"] == 15
        assert result["scale_factor"] == 0.5

    def test_high_motion_goes_high(self):
        """Frames avec bruit élevé → mode HIGH."""
        opt = BandwidthOptimizer()
        for _ in range(10):
            frame = make_frame(noise=80)
            result = opt.update(frame)
        assert result["bandwidth_mode"] == "HIGH"
        assert result["target_fps"] == 30
        assert result["scale_factor"] == 1.0

    def test_processing_time_under_5ms(self):
        """Module A doit traiter en < 5 ms (vectorisé NumPy)."""
        frame = make_frame(noise=20)
        self.opt.update(frame)  # warmup
        times = []
        for _ in range(30):
            r = self.opt.update(make_frame(noise=20))
            times.append(r["processing_ms"])
        avg = sum(times) / len(times)
        assert avg < 5.0, f"Trop lent : {avg:.2f} ms (attendu < 5 ms)"

    def test_apply_scaling_downscale(self):
        frame = make_frame(w=640, h=480)
        small = self.opt.apply_scaling(frame, 0.5)
        assert small.shape == (240, 320, 3)

    def test_apply_scaling_no_upscale(self):
        frame = make_frame(w=640, h=480)
        same  = self.opt.apply_scaling(frame, 1.5)
        assert same.shape == frame.shape  # pas d'upscale

    def test_stats_tracking(self):
        frame = make_frame()
        for _ in range(20):
            self.opt.update(frame.copy())
        stats = self.opt.stats
        assert "eco_frames"    in stats
        assert "normal_frames" in stats
        assert stats["eco_frames"] + stats["normal_frames"] + stats["high_frames"] == 19


# ─── Module B ────────────────────────────────────────────────────────────────

class TestPrivacyMasker:

    def test_import(self):
        """Module doit s'importer sans erreur."""
        from modules.privacy_masker import PrivacyMasker
        masker = PrivacyMasker()
        assert masker is not None

    def test_process_no_crash_empty_frame(self):
        from modules.privacy_masker import PrivacyMasker
        masker = PrivacyMasker()
        frame = make_frame(w=320, h=240)
        result, stats = masker.process(frame)
        assert result.shape == frame.shape
        assert "faces_detected"   in stats
        assert "processing_ms"    in stats

    def test_toggle(self):
        from modules.privacy_masker import PrivacyMasker
        masker = PrivacyMasker()
        assert masker.enabled == True
        masker.toggle()
        assert masker.enabled == False
        masker.toggle()
        assert masker.enabled == True

    def test_mode_change(self):
        from modules.privacy_masker import PrivacyMasker
        for mode in ("blur", "black", "pixel"):
            m = PrivacyMasker(mode=mode)
            assert m.mode == mode

    def test_invalid_mode_raises(self):
        from modules.privacy_masker import PrivacyMasker
        m = PrivacyMasker()
        with pytest.raises(AssertionError):
            m.set_mode("invisible")


# ─── Module C ────────────────────────────────────────────────────────────────

class TestARDrawer:

    def test_import(self):
        from modules.ar_drawer import ARDrawer
        drawer = ARDrawer()
        assert drawer is not None

    def test_process_frame_no_hands(self):
        from modules.ar_drawer import ARDrawer
        drawer = ARDrawer()
        frame  = make_frame(w=640, h=480)
        result, stats = drawer.process(frame)
        assert result.shape == frame.shape
        assert stats["is_drawing"] == False
        assert "processing_ms"   in stats

    def test_clear_canvas(self):
        from modules.ar_drawer import ARDrawer
        drawer = ARDrawer()
        frame  = make_frame(w=640, h=480)
        drawer.process(frame)  # init canvas
        drawer.clear_canvas()
        assert drawer._canvas is None or np.all(drawer._canvas == 0)

    def test_toggle(self):
        from modules.ar_drawer import ARDrawer
        drawer = ARDrawer()
        assert drawer.enabled == True
        drawer.toggle()
        assert drawer.enabled == False

    def test_color_change(self):
        from modules.ar_drawer import ARDrawer
        drawer = ARDrawer()
        drawer.set_color((255, 0, 128))
        assert drawer.brush_color == (255, 0, 128)

    def test_thickness_clamped(self):
        from modules.ar_drawer import ARDrawer
        drawer = ARDrawer()
        drawer.set_thickness(100)
        assert drawer.brush_thickness <= 20
        drawer.set_thickness(-5)
        assert drawer.brush_thickness >= 1


# ─── Latency Profiler ─────────────────────────────────────────────────────────

class TestLatencyProfiler:

    def test_record_and_summary(self, tmp_path):
        profiler = LatencyProfiler(log_dir=str(tmp_path))
        for ms in range(10, 35):
            profiler.record(2.0, 12.0, 5.0, float(ms))

        summary = profiler.get_summary()
        assert "p50_ms" in summary
        assert "p95_ms" in summary
        assert "under_33ms_pct" in summary
        assert summary["frame_count"] == 25
        assert 0 <= summary["under_33ms_pct"] <= 100

    def test_csv_created(self, tmp_path):
        profiler = LatencyProfiler(log_dir=str(tmp_path))
        profiler.record(1, 10, 4, 15)
        assert (tmp_path / "latency.csv").exists()


# ─── Pipeline intégration ────────────────────────────────────────────────────

class TestPipelineIntegration:

    def test_full_pipeline_latency(self):
        """
        Simule le pipeline complet (sans GPU, sans MediaPipe).
        Vérifie que A seul + overhead reste sous 10 ms.
        """
        from modules.bandwidth_optimizer import BandwidthOptimizer
        opt = BandwidthOptimizer()

        times = []
        for _ in range(60):
            frame = make_frame(w=640, h=480, noise=20)
            t0    = time.perf_counter()
            result = opt.update(frame)
            frame  = opt.apply_scaling(frame, result["scale_factor"])
            t1    = time.perf_counter()
            times.append((t1 - t0) * 1000)

        p95 = sorted(times)[int(60 * 0.95)]
        print(f"\n  Pipeline A P95 = {p95:.2f} ms")
        assert p95 < 10, f"Module A P95 trop lent : {p95:.2f} ms"

    def test_mode_transitions(self):
        """Vérifie les transitions ECO→NORMAL→HIGH."""
        opt = BandwidthOptimizer()
        static = make_frame()

        for _ in range(10):
            r = opt.update(static.copy())
        assert r["bandwidth_mode"] == "ECO"

        for _ in range(15):
            r = opt.update(make_frame(noise=80))
        assert r["bandwidth_mode"] in ("NORMAL", "HIGH")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
