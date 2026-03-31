"""
VideoTransformTrack — Cœur du pipeline AI
==========================================
Reçoit chaque frame vidéo depuis aiortc, applique les 3 modules AI
dans l'ordre, puis retourne la frame transformée au peer.

Pipeline par frame :
  [RTP recv] → decode → Module A → Module B → Module C → encode → [RTP send]

Concurrence :
  • L'inférence MediaPipe (B et C) tourne dans un ThreadPoolExecutor
    pour ne pas bloquer la boucle asyncio principale.
  • Module A (OpenCV pur NumPy) tourne en inline (< 3 ms, pas de GIL).
"""

import asyncio
import time
import fractions
import concurrent.futures
from typing import Optional

import numpy as np
import cv2
import av

from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

from modules.bandwidth_optimizer import BandwidthOptimizer
from modules.privacy_masker      import PrivacyMasker
from modules.ar_drawer           import ARDrawer
from modules.latency_profiler    import LatencyProfiler


# Pool de threads pour l'inférence IA (évite le GIL sur les appels C++)
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3,
                                                  thread_name_prefix="ai-worker")


class VideoTransformTrack(MediaStreamTrack):
    """
    Wraps une MediaStreamTrack source et applique le pipeline AI.
    Compatible avec le format aiortc (VideoFrame av).
    """

    kind = "video"

    def __init__(self, source_track: MediaStreamTrack, config: dict):
        super().__init__()
        self._source   = source_track
        self._config   = config   # activations des modules

        # ── Modules AI ───────────────────────────────────────────────────
        self.optimizer = BandwidthOptimizer()
        self.masker    = PrivacyMasker(mode=config.get("mask_mode", "blur"))
        self.drawer    = ARDrawer()
        self.profiler  = LatencyProfiler()

        # État runtime partagé avec le dashboard WS
        self.runtime_stats = {
            "bandwidth":  {},
            "privacy":    {},
            "ar":         {},
            "latency":    {},
        }

        self._frame_count = 0
        self._skip_frame  = False   # throttle selon bandwidth mode

    # ─────────────────────────────────────────────────────────────────────

    async def recv(self):
        """
        Appelé par aiortc pour chaque frame entrante.
        Retourne une av.VideoFrame transformée.
        """
        # Récupérer la frame source
        frame: av.VideoFrame = await self._source.recv()
        self._frame_count += 1

        t_total_start = time.perf_counter()

        # ── Convertir av.VideoFrame → numpy BGR ──────────────────────────
        img: np.ndarray = frame.to_ndarray(format="bgr24")

        # ─── MODULE A : Bandwidth Optimizer (inline, < 3 ms) ─────────────
        bw_stats   = self.optimizer.update(img)
        bw_ms      = bw_stats["processing_ms"]

        # Throttle : en mode ECO, on skippe 1 frame sur 2
        mode = bw_stats["bandwidth_mode"]
        if mode == "ECO" and self._frame_count % 2 == 0:
            self._skip_frame = True
        else:
            self._skip_frame = False

        # Downscale si nécessaire
        scale = bw_stats["scale_factor"]
        img   = self.optimizer.apply_scaling(img, scale)

        # ─── MODULE B : Privacy Masking (ThreadExecutor) ──────────────────
        loop   = asyncio.get_event_loop()

        if self._config.get("privacy_enabled", True):
            img, prv_stats = await loop.run_in_executor(
                _EXECUTOR, self.masker.process, img
            )
        else:
            prv_stats = {"processing_ms": 0, "faces_detected": 0,
                         "enabled": False}

        prv_ms = prv_stats["processing_ms"]

        # ─── MODULE C : AR Drawing (ThreadExecutor) ───────────────────────
        if self._config.get("ar_enabled", True):
            img, ar_stats = await loop.run_in_executor(
                _EXECUTOR, self.drawer.process, img
            )
        else:
            ar_stats = {"processing_ms": 0, "is_drawing": False,
                        "enabled": False}

        ar_ms = ar_stats["processing_ms"]

        # ─── Remettre à la résolution originale si downscalé ─────────────
        if scale < 1.0:
            orig_h, orig_w = frame.to_ndarray(format="bgr24").shape[:2]
            img = cv2.resize(img, (orig_w, orig_h),
                             interpolation=cv2.INTER_LINEAR)

        # ─── Overlay HUD (debug info sur la frame) ───────────────────────
        if self._config.get("show_hud", True):
            img = self._draw_hud(img, bw_stats, prv_stats, ar_stats)

        # ─── Convertir numpy → av.VideoFrame ─────────────────────────────
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts      = frame.pts
        new_frame.time_base = frame.time_base

        # ─── Profiling total ─────────────────────────────────────────────
        total_ms = (time.perf_counter() - t_total_start) * 1000
        self.profiler.record(bw_ms, prv_ms, ar_ms, total_ms)

        # ─── Mise à jour stats dashboard ─────────────────────────────────
        self.runtime_stats.update({
            "bandwidth": bw_stats,
            "privacy":   prv_stats,
            "ar":        ar_stats,
            "latency":   self.profiler.get_summary(),
            "total_ms":  round(total_ms, 2),
        })

        return new_frame

    # ─── HUD overlay ──────────────────────────────────────────────────────

    def _draw_hud(self, img: np.ndarray, bw: dict,
                  prv: dict, ar: dict) -> np.ndarray:
        """Affiche un overlay de debug semi-transparent sur la frame."""
        overlay = img.copy()
        h, w = img.shape[:2]

        # Fond semi-transparent
        cv2.rectangle(overlay, (8, 8), (260, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        font   = cv2.FONT_HERSHEY_SIMPLEX
        colors = {"ECO": (0, 200, 255), "NORMAL": (0, 255, 128),
                  "HIGH": (0, 128, 255)}
        mode   = bw.get("bandwidth_mode", "NORMAL")
        col    = colors.get(mode, (255, 255, 255))

        lines = [
            (f"[A] {mode}  score={bw.get('motion_score',0):.1f}", col),
            (f"    fps={bw.get('target_fps',30)} scale={bw.get('scale_factor',1.0)}", (180,180,180)),
            (f"[B] faces={prv.get('faces_detected',0)}  {prv.get('mode','blur')}", (255,180,80)),
            (f"[C] draw={ar.get('is_drawing',False)}", (80,255,180)),
            (f"LAT {self.profiler.get_summary().get('p95_ms','?')} ms p95", (200,200,200)),
        ]

        for i, (text, color) in enumerate(lines):
            cv2.putText(img, text, (14, 30 + i * 20),
                        font, 0.44, color, 1, cv2.LINE_AA)

        return img
