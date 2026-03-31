"""
Module C — Virtual Augmented Streamer (AR Drawing)
====================================================
Utilise mediapipe 0.10 HandLandmarker (nouvelle API mp.tasks) avec un modèle
embarqué en mémoire via un stub léger, OU détection approchée par couleur
de peau si mediapipe n'est pas disponible.

Gestes reconnus :
  • Pinch (pouce + index rapprochés) -> dessiner
  • Poing fermé (tous doigts vers paume) -> effacer le canvas
  • Index levé seul -> curseur visible, pas de dessin

Canvas persistant entre frames. Lissage EMA sur le tip de l'index.
"""

import time
import math
import numpy as np
import cv2
from collections import deque

# ── Tentative mediapipe 0.10 HandLandmarker ───────────────────────────────
_MP_HANDS_AVAILABLE = False
_HandLandmarker     = None
_HandLandmarkerOpts = None
_RunningMode        = None
_mp_image_cls       = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as _mpv
    from mediapipe.tasks.python.core import base_options as _mpbo
    _mp_image_cls       = mp.Image
    _HandLandmarker     = _mpv.HandLandmarker
    _HandLandmarkerOpts = _mpv.HandLandmarkerOptions
    _RunningMode        = _mpv.RunningMode
    _MP_HANDS_AVAILABLE = True   # API disponible, modèle .task requis au runtime
except Exception:
    pass

# ─── Paramètres ──────────────────────────────────────────────────────────────
PINCH_THRESHOLD  = 0.06
ERASE_THRESHOLD  = 0.08
BRUSH_COLOR_BGR  = (0, 255, 128)
BRUSH_THICKNESS  = 4
CANVAS_ALPHA     = 0.75
SMOOTHING        = 0.45
TRAIL_LEN        = 6


class ARDrawer:
    """
    Usage :
        drawer = ARDrawer(model_path="hand_landmarker.task")  # optionnel
        out_frame, stats = drawer.process(frame_bgr)
    """

    def __init__(self, model_path: str = "hand_landmarker.task"):
        self._enabled        = True
        self._canvas         = None
        self._prev_point     = None
        self._smooth_x       = 0.0
        self._smooth_y       = 0.0
        self._trail          = deque(maxlen=TRAIL_LEN)
        self._stroke_count   = 0
        self._frame_count    = 0
        self.brush_color     = BRUSH_COLOR_BGR
        self.brush_thickness = BRUSH_THICKNESS
        self._detector_type  = "none"
        self._landmarker     = None

        # ── Essayer de charger le landmarker mediapipe (modèle fichier) ──
        if _MP_HANDS_AVAILABLE:
            import os
            if os.path.exists(model_path):
                try:
                    opts = _HandLandmarkerOpts(
                        base_options=__import__(
                            'mediapipe.tasks.python.core.base_options',
                            fromlist=['BaseOptions']
                        ).BaseOptions(model_asset_path=model_path),
                        running_mode=_RunningMode.IMAGE,
                        num_hands=1,
                        min_hand_detection_confidence=0.6,
                        min_tracking_confidence=0.5,
                    )
                    self._landmarker    = _HandLandmarker.create_from_options(opts)
                    self._detector_type = "mediapipe_tasks"
                except Exception as e:
                    print(f"[ARDrawer] HandLandmarker init échoué : {e}")

        # ── Fallback : détection de mouvement par contour de main (couleur) ─
        if self._detector_type == "none":
            self._detector_type = "skin_contour"

    # ── API publique ──────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        self._frame_count += 1
        h, w = frame.shape[:2]

        if self._canvas is None or self._canvas.shape[:2] != (h, w):
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if not self._enabled:
            return self._composite(frame), self._stats(False, False, None, t0)

        # Choix du détecteur
        if self._landmarker is not None:
            drawing, erasing, tip_px = self._detect_mp(frame, w, h)
        else:
            drawing, erasing, tip_px = self._detect_skin(frame, w, h)

        # Dessiner sur le canvas si pinch actif
        if drawing and tip_px:
            if self._prev_point:
                cv2.line(self._canvas, self._prev_point, tip_px,
                         self.brush_color, self.brush_thickness,
                         lineType=cv2.LINE_AA)
                self._stroke_count += 1
            self._prev_point = tip_px
        else:
            self._prev_point = None

        if erasing:
            self._canvas[:] = 0
            self._prev_point = None

        # Curseur visuel
        if tip_px:
            col = (0, 255, 0) if drawing else (80, 80, 200)
            cv2.circle(frame, tip_px, 7, col, 2, cv2.LINE_AA)

        return self._composite(frame), self._stats(drawing, erasing, tip_px, t0)

    def clear_canvas(self):
        if self._canvas is not None:
            self._canvas[:] = 0
        self._prev_point = None
        self._trail.clear()

    def set_color(self, bgr: tuple):
        self.brush_color = bgr

    def set_thickness(self, px: int):
        self.brush_thickness = max(1, min(px, 20))

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Détection mediapipe tasks ─────────────────────────────────────────

    def _detect_mp(self, frame, w, h):
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = _mp_image_cls(image_format=__import__('mediapipe').ImageFormat.SRGB,
                                 data=rgb)
        result   = self._landmarker.detect(mp_img)

        if not result.hand_landmarks:
            return False, False, None

        lm = result.hand_landmarks[0]

        # Lissage EMA
        self._smooth_x = SMOOTHING * lm[8].x + (1-SMOOTHING) * self._smooth_x
        self._smooth_y = SMOOTHING * lm[8].y + (1-SMOOTHING) * self._smooth_y
        tip_px = (int(self._smooth_x * w), int(self._smooth_y * h))

        pinch  = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
        erase  = np.mean([math.hypot(lm[i].x - lm[4].x, lm[i].y - lm[4].y)
                          for i in [8, 12, 16, 20]])

        return (pinch < PINCH_THRESHOLD,
                erase < ERASE_THRESHOLD,
                tip_px)

    # ── Fallback : détection contour de peau ─────────────────────────────

    def _detect_skin(self, frame, w, h):
        """
        Détecte une zone peau dans le coin supérieur droit (ROI de saisie).
        Si le blob est assez grand -> "main présente" -> tip = centroïde.
        Pinch simulé : blob dans la moitié basse du ROI.
        """
        roi_x, roi_y = w * 2 // 3, 0
        roi = frame[roi_y:h // 2, roi_x:]

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([0,  25, 50]),
                           np.array([20, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5,5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, False, None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 1500:
            return False, False, None

        M   = cv2.moments(cnt)
        if M["m00"] == 0:
            return False, False, None
        cx  = int(M["m10"] / M["m00"]) + roi_x
        cy  = int(M["m01"] / M["m00"])

        self._smooth_x = SMOOTHING * (cx/w) + (1-SMOOTHING) * self._smooth_x
        self._smooth_y = SMOOTHING * (cy/h) + (1-SMOOTHING) * self._smooth_y
        tip_px = (int(self._smooth_x * w), int(self._smooth_y * h))

        # Pinch simulé : blob compact (ratio hauteur/largeur proche de 1)
        x,y,bw,bh = cv2.boundingRect(cnt)
        compact = min(bw, bh) / max(bw, bh, 1)
        drawing = compact > 0.55

        return drawing, False, tip_px

    # ── Composite ────────────────────────────────────────────────────────

    def _composite(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._canvas is None or self._canvas.shape[:2] != (h, w):
            self._canvas = np.zeros((h, w, 3), dtype=frame.dtype)
            return frame
        mask = self._canvas.astype(bool).any(axis=2)
        out  = frame.copy()
        if mask.any():
            blended = cv2.addWeighted(frame, 1 - CANVAS_ALPHA,
                                      self._canvas, CANVAS_ALPHA, 0)
            out[mask] = blended[mask]
        return out

    def _stats(self, drawing, erasing, tip, t0) -> dict:
        return {
            "is_drawing":    drawing,
            "is_erasing":    erasing,
            "finger_tip_px": tip,
            "stroke_count":  self._stroke_count,
            "enabled":       self._enabled,
            "detector":      self._detector_type,
            "processing_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
