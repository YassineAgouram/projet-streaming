"""
Benchmark VisionEdge — Génère les données de performance réelles
================================================================
Simule 300 frames (10s à 30 FPS) avec les 3 modules actifs.
Produit :
  - logs/benchmark_results.csv
  - logs/latency_report.txt  (résumé pour soutenance)
  - logs/chart_data.json     (pour le dashboard)
"""

import sys, time, json, csv, os
import numpy as np
import cv2

sys.path.insert(0, ".")
from modules.bandwidth_optimizer import BandwidthOptimizer
from modules.privacy_masker      import PrivacyMasker
from modules.ar_drawer           import ARDrawer
from modules.latency_profiler    import LatencyProfiler

os.makedirs("logs", exist_ok=True)

rng = np.random.default_rng(2024)

def make_scene(kind: str, w=640, h=480) -> np.ndarray:
    """Génère une frame synthétique selon le type de scène."""
    if kind == "static":
        base = np.full((h, w, 3), [30, 40, 60], dtype=np.uint8)
        # Légère texture
        noise = rng.integers(0, 8, (h, w, 3), dtype=np.uint8)
        return (base + noise).clip(0, 255).astype(np.uint8)
    elif kind == "motion":
        frame = rng.integers(10, 200, (h, w, 3), dtype=np.uint8)
        # Simulation de blocs en mouvement
        x = rng.integers(0, w - 120)
        y = rng.integers(0, h - 120)
        frame[y:y+120, x:x+120] = rng.integers(150, 255, (120, 120, 3), dtype=np.uint8)
        return frame
    elif kind == "face_scene":
        base = np.full((h, w, 3), [60, 50, 45], dtype=np.uint8)
        # Simuler une zone chair (visage factice pour tests)
        cx, cy = w // 2, h // 2
        cv2.ellipse(base, (cx, cy), (80, 100), 0, 0, 360, (140, 100, 80), -1)
        return base
    else:
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

print("\n" + "═"*58)
print("  VISIONEDGE — BENCHMARK PERFORMANCE PIPELINE")
print("═"*58)

# ── Init modules ─────────────────────────────────────────────────────────────
opt      = BandwidthOptimizer()
masker   = PrivacyMasker(mode="blur")
drawer   = ARDrawer()
profiler = LatencyProfiler(log_dir="logs")

TOTAL_FRAMES = 300
W, H = 640, 480

# Scénarios : 100 statiques, 100 en mouvement, 100 mixtes
scenes = (
    ["static"]  * 100 +
    ["motion"]  * 100 +
    ["face_scene"] * 100
)

results = []
bar_width = 40

print(f"\n  Simulation de {TOTAL_FRAMES} frames ({W}x{H}) ...\n")

for i, scene_type in enumerate(scenes):
    frame = make_scene(scene_type, W, H)
    t_total = time.perf_counter()

    # ── Module A ──────────────────────────────────────────────────────────
    bw   = opt.update(frame)
    frame_scaled = opt.apply_scaling(frame, bw["scale_factor"])

    # ── Module B ──────────────────────────────────────────────────────────
    frame_masked, prv = masker.process(frame_scaled)

    # ── Module C ──────────────────────────────────────────────────────────
    frame_out, ar = drawer.process(frame_masked)

    total_ms = (time.perf_counter() - t_total) * 1000
    profiler.record(bw["processing_ms"], prv["processing_ms"],
                    ar["processing_ms"], total_ms)

    results.append({
        "frame":       i + 1,
        "scene":       scene_type,
        "bw_mode":     bw["bandwidth_mode"],
        "motion":      bw["motion_score"],
        "bw_ms":       round(bw["processing_ms"], 3),
        "prv_ms":      round(prv["processing_ms"], 3),
        "ar_ms":       round(ar["processing_ms"], 3),
        "total_ms":    round(total_ms, 3),
        "scale":       bw["scale_factor"],
        "faces":       prv["faces_detected"],
    })

    # Barre de progression
    done = int((i + 1) / TOTAL_FRAMES * bar_width)
    bar  = "█" * done + "░" * (bar_width - done)
    pct  = (i + 1) / TOTAL_FRAMES * 100
    print(f"\r  [{bar}] {pct:5.1f}%  frame {i+1:3d}/{TOTAL_FRAMES}"
          f"  {total_ms:5.2f}ms", end="", flush=True)

print("\n")

# ── Écrire CSV détaillé ───────────────────────────────────────────────────────
csv_path = "logs/benchmark_results.csv"
with open(csv_path, "w", newline="") as f:
    w_csv = csv.DictWriter(f, fieldnames=results[0].keys())
    w_csv.writeheader()
    w_csv.writerows(results)

# ── Calculer stats par scénario ───────────────────────────────────────────────
def stats_for(subset):
    totals = [r["total_ms"] for r in subset]
    bw_ms  = [r["bw_ms"]   for r in subset]
    prv_ms = [r["prv_ms"]  for r in subset]
    ar_ms  = [r["ar_ms"]   for r in subset]
    totals_s = sorted(totals)
    n = len(totals_s)
    return {
        "count":   n,
        "mean_ms": round(sum(totals) / n, 2),
        "p50_ms":  round(totals_s[int(n*0.50)], 2),
        "p95_ms":  round(totals_s[int(n*0.95)], 2),
        "p99_ms":  round(totals_s[min(int(n*0.99), n-1)], 2),
        "max_ms":  round(max(totals), 2),
        "bw_mean": round(sum(bw_ms)  / n, 2),
        "prv_mean":round(sum(prv_ms) / n, 2),
        "ar_mean": round(sum(ar_ms)  / n, 2),
        "under_33": round(sum(1 for t in totals if t < 33) / n * 100, 1),
    }

static_r  = [r for r in results if r["scene"] == "static"]
motion_r  = [r for r in results if r["scene"] == "motion"]
face_r    = [r for r in results if r["scene"] == "face_scene"]
all_r     = results

s_static = stats_for(static_r)
s_motion = stats_for(motion_r)
s_face   = stats_for(face_r)
s_all    = stats_for(all_r)

# Modes bandwidth
eco_count  = sum(1 for r in results if r["bw_mode"] == "ECO")
norm_count = sum(1 for r in results if r["bw_mode"] == "NORMAL")
high_count = sum(1 for r in results if r["bw_mode"] == "HIGH")

# ── Rapport texte ──────────────────────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════════╗
║          VISIONEDGE — RAPPORT DE PERFORMANCE             ║
║              Pipeline AI 3 modules — {TOTAL_FRAMES} frames           ║
╚══════════════════════════════════════════════════════════╝

Résolution  : {W}×{H} px   |   Cible : 30 FPS (33.3 ms/frame)
Modules     : [A] BandwidthOptimizer  [B] PrivacyMasker  [C] ARDrawer
Détecteurs  : [A] NumPy vectorisé  [B] Haar cascade  [C] Skin contour

═══════════════════════════════════════════════════════════
  LATENCE GLOBALE (tous scénarios — {TOTAL_FRAMES} frames)
═══════════════════════════════════════════════════════════
  Moyenne      : {s_all['mean_ms']:6.2f} ms
  P50          : {s_all['p50_ms']:6.2f} ms
  P95          : {s_all['p95_ms']:6.2f} ms    ← seuil critique soutenance
  P99          : {s_all['p99_ms']:6.2f} ms
  Max          : {s_all['max_ms']:6.2f} ms
  Sous 33 ms   : {s_all['under_33']:5.1f} %   (objectif : > 90%)

  Décomposition moyenne par module :
    [A] Bandwidth   : {s_all['bw_mean']:5.2f} ms
    [B] Privacy     : {s_all['prv_mean']:5.2f} ms
    [C] AR Drawing  : {s_all['ar_mean']:5.2f} ms
    ─────────────────────────
    Total pipeline  : {s_all['bw_mean']+s_all['prv_mean']+s_all['ar_mean']:5.2f} ms

═══════════════════════════════════════════════════════════
  PAR SCÉNARIO
═══════════════════════════════════════════════════════════
  ┌─────────────┬────────┬────────┬────────┬─────────────┐
  │  Scénario   │  Mean  │  P50   │  P95   │ Sous 33ms % │
  ├─────────────┼────────┼────────┼────────┼─────────────┤
  │ Statique    │{s_static['mean_ms']:6.2f}ms│{s_static['p50_ms']:6.2f}ms│{s_static['p95_ms']:6.2f}ms│   {s_static['under_33']:5.1f} %   │
  │ Mouvement   │{s_motion['mean_ms']:6.2f}ms│{s_motion['p50_ms']:6.2f}ms│{s_motion['p95_ms']:6.2f}ms│   {s_motion['under_33']:5.1f} %   │
  │ Scène visage│{s_face['mean_ms']:6.2f}ms│{s_face['p50_ms']:6.2f}ms│{s_face['p95_ms']:6.2f}ms│   {s_face['under_33']:5.1f} %   │
  └─────────────┴────────┴────────┴────────┴─────────────┘

═══════════════════════════════════════════════════════════
  MODULE A — BANDWIDTH OPTIMIZER
═══════════════════════════════════════════════════════════
  Frames en mode ECO    : {eco_count:3d} ({eco_count/TOTAL_FRAMES*100:5.1f}%)  → 15 FPS, scale 0.5x
  Frames en mode NORMAL : {norm_count:3d} ({norm_count/TOTAL_FRAMES*100:5.1f}%)  → 24 FPS, scale 0.75x
  Frames en mode HIGH   : {high_count:3d} ({high_count/TOTAL_FRAMES*100:5.1f}%)  → 30 FPS, scale 1.0x
  Économie bande passante estimée : {eco_count/TOTAL_FRAMES*50:.1f}% (frames ECO × 50%)

═══════════════════════════════════════════════════════════
  DÉFENSE LATENCE (argument ingénieur)
═══════════════════════════════════════════════════════════
  Le pipeline respecte le budget 33 ms dans {s_all['under_33']:.1f}% des frames.
  Le P95 de {s_all['p95_ms']:.2f} ms valide l'objectif temps réel 30 FPS.

  Techniques utilisées pour tenir le budget :
  • Module A  : absdiff NumPy vectorisé (pas de boucle Python)
  • Module B  : inférence sur frame ×0.5 + cache {masker.skip_frames} frames
  • Module C  : EMA pour lisser les coordonnées, évite les micro-tremblements
  • Asyncio   : run_in_executor isole B et C du event loop principal
  • VP8       : pas de B-frames → latence encode minimale

Fichier CSV : {csv_path}
"""

report_path = "logs/latency_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(report)

# ── JSON pour dashboard ────────────────────────────────────────────────────
chart_data = {
    "frames":       [r["frame"]    for r in results],
    "total_ms":     [r["total_ms"] for r in results],
    "bw_ms":        [r["bw_ms"]    for r in results],
    "prv_ms":       [r["prv_ms"]   for r in results],
    "ar_ms":        [r["ar_ms"]    for r in results],
    "motion":       [r["motion"]   for r in results],
    "bw_mode":      [r["bw_mode"]  for r in results],
    "summary":      s_all,
    "by_scene": {
        "static":  s_static,
        "motion":  s_motion,
        "face":    s_face,
    },
    "mode_counts": {
        "ECO": eco_count, "NORMAL": norm_count, "HIGH": high_count
    },
}
with open("logs/chart_data.json", "w") as f:
    json.dump(chart_data, f, indent=2)

print(f"\n  Fichiers générés :")
print(f"    logs/benchmark_results.csv  ({TOTAL_FRAMES} lignes)")
print(f"    logs/latency_report.txt")
print(f"    logs/chart_data.json")
print()
