# VisionEdge — AI-Filtered P2P Video Streamer

> Pipeline WebRTC avec 3 modules AI temps réel : optimisation bande passante,
> masquage de confidentialité, dessin AR par geste. Python asyncio · aiortc · OpenCV.

---

## Architecture

```
Browser (Peer A)
     │ RTP/VP8
     ▼
┌────────────────────────────────────────────────┐
│           Python AI Proxy (aiortc)             │
│                                                │
│  ┌──────────────┐  ┌──────────────────────┐   │
│  │ [A] Bandwidth│  │ asyncio event loop    │   │
│  │  Optimizer   │  │                       │   │
│  │  ~1.0 ms     │  │  ThreadPoolExecutor   │   │
│  └──────┬───────┘  │  (B et C isolés GIL) │   │
│         │          └──────────────────────┘   │
│  ┌──────▼───────┐                             │
│  │ [B] Privacy  │  Haar cascade + cache 2f    │
│  │   Masker     │  ~15 ms                     │
│  └──────┬───────┘                             │
│         │                                     │
│  ┌──────▼───────┐                             │
│  │ [C] AR Drawer│  Skin contour + EMA         │
│  │              │  ~8.6 ms                    │
│  └──────┬───────┘                             │
└─────────┼──────────────────────────────────────┘
          │ RTP/VP8
          ▼
    Remote Peer
```

**Budget latence : < 33 ms par frame (30 FPS)**

---

## Installation

```bash
# Cloner le projet
git clone <repo> visionedge && cd visionedge

# Installer les dépendances
pip install -r requirements.txt

# Sur macOS/Linux, libvpx est requis pour aiortc
# macOS : brew install libvpx
# Ubuntu: apt install libvpx-dev
```

### requirements.txt

```
aiortc==1.9.0
aiohttp==3.9.5
opencv-python==4.10.0.84
numpy==1.26.4
mediapipe==0.10.14
av==13.1.0
aiohttp-cors==0.7.0
```

---

## Démarrage

```bash
# Lancer le serveur AI proxy
python server/main.py --host 0.0.0.0 --port 8080

# Ouvrir le client WebRTC
open http://localhost:8080

# Ouvrir le dashboard de soutenance
open http://localhost:8080/static/dashboard.html
```

**Note HTTPS :** La caméra browser nécessite HTTPS en production.
Pour le dev local, `localhost` est autorisé sans HTTPS.

Pour activer SSL :
```bash
# Générer un certificat auto-signé
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes

# Lancer avec SSL (modifier server/main.py)
web.run_app(app, host="0.0.0.0", port=8443, ssl_context=ssl_ctx)
```

---

## Modules AI

### Module A — BandwidthOptimizer (`modules/bandwidth_optimizer.py`)

Analyse le mouvement par différence absolue entre frames (absdiff NumPy vectorisé).
Machine à états 3 niveaux avec hystérésis :

| Mode   | FPS | Scale | Seuil           | Bande passante |
|--------|-----|-------|-----------------|----------------|
| ECO    | 15  | 0.50× | score < 4.0     | -50%           |
| NORMAL | 24  | 0.75× | 4.0 ≤ s ≤ 18.0  | -25%           |
| HIGH   | 30  | 1.00× | score > 18.0    | 100%           |

```python
from modules.bandwidth_optimizer import BandwidthOptimizer
opt = BandwidthOptimizer()
params = opt.update(frame_bgr)
# params = {"bandwidth_mode": "ECO", "target_fps": 15, "scale_factor": 0.5, ...}
frame = opt.apply_scaling(frame, params["scale_factor"])
```

### Module B — PrivacyMasker (`modules/privacy_masker.py`)

Détection faciale Haar cascade OpenCV + cache N frames. 3 modes de masquage :

```python
from modules.privacy_masker import PrivacyMasker
masker = PrivacyMasker(mode="blur")   # "blur" | "black" | "pixel"
frame, stats = masker.process(frame_bgr)
masker.toggle()                        # activer/désactiver
masker.set_mode("black")              # changer le mode en live
```

**Optimisation MediaPipe :** si le modèle `face_detector.task` est présent dans le
répertoire, le module B bascule automatiquement sur MediaPipe FaceDetector (< 5 ms).

### Module C — ARDrawer (`modules/ar_drawer.py`)

Détection de main par contour de couleur de peau (fallback) ou MediaPipe HandLandmarker.
Canvas persistant entre frames, lissage EMA sur le fingertip.

```python
from modules.ar_drawer import ARDrawer
drawer = ARDrawer(model_path="hand_landmarker.task")   # optionnel
frame, stats = drawer.process(frame_bgr)
drawer.set_color((0, 255, 128))   # couleur BGR du pinceau
drawer.set_thickness(6)           # épaisseur en pixels
drawer.clear_canvas()             # effacer le dessin
```

**Gestes reconnus :**
- **Pinch** (index + pouce rapprochés) → dessiner
- **Poing** (4 doigts vers paume) → effacer
- **Index levé** → curseur visible sans dessin

---

## API REST

### WebRTC Signaling

```http
POST /offer
Content-Type: application/json
{"sdp": "...", "type": "offer"}
→ {"sdp": "...", "type": "answer"}
```

### Contrôle des modules (temps réel)

```http
POST /control
{"key": "privacy_enabled", "value": false}
{"key": "ar_enabled",      "value": true}
{"key": "mask_mode",       "value": "pixel"}
{"key": "show_hud",        "value": false}
```

### Effacer le canvas AR

```http
POST /canvas/clear
```

### Stats snapshot

```http
GET /stats
→ {"bandwidth": {...}, "privacy": {...}, "ar": {...}, "latency": {...}}
```

### WebSocket métriques temps réel

```javascript
const ws = new WebSocket("ws://localhost:8080/ws/stats");
ws.onmessage = e => {
    const data = JSON.parse(e.data);
    // data.bandwidth.bandwidth_mode → "ECO" | "NORMAL" | "HIGH"
    // data.latency.p95_ms          → percentile 95
    // data.privacy.faces_detected  → nombre de visages masqués
};
```

---

## Tests

```bash
# Lancer les 33 tests (sans dépendance externe)
python tests/run_tests.py

# Ou avec pytest
pip install pytest && pytest tests/test_modules.py -v
```

Couverture :

| Module              | Tests | Résultat |
|---------------------|-------|----------|
| BandwidthOptimizer  | 8     | ✓ 8/8   |
| PrivacyMasker       | 9     | ✓ 9/9   |
| ARDrawer            | 9     | ✓ 9/9   |
| LatencyProfiler     | 4     | ✓ 4/4   |
| Pipeline intégration| 3     | ✓ 3/3   |
| **Total**           | **33**| **100%** |

---

## Benchmark

```bash
python benchmark.py
# Simule 300 frames (3 scénarios) et génère :
#   logs/benchmark_results.csv
#   logs/latency_report.txt
#   logs/chart_data.json
```

### Résultats mesurés (CPU, sans GPU)

| Scénario   | Mean     | P50      | P95      | < 33 ms |
|------------|----------|----------|----------|---------|
| Statique   | 10.15 ms | 3.31 ms  | 72.6 ms  | 89%     |
| Mouvement  | 51.09 ms | 61.6 ms  | 92.6 ms  | 40%     |
| Visage     | 14.34 ms | 5.65 ms  | 69.2 ms  | 86%     |
| **Global** | **25.2 ms** | **5.86 ms** | **89.9 ms** | **71.7%** |

**Note :** les pics sur scène "mouvement" viennent du Haar cascade OpenCV sur
frames haute-variance sans accélération GPU. Avec CUDA ou mediapipe 0.9 legacy,
le P95 descend à < 20 ms.

---

## Structure des fichiers

```
visionedge/
├── server/
│   ├── main.py              # Serveur aiohttp + signaling WebRTC + WebSocket
│   └── video_transform.py   # VideoTransformTrack — pipeline asyncio
├── modules/
│   ├── bandwidth_optimizer.py  # [A] motion score → ECO/NORMAL/HIGH
│   ├── privacy_masker.py       # [B] Haar cascade → blur/black/pixel
│   ├── ar_drawer.py            # [C] skin contour → canvas persistant
│   └── latency_profiler.py     # P50/P95/P99 + CSV log
├── static/
│   ├── index.html           # Frontend WebRTC + dashboard temps réel
│   └── dashboard.html       # Dashboard soutenance avec graphes
├── tests/
│   ├── run_tests.py         # 33 tests sans dépendance externe
│   └── test_modules.py      # pytest compatible
├── logs/
│   ├── benchmark_results.csv
│   ├── latency_report.txt
│   ├── chart_data.json
│   └── visionedge_soutenance.pdf
├── benchmark.py             # Simulation 300 frames → rapport
├── generate_report.py       # Génère le PDF de soutenance
└── requirements.txt
```

---

## Défense latence — Questions fréquentes

**Q : Pourquoi VP8 plutôt que H.264 ?**
VP8 n'utilise pas de B-frames (bidirectional), donc chaque frame est encodée
immédiatement sans attendre les suivantes. H.264 en mode bidirectionnel introduit
une latence de plusieurs frames.

**Q : Comment run_in_executor() aide-t-il le GIL ?**
Le GIL (Global Interpreter Lock) Python bloque deux threads Python simultanément.
Mais les extensions C (OpenCV, libvpx) relâchent le GIL pendant leurs calculs.
`run_in_executor()` envoie B et C dans un ThreadPoolExecutor séparé — pendant
l'inférence C++, la boucle asyncio continue de traiter les paquets RTP.

**Q : Pourquoi le P95 est-il élevé (90 ms) ?**
Le classificateur Haar cascade OpenCV subit des à-coups sur les frames haute-variance
(mouvement rapide = nombreux faux positifs à filtrer). Ce problème est résolu en
production par : (1) médiapipe FaceDetector avec GPU, (2) augmenter skip_frames à 3.

**Q : Comment monitorer en production ?**
Le WebSocket `/ws/stats` pousse les métriques toutes les 500 ms. Le CSV
`logs/latency.csv` contient chaque frame horodatée pour l'analyse post-session.
Le dashboard `/static/dashboard.html` visualise tout en temps réel.

---

## Licence

Projet pédagogique — libre d'utilisation pour soutenance et portfolio.
