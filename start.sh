#!/usr/bin/env bash
# ============================================================
# VisionEdge — Script de démarrage
# Usage : bash start.sh [--port 8080] [--benchmark] [--tests]
# ============================================================

set -e
CYAN="\033[96m" GREEN="\033[92m" AMBER="\033[93m" RED="\033[91m" NC="\033[0m" BOLD="\033[1m"

PORT=8080
RUN_TESTS=false
RUN_BENCH=false

for arg in "$@"; do
  case $arg in
    --port) PORT="${2}"; shift ;;
    --tests) RUN_TESTS=true ;;
    --benchmark) RUN_BENCH=true ;;
  esac
done

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║          VISIONEDGE — AI Video Proxy             ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Vérification Python ─────────────────────────────────────────────────
PY=$(python3 --version 2>&1)
echo -e "  ${CYAN}▸ Python   :${NC} $PY"

# ── Vérification dépendances clés ────────────────────────────────────────
check_dep() {
  python3 -c "import $1; print('  ${CYAN}▸ $1${NC} :', $1.__version__)" 2>/dev/null \
    || echo -e "  ${RED}✗ $1 non installé — pip install $1${NC}"
}
check_dep cv2
check_dep numpy
check_dep aiortc 2>/dev/null || echo -e "  ${AMBER}⚠  aiortc non installé — pip install aiortc${NC}"
check_dep aiohttp 2>/dev/null || echo -e "  ${AMBER}⚠  aiohttp non installé — pip install aiohttp${NC}"

echo ""
mkdir -p logs

# ── Tests ────────────────────────────────────────────────────────────────
if [ "$RUN_TESTS" = true ]; then
  echo -e "${BOLD}  Lancement des tests...${NC}"
  python3 tests/run_tests.py
  echo ""
fi

# ── Benchmark ────────────────────────────────────────────────────────────
if [ "$RUN_BENCH" = true ]; then
  echo -e "${BOLD}  Lancement du benchmark (300 frames)...${NC}"
  python3 benchmark.py
  echo ""
fi

# ── Rapport PDF ──────────────────────────────────────────────────────────
if [ -f "logs/chart_data.json" ]; then
  echo -e "  ${CYAN}▸ Génération du rapport PDF...${NC}"
  python3 generate_report.py 2>/dev/null && \
    echo -e "  ${GREEN}✓ logs/visionedge_soutenance.pdf${NC}" || \
    echo -e "  ${AMBER}⚠  reportlab non installé — pip install reportlab${NC}"
fi

# ── Serveur ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  Démarrage du serveur AI Proxy...${NC}"
echo -e "  ${CYAN}▸ Client WebRTC  :${NC} http://localhost:${PORT}"
echo -e "  ${CYAN}▸ Dashboard      :${NC} http://localhost:${PORT}/static/dashboard.html"
echo -e "  ${CYAN}▸ Stats API      :${NC} http://localhost:${PORT}/stats"
echo -e "  ${CYAN}▸ WebSocket      :${NC} ws://localhost:${PORT}/ws/stats"
echo ""
echo -e "  ${AMBER}Ctrl+C pour arrêter${NC}"
echo ""

python3 server/main.py --host 0.0.0.0 --port "$PORT"
