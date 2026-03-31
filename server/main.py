"""
VisionEdge — Serveur Principal
================================
• Signaling WebRTC (SDP offer/answer) via HTTP POST /offer
• WebSocket /ws/stats → pousse les métriques au dashboard toutes les 500 ms
• Serve le frontend statique depuis /static
• Contrôle des modules via POST /control

Démarrage :
    python server/main.py --host 0.0.0.0 --port 8080
"""

import argparse
import asyncio
import json
import logging
import ssl
import sys
import time
from pathlib import Path

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay

# Ajoute le root du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))
from server.video_transform import VideoTransformTrack

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("visionedge")

# ─── State global ─────────────────────────────────────────────────────────────
pcs: set[RTCPeerConnection]     = set()
transforms: list[VideoTransformTrack] = []
relay  = MediaRelay()
config = {
    "privacy_enabled": True,
    "ar_enabled":      True,
    "show_hud":        True,
    "mask_mode":       "blur",
}


# ─────────────────────────────────────────────────────────────────────────────
# Routes HTTP
# ─────────────────────────────────────────────────────────────────────────────

async def handle_offer(request: web.Request) -> web.Response:
    """
    POST /offer
    Body JSON : { sdp, type }
    Retourne : { sdp, type }
    """
    params = await request.json()
    offer  = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    log.info("New peer connection — total: %d", len(pcs))

    @pc.on("connectionstatechange")
    async def on_state():
        log.info("PC state → %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log.info("Track received: %s", track.kind)
        if track.kind == "video":
            transform = VideoTransformTrack(relay.subscribe(track), config)
            transforms.append(transform)
            pc.addTrack(transform)

            @track.on("ended")
            async def on_ended():
                log.info("Track ended")
                if transform in transforms:
                    transforms.remove(transform)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        body=json.dumps({
            "sdp":  pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }),
    )


async def handle_control(request: web.Request) -> web.Response:
    """
    POST /control
    Body : { "key": "privacy_enabled", "value": false }
    Permet au dashboard de toggle les modules en temps réel.
    """
    body = await request.json()
    key, val = body.get("key"), body.get("value")

    if key in config:
        config[key] = val
        # Propager aux transforms actifs
        for t in transforms:
            if key == "privacy_enabled":
                if val != t.masker.enabled:
                    t.masker.toggle()
            elif key == "ar_enabled":
                if val != t.drawer.enabled:
                    t.drawer.toggle()
            elif key == "mask_mode":
                t.masker.set_mode(val)
            elif key == "show_hud":
                t._config["show_hud"] = val
        log.info("Config updated: %s = %s", key, val)
        return web.json_response({"ok": True, "config": config})

    return web.json_response({"ok": False, "error": "Unknown key"}, status=400)


async def handle_clear_canvas(request: web.Request) -> web.Response:
    """POST /canvas/clear — efface le dessin AR"""
    for t in transforms:
        t.drawer.clear_canvas()
    return web.json_response({"ok": True})


async def handle_stats_rest(request: web.Request) -> web.Response:
    """GET /stats — snapshot JSON des métriques"""
    if transforms:
        return web.json_response(transforms[-1].runtime_stats)
    return web.json_response({})


async def handle_ws_stats(request: web.Request) -> web.WebSocketResponse:
    """
    WebSocket /ws/stats
    Pousse les métriques toutes les 500 ms → dashboard temps réel.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    log.info("WS client connected")

    try:
        while not ws.closed:
            if transforms:
                stats = transforms[-1].runtime_stats
                await ws.send_str(json.dumps(stats))
            await asyncio.sleep(0.5)
    except Exception:
        pass
    finally:
        log.info("WS client disconnected")

    return ws


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(Path(__file__).parent.parent / "static" / "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

async def on_shutdown(app):
    log.info("Shutting down — closing %d peer connections", len(pcs))
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
    pcs.clear()
    transforms.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    # Servir les fichiers statiques
    app.router.add_get("/",              handle_index)
    app.router.add_static("/static",     Path(__file__).parent.parent / "static")

    # API WebRTC
    app.router.add_post("/offer",        handle_offer)

    # API contrôle
    app.router.add_post("/control",      handle_control)
    app.router.add_post("/canvas/clear", handle_clear_canvas)

    # Stats
    app.router.add_get("/stats",         handle_stats_rest)
    app.router.add_get("/ws/stats",      handle_ws_stats)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionEdge AI Proxy Server")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    default=8080, type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    app = create_app()
    log.info("🚀 VisionEdge server → http://%s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, access_log=None)
