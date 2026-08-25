from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .config import settings
from .pose_backend import ModelNotFoundError
from .schemas import ErrorMessage
from .session import GestureSession
from .shared_detector import SharedPoseDetector

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading pose model from %s", settings.model_path)
    try:
        _state["detector"] = SharedPoseDetector(settings)
    except ModelNotFoundError as exc:
        log.error(str(exc))
        raise
    log.info("Model loaded, ready to accept connections.")
    yield
    _state["detector"].close()
    log.info("Shut down cleanly.")


app = FastAPI(title="Real-Time Gesture Recognition", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": "detector" in _state}


@app.websocket("/ws/gestures")
async def gestures_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session = GestureSession(settings, _state["detector"])
    min_interval = 1.0 / settings.max_process_fps
    last_processed = 0.0
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    log.info("Client connected: %s", client)

    try:
        while True:
            payload = await websocket.receive_bytes()

            if len(payload) > settings.max_frame_bytes:
                await websocket.send_json(
                    ErrorMessage(error="frame too large").model_dump()
                )
                continue

            now = time.time()
            if now - last_processed < min_interval:
                # Client is sending faster than we process; drop this frame
                # instead of queueing work the server can never catch up on.
                continue
            last_processed = now

            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json(
                    ErrorMessage(error="could not decode frame").model_dump()
                )
                continue

            try:
                result = await session.process_frame(frame)
            except Exception:
                log.exception("Frame processing failed for %s", client)
                await websocket.send_json(
                    ErrorMessage(error="processing failed").model_dump()
                )
                continue

            await websocket.send_json(result.model_dump())

    except WebSocketDisconnect:
        pass
    finally:
        log.info("Client disconnected: %s", client)


# backend/app/main.py -> repo_root/frontend, in both local dev and the
# Docker image (which mirrors the repo's backend/ + frontend/ layout).
_frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
else:
    log.warning("Frontend directory not found at %s; serving API only.", _frontend_dir)
