# Real-Time Gesture Recognition

Point a webcam at people and it'll call out simple gestures live — raised hand, wave, clap, hug — with a pose skeleton and label drawn over each person. Pose tracking runs on MediaPipe's `PoseLandmarker` (CPU only), and gestures are detected with plain rule-based geometry, so it works regardless of how far someone is from the camera.

## Why the browser does the camera work

Docker Desktop on macOS can't reach the host webcam — no `/dev/video*` passthrough into its Linux VM. So capture happens in the browser (which macOS *does* grant camera access to), and only the CPU-heavy inference runs in the container. The browser streams frames over a WebSocket; the backend sends back skeleton + gesture data to draw.

```
Browser (camera + drawing)  ──frames──▶  FastAPI + MediaPipe (Docker)
                             ◀──JSON────
```

- `backend/` — FastAPI service. `/health`, `/ws/gestures` (the actual pipeline), serves `frontend/` at `/`.
- `frontend/` — one plain HTML/JS page, no build step. Captures camera, sends frames, draws the response.
- `src/vision/` — the original single-process script, kept for reference/local testing.

Each WebSocket connection gets its own tracking state, so multiple viewers don't interfere with each other. Only the MediaPipe model itself is shared (behind a lock).

## Running it

**Docker (recommended):**
```bash
docker compose up --build
```
Open http://localhost:8000 and click "Start camera". (Anything beyond `localhost` needs HTTPS — browsers block camera access on plain HTTP.)

**Locally, no Docker:**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

**Tests:**
```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ -v
```
No camera/mediapipe needed — runs in under a second.

**Legacy single-file demo:**
```bash
cd src/vision
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python gesture_app.py   # press 'q' to quit
```

## Tuning

All thresholds live in `backend/app/config.py`, overridable via env vars:

- `raise_hand_min_frames` — how long a hand must stay up.
- `wave_min_reversals` / `wave_min_amplitude_ratio` — how much back-and-forth counts as a wave.
- `clap_close_ratio` / `clap_open_ratio` — how close/apart hands must get for a clap.
- `hug_torso_ratio` / `hug_wrist_reach_ratio` — how close two people and how far a wrist reaches for a hug.

## Limitations

- Needs decent lighting and people mostly facing the camera.
- Rule-based, not learned — easy to reason about, but less robust than a trained model on unusual poses.
- Hug detection compares every pair of people, fine for a handful but not a crowd (swap in ByteTrack/DeepSORT if you need real crowd tracking).

## Cleanup

If you have a stray `src/.vision/` virtualenv lying around (accidentally created inside `src/`), it's safe to delete — it's gitignored and not used by anything:
```bash
rm -rf src/.vision
```
