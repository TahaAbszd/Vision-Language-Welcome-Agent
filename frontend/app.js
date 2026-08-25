"use strict";

// Same 33-point BlazePose connection set the backend draws, kept identical
// here (see backend/app/landmarks.py) so the skeleton drawn client-side
// matches what the model actually returned.
const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];

const TARGET_FPS = 20;
const SEND_WIDTH = 640;
const SEND_HEIGHT = 480;
const JPEG_QUALITY = 0.7;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const clientFpsEl = document.getElementById("clientFps");
const serverFpsEl = document.getElementById("serverFps");
const processingMsEl = document.getElementById("processingMs");
const peopleCountEl = document.getElementById("peopleCount");

const sendCanvas = document.createElement("canvas");
sendCanvas.width = SEND_WIDTH;
sendCanvas.height = SEND_HEIGHT;
const sendCtx = sendCanvas.getContext("2d");

let stream = null;
let ws = null;
let running = false;
let awaitingResponse = false;
let lastSendTime = 0;
let lastResult = null;
let clientFrameTimes = [];
let reconnectAttempts = 0;
let reconnectTimer = null;

function setStatus(online, text) {
  statusDot.classList.toggle("online", online);
  statusText.textContent = text;
}

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws/gestures`;
}

function connect() {
  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    reconnectAttempts = 0;
    setStatus(true, "connected");
  };

  ws.onmessage = (event) => {
    awaitingResponse = false;
    try {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.warn("server reported:", data.error);
        return;
      }
      lastResult = data;
      serverFpsEl.textContent = data.server_fps.toFixed(1);
      processingMsEl.textContent = `${data.processing_ms.toFixed(0)} ms`;
      peopleCountEl.textContent = data.tracks.length;
    } catch (err) {
      console.error("bad message from server", err);
    }
  };

  ws.onclose = () => {
    setStatus(false, "disconnected");
    awaitingResponse = false;
    if (running) scheduleReconnect();
  };

  ws.onerror = () => {
    ws.close();
  };
}

function scheduleReconnect() {
  reconnectAttempts += 1;
  const delay = Math.min(1000 * 2 ** reconnectAttempts, 10000);
  setStatus(false, `reconnecting in ${Math.round(delay / 1000)}s`);
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    if (running) connect();
  }, delay);
}

function drawSkeleton(track) {
  const [r, g, b] = track.color;
  const strokeStyle = `rgb(${r}, ${g}, ${b})`;
  const pts = track.keypoints.map((k) => [
    k.x * overlay.width,
    k.y * overlay.height,
  ]);

  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 2;
  for (const [a, b_] of POSE_CONNECTIONS) {
    if (a < pts.length && b_ < pts.length) {
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b_][0], pts[b_][1]);
      ctx.stroke();
    }
  }
  ctx.fillStyle = strokeStyle;
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  const nose = pts[0];
  if (nose) {
    const label = `ID ${track.id}` + (track.gestures.length ? ` | ${track.gestures.join(" + ")}` : "");
    ctx.font = "16px -apple-system, sans-serif";
    ctx.fillStyle = strokeStyle;
    ctx.fillText(label, Math.max(nose[0], 5), Math.max(nose[1] - 12, 15));
  }
}

function renderLoop() {
  if (!running) return;

  // Draw the live (mirrored) camera feed every animation frame so the
  // video looks smooth regardless of the server round-trip time; the
  // skeleton overlay just reflects whatever the last response was.
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -overlay.width, 0, overlay.width, overlay.height);
  ctx.restore();

  if (lastResult) {
    for (const track of lastResult.tracks) drawSkeleton(track);
  }

  const now = performance.now();
  clientFrameTimes.push(now);
  clientFrameTimes = clientFrameTimes.filter((t) => now - t < 1000);
  clientFpsEl.textContent = clientFrameTimes.length;

  maybeSendFrame(now);
  requestAnimationFrame(renderLoop);
}

function maybeSendFrame(now) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (awaitingResponse) return; // one frame in flight at a time (backpressure)
  if (now - lastSendTime < 1000 / TARGET_FPS) return;
  lastSendTime = now;

  sendCtx.save();
  sendCtx.scale(-1, 1);
  sendCtx.drawImage(video, -SEND_WIDTH, 0, SEND_WIDTH, SEND_HEIGHT);
  sendCtx.restore();

  sendCanvas.toBlob(
    (blob) => {
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
      awaitingResponse = true;
      blob.arrayBuffer().then((buf) => ws.send(buf));
    },
    "image/jpeg",
    JPEG_QUALITY
  );
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: SEND_WIDTH, height: SEND_HEIGHT },
      audio: false,
    });
  } catch (err) {
    setStatus(false, `camera denied: ${err.message}`);
    return;
  }

  video.srcObject = stream;
  await video.play();

  overlay.width = SEND_WIDTH;
  overlay.height = SEND_HEIGHT;

  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  connect();
  requestAnimationFrame(renderLoop);
}

function stop() {
  running = false;
  clearTimeout(reconnectTimer);
  if (ws) ws.close();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  lastResult = null;
  setStatus(false, "stopped");
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  setStatus(false, "getUserMedia unsupported (needs https or localhost)");
  startBtn.disabled = true;
}
