# Real-time Gesture Recognition

A small webcam app that watches people in the frame and calls out a few basic gestures live: raising a hand, waving, clapping, and hugging. It draws the pose skeleton on top of the video and prints a label next to each person.

I put this together as the vision piece of a welcome-agent project. The idea was to keep it lightweight enough to run on a laptop with no GPU, so it leans on MediaPipe for pose estimation and plain geometry for the gesture logic — no model training, no dataset to collect.

## What it does

- Tracks up to a few people at once (each gets their own ID and colour).
- Detects four gestures:
  - **raise_hand** — a wrist held above the shoulder for a moment.
  - **wave** — a raised hand swinging left and right.
  - **clap** — both hands coming together quickly in front of the chest.
  - **hug** — two people whose torsos get close and at least one arm reaches around the other.
- Overlays the skeleton, the person's ID + current gesture, and a running FPS counter.

All the thresholds are measured relative to each person's shoulder width, so it doesn't fall apart when someone stands closer to or further from the camera.

## Why MediaPipe and not YOLO-Pose

I started with YOLO-Pose but it wanted a GPU to stay smooth, and the whole point here was to run on a normal machine. MediaPipe's newer `PoseLandmarker` (the Tasks API) runs fine on CPU, and it does multi-person detection out of the box via `num_poses`, which the hug gesture needs since it's a two-person thing. 33 landmarks per person is plenty for what I'm doing.

The old YOLO detector is still in `yolo_detector.py` if you want to go that route for busier scenes, but for this app MediaPipe was the better fit.

## Setup

You'll need Python 3.9+ and a webcam.

```bash
pip install mediapipe opencv-python numpy
```

Then grab the pose model once (the lite version is the default):

```bash
curl -L -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

If you want more accuracy and don't mind it running slower, swap `_lite` for `_full` or `_heavy` in both the URL and the `model_path` in the config.

## Running it

```bash
python gesture_app.py
```

A window pops up with your camera feed. Press **q** to quit. The view is mirrored on purpose — it just feels more natural when you're standing in front of it.

## Tuning

Everything you'd want to tweak lives in the `Config` dataclass near the top of the file — camera resolution, how many people to track, and the thresholds for each gesture. A few of the ones I ended up adjusting the most:

- `raise_hand_min_frames` — how long a hand has to stay up before it counts (stops flickery false positives).
- `wave_min_reversals` / `wave_min_amplitude_ratio` — how much and how many times the hand has to change direction to read as a wave.
- `clap_close_ratio` and `clap_open_ratio` — how close the hands have to get, and how far apart they had to be just before, so a clap only fires on an actual clapping motion.
- `hug_torso_ratio` / `hug_wrist_reach_ratio` — how close two people and their arms have to be.

If a gesture triggers too easily or not enough, these are the first knobs to turn.

## How it works, briefly

Each frame goes: camera → MediaPipe pose → a tiny centroid tracker that keeps people's IDs stable across frames → a short rolling history of landmarks per person → the gesture rules read that history and decide what's happening.

The tracker is deliberately simple (it matches people frame-to-frame by how close their torso centres are). It's fine for a handful of people; if you needed to handle a crowd or heavy occlusion you'd want something like ByteTrack or DeepSORT instead.

## Notes / limitations

- Best with good lighting and people mostly facing the camera.
- The hug check is pairwise, so very crowded frames will do more comparisons.
- Gestures are rule-based, not learned — they're readable and need no training, but they won't be as robust as a trained classifier on unusual poses.
