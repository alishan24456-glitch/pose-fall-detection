# Pose Fall Detection — Posture States + Fall Alert

Watches a fixed camera and raises an alarm when a person falls and stays
down. It tracks the body's posture from a pose skeleton and classifies
every frame as **STANDING**, **SITTING** or **FALLEN**, with a flashing
**FALL ALERT** when a fall lasts 2 seconds. The elderly-care and
workplace-safety use case that dominates pose-estimation demand.

> **Honest framing:** this is a geometry heuristic on pose landmarks
> (torso angle + body shape), not a trained fall classifier. A fall is
> only a fall if the person stays down — a quick bend, a crouch or a
> stumble that recovers never raises the alert. Real-mode reliability
> depends on a clear, mostly full-body view of one person with the
> camera roughly level; a person half out of frame, or two people
> overlapping, will confuse it.

## How posture is decided

| Step | What the pipeline actually measures |
|---|---|
| **Torso angle** | Angle of the shoulder-mid to hip-mid line away from vertical (real mode, MediaPipe Pose) or of the head-to-body-centroid line (simulation pixel estimator). Near 0° the body is upright; near 90° it is horizontal |
| **FALLEN** | Torso angle ≥ 55° (`--fall-angle`), sustained for at least 0.5 s (15 frames) — a single noisy frame never counts |
| **SITTING** | Torso near-upright (< 20°) **and** a seated-leg signal: in real mode the knee sits level with the hip (knee-minus-hip drop under 0.35 torso lengths, vs ~0.6–0.8 standing); in simulation, where knees are not measurable from pixels, a short/wide body box (height/width < 2.0) with the hips low in the frame (hip height > 58%). All together, so a stumbling body never flashes SITTING on its way to a fall, and a far-away standing person is not misread as seated |
| **STANDING** | Everything else upright |
| **FALL ALERT** | A FALLEN spell lasting ≥ 2.0 s (`--alert-seconds`): flashing banner, alert counter, timestamped event-log entry. It clears only once the person is upright again |

The torso angle is median-filtered over the last 5 frames, so jitter in
the landmarks never flips the state.

## Run it (Windows CMD)

```
pip install -r requirements.txt
python pose_fall_detection.py                  REM opens a video file chooser
python pose_fall_detection.py --source pick    REM same chooser, explicitly
python pose_fall_detection.py --source living_room.mp4
python pose_fall_detection.py --source 0       REM webcam
python pose_fall_detection.py --simulate       REM rendered room
```

Press **Q** to quit. The annotated video is written to `output.mp4`
(change with `--output`); the timestamped event log goes to `events.log`
(change with `--log`). Useful options: `--fall-angle 55`,
`--fall-frames 15`, `--alert-seconds 2.0`, `--max-seconds 30`,
`--no-display` for headless runs.

MediaPipe Pose is needed **only for real mode**. The script supports
both the classic `mp.solutions` Pose and the current Tasks API
PoseLandmarker (newer mediapipe builds removed `mp.solutions`); with
the Tasks API the `pose_landmarker_lite.task` model auto-downloads on
first use (it is also included in this repo). On Linux the Tasks API
needs the system OpenGL libraries (`libegl1`, `libgles2`) installed;
without them real mode fails at startup with a `libEGL.so.1` error and
the script says so plainly. Simulation mode (`--simulate`) never needs
mediapipe at all.

## Modes

- **Real mode** — MediaPipe Pose tracks the body in a video file or
  webcam feed and supplies shoulder/hip landmarks to the same posture
  analyser.
- **Simulation mode** (`--simulate`) — a room is rendered with OpenCV:
  a person walks in, sits on a chair for ~6 s, stands, walks across the
  room, trips at ~18 s and falls, and stays down on the floor while the
  alert fires. The scene publishes a pose sample estimated from the
  rendered pixels only — head from the skin-coloured blob, torso axis
  from head to body centroid, body shape from the silhouette bounding
  box. The posture analyser only ever sees that pixel-derived sample;
  it never peeks at the scene's scripted choreography.

## Verified demo runs

**Simulation (30 s, 900 frames):** pose estimated on **900/900 frames
(100%)**; **370 standing, 202 sitting, 328 fallen** frames. The person
sits 6.7–13.5 s, stands again, trips at ~18 s; **1 fall detected**
(torso 86° at 19.5 s) and **1 FALL ALERT raised at 21.0 s** (down for
2.0 s); longest fallen spell **10.9 s**; the run ends with the person
still down and the alert banner showing. No false SITTING during the
stumble, no alert for the chair-sitting spell.

**Real footage — warehouse aisle (Pexels, free license: "Woman Shopping
in a Grocery Store Aisle", 1280×676, first 10 s / 250 frames):** pose
found on **250/250 frames (100%)**, all **250 STANDING**, 0 falls, 0
alerts. The shopper walks the aisle pushing a cart the whole clip, so
STANDING throughout is the correct reading.

**Real footage — busy street (Pexels, free license: "People Walking on
the Street" by George Morina, 1920×1080, first 10 s / 300 frames):**
pose found on **154/300 frames (51.3%)** — this is a wide, crowded
street shot with small, overlapping pedestrians and the tracker follows
one person at a time, so it loses the subject for stretches (nothing
detected 3.8–8.3 s). **127 standing, 27 sitting, 0 fallen, 0 alerts.**
The brief SITTING spell at 8.4–9.2 s is a person low in frame whose
knees read level with the hips; on a shot this busy it is as likely a
seated passer-by as a tracking wobble, and it is reported as measured.
No fall, no alert — correct for a street full of walking people.

## Real-mode note (Linux)

On this build machine the Tasks API PoseLandmarker would not start at
first: mediapipe 1.1.1 loads `libEGL.so.1` / `libGLESv2.so.2` at
startup and neither is installed in this Linux VM (the bundled
mesa/Chromium copies on the box are incomplete — Chromium's
`libGLESv2.so` is a stub missing `glFenceSync`). The real-mode runs
above were done with the stock Ubuntu 24.04 mesa libraries
(`libegl1`, `libgles2`, `libegl-mesa0`, `libgbm1`, `libglapi-mesa`,
`libglvnd0`) extracted to a throwaway folder and pointed at with
`LD_LIBRARY_PATH` — nothing was installed system-wide. On a normal
Windows PC, or a Linux box with `libegl1`/`libgles2` installed, real
mode starts as-is and the `pose_landmarker_lite.task` model in this
repo is used directly.

## Files

- `pose_fall_detection.py` — the whole system (single file)
- `pose_landmarker_lite.task` — MediaPipe Pose model (auto-downloaded
  if missing; only used by real mode)
- `requirements.txt`, `.gitignore`

Author: Abdul Hanan — Computer Vision & AI Developer
