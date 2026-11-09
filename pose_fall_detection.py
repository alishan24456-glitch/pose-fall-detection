#!/usr/bin/env python3
"""
Pose Fall Detection
===================

Watches a fixed camera and raises an alarm when a person falls and stays
down. It tracks the body's posture from a pose skeleton:

* **Torso angle** -- the angle of the shoulder-to-hip line away from
  vertical. Near 0 degrees the person is upright; near 90 degrees the
  body is horizontal.
* **Posture states** -- STANDING, SITTING or FALLEN, classified from the
  torso angle plus the body's bounding-box shape (a seated body is short
  and wide compared with a standing one; a fallen body is nearly flat).
* **Fall alert** -- a FALLEN state that lasts at least ``--alert-seconds``
  (default 2.0 s) raises a FALL ALERT (flashing banner + alert counter +
  timestamped event log) that only clears once the person is upright
  again. A fall is only a fall if the person stays down.

One shared posture pipeline serves both run modes:

* **Real mode** (``--source``) -- MediaPipe Pose tracks the body in a
  video file or webcam feed and supplies shoulder/hip landmarks.
* **Simulation mode** (``--simulate``) -- a room is rendered with OpenCV
  (a person walks in, sits on a chair, stands, walks across the room,
  trips and falls, and stays down) and the scene publishes a pose sample
  estimated from the rendered pixels only -- head position from the
  skin-coloured blob, torso axis from head to body centroid, body shape
  from the silhouette bounding box. The posture analyser only ever sees
  that pixel-derived sample; it never peeks at the scene's scripted
  choreography (it does not know "a fall is scheduled at 18 s").

How posture is judged -- and what it is NOT
--------------------------------------------
This is a geometry heuristic on pose landmarks (or on a pixel-estimated
torso axis), not a trained fall classifier. It is honest about that in
the README, and real-mode reliability depends on a clear, mostly
full-body view of one person with the camera roughly level.

* ``torso_angle`` = angle of the shoulder-mid to hip-mid line from
  vertical (real mode), or of the head-to-body-centroid line from
  vertical (simulation pixel estimator).
* **FALLEN** = torso angle at or above ``--fall-angle`` (default 55
  degrees) sustained for at least 0.5 s (15 frames at 30 fps).
* **SITTING** = torso near-upright (< 20°) **and** a seated-leg signal:
  in real mode the knee sits level with the hip (knee-minus-hip drop
  under 0.35 torso lengths, vs ~0.6–0.8 standing); in simulation, where
  knees are not measurable from pixels, a short/wide body box (aspect
  below 2.0) with the hips low in the frame (hip height above 58%).
* **STANDING** = everything else upright.
* A **fall** is one entry into the FALLEN state. A **FALL ALERT** is a
  FALLEN spell lasting at least ``--alert-seconds``.
* STANDING/SITTING changes only commit after holding for 10 frames
  (~0.33 s), so a distant or partly occluded walker cannot flicker the
  board frame to frame.

The rendered person in the simulation is drawn from the same joint
geometry the pixel estimator measures (a torso drawn horizontal has a
near-horizontal head-to-centroid line), so what the viewer sees matches
what the analyser measures.

Examples
--------
    python pose_fall_detection.py --simulate
    python pose_fall_detection.py --simulate --output demo.mp4 --no-display
    python pose_fall_detection.py --source living_room.mp4
    python pose_fall_detection.py --source 0
    python pose_fall_detection.py --source pick
    python pose_fall_detection.py            (opens the file picker)

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_WIDTH, SIM_HEIGHT = 1280, 720   # simulation canvas
SIM_FPS = 30                        # simulation frame rate
FLOOR_Y = 620                       # floor line in the simulation room

FALL_ANGLE = 55.0        # torso angle (deg from vertical) counted as fallen
FALL_FRAMES = 15         # fallen must hold this long (~0.5 s) to count
ALERT_SECONDS = 2.0      # fallen this long raises the FALL ALERT
SIT_ASPECT = 2.0         # bbox height/width below this + upright = sitting
SMOOTH_WINDOW = 5        # median filter over recent torso angles

# Person palette (saturated, so the pixel estimator can separate the
# figure from the deliberately desaturated room).
COLOUR_SKIN = (140, 180, 240)       # BGR peach
COLOUR_SHIRT = (230, 120, 30)       # BGR strong blue
COLOUR_TROUSERS = (70, 70, 110)     # BGR dark slate
COLOUR_SHOES = (40, 40, 40)

COLOUR_STANDING = (0, 190, 70)      # green (BGR)
COLOUR_SITTING = (0, 200, 230)      # amber (BGR)
COLOUR_FALLEN = (0, 55, 230)        # red (BGR)
COLOUR_TEXT = (255, 255, 255)
COLOUR_SKELETON = (0, 255, 255)     # yellow estimated skeleton

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Pose sample -- the one interface both providers speak
# ---------------------------------------------------------------------------

@dataclass
class PoseSample:
    """Everything the posture analyser needs, in image coordinates.

    ``torso_angle`` is the angle of the shoulder-mid to hip-mid line
    (real mode) or head-to-body-centroid line (simulation) away from
    vertical, in degrees. ``bbox`` is the body's bounding box
    (x1, y1, x2, y2); ``aspect`` is its height / width.
    ``hip_y_frac`` is the hip (or body-centre) height as a fraction of
    the frame height. ``leg_drop`` is (knee height - hip height) /
    torso length in real mode -- near 1.0 standing (knee a full thigh
    below the hip), near 0.0 sitting (thigh horizontal, knee level with
    the hip); None when the knees are not measurable (simulation).
    ``points`` are the joints drawn on the overlay.
    """

    torso_angle: float
    bbox: tuple[int, int, int, int]
    hip_y_frac: float
    points: dict[str, tuple[int, int]]
    leg_drop: float | None = None

    @property
    def aspect(self) -> float:
        x1, y1, x2, y2 = self.bbox
        w = max(1, x2 - x1)
        return (y2 - y1) / w


class PoseProvider:
    """Interface every pose source implements: frame in, sample out."""

    def sample(self, frame: np.ndarray) -> PoseSample | None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real mode: MediaPipe Pose (Tasks API, classic solutions as fallback)
# ---------------------------------------------------------------------------

class MediaPipePoseProvider(PoseProvider):
    """Real-mode provider backed by MediaPipe Pose (lazily imported).

    Works with BOTH mediapipe APIs: the classic solutions Pose when
    present, and the current Tasks API PoseLandmarker on newer builds
    where ``mp.solutions`` no longer exists. Landmark indices are the
    standard Pose layout in both APIs: nose 0, shoulders 11/12, hips
    23/24, knees 25/26, ankles 27/28.
    """

    NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28

    TASKS_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
    TASKS_MODEL_FILE = "pose_landmarker_lite.task"

    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Real mode needs the 'mediapipe' package.\n"
                "Install it with:  pip install mediapipe\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc
        self._mp = mp
        self._pose = None        # legacy solutions Pose, when present
        self._landmarker = None  # Tasks API PoseLandmarker, the fallback
        self._frame_index = 0

        # Backend 1 (preferred): the classic solutions Pose, present in
        # mediapipe <= 0.10.14 and many later builds.
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "pose"):
            self._pose = solutions.pose.Pose(
                static_image_mode=False, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            return

        # Backend 2: the current MediaPipe Tasks API. Newer mediapipe
        # builds removed mp.solutions entirely and ship only this API.
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError(
                "This mediapipe build has neither the classic Pose "
                "(mp.solutions) nor the Tasks API.\n"
                "Fix:  pip install --upgrade mediapipe\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc
        model_path = self._ensure_tasks_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        try:
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                "The MediaPipe Tasks PoseLandmarker failed to start "
                f"({exc}). Try 'pip install --upgrade mediapipe'; on "
                "Linux the system OpenGL libraries may also be missing.\n"
                "(Simulation mode --simulate does not need mediapipe.)"
            ) from exc

    def _ensure_tasks_model(self) -> str:
        """Download the PoseLandmarker model next to this script if absent."""
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, self.TASKS_MODEL_FILE)
        if not os.path.exists(path):
            print(f"Downloading pose model ({self.TASKS_MODEL_FILE}) ...")
            urllib.request.urlretrieve(self.TASKS_MODEL_URL, path)
            print("Pose model downloaded.")
        return path

    # -- landmark plumbing -------------------------------------------------
    def _landmarks(self, frame):
        """Return a list of (x, y, visible) in pixels, or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        if self._pose is not None:
            res = self._pose.process(rgb)
            if not res.pose_landmarks:
                return None
            return [(lm.x * w, lm.y * h, lm.visibility >= 0.5)
                    for lm in res.pose_landmarks.landmark]
        # Tasks API path.
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_index += 1
        result = self._landmarker.detect_for_video(
            mp_image, self._frame_index * 33)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        return [(p.x * w, p.y * h, getattr(p, "visibility", 1.0) >= 0.5)
                for p in lm]

    def sample(self, frame: np.ndarray) -> PoseSample | None:
        lm = self._landmarks(frame)
        if lm is None:
            return None
        h, w = frame.shape[:2]

        def mid(a, b):
            return ((lm[a][0] + lm[b][0]) / 2, (lm[a][1] + lm[b][1]) / 2)

        if not (lm[self.L_SHOULDER][2] and lm[self.R_SHOULDER][2]
                and lm[self.L_HIP][2] and lm[self.R_HIP][2]):
            return None
        shoulder = mid(self.L_SHOULDER, self.R_SHOULDER)
        hip = mid(self.L_HIP, self.R_HIP)
        dx = shoulder[0] - hip[0]
        dy = shoulder[1] - hip[1]
        torso_angle = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 90.0

        xs = [p[0] for p in lm if p[2]]
        ys = [p[1] for p in lm if p[2]]
        bbox = (int(max(0, min(xs) - 10)), int(max(0, min(ys) - 10)),
                int(min(w, max(xs) + 10)), int(min(h, max(ys) + 10)))
        points = {
            "head": (int(lm[self.NOSE][0]), int(lm[self.NOSE][1])),
            "shoulder": (int(shoulder[0]), int(shoulder[1])),
            "hip": (int(hip[0]), int(hip[1])),
            "knee": (int(mid(self.L_KNEE, self.R_KNEE)[0]),
                     int(mid(self.L_KNEE, self.R_KNEE)[1])),
            "ankle": (int(mid(self.L_ANKLE, self.R_ANKLE)[0]),
                      int(mid(self.L_ANKLE, self.R_ANKLE)[1])),
        }
        # Knee-vs-hip geometry: the sitting test in real mode. A
        # standing knee sits about a thigh-length below the hip
        # (leg_drop ~0.6-0.8); a seated knee is level with the hip
        # (leg_drop ~0). The knee coordinates are used whenever they
        # are in a sane range -- MediaPipe's visibility score drops
        # below 0.5 for knees near the frame edge even while the
        # tracked position stays smooth and correct, so visibility
        # is NOT the gate here (shoulders/hips above still are).
        leg_drop = None
        h_frame = frame.shape[0]
        knee = mid(self.L_KNEE, self.R_KNEE)
        torso_len = abs(shoulder[1] - hip[1])
        if torso_len > 20 and -0.5 * h_frame < knee[1] < 1.5 * h_frame:
            leg_drop = (knee[1] - hip[1]) / torso_len
        return PoseSample(torso_angle=torso_angle, bbox=bbox,
                          hip_y_frac=hip[1] / h, points=points,
                          leg_drop=leg_drop)

# ---------------------------------------------------------------------------
# Simulation mode: rendered room + pixel-only pose estimation
# ---------------------------------------------------------------------------

def _smoothstep(a: float, b: float, t: float) -> float:
    """Smooth 0..1 interpolation between keyframe values a and b."""
    t = max(0.0, min(1.0, t))
    t = t * t * (3 - 2 * t)
    return a + (b - a) * t


@dataclass
class FigurePose:
    """Joint angles of the rendered figure (degrees).

    hip: (x, y) of the hip joint. torso: torso angle from vertical,
    positive = leaning toward the facing direction (left). thigh/shin:
    leg segment angles from straight down, positive = forward (left).
    arm/forearm: arm segment angles from hanging straight down.
    """

    hip: tuple[float, float]
    torso: float = 0.0
    thigh: float = 0.0
    shin: float = 0.0
    arm: float = 10.0
    forearm: float = 5.0


class SimScene:
    """Renders the room and the figure. Knows the choreography; the
    analyser never sees it -- it only ever receives rendered pixels."""

    TORSO_LEN, THIGH_LEN, SHIN_LEN = 150, 120, 120
    ARM_LEN, FOREARM_LEN = 100, 95
    HEAD_R = 26
    CHAIR_X = 850
    CHAIR_SEAT_Y = 490

    def figure_pose(self, t: float) -> FigurePose:
        """Keyframe choreography for time t (seconds into the run)."""
        # Standing idle at the door (0-3 s).
        if t < 3.0:
            return FigurePose(hip=(430, 380), torso=2.0,
                              thigh=0, shin=0, arm=10, forearm=5)
        # Walk to the chair (3-6 s), legs swinging.
        if t < 6.0:
            k = (t - 3.0) / 3.0
            x = _smoothstep(430, self.CHAIR_X, k)
            swing = 25 * math.sin(k * math.pi * 4)
            return FigurePose(hip=(x, 380 - 5 * abs(math.sin(k * math.pi * 4))),
                              torso=4.0, thigh=swing, shin=-swing * 0.6,
                              arm=-swing * 0.6, forearm=10)
        # Sit down (6-7 s).
        if t < 7.0:
            k = (t - 6.0)
            return FigurePose(
                hip=(_smoothstep(self.CHAIR_X, self.CHAIR_X, k),
                     _smoothstep(380, self.CHAIR_SEAT_Y, k)),
                torso=_smoothstep(4, 5, k), thigh=_smoothstep(0, 85, k),
                shin=_smoothstep(0, 5, k), arm=_smoothstep(10, 45, k),
                forearm=_smoothstep(5, 75, k))
        # Sitting on the chair (7-13 s), breathing sway.
        if t < 13.0:
            return FigurePose(hip=(self.CHAIR_X, self.CHAIR_SEAT_Y),
                              torso=5 + 2 * math.sin(t * 1.5),
                              thigh=85, shin=5, arm=45, forearm=75)
        # Stand up (13-14.5 s).
        if t < 14.5:
            k = (t - 13.0) / 1.5
            return FigurePose(
                hip=(self.CHAIR_X, _smoothstep(self.CHAIR_SEAT_Y, 380, k)),
                torso=_smoothstep(5, 8, k), thigh=_smoothstep(85, 0, k),
                shin=_smoothstep(5, 0, k), arm=_smoothstep(45, 10, k),
                forearm=_smoothstep(75, 5, k))
        # Walk left across the room (14.5-17.5 s).
        if t < 17.5:
            k = (t - 14.5) / 3.0
            x = _smoothstep(self.CHAIR_X, 560, k)
            swing = 25 * math.sin(k * math.pi * 4)
            return FigurePose(hip=(x, 380 - 5 * abs(math.sin(k * math.pi * 4))),
                              torso=4.0, thigh=swing, shin=-swing * 0.6,
                              arm=-swing * 0.6, forearm=10)
        # Standing, about to trip (17.5-18.0 s).
        if t < 18.0:
            return FigurePose(hip=(560, 380), torso=3.0,
                              thigh=0, shin=0, arm=10, forearm=5)
        # Trip / stumble (18.0-18.6 s): torso pitches forward, arms flail.
        if t < 18.6:
            k = (t - 18.0) / 0.6
            return FigurePose(
                hip=(560, _smoothstep(380, 420, k)),
                torso=_smoothstep(3, 35, k), thigh=_smoothstep(0, 45, k),
                shin=_smoothstep(0, 10, k), arm=_smoothstep(10, -60, k),
                forearm=_smoothstep(5, -40, k))
        # Fall to the floor (18.6-19.3 s): body rotates to horizontal.
        if t < 19.3:
            k = (t - 18.6) / 0.7
            return FigurePose(
                hip=(560, _smoothstep(420, 592, k)),
                torso=_smoothstep(35, 88, k), thigh=_smoothstep(45, -85, k),
                shin=_smoothstep(10, -85, k), arm=_smoothstep(-60, -80, k),
                forearm=_smoothstep(-40, -60, k))
        # Lying on the floor (19.3-30 s): still, faint breathing bob,
        # one weak arm lift around 24 s.
        bob = 2 * math.sin(t * 2.0)
        arm = -80
        if 24.0 <= t < 25.0:
            arm = _smoothstep(-80, -70, (t - 24.0)) if t < 24.5 else \
                _smoothstep(-70, -80, (t - 24.5) / 0.5)
        return FigurePose(hip=(560, 592 + bob), torso=88 + 0.5 * math.sin(t * 2.0),
                          thigh=-85, shin=-85, arm=arm, forearm=-60)

    def _segment(self, hip, length, angle_deg, up=False):
        """End point of a segment starting at hip, angle from vertical
        (positive = toward the facing direction, left). ``up=True``
        points the segment upward (torso/head); the default points it
        downward (legs, hanging arms)."""
        a = math.radians(angle_deg)
        dx = -math.sin(a)
        dy = -math.cos(a) if up else math.cos(a)
        return (hip[0] + length * dx, hip[1] + length * dy)

    def render(self, t: float) -> np.ndarray:
        frame = np.full((SIM_HEIGHT, SIM_WIDTH, 3), (205, 200, 195), np.uint8)
        # Wall / floor / skirting.
        cv2.rectangle(frame, (0, FLOOR_Y), (SIM_WIDTH, SIM_HEIGHT),
                      (150, 140, 130), -1)
        cv2.line(frame, (0, FLOOR_Y), (SIM_WIDTH, FLOOR_Y), (90, 85, 80), 4)
        # Window, wall clock, rug, plant (all desaturated so the pixel
        # estimator never mistakes them for the person).
        cv2.rectangle(frame, (80, 90), (300, 330), (170, 175, 180), -1)
        cv2.rectangle(frame, (80, 90), (300, 330), (120, 120, 120), 3)
        cv2.line(frame, (190, 90), (190, 330), (120, 120, 120), 3)
        cv2.line(frame, (80, 210), (300, 210), (120, 120, 120), 3)
        cv2.circle(frame, (1120, 140), 55, (185, 185, 185), -1)
        cv2.circle(frame, (1120, 140), 55, (110, 110, 110), 3)
        cv2.line(frame, (1120, 140), (1120, 102), (110, 110, 110), 3)
        cv2.line(frame, (1120, 140), (1148, 140), (110, 110, 110), 3)
        cv2.rectangle(frame, (430, 560), (760, 640), (165, 150, 145), -1)
        cv2.rectangle(frame, (430, 560), (760, 640), (130, 120, 115), 2)
        # Chair.
        cv2.rectangle(frame, (self.CHAIR_X - 15, 330),
                      (self.CHAIR_X + 75, self.CHAIR_SEAT_Y), (120, 115, 110), -1)
        cv2.rectangle(frame, (self.CHAIR_X - 15, self.CHAIR_SEAT_Y),
                      (self.CHAIR_X + 75, self.CHAIR_SEAT_Y + 12), (100, 95, 90), -1)
        for lx in (self.CHAIR_X - 10, self.CHAIR_X + 65):
            cv2.line(frame, (lx, self.CHAIR_SEAT_Y + 12), (lx, FLOOR_Y),
                     (100, 95, 90), 6)
        # Door (left).
        cv2.rectangle(frame, (0, 180), (60, FLOOR_Y), (160, 155, 150), -1)
        cv2.rectangle(frame, (0, 180), (60, FLOOR_Y), (110, 105, 100), 3)

        # -- the figure -------------------------------------------------
        p = self.figure_pose(t)
        hx, hy = p.hip
        shoulder = self._segment((hx, hy), self.TORSO_LEN, p.torso, up=True)
        head = self._segment(shoulder, 32, p.torso, up=True)
        knee = self._segment((hx, hy), self.THIGH_LEN, p.thigh)
        ankle = self._segment(knee, self.SHIN_LEN, p.shin)
        elbow = self._segment(shoulder, self.ARM_LEN, p.arm)
        wrist = self._segment(elbow, self.FOREARM_LEN, p.forearm)

        def line(a, b, colour, thick):
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     colour, thick, cv2.LINE_AA)

        line((hx, hy), shoulder, COLOUR_SHIRT, 34)      # torso
        line(shoulder, elbow, COLOUR_SHIRT, 20)         # upper arm
        line(elbow, wrist, COLOUR_SKIN, 16)             # forearm + hand
        line((hx, hy), knee, COLOUR_TROUSERS, 26)       # thigh
        line(knee, ankle, COLOUR_TROUSERS, 22)          # shin
        cv2.circle(frame, (int(ankle[0]), int(ankle[1])), 12,
                   COLOUR_SHOES, -1)
        cv2.circle(frame, (int(head[0]), int(head[1])), self.HEAD_R,
                   COLOUR_SKIN, -1)                     # head
        return frame


class PixelPoseEstimator(PoseProvider):
    """Simulation-mode provider: estimates a pose sample from the
    rendered pixels only. The room is deliberately desaturated, the
    person saturated, so the figure separates cleanly by saturation;
    the head is the largest skin-coloured blob."""

    def sample(self, frame: np.ndarray) -> PoseSample | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        person = cv2.inRange(hsv, np.array([0, 85, 40]),
                             np.array([180, 255, 255]))
        person = cv2.morphologyEx(
            person, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        contours, _ = cv2.findContours(person, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 2000:
            return None
        x, y, w, h = cv2.boundingRect(cnt)
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Head: largest skin-coloured blob (hue 5-30, saturated).
        skin = cv2.inRange(hsv, np.array([5, 80, 120]),
                           np.array([30, 255, 255]))
        skin_contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        if not skin_contours:
            return None
        head_cnt = max(skin_contours, key=cv2.contourArea)
        if cv2.contourArea(head_cnt) < 300:
            return None
        hm = cv2.moments(head_cnt)
        head = (hm["m10"] / hm["m00"], hm["m01"] / hm["m00"])

        # Torso angle: head-to-body-centroid line away from vertical.
        dx = head[0] - cx
        dy = head[1] - cy
        torso_angle = abs(math.degrees(math.atan2(dx, -dy))) if dy != 0 else 90.0
        h_frame = frame.shape[0]
        points = {
            "head": (int(head[0]), int(head[1])),
            "shoulder": (int(head[0] + (cx - head[0]) * 0.35),
                         int(head[1] + (cy - head[1]) * 0.35)),
            "hip": (int(cx), int(cy)),
            "knee": (int(cx), int(min(h_frame - 5, cy + (y + h - cy) * 0.5))),
            "ankle": (int(cx), int(y + h)),
        }
        return PoseSample(torso_angle=torso_angle,
                          bbox=(x, y, x + w, y + h),
                          hip_y_frac=cy / h_frame, points=points)

# ---------------------------------------------------------------------------
# Posture analyser -- shared by both modes
# ---------------------------------------------------------------------------

class FallAnalyzer:
    """Classifies posture from PoseSamples and raises fall alerts.

    The analyser only ever sees PoseSamples -- it has no access to the
    simulation choreography or to the raw video. State changes require
    a sustained signal (median-filtered torso angle, 0.5 s to enter
    FALLEN, 2 s fallen to raise the alert), so a single noisy frame
    never triggers anything.
    """

    def __init__(self, fall_angle=FALL_ANGLE, fall_frames=FALL_FRAMES,
                 alert_seconds=ALERT_SECONDS, sit_aspect=SIT_ASPECT,
                 log_path="events.log"):
        self.fall_angle = fall_angle
        self.fall_frames = fall_frames
        self.alert_frames = int(alert_seconds * SIM_FPS)
        self.sit_aspect = sit_aspect
        self.log_path = log_path
        self.angles = deque(maxlen=SMOOTH_WINDOW)
        self.state = "NO PERSON"
        self.fallen_frames = 0        # consecutive frames in FALLEN
        self.fall_start_t = None
        self.alert_raised = False
        self.counters = {
            "frames": 0, "pose_frames": 0,
            "standing_frames": 0, "sitting_frames": 0, "fallen_frames": 0,
            "falls": 0, "alerts": 0,
        }
        self.longest_fallen_s = 0.0
        self._pending_posture = None   # hysteresis for STANDING/SITTING flips
        self._pending_frames = 0
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._log("Fall detection started.")

    def _log(self, message: str, t: float | None = None):
        stamp = f"{t:6.1f}s" if t is not None else "      "
        self._log_file.write(f"[{stamp}] {message}\n")
        self._log_file.flush()

    def classify(self, sample: PoseSample) -> str:
        """Posture from one (smoothed) sample."""
        self.angles.append(sample.torso_angle)
        angle = sorted(self.angles)[len(self.angles) // 2]
        if angle >= self.fall_angle:
            return "FALLEN"
        # Sitting needs a near-upright torso plus a seated-leg signal.
        # Real mode measures the legs directly: a seated knee is level
        # with the hip (leg_drop near 0), a standing knee a full thigh
        # below it (leg_drop ~0.6-0.8). The simulation has no measurable
        # knees, so it falls back to body shape + hip height -- valid
        # there because the framing is fixed, unlike a real camera
        # where a far-away standing person also looks short and wide.
        if angle < 20.0:
            if sample.leg_drop is not None:
                if sample.leg_drop < 0.35:
                    return "SITTING"
            elif (sample.aspect < self.sit_aspect
                    and sample.hip_y_frac > 0.58):
                return "SITTING"
        return "STANDING"

    def update(self, sample: PoseSample | None, t: float):
        self.counters["frames"] += 1
        if sample is None:
            if self.state != "NO PERSON":
                self._log(f"Person lost ({self.state} -> NO PERSON)", t)
                self.state = "NO PERSON"
                self.fallen_frames = 0
                self.alert_raised = False
                self._pending_posture = None
                self._pending_frames = 0
            return None, self.state
        self.counters["pose_frames"] += 1
        posture = self.classify(sample)
        key = {"STANDING": "standing_frames", "SITTING": "sitting_frames",
               "FALLEN": "fallen_frames"}[posture]
        self.counters[key] += 1

        if posture == "FALLEN":
            self._pending_posture = None
            self._pending_frames = 0
            if self.fallen_frames == 0:
                self.fall_start_t = t
            self.fallen_frames += 1
            if self.fallen_frames == self.fall_frames:
                self.state = "FALLEN"
                self.counters["falls"] += 1
                self._log("FALLEN detected "
                          f"(torso {sample.torso_angle:.0f} deg)", t)
            spell = t - (self.fall_start_t if self.fall_start_t is not None
                         else t)
            self.longest_fallen_s = max(self.longest_fallen_s, spell)
            if (self.state == "FALLEN" and not self.alert_raised
                    and self.fallen_frames >= self.alert_frames):
                self.alert_raised = True
                self.counters["alerts"] += 1
                self._log("!!! FALL ALERT -- person down for "
                          f"{spell:.1f} s, help needed", t)
        else:
            if self.state == "FALLEN" and self.alert_raised:
                self._log("Alert cleared -- person upright again", t)
            elif self.state == "FALLEN":
                self._log("Person got up before the alert fired", t)
            self.fallen_frames = 0
            self.alert_raised = False
            # STANDING/SITTING flips need to hold for 10 frames (0.33 s)
            # before the state commits -- on real footage a distant or
            # partly occluded walker can flip the raw classification
            # frame to frame, and the board should not flicker with it.
            if posture != self.state:
                if self._pending_posture != posture:
                    self._pending_posture = posture
                    self._pending_frames = 0
                self._pending_frames += 1
                if self._pending_frames >= 10:
                    self._log(f"Posture: {self.state} -> {posture}", t)
                    self.state = posture
                    self._pending_posture = None
                    self._pending_frames = 0
            else:
                self._pending_posture = None
                self._pending_frames = 0
        return posture, self.state

    def close(self):
        self._log("Fall detection stopped.")
        self._log_file.close()


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

STATE_COLOURS = {"STANDING": COLOUR_STANDING, "SITTING": COLOUR_SITTING,
                 "FALLEN": COLOUR_FALLEN, "NO PERSON": (200, 200, 200)}


def draw_overlay(frame, sample, posture, state, analyzer, t, fps):
    h, w = frame.shape[:2]
    colour = STATE_COLOURS.get(state, COLOUR_TEXT)
    if sample is not None:
        pts = sample.points
        for a, b in (("head", "shoulder"), ("shoulder", "hip"),
                     ("hip", "knee"), ("knee", "ankle")):
            cv2.line(frame, pts[a], pts[b], COLOUR_SKELETON, 3, cv2.LINE_AA)
        for name, pt in pts.items():
            cv2.circle(frame, pt, 6 if name == "head" else 5,
                       COLOUR_SKELETON, -1)
        x1, y1, x2, y2 = sample.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"torso {sample.torso_angle:.0f} deg  "
                           f"aspect {sample.aspect:.2f}",
                    (20, h - 20), FONT, 0.6, COLOUR_TEXT, 2, cv2.LINE_AA)
    # Status panel.
    cv2.rectangle(frame, (0, 0), (w, 74), (0, 0, 0), -1)
    cv2.putText(frame, f"POSE FALL DETECTION   {state}",
                (20, 32), FONT, 0.9, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, f"t={t:5.1f}s   fps={fps:4.1f}   "
                       f"falls={analyzer.counters['falls']}   "
                       f"alerts={analyzer.counters['alerts']}",
                (20, 62), FONT, 0.6, COLOUR_TEXT, 2, cv2.LINE_AA)
    if state == "FALLEN":
        start = analyzer.fall_start_t if analyzer.fall_start_t is not None else t
        down_for = t - start
        cv2.putText(frame, f"PERSON DOWN {down_for:.1f}s",
                    (w - 430, 32), FONT, 0.9, COLOUR_FALLEN, 2, cv2.LINE_AA)
    if analyzer.alert_raised and int(t * 4) % 2 == 0:
        cv2.rectangle(frame, (0, 90), (w, 150), (0, 0, 200), -1)
        cv2.putText(frame, "!!!  FALL ALERT -- PERSON DOWN, HELP NEEDED  !!!",
                    (90, 130), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Source selection (Windows CMD file chooser, per the standing rule)
# ---------------------------------------------------------------------------

def pick_video_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pose fall detection: posture states + fall alert.")
    parser.add_argument("--source", default=None,
                        help="video file, webcam index, or 'pick' "
                             "(default: opens the file chooser)")
    parser.add_argument("--simulate", action="store_true",
                        help="rendered room simulation instead of a camera")
    parser.add_argument("--output", default="output.mp4",
                        help="annotated video written here")
    parser.add_argument("--log", default="events.log",
                        help="timestamped event log")
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="stop after this many seconds (simulation)")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a preview window")
    parser.add_argument("--fall-angle", type=float, default=FALL_ANGLE)
    parser.add_argument("--fall-frames", type=int, default=FALL_FRAMES)
    parser.add_argument("--alert-seconds", type=float, default=ALERT_SECONDS)
    args = parser.parse_args(argv)

    if args.simulate:
        provider: PoseProvider = PixelPoseEstimator()
        scene = SimScene()
        fps = SIM_FPS
        total_frames = int(args.max_seconds * SIM_FPS)
        cap = None
        width, height = SIM_WIDTH, SIM_HEIGHT
    else:
        source = args.source
        if source is None or str(source).lower() == "pick":
            picked = pick_video_file()
            if not picked:
                print("No video file chosen. Exiting.")
                return 1
            source = picked
        if str(source).isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open video source: {source}")
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        provider = MediaPipePoseProvider()
        scene = None
        total_frames = None

    analyzer = FallAnalyzer(fall_angle=args.fall_angle,
                            fall_frames=args.fall_frames,
                            alert_seconds=args.alert_seconds,
                            log_path=args.log)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"Could not open video writer: {args.output}")
        return 1

    frame_index = 0
    t_start = time.time()
    try:
        while True:
            if scene is not None:
                if frame_index >= total_frames:
                    break
                t = frame_index / SIM_FPS
                frame = scene.render(t)
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_index / fps
                if args.max_seconds and t >= args.max_seconds:
                    break
            sample = provider.sample(frame)
            posture, state = analyzer.update(sample, t)
            elapsed = time.time() - t_start
            run_fps = (frame_index + 1) / max(elapsed, 1e-6)
            frame = draw_overlay(frame, sample, posture, state,
                                 analyzer, t, run_fps)
            writer.write(frame)
            if not args.no_display:
                cv2.imshow("Pose Fall Detection  (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            frame_index += 1
    finally:
        writer.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        analyzer.close()

    c = analyzer.counters
    duration = c["frames"] / fps
    print("\n--- Pose Fall Detection: end-of-run summary ---")
    print(f"Frames processed:      {c['frames']} "
          f"({duration:.1f} s at {fps:.0f} fps)")
    print(f"Frames with a pose:    {c['pose_frames']} "
          f"({100.0 * c['pose_frames'] / max(1, c['frames']):.1f}%)")
    print(f"Standing frames:       {c['standing_frames']}")
    print(f"Sitting frames:        {c['sitting_frames']}")
    print(f"Fallen frames:         {c['fallen_frames']}")
    print(f"Falls detected:        {c['falls']}")
    print(f"FALL ALERTS raised:    {c['alerts']}")
    print(f"Longest fallen spell:  {analyzer.longest_fallen_s:.1f} s")
    print(f"Final state:           {analyzer.state}")
    print(f"Annotated video:       {args.output}")
    print(f"Event log:             {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
