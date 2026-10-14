import os
import io
import math
import time
import uuid
import queue
import socket
import shutil
import logging
import tempfile
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import qrcode
from flask import Flask, Response, abort, send_file, stream_with_context

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False


# =============================================================================
# CONFIG
# Every threshold lives here with a one-line comment.
# =============================================================================
CONFIG = {
    "idle_timeout_s": 60.0,                 # seconds of no activity before returning to ATTRACT
    "session_timeout_s": 180.0,             # maximum seconds for one guided kiosk session
    "photos_per_session": 3,                # number of photos captured in one session
    "delete_after_min": 5.0,                # minutes before local files are automatically deleted
    "greeting_s": 3.0,                      # seconds to show greeting before frame selection
    "countdown_s": 3.0,                     # seconds before first photo capture
    "capture_interval_s": 1.0,              # seconds between captured photos
    "flash_ms": 150,                        # milliseconds of white flash effect after each capture
    "camera_width": 640,                    # webcam capture width in pixels
    "camera_height": 480,                   # webcam capture height in pixels
    "target_fps": 30,                       # target capture/tracking frames per second
    "stream_fps": 20,                       # MJPEG preview frames per second
    "jpeg_quality": 82,                     # JPEG quality for live preview and saved photos
    "ui_refresh_s": 0.5,                    # Streamlit panel refresh interval in seconds
    "wave_window_s": 1.2,                   # seconds window used to count wave reversals
    "wave_min_reversals": 3,                # minimum direction reversals required for a wave
    "wave_amplitude": 0.15,                 # minimum palm-x amplitude as fraction of frame width
    "wave_min_step": 0.02,                  # minimum palm-x step to count a direction reversal
    "swipe_window_s": 0.5,                  # seconds window used to detect a swipe
    "swipe_amplitude": 0.25,                # minimum palm-x travel as fraction of frame width
    "swipe_min_velocity": 0.45,             # minimum normalized palm-x velocity for a swipe
    "arm_hold_s": 1.0,                      # seconds an open palm must be held to arm risky actions
    "arm_valid_s": 8.0,                     # seconds the arming gesture remains valid
    "open_palm_enter": 0.70,                # open-palm score needed to begin arming hold
    "open_palm_exit": 0.55,                 # open-palm score needed to stay armed before release
    "refractory_s": 1.2,                    # seconds before the same gesture can fire again
    "one_euro_min_cutoff": 0.007,           # One Euro filter minimum cutoff frequency
    "one_euro_beta": 0.02,                  # One Euro filter speed adaptation factor
    "mp_min_detection_confidence": 0.50,    # MediaPipe hand detection confidence threshold
    "mp_min_tracking_confidence": 0.50,     # MediaPipe hand tracking confidence threshold
    "collage_width": 1800,                  # collage width in pixels at 300 DPI for 6 inches
    "collage_height": 2400,                 # collage height in pixels at 300 DPI for 8 inches
    "download_port": 5001,                  # preferred local Flask download/stream port
    "operator_code": "BOOTH-RESET",         # hidden operator code to force reset/override
    "max_log_mb": 2,                        # approximate maximum log file size in megabytes
}


# =============================================================================
# LOCAL DATA PATHS AND LOGGING
# No network upload. Logs contain no images.
# =============================================================================
DATA_DIR = Path(os.environ.get("BOOTHKIOSK_DATA", os.path.join(tempfile.gettempdir(), "boothkiosk")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "sessions").mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_DIR / "boothkiosk.log"

logger = logging.getLogger("boothkiosk")
logger.setLevel(logging.INFO)

if not logger.handlers:
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > CONFIG["max_log_mb"] * 1024 * 1024:
            LOG_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)


# =============================================================================
# STATES, EVENTS, AND BASIC TYPES
# =============================================================================
class KioskState(str, Enum):
    ATTRACT = "ATTRACT"
    GREETING = "GREETING"
    FRAME_SELECT = "FRAME_SELECT"
    COUNTDOWN = "COUNTDOWN"
    REVIEW = "REVIEW"
    SHARE = "SHARE"
    RESET = "RESET"


@dataclass
class GestureEvent:
    type: str
    hand: str
    t_ms: int
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)


class FrameBus:
    """Thread-safe newest-frame-only bus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Optional[np.ndarray] = None

    def put(self, item: np.ndarray) -> None:
        with self._lock:
            self._item = item

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._item


# =============================================================================
# ONE EURO FILTER
# Reduces landmark jitter without adding large lag.
# =============================================================================
class OneEuro:
    def __init__(self, min_cutoff: float = 0.007, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        if cutoff <= 0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-5))

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        x = float(x)
        t = float(t if t is not None else time.time())

        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-5)
        dx = (x - self.x_prev) / dt

        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha_x = self._alpha(cutoff, dt)
        x_hat = alpha_x * x + (1.0 - alpha_x) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class LandmarkSmoother:
    def __init__(self, cfg: Dict[str, Any], n: int = 21) -> None:
        self.n = n
        self.filters = [
            [
                OneEuro(min_cutoff=cfg["one_euro_min_cutoff"], beta=cfg["one_euro_beta"]),
                OneEuro(min_cutoff=cfg["one_euro_min_cutoff"], beta=cfg["one_euro_beta"]),
            ]
            for _ in range(n)
        ]

    def filter(self, lm: np.ndarray, t: float) -> np.ndarray:
        lm = np.asarray(lm, dtype=float)
        out = lm.copy()
        count = min(self.n, len(lm))

        for i in range(count):
            out[i, 0] = self.filters[i][0](float(lm[i, 0]), t)
            out[i, 1] = self.filters[i][1](float(lm[i, 1]), t)

        return out


# =============================================================================
# HAND GEOMETRY
# Index map:
# 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip
# 5 index MCP, 9 middle MCP, 13 ring MCP, 17 pinky MCP
# =============================================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def landmark_point(lm: np.ndarray, idx: int, default: Tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    if lm is None or idx >= len(lm):
        return np.array(default, dtype=float)
    return np.asarray(lm[idx], dtype=float)


def safe_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def palm_width(lm: np.ndarray) -> float:
    return max(safe_dist(landmark_point(lm, 5), landmark_point(lm, 17)), 1e-6)


def palm_center(lm: np.ndarray) -> np.ndarray:
    pts = [landmark_point(lm, i) for i in (0, 5, 9, 13, 17)]
    return np.mean(np.array(pts, dtype=float), axis=0)


def pinch_ratio(lm: np.ndarray) -> float:
    return safe_dist(landmark_point(lm, 4), landmark_point(lm, 8)) / palm_width(lm)


def open_palm_score(lm: np.ndarray) -> float:
    if lm is None or len(lm) < 21:
        return 0.0

    pw = palm_width(lm)
    wrist = landmark_point(lm, 0)

    extended = 0
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        tip_d = safe_dist(landmark_point(lm, tip), wrist)
        pip_d = safe_dist(landmark_point(lm, pip), wrist)
        if tip_d > pip_d * 1.10 + 1e-6:
            extended += 1

    tips = [landmark_point(lm, i) for i in (8, 12, 16, 20)]
    spreads = [safe_dist(tips[i], tips[i + 1]) for i in range(3)]
    spread = float(np.mean(spreads)) / pw if pw > 0 else 0.0

    thumb_out = safe_dist(landmark_point(lm, 4), landmark_point(lm, 17)) / pw if pw > 0 else 0.0

    score = (
        (extended / 4.0) * 0.60
        + min(spread / 0.55, 1.0) * 0.30
        + min(thumb_out / 1.10, 1.0) * 0.10
    )
    return float(score)


def synthetic_landmarks(cx: float, cy: float, open_palm: bool = False) -> np.ndarray:
    """Create a realistic 21-point hand for demo simulation and unit tests."""
    lm = np.zeros((21, 2), dtype=float)

    lm[0] = (cx, cy + 0.07)
    lm[5] = (cx - 0.050, cy)
    lm[9] = (cx - 0.017, cy - 0.004)
    lm[13] = (cx + 0.017, cy - 0.004)
    lm[17] = (cx + 0.050, cy)

    if open_palm:
        tips = [
            (cx - 0.065, cy - 0.090),
            (cx - 0.020, cy - 0.100),
            (cx + 0.020, cy - 0.095),
            (cx + 0.065, cy - 0.075),
        ]
        thumb = (cx - 0.090, cy + 0.020)
    else:
        tips = [
            (cx - 0.055, cy + 0.015),
            (cx - 0.025, cy + 0.020),
            (cx + 0.025, cy + 0.015),
            (cx + 0.055, cy + 0.010),
        ]
        thumb = (cx - 0.045, cy + 0.045)

    mcp_idxs = [5, 9, 13, 17]
    pip_idxs = [6, 10, 14, 18]
    dip_idxs = [7, 11, 15, 19]
    tip_idxs = [8, 12, 16, 20]

    for mcp_i, pip_i, dip_i, tip_i, tip_xy in zip(mcp_idxs, pip_idxs, dip_idxs, tip_idxs, tips):
        mcp = lm[mcp_i]
        tip = np.array(tip_xy, dtype=float)
        lm[pip_i] = 0.65 * mcp + 0.35 * tip
        lm[dip_i] = 0.30 * mcp + 0.70 * tip
        lm[tip_i] = tip

    lm[1] = 0.75 * lm[0] + 0.25 * np.array(thumb, dtype=float)
    lm[2] = 0.50 * lm[0] + 0.50 * np.array(thumb, dtype=float)
    lm[3] = 0.25 * lm[0] + 0.75 * np.array(thumb, dtype=float)
    lm[4] = np.array(thumb, dtype=float)

    return np.clip(lm, 0.01, 0.99)


# =============================================================================
# GESTURE ENGINE
# Emits GestureEvent objects. Uses hysteresis, hold timers, and refractory.
# =============================================================================
class GestureEngine:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.smoothers: Dict[str, LandmarkSmoother] = {}
        self.wave_buf: Dict[str, deque] = defaultdict(deque)
        self.swipe_buf: Dict[str, deque] = defaultdict(deque)
        self.last_emit: Dict[Tuple[str, str], float] = {}
        self.open_state: Dict[str, bool] = {}
        self.open_start: Dict[str, float] = {}
        self.arm_emitted: Dict[str, bool] = {}
        self.armed: bool = False
        self.armed_until: float = 0.0

    def _can_emit(self, typ: str, hand: str, t: float, cooldown: Optional[float] = None) -> bool:
        cooldown = self.cfg["refractory_s"] if cooldown is None else cooldown
        key = (typ, hand)
        if t - self.last_emit.get(key, -1e9) < cooldown:
            return False
        self.last_emit[key] = t
        return True

    def _update_wave(self, hand: str, lm: np.ndarray, t: float) -> List[GestureEvent]:
        pc = palm_center(lm)
        x = float(pc[0])
        buf = self.wave_buf[hand]
        buf.append((t, x))

        while buf and t - buf[0][0] > self.cfg["wave_window_s"]:
            buf.popleft()

        if len(buf) < 5:
            return []

        xs = [xx for _, xx in buf]
        amp = max(xs) - min(xs)

        reversals = 0
        dir_sign = 0
        last_x = buf[0][1]

        for _, xx in list(buf)[1:]:
            dx = xx - last_x
            if abs(dx) >= self.cfg["wave_min_step"]:
                s = 1 if dx > 0 else -1
                if dir_sign != 0 and s != dir_sign:
                    reversals += 1
                dir_sign = s
                last_x = xx

        if reversals >= self.cfg["wave_min_reversals"] and amp > self.cfg["wave_amplitude"]:
            if self._can_emit("wave", hand, t):
                self.wave_buf[hand].clear()
                conf = min(0.98, 0.72 + amp)
                return [
                    GestureEvent(
                        "wave",
                        hand,
                        int(t * 1000),
                        conf,
                        {
                            "amplitude": amp,
                            "reversals": reversals,
                            "palm_width": palm_width(lm),
                            "pinch_ratio": pinch_ratio(lm),
                        },
                    )
                ]

        return []

    def _update_swipe(self, hand: str, lm: np.ndarray, t: float) -> List[GestureEvent]:
        pc = palm_center(lm)
        x = float(pc[0])
        buf = self.swipe_buf[hand]
        buf.append((t, x))

        while buf and t - buf[0][0] > self.cfg["swipe_window_s"]:
            buf.popleft()

        if len(buf) < 3:
            return []

        t0, x0 = buf[0]
        t1, x1 = buf[-1]
        dt = max(t1 - t0, 1e-5)
        dx = x1 - x0

        if abs(dx) > self.cfg["swipe_amplitude"] and abs(dx) / dt > self.cfg["swipe_min_velocity"]:
            if self._can_emit("swipe", hand, t):
                self.swipe_buf[hand].clear()
                direction = "right" if dx > 0 else "left"
                conf = min(0.98, 0.70 + abs(dx))
                return [
                    GestureEvent(
                        "swipe",
                        hand,
                        int(t * 1000),
                        conf,
                        {
                            "direction": direction,
                            "dx": dx,
                            "velocity": abs(dx) / dt,
                            "palm_width": palm_width(lm),
                        },
                    )
                ]

        return []

    def _update_arm(self, hand: str, lm: np.ndarray, t: float) -> List[GestureEvent]:
        score = open_palm_score(lm)
        prev = self.open_state.get(hand, False)

        if prev:
            is_open = score >= self.cfg["open_palm_exit"]
        else:
            is_open = score >= self.cfg["open_palm_enter"]

        self.open_state[hand] = is_open

        if is_open:
            start = self.open_start.setdefault(hand, t)
            if not self.arm_emitted.get(hand, False) and t - start >= self.cfg["arm_hold_s"]:
                self.arm_emitted[hand] = True
                self.armed = True
                self.armed_until = t + self.cfg["arm_valid_s"]
                return [
                    GestureEvent(
                        "arm",
                        hand,
                        int(t * 1000),
                        0.88,
                        {
                            "score": score,
                            "hold_s": t - start,
                        },
                    )
                ]
        else:
            self.open_start.pop(hand, None)
            self.arm_emitted[hand] = False

        return []

    def process(
        self,
        raw_landmarks_list: List[np.ndarray],
        handedness_list: List[str],
        t: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[GestureEvent]]:
        t = time.time() if t is None else float(t)
        events: List[GestureEvent] = []
        filtered: List[np.ndarray] = []

        for i, raw in enumerate(raw_landmarks_list):
            hand = handedness_list[i] if i < len(handedness_list) else f"hand{i}"
            smoother = self.smoothers.setdefault(hand, LandmarkSmoother(self.cfg))
            lm = smoother.filter(raw, t)
            filtered.append(lm)

            events.extend(self._update_wave(hand, lm, t))
            events.extend(self._update_swipe(hand, lm, t))
            events.extend(self._update_arm(hand, lm, t))

        if self.armed and t > self.armed_until:
            self.armed = False

        return filtered, events


# =============================================================================
# FRAME TEMPLATES AND COLLAGE
# =============================================================================
FRAME_TEMPLATES = [
    {
        "name": "Classic Grid",
        "bg": (248, 248, 245),
        "border": (30, 30, 30),
        "accent": (180, 30, 60),
        "layout": "classic",
    },
    {
        "name": "Polaroid Strip",
        "bg": (255, 255, 255),
        "border": (20, 20, 20),
        "accent": (70, 130, 180),
        "layout": "polaroid",
    },
    {
        "name": "Neon Booth",
        "bg": (15, 15, 25),
        "border": (0, 255, 180),
        "accent": (255, 80, 160),
        "layout": "neon",
    },
]

COLLAGE_BOXES = {
    "classic": [
        (150, 260, 1500, 820),
        (150, 1150, 720, 720),
        (930, 1150, 720, 720),
    ],
    "polaroid": [
        (450, 220, 900, 620),
        (450, 920, 900, 620),
        (450, 1620, 900, 620),
    ],
    "neon": [
        (120, 260, 760, 1120),
        (920, 260, 760, 1120),
        (520, 1480, 760, 760),
    ],
}

STATE_COLORS = {
    KioskState.ATTRACT.value: (90, 40, 120),
    KioskState.GREETING.value: (0, 120, 200),
    KioskState.FRAME_SELECT.value: (0, 160, 0),
    KioskState.COUNTDOWN.value: (0, 0, 220),
    KioskState.REVIEW.value: (0, 120, 220),
    KioskState.SHARE.value: (200, 200, 0),
    KioskState.RESET.value: (0, 0, 0),
}

_FONT_CACHE: Dict[int, Any] = {}


def load_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[size] = font
            return font
        except Exception:
            pass

    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def add_cover(
    canvas: Image.Image,
    img: Image.Image,
    box: Tuple[int, int, int, int],
    border_color: Optional[Tuple[int, int, int]] = None,
    border_width: int = 0,
) -> None:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return

    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return

    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale + 0.5), int(ih * scale + 0.5)
    img = img.resize((nw, nh), Image.LANCZOS)

    left = (nw - w) // 2
    top = (nh - h) // 2
    img = img.crop((left, top, left + w, top + h))
    canvas.paste(img, (x, y))

    if border_width > 0 and border_color is not None:
        d = ImageDraw.Draw(canvas)
        d.rectangle(
            [x - border_width, y - border_width, x + w + border_width - 1, y + h + border_width - 1],
            outline=border_color,
            width=border_width,
        )


def create_collage_from_paths(photo_paths: List[str], template_idx: int, out_path: str) -> None:
    template = FRAME_TEMPLATES[int(template_idx) % len(FRAME_TEMPLATES)]
    W, H = CONFIG["collage_width"], CONFIG["collage_height"]

    canvas = Image.new("RGB", (W, H), template["bg"])
    draw = ImageDraw.Draw(canvas)
    boxes = COLLAGE_BOXES.get(template["layout"], COLLAGE_BOXES["classic"])

    if not photo_paths:
        photo_paths = []

    for i, box in enumerate(boxes):
        path = photo_paths[i % len(photo_paths)] if photo_paths else None

        if path and os.path.exists(path):
            try:
                pil = Image.open(path).convert("RGB")
            except Exception:
                pil = Image.new("RGB", (CONFIG["camera_width"], CONFIG["camera_height"]), (40, 40, 40))
        else:
            pil = Image.new("RGB", (CONFIG["camera_width"], CONFIG["camera_height"]), (40, 40, 40))

        add_cover(canvas, pil, box, template["border"], 10)

    draw.rectangle([80, 120, W - 80, 170], outline=template["accent"], width=6)

    text_color = (25, 25, 25) if sum(template["bg"]) > 380 else (245, 245, 245)
    title_font = load_font(72)
    sub_font = load_font(42)
    tiny_font = load_font(30)

    draw.text((100, 35), "BoothKiosk", font=title_font, fill=template["accent"])
    draw.text((100, 188), template["name"], font=sub_font, fill=text_color)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    draw.text((100, H - 112), f"Session created locally: {stamp}", font=tiny_font, fill=text_color)
    draw.text((100, H - 72), "No cloud upload • auto-delete enabled", font=tiny_font, fill=text_color)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=95, optimize=True)


# =============================================================================
# STATE MACHINE
# =============================================================================
@dataclass
class DisplayState:
    state: str = KioskState.ATTRACT.value
    status: str = "Demo simulation running. Wave to start."
    template_name: str = FRAME_TEMPLATES[0]["name"]
    armed: bool = False
    captured: int = 0
    total_photos: int = CONFIG["photos_per_session"]
    countdown_label: str = ""
    flash_until: float = 0.0
    recording: bool = False
    session_id: str = ""


class StateMachine:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.state = KioskState.ATTRACT
        self.state_since = time.time()
        self.last_activity = time.time()
        self.session_start: Optional[float] = None
        self.session_id = uuid.uuid4().hex[:8]
        self.session_dir: Optional[Path] = None
        self.template = 0
        self.photo_paths: List[str] = []
        self.capture_schedule: List[float] = []
        self.next_capture_index = 0
        self.flash_until = 0.0
        self.collage_path: Optional[str] = None
        self.token: Optional[str] = None

    def go(self, state: KioskState, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self.state = state
        self.state_since = now
        self.last_activity = now

        if state != KioskState.ATTRACT and self.session_start is None:
            self.session_start = now

        if state == KioskState.ATTRACT:
            self.session_start = None

    def trigger(self, event: str, params: Optional[Dict[str, Any]] = None, now: Optional[float] = None) -> None:
        params = params or {}
        now = time.time() if now is None else now

        if event == "error":
            self.request_reset(now)
            return

        if self.state == KioskState.ATTRACT:
            if event == "wave":
                self.go(KioskState.GREETING, now)

        elif self.state == KioskState.GREETING:
            if event in ("start", "wave", "arm"):
                self.go(KioskState.FRAME_SELECT, now)

        elif self.state == KioskState.FRAME_SELECT:
            if event == "swipe":
                direction = params.get("direction", "right")
                step = 1 if direction == "right" else -1
                self.template = (self.template + step) % len(FRAME_TEMPLATES)
                self.last_activity = now
            elif event in ("arm", "confirm"):
                self.start_countdown(now)
            elif event == "reset":
                self.request_reset(now)

        elif self.state == KioskState.COUNTDOWN:
            if event == "capture_complete":
                self.go(KioskState.REVIEW, now)
            elif event == "reset":
                self.request_reset(now)

        elif self.state == KioskState.REVIEW:
            if event == "share":
                self.go(KioskState.SHARE, now)
            elif event == "retake":
                self.reset_session(keep_template=True)
                self.start_countdown(now)
            elif event == "reset":
                self.request_reset(now)

        elif self.state == KioskState.SHARE:
            if event == "reset":
                self.request_reset(now)

        elif self.state == KioskState.RESET:
            if event == "reset_done":
                self.finish_reset(now)

    def start_countdown(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self.go(KioskState.COUNTDOWN, now)
        self.photo_paths = []
        self.next_capture_index = 0
        self.flash_until = 0.0

        if self.session_dir is None:
            self.session_dir = DATA_DIR / "sessions" / self.session_id

        self.session_dir.mkdir(parents=True, exist_ok=True)

        first = now + self.cfg["countdown_s"]
        self.capture_schedule = [
            first + i * self.cfg["capture_interval_s"]
            for i in range(self.cfg["photos_per_session"])
        ]

    def request_reset(self, now: Optional[float] = None) -> None:
        self.go(KioskState.RESET, now)

    def finish_reset(self, now: Optional[float] = None) -> None:
        self.reset_session(keep_template=False)
        self.go(KioskState.ATTRACT, now)

    def reset_session(self, keep_template: bool = False) -> None:
        old_template = self.template
        self.session_id = uuid.uuid4().hex[:8]
        self.session_dir = None
        self.photo_paths = []
        self.capture_schedule = []
        self.next_capture_index = 0
        self.flash_until = 0.0
        self.collage_path = None
        self.token = None
        self.session_start = None
        self.template = old_template if keep_template else 0

    def tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now

        if self.state == KioskState.GREETING and now - self.state_since >= self.cfg["greeting_s"]:
            self.go(KioskState.FRAME_SELECT, now)

        if self.state in (KioskState.ATTRACT, KioskState.RESET):
            return

        if now - self.last_activity >= self.cfg["idle_timeout_s"]:
            self.request_reset(now)
        elif self.session_start is not None and now - self.session_start >= self.cfg["session_timeout_s"]:
            self.request_reset(now)


# =============================================================================
# LOCAL DOWNLOAD SERVER AND MJPEG STREAM
# One-time token, 128-bit random UUID hex, local network only.
# =============================================================================
app = Flask(__name__)
TOKENS: Dict[str, Dict[str, Any]] = {}
TOKEN_LOCK = threading.Lock()

FLASK_STATE = {
    "started": False,
    "port": None,
    "url_base": None,
    "lock": threading.Lock(),
}

_RUNTIME: Optional["KioskRuntime"] = None


@app.route("/d/<token>")
def download_route(token: str):
    now = time.time()

    with TOKEN_LOCK:
        meta = TOKENS.get(token)
        if not meta or meta.get("used") or now > meta.get("expires", 0.0) or not os.path.exists(meta["path"]):
            abort(404)

        meta["used"] = True
        meta["used_at"] = now
        path = meta["path"]

    logger.info("one-time download served token_prefix=%s", token[:8])

    return send_file(
        path,
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="boothkiosk-collage.jpg",
    )


@app.route("/stream")
def stream_route():
    if _RUNTIME is None:
        abort(503)

    token = request_args_get("t")
    if token != _RUNTIME.stream_token:
        abort(403)

    def generate():
        while not _RUNTIME.stop_event.is_set():
            frame = _RUNTIME.display_bus.get_latest()
            if frame is None:
                frame = make_demo_frame(time.time())

            ok, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG["jpeg_quality"])],
            )

            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )

            time.sleep(1.0 / max(1, CONFIG["stream_fps"]))

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.route("/health")
def health_route():
    return {"ok": True, "time": time.time()}


def request_args_get(name: str) -> Optional[str]:
    # Small helper to avoid importing request directly at top in some environments.
    try:
        from flask import request
        return request.args.get(name)
    except Exception:
        return None


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        # This does not send payload; it only selects the local route.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def find_free_port(start_port: int) -> int:
    for port in range(start_port, start_port + 25):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start_port


def run_flask(port: int) -> None:
    try:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
    except Exception as exc:
        logger.error("Flask server failed: %s", exc)


def ensure_flask(rt: "KioskRuntime") -> str:
    global _RUNTIME
    _RUNTIME = rt

    with FLASK_STATE["lock"]:
        if FLASK_STATE["started"] and FLASK_STATE["url_base"]:
            return str(FLASK_STATE["url_base"])

        port = find_free_port(CONFIG["download_port"])
        threading.Thread(target=run_flask, args=(port,), daemon=True).start()
        time.sleep(1.0)

        url_base = f"http://{get_lan_ip()}:{port}"
        FLASK_STATE["started"] = True
        FLASK_STATE["port"] = port
        FLASK_STATE["url_base"] = url_base
        return url_base


def issue_download_token(path: str) -> str:
    token = uuid.uuid4().hex  # 128-bit random token
    expires = time.time() + CONFIG["delete_after_min"] * 60.0

    with TOKEN_LOCK:
        TOKENS[token] = {
            "path": str(path),
            "expires": expires,
            "used": False,
            "used_at": None,
        }

    return token


def cleanup_tokens(force: bool = False) -> None:
    now = time.time()
    removed: List[Tuple[str, Optional[str]]] = []

    with TOKEN_LOCK:
        for tok, meta in list(TOKENS.items()):
            expired = now > meta.get("expires", 0.0)
            used_old = (
                meta.get("used", False)
                and meta.get("used_at") is not None
                and now - float(meta["used_at"]) > 10.0
            )

            if force or expired or used_old:
                removed.append((tok, meta.get("path")))
                TOKENS.pop(tok, None)

    for tok, path in removed:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logger.info("deleted token file token_prefix=%s", tok[:8])
        except Exception as exc:
            logger.warning("could not delete token file: %s", exc)


def cleanup_old_sessions() -> None:
    cutoff = time.time() - CONFIG["delete_after_min"] * 60.0
    sessions_root = DATA_DIR / "sessions"

    if not sessions_root.exists():
        return

    for sd in sessions_root.iterdir():
        if not sd.is_dir():
            continue

        try:
            file_times = [f.stat().st_mtime for f in sd.rglob("*") if f.is_file()]
            dir_time = sd.stat().st_mtime
            latest = max(file_times + [dir_time], default=0.0)
        except Exception:
            latest = 0.0

        if latest < cutoff:
            shutil.rmtree(sd, ignore_errors=True)
            logger.info("auto-deleted old session directory session=%s", sd.name)


def delete_session_files(session_id: str) -> None:
    sd = DATA_DIR / "sessions" / str(session_id)

    if sd.exists():
        shutil.rmtree(sd, ignore_errors=True)
        logger.info("deleted session directory session=%s", session_id)

    with TOKEN_LOCK:
        for tok, meta in list(TOKENS.items()):
            if str(meta.get("path", "")).startswith(str(sd)):
                TOKENS.pop(tok, None)
                try:
                    if os.path.exists(meta["path"]):
                        os.remove(meta["path"])
                except Exception:
                    pass


def make_qr_bytes(data: str) -> bytes:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =============================================================================
# DEMO SIMULATION
# Provides an immediate non-blank live demo without requiring a webcam.
# =============================================================================
def demo_landmarks(t: float) -> np.ndarray:
    phase = t % 24.0
    open_palm = False

    if phase < 5.0:
        # Wave: repeated palm-x reversals with enough amplitude.
        x = 0.5 + 0.23 * math.sin(2 * math.pi * (t % 0.75) / 0.75)
        y = 0.55 + 0.015 * math.sin(2 * math.pi * t / 1.7)
    elif phase < 7.0:
        # Greeting pause.
        x = 0.5 + 0.01 * math.sin(t * 2.0)
        y = 0.55
    elif phase < 8.0:
        # Swipe right.
        p = min(max((phase - 7.0) / 0.8, 0.0), 1.0)
        x = 0.18 + 0.64 * p
        y = 0.52
    elif phase < 9.0:
        # Swipe left.
        p = min(max((phase - 8.0) / 0.8, 0.0), 1.0)
        x = 0.82 - 0.64 * p
        y = 0.52
    elif phase < 12.0:
        # Open palm hold for arming.
        x = 0.5 + 0.01 * math.sin(t * 1.5)
        y = 0.52
        open_palm = True
    elif phase < 16.0:
        # Capture/review idle.
        x = 0.5
        y = 0.52
    else:
        # Return toward attract.
        x = 0.5 + 0.03 * math.sin(t * 0.7)
        y = 0.55 + 0.02 * math.cos(t * 0.5)

    return synthetic_landmarks(x, y, open_palm)


def make_demo_frame(t: float) -> np.ndarray:
    w, h = CONFIG["camera_width"], CONFIG["camera_height"]
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    b = int(90 + 50 * math.sin(t * 0.7))
    g = int(70 + 40 * math.sin(t * 0.5 + 1.0))
    r = int(110 + 50 * math.cos(t * 0.4))
    frame[:] = (np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255))

    cv2.line(frame, (0, int(h * 0.75)), (w, int(h * 0.75)), (245, 245, 245), 2)

    for k in range(3):
        x = int(w * (0.20 + 0.20 * k + 0.05 * math.sin(t * 0.6 + k)))
        y = int(h * (0.55 + 0.03 * math.cos(t * 0.8 + k)))
        cv2.ellipse(frame, (x, y), (35, 70), 0, 0, 360, (165, 155, 145), -1)

    cv2.putText(
        frame,
        "DEMO SIMULATION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame


def make_error_frame(msg: str) -> np.ndarray:
    w, h = CONFIG["camera_width"], CONFIG["camera_height"]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    cv2.putText(
        frame,
        "ERROR",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )

    for i, line in enumerate(textwrap_wrap(msg, 55)[:5]):
        cv2.putText(
            frame,
            line,
            (20, 140 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    return frame


def textwrap_wrap(text: str, width: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines or [""]


# =============================================================================
# RUNTIME
# =============================================================================
class KioskRuntime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.worker_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.raw_bus = FrameBus()
        self.display_bus = FrameBus()
        self.event_queue: queue.Queue[GestureEvent] = queue.Queue()

        self.gesture_engine = GestureEngine(CONFIG)
        self.sm = StateMachine(CONFIG)
        self.display_state = DisplayState()

        self.metrics: Dict[str, Any] = {
            "fps": 0.0,
            "tracker_latency_ms": 0.0,
            "display_latency_ms": 0.0,
            "frames": 0,
            "events": 0,
            "captures": 0,
            "sessions_started": 0,
            "sessions_completed": 0,
            "false_starts": 0,
        }

        self.use_demo = True
        self.use_camera = False
        self.consent = False
        self.operator_override = False

        self.last_frame_time = time.time()
        self.stream_token = uuid.uuid4().hex
        self.status = self.display_state.status

        self.share: Dict[str, Any] = {
            "url": None,
            "qr": None,
            "collage_path": None,
            "token": None,
        }

        self.test_results: Optional[Dict[str, Any]] = None
        self.workers_started = False
        self.thread_registry: Dict[str, threading.Thread] = {}


_RUNTIME_SINGLETON: Optional[KioskRuntime] = None
_RUNTIME_CREATE_LOCK = threading.Lock()


def get_runtime() -> KioskRuntime:
    global _RUNTIME_SINGLETON

    with _RUNTIME_CREATE_LOCK:
        if _RUNTIME_SINGLETON is None:
            _RUNTIME_SINGLETON = KioskRuntime()

        st.session_state["runtime"] = _RUNTIME_SINGLETON
        return _RUNTIME_SINGLETON


# =============================================================================
# BACKGROUND THREADS
# =============================================================================
def capture_loop(stop: threading.Event, rt: KioskRuntime) -> None:
    cap: Optional[cv2.VideoCapture] = None

    while not stop.is_set():
        try:
            use_camera = rt.use_camera and not rt.use_demo

            if use_camera:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_width"])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])

                ok, frame = cap.read() if cap is not None else (False, None)

                if not ok or frame is None:
                    frame = make_error_frame("Camera unavailable. Close other apps using the webcam or switch back to demo mode.")
                else:
                    frame = cv2.resize(
                        frame,
                        (CONFIG["camera_width"], CONFIG["camera_height"]),
                        interpolation=cv2.INTER_LINEAR,
                    )
            else:
                frame = make_demo_frame(time.time())

            rt.raw_bus.put(frame)
            rt.last_frame_time = time.time()

        except Exception as exc:
            logger.exception("capture loop error")
            rt.raw_bus.put(make_error_frame(str(exc)))
            rt.last_frame_time = time.time()

        time.sleep(1.0 / max(1, CONFIG["target_fps"]))

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def tracking_loop(stop: threading.Event, rt: KioskRuntime) -> None:
    hands = None

    if MP_AVAILABLE:
        try:
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=CONFIG["mp_min_detection_confidence"],
                min_tracking_confidence=CONFIG["mp_min_tracking_confidence"],
            )
        except Exception as exc:
            logger.error("MediaPipe Hands init failed: %s", exc)
            hands = None

    fps_times: deque = deque()

    while not stop.is_set():
        try:
            raw = rt.raw_bus.get_latest()
            if raw is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            now = time.time()

            if rt.use_demo or not rt.use_camera:
                raw_landmarks = [demo_landmarks(now)]
                handedness = ["demo"]
            else:
                raw_landmarks = []
                handedness = []

                if hands is not None:
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = hands.process(rgb)

                    hand_lms = getattr(result, "multi_hand_landmarks", None) or []
                    hand_meta = getattr(result, "multi_handedness", None) or []

                    for idx, hl in enumerate(hand_lms):
                        lm = np.array([[p.x, p.y] for p in hl.landmark], dtype=float)
                        raw_landmarks.append(lm)

                        label = "right"
                        if idx < len(hand_meta):
                            try:
                                label = hand_meta[idx].classification[0].label.lower()
                            except Exception:
                                label = "right"

                        handedness.append(label)
                else:
                    rt.status = "MediaPipe unavailable; camera preview only."

            filtered, events = rt.gesture_engine.process(raw_landmarks, handedness, now)

            for ev in events:
                rt.event_queue.put(ev)
                rt.metrics["events"] += 1

            latency_ms = (time.perf_counter() - t0) * 1000.0

            fps_times.append(now)
            while fps_times and now - fps_times[0] > 1.0:
                fps_times.popleft()

            rt.metrics["fps"] = float(len(fps_times))
            rt.metrics["tracker_latency_ms"] = latency_ms
            rt.metrics["frames"] += 1

            annotated = draw_annotations(raw, filtered, rt.display_state, now, rt)
            rt.display_bus.put(annotated)

        except Exception as exc:
            logger.exception("tracking loop error")
            rt.display_bus.put(make_error_frame(str(exc)))

        time.sleep(0.001)

    if hands is not None:
        try:
            hands.close()
        except Exception:
            pass


def draw_annotations(
    frame: np.ndarray,
    landmarks_list: List[np.ndarray],
    disp: DisplayState,
    now: float,
    rt: KioskRuntime,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    for lm in landmarks_list:
        for a, b in HAND_CONNECTIONS:
            if a < len(lm) and b < len(lm):
                p1 = (int(lm[a][0] * w), int(lm[a][1] * h))
                p2 = (int(lm[b][0] * w), int(lm[b][1] * h))
                cv2.line(img, p1, p2, (0, 255, 180), 2)

        for i, p in enumerate(lm):
            color = (80, 220, 255) if i in {0, 4, 8, 12, 16, 20} else (255, 255, 255)
            cv2.circle(img, (int(p[0] * w), int(p[1] * h)), 4, color, -1)

    banner_color = STATE_COLORS.get(disp.state, (60, 60, 60))
    cv2.rectangle(img, (0, 0), (w, 68), banner_color, -1)

    cv2.putText(
        img,
        f"STATE: {disp.state}",
        (14, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    if rt.use_demo or not rt.use_camera:
        cv2.putText(
            img,
            "DEMO",
            (w - 110, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    else:
        cv2.circle(img, (w - 145, 36), 10, (0, 0, 255), -1)
        cv2.putText(
            img,
            "CAMERA ON",
            (w - 125, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    if disp.armed:
        cv2.putText(
            img,
            "ARMED",
            (14, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if disp.recording:
        cv2.circle(img, (w - 45, 100), 16, (0, 0, 255), -1)
        cv2.putText(
            img,
            "REC",
            (w - 115, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if disp.countdown_label:
        label = disp.countdown_label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)
        cv2.rectangle(
            img,
            ((w - tw) // 2 - 20, (h - th) // 2 - 20),
            ((w + tw) // 2 + 20, (h + th) // 2 + 20),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            label,
            ((w - tw) // 2, (h + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (255, 255, 255),
            6,
        )

    if disp.state == KioskState.FRAME_SELECT.value:
        cv2.rectangle(img, (10, h - 92), (w - 10, h - 22), (0, 0, 0), -1)
        cv2.putText(
            img,
            f"Frame: {disp.template_name}  |  swipe to change",
            (20, h - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    if disp.state == KioskState.REVIEW.value:
        cv2.rectangle(img, (10, h - 92), (w - 10, h - 22), (0, 0, 0), -1)
        cv2.putText(
            img,
            f"Review: captured {disp.captured}/{disp.total_photos}",
            (20, h - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    if disp.state == KioskState.SHARE.value:
        cv2.rectangle(img, (10, h - 92), (w - 10, h - 22), (0, 0, 0), -1)
        cv2.putText(
            img,
            "Scan QR on same Wi-Fi or use Download link",
            (20, h - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

    if now < disp.flash_until:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    return img


def handle_event_locked(rt: KioskRuntime, ev: GestureEvent, now: float) -> Tuple[bool, bool, bool]:
    pending_share = False
    pending_delete = False
    pending_delete_all = False
    sm = rt.sm

    if ev.type == "delete_all":
        rt.status = "Deleting all local data..."
        pending_delete_all = True
        return pending_share, pending_delete, pending_delete_all

    if ev.type == "error":
        reason = ev.params.get("reason", "unknown fault")
        rt.status = f"Fault detected by {ev.hand}: {reason}. Resetting to ATTRACT."
        sm.request_reset(now)
        pending_delete = True
        return pending_share, pending_delete, pending_delete_all

    if ev.type == "reset":
        if sm.state != KioskState.RESET:
            sm.request_reset(now)
            pending_delete = True
        return pending_share, pending_delete, pending_delete_all

    if ev.type == "arm":
        rt.gesture_engine.armed = True
        rt.gesture_engine.armed_until = now + CONFIG["arm_valid_s"]

    if ev.type == "wave" and sm.state == KioskState.ATTRACT:
        rt.metrics["sessions_started"] += 1

    if sm.state == KioskState.ATTRACT:
        if ev.type == "wave":
            sm.go(KioskState.GREETING, now)
            rt.status = "Greeting. Get ready for frame selection."

    elif sm.state == KioskState.GREETING:
        if ev.type in ("start", "wave", "arm"):
            sm.go(KioskState.FRAME_SELECT, now)
            rt.status = "Choose a frame by swiping, then hold open palm to start capture."

    elif sm.state == KioskState.FRAME_SELECT:
        if ev.type == "swipe":
            direction = ev.params.get("direction", "right")
            step = 1 if direction == "right" else -1
            sm.template = (sm.template + step) % len(FRAME_TEMPLATES)
            sm.last_activity = now
            rt.status = f"Frame changed to {FRAME_TEMPLATES[sm.template]['name']}."

        elif ev.type == "arm":
            sm.start_countdown(now)
            rt.status = "Capture armed. Countdown started."

        elif ev.type == "confirm":
            if rt.gesture_engine.armed or rt.operator_override:
                sm.start_countdown(now)
                rt.status = "Countdown started."
            else:
                rt.status = "Risky action blocked. Hold open palm for 1 second to arm capture."

    elif sm.state == KioskState.COUNTDOWN:
        if ev.type == "capture_complete":
            sm.go(KioskState.REVIEW, now)
            rt.status = "Photos captured. Choose Download link or Retake."

    elif sm.state == KioskState.REVIEW:
        if ev.type == "share":
            pending_share = True
            rt.status = "Preparing local one-time download..."
        elif ev.type == "retake":
            sm.reset_session(keep_template=True)
            sm.start_countdown(now)
            rt.status = "Retake countdown started."

    elif sm.state == KioskState.SHARE:
        if ev.type == "reset":
            sm.request_reset(now)
            pending_delete = True

    elif sm.state == KioskState.RESET:
        if ev.type == "reset_done":
            sm.finish_reset(now)

    return pending_share, pending_delete, pending_delete_all


def capture_due_locked(rt: KioskRuntime, now: float) -> None:
    sm = rt.sm

    if sm.state != KioskState.COUNTDOWN:
        return

    if sm.session_dir is None:
        sm.session_dir = DATA_DIR / "sessions" / sm.session_id
        sm.session_dir.mkdir(parents=True, exist_ok=True)

    while (
        sm.next_capture_index < len(sm.capture_schedule)
        and now >= sm.capture_schedule[sm.next_capture_index]
    ):
        frame = rt.raw_bus.get_latest()
        if frame is None:
            frame = make_demo_frame(now)

        path = sm.session_dir / f"photo_{sm.next_capture_index}.jpg"

        try:
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(CONFIG["jpeg_quality"])])
            sm.photo_paths.append(str(path))
            rt.metrics["captures"] += 1
        except Exception as exc:
            logger.exception("photo capture save failed")
            rt.status = f"Capture failed: {exc}"
            sm.request_reset(now)
            return

        sm.flash_until = now + CONFIG["flash_ms"] / 1000.0
        sm.next_capture_index += 1
        sm.last_activity = now

        logger.info(
            "captured photo %d/%d session=%s",
            sm.next_capture_index,
            CONFIG["photos_per_session"],
            sm.session_id,
        )

    if sm.next_capture_index >= CONFIG["photos_per_session"]:
        sm.go(KioskState.REVIEW, now)
        rt.status = "Photos captured. Choose Download link or Retake."


def update_display_state_locked(rt: KioskRuntime, now: float) -> None:
    sm = rt.sm
    label = ""

    if sm.state == KioskState.COUNTDOWN and sm.next_capture_index < len(sm.capture_schedule):
        rem = sm.capture_schedule[sm.next_capture_index] - now
        label = str(max(1, int(math.ceil(rem)))) if rem > 0 else "CHEESE"

    rt.display_state = DisplayState(
        state=sm.state.value,
        status=rt.status,
        template_name=FRAME_TEMPLATES[sm.template % len(FRAME_TEMPLATES)]["name"],
        armed=bool(rt.gesture_engine.armed),
        captured=len(sm.photo_paths),
        total_photos=CONFIG["photos_per_session"],
        countdown_label=label,
        flash_until=sm.flash_until,
        recording=(sm.state == KioskState.COUNTDOWN),
        session_id=sm.session_id,
    )


def controller_loop(stop: threading.Event, rt: KioskRuntime) -> None:
    with rt.lock:
        update_display_state_locked(rt, time.time())

    while not stop.is_set():
        now = time.time()
        pending_share = False
        pending_delete = False
        pending_delete_all = False

        try:
            ev = rt.event_queue.get(timeout=0.05)
        except queue.Empty:
            ev = None

        with rt.lock:
            if ev is not None:
                pending_share, pending_delete, pending_delete_all = handle_event_locked(rt, ev, now)

            rt.sm.tick(now)

            if rt.sm.state == KioskState.COUNTDOWN:
                capture_due_locked(rt, now)

            if rt.sm.state == KioskState.RESET:
                pending_delete = True

            update_display_state_locked(rt, now)

        if pending_delete_all:
            delete_all_data(rt)
            with rt.lock:
                rt.sm.reset_session(keep_template=False)
                rt.sm.go(KioskState.ATTRACT, time.time())
                rt.status = "Delete-all-data completed. Local media and one-time tokens removed."
                update_display_state_locked(rt, time.time())

        if pending_share:
            try:
                prepare_share(rt)
                with rt.lock:
                    rt.sm.go(KioskState.SHARE, time.time())
                    rt.metrics["sessions_completed"] += 1
                    rt.status = "Local one-time download link ready."
                    update_display_state_locked(rt, time.time())
            except Exception as exc:
                logger.exception("share failed")
                with rt.lock:
                    rt.status = f"Share failed: {exc}"
                    rt.sm.request_reset(time.time())
                    update_display_state_locked(rt, time.time())

        if pending_delete:
            sid = None
            with rt.lock:
                sid = rt.sm.session_id

            if sid:
                delete_session_files(sid)

            with rt.lock:
                if rt.sm.state == KioskState.RESET:
                    rt.sm.finish_reset(time.time())
                    rt.status = "Session reset. Returned to ATTRACT."
                    update_display_state_locked(rt, time.time())

        time.sleep(0.01)


def cleanup_loop(stop: threading.Event, rt: KioskRuntime) -> None:
    while not stop.is_set():
        try:
            cleanup_tokens(False)
            cleanup_old_sessions()
        except Exception:
            logger.exception("cleanup loop error")

        time.sleep(5.0)


def watchdog_loop(stop: threading.Event, rt: KioskRuntime) -> None:
    while not stop.is_set():
        time.sleep(1.0)

        try:
            for name in ("capture", "tracking", "controller"):
                th = rt.thread_registry.get(name)
                if th is not None and not th.is_alive() and not stop.is_set():
                    logger.warning("watchdog restarting thread=%s", name)

                    with rt.worker_lock:
                        th = rt.thread_registry.get(name)
                        if th is not None and not th.is_alive():
                            if name == "capture":
                                target = capture_loop
                            elif name == "tracking":
                                target = tracking_loop
                            else:
                                target = controller_loop

                            new_th = threading.Thread(
                                target=target,
                                args=(rt.stop_event, rt),
                                daemon=True,
                                name=f"booth-{name}-restarted",
                            )
                            rt.thread_registry[name] = new_th
                            new_th.start()

            if rt.use_camera and not rt.use_demo:
                if time.time() - rt.last_frame_time > 5.0:
                    rt.event_queue.put(
                        GestureEvent(
                            "error",
                            "watchdog",
                            int(time.time() * 1000),
                            1.0,
                            {"reason": "camera stall"},
                        )
                    )
                    rt.last_frame_time = time.time()

        except Exception:
            logger.exception("watchdog loop error")


def ensure_workers(rt: KioskRuntime) -> None:
    with rt.worker_lock:
        if rt.workers_started:
            return

        targets = {
            "capture": capture_loop,
            "tracking": tracking_loop,
            "controller": controller_loop,
            "cleanup": cleanup_loop,
            "watchdog": watchdog_loop,
        }

        for name, target in targets.items():
            th = threading.Thread(
                target=target,
                args=(rt.stop_event, rt),
                daemon=True,
                name=f"booth-{name}",
            )
            rt.thread_registry[name] = th
            th.start()

        rt.workers_started = True
        logger.info("background workers started")


# =============================================================================
# SHARE, DELETE, AND HELPERS
# =============================================================================
def prepare_share(rt: KioskRuntime) -> None:
    with rt.lock:
        paths = list(rt.sm.photo_paths)
        template = rt.sm.template
        session_dir = rt.sm.session_dir
        session_id = rt.sm.session_id

    if not paths:
        raise RuntimeError("No photos captured.")

    if session_dir is None:
        session_dir = DATA_DIR / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

    collage_path = session_dir / "collage.jpg"
    create_collage_from_paths(paths, template, str(collage_path))

    base = ensure_flask(rt)
    token = issue_download_token(str(collage_path))
    url = f"{base}/d/{token}"
    qr = make_qr_bytes(url)

    with rt.lock:
        rt.sm.collage_path = str(collage_path)
        rt.sm.token = token
        rt.share = {
            "url": url,
            "qr": qr,
            "collage_path": str(collage_path),
            "token": token,
        }

    logger.info(
        "prepared share session=%s template=%d token_prefix=%s",
        session_id,
        template,
        token[:8],
    )


def delete_all_data(rt: KioskRuntime) -> None:
    sessions_root = DATA_DIR / "sessions"

    if sessions_root.exists():
        for p in sessions_root.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    with TOKEN_LOCK:
        TOKENS.clear()

    with rt.lock:
        rt.sm.reset_session(keep_template=False)
        rt.share = {
            "url": None,
            "qr": None,
            "collage_path": None,
            "token": None,
        }
        rt.status = "Delete-all-data completed. Local media and one-time tokens removed."

    logger.info("delete-all-data executed")


def enqueue_event(
    rt: KioskRuntime,
    typ: str,
    hand: str = "ui",
    confidence: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    rt.event_queue.put(
        GestureEvent(
            typ,
            hand,
            int(time.time() * 1000),
            confidence,
            params or {},
        )
    )


# =============================================================================
# SELF TESTS
# =============================================================================
def test_state_machine() -> None:
    sm = StateMachine(CONFIG)
    assert sm.state == KioskState.ATTRACT

    sm.trigger("wave", now=0.0)
    assert sm.state == KioskState.GREETING

    sm.tick(3.2)
    assert sm.state == KioskState.FRAME_SELECT

    sm.trigger("swipe", params={"direction": "right"}, now=4.0)
    assert sm.template == 1

    sm.trigger("arm", now=5.0)
    assert sm.state == KioskState.COUNTDOWN

    sm.trigger("capture_complete", now=10.0)
    assert sm.state == KioskState.REVIEW

    sm.trigger("share", now=11.0)
    assert sm.state == KioskState.SHARE

    sm.trigger("reset", now=12.0)
    assert sm.state == KioskState.RESET

    sm.trigger("reset_done", now=12.1)
    assert sm.state == KioskState.ATTRACT


def test_gesture_math() -> None:
    lm = synthetic_landmarks(0.5, 0.5, True)
    pw = palm_width(lm)
    assert pw > 0.0

    pc = palm_center(lm)
    assert 0.0 <= pc[0] <= 1.0
    assert 0.0 <= pc[1] <= 1.0

    pr = pinch_ratio(lm)
    assert pr > 0.0

    normalized_palm_width = safe_dist(lm[5], lm[17]) / pw
    assert abs(normalized_palm_width - 1.0) < 1e-6


def test_wave_detection() -> None:
    eng = GestureEngine(CONFIG)
    events: List[GestureEvent] = []

    for i in range(90):
        t = i / 30.0
        x = 0.5 + 0.25 * math.sin(2 * math.pi * t / 0.55)
        lm = synthetic_landmarks(x, 0.55, False)
        _, evs = eng.process([lm], ["right"], t)
        events.extend(evs)

    assert any(e.type == "wave" for e in events), "No wave event emitted"


def test_swipe_detection() -> None:
    eng = GestureEngine(CONFIG)
    events: List[GestureEvent] = []

    for i in range(30):
        t = i / 30.0
        x = 0.20 + 0.65 * min(t / 0.45, 1.0)
        lm = synthetic_landmarks(x, 0.52, False)
        _, evs = eng.process([lm], ["right"], t)
        events.extend(evs)

    assert any(e.type == "swipe" for e in events), "No swipe event emitted"


def test_arm_detection() -> None:
    eng = GestureEngine(CONFIG)
    events: List[GestureEvent] = []

    for i in range(30):
        t = i / 15.0
        lm = synthetic_landmarks(0.5, 0.52, True)
        _, evs = eng.process([lm], ["right"], t)
        events.extend(evs)

    assert any(e.type == "arm" for e in events), "No arm event emitted"
    assert eng.armed, "Engine did not become armed"


def test_idle_no_false_starts() -> int:
    eng = GestureEngine(CONFIG)
    false_starts = 0

    # 30-minute equivalent sweep at 5 fps.
    for i in range(30 * 60 * 5):
        t = i / 5.0
        x = 0.5 + 0.035 * math.sin(2 * math.pi * t / 2.2)
        y = 0.55 + 0.010 * math.cos(2 * math.pi * t / 3.1)
        lm = synthetic_landmarks(x, y, False)
        _, evs = eng.process([lm], ["right"], t)
        false_starts += sum(1 for e in evs if e.type == "wave")

    assert false_starts == 0, f"False wave starts detected: {false_starts}"
    return false_starts


def test_token_expiry() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "token_test.jpg"
        p.write_bytes(b"test")
        tok = uuid.uuid4().hex

        with TOKEN_LOCK:
            TOKENS[tok] = {
                "path": str(p),
                "expires": time.time() - 1.0,
                "used": False,
                "used_at": None,
            }

        cleanup_tokens(True)

        assert not p.exists(), "Expired token file was not deleted"
        assert tok not in TOKENS, "Expired token was not removed"


def test_one_time_token_used() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "one_time.jpg"
        p.write_bytes(b"test")
        tok = uuid.uuid4().hex

        with TOKEN_LOCK:
            TOKENS[tok] = {
                "path": str(p),
                "expires": time.time() + 60.0,
                "used": True,
                "used_at": time.time() - 20.0,
            }

        cleanup_tokens(False)

        assert not p.exists(), "Used token file was not deleted"
        assert tok not in TOKENS, "Used token was not removed"


def test_deletion() -> None:
    sid = "selftest_" + uuid.uuid4().hex[:6]
    sd = DATA_DIR / "sessions" / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "a.jpg").write_bytes(b"x")

    delete_session_files(sid)

    assert not sd.exists(), "Session directory was not deleted"


def test_evaluation_simulation() -> Tuple[float, float]:
    completions = 0
    total_time = 0.0

    for _ in range(20):
        sm = StateMachine(CONFIG)
        t = 0.0

        sm.trigger("wave", now=t)
        t += 0.2

        t += CONFIG["greeting_s"] + 0.1
        sm.tick(t)
        assert sm.state == KioskState.FRAME_SELECT

        sm.trigger("swipe", params={"direction": "right"}, now=t)
        t += 0.3

        sm.trigger("arm", now=t)
        assert sm.state == KioskState.COUNTDOWN

        t += (
            CONFIG["countdown_s"]
            + (CONFIG["photos_per_session"] - 1) * CONFIG["capture_interval_s"]
            + 0.2
        )

        sm.trigger("capture_complete", now=t)
        assert sm.state == KioskState.REVIEW

        sm.trigger("share", now=t + 0.1)
        assert sm.state == KioskState.SHARE

        completions += 1
        total_time += t + 0.1

    completion_rate = 100.0 * completions / 20.0
    avg_session_s = total_time / 20.0
    return completion_rate, avg_session_s


def run_self_tests() -> Dict[str, Any]:
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            results.append((name, True, "passed"))
        except AssertionError as exc:
            results.append((name, False, str(exc) or "assertion failed"))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    check("State machine transitions", test_state_machine)
    check("Gesture math normalization", test_gesture_math)
    check("Wave detection", test_wave_detection)
    check("Swipe detection", test_swipe_detection)
    check("Open-palm arming", test_arm_detection)
    check("Token expiry deletion", test_token_expiry)
    check("One-time token used deletion", test_one_time_token_used)
    check("Session deletion", test_deletion)

    completion_rate = 0.0
    avg_session_s = 0.0

    try:
        completion_rate, avg_session_s = test_evaluation_simulation()
        results.append((
            "20-session simulation",
            True,
            f"{completion_rate:.0f}% completion, {avg_session_s:.1f}s average",
        ))
    except Exception as exc:
        results.append(("20-session simulation", False, f"{type(exc).__name__}: {exc}"))

    false_starts = -1

    try:
        false_starts = test_idle_no_false_starts()
        results.append((
            "30-minute false-start sweep",
            false_starts == 0,
            f"{false_starts} false starts",
        ))
    except Exception as exc:
        results.append(("30-minute false-start sweep", False, f"{type(exc).__name__}: {exc}"))

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    accuracy = 100.0 * passed / total if total else 0.0

    return {
        "results": results,
        "passed": passed,
        "total": total,
        "accuracy_pct": accuracy,
        "completion_rate_pct": completion_rate,
        "avg_session_s": avg_session_s,
        "false_starts": false_starts,
    }


# =============================================================================
# STREAMLIT UI HELPERS
# =============================================================================
def get_ui_snapshot() -> Dict[str, Any]:
    rt = st.session_state["runtime"]

    with rt.lock:
        return {
            "state": rt.display_state.state,
            "status": rt.display_state.status,
            "template_name": rt.display_state.template_name,
            "template_index": rt.sm.template,
            "armed": rt.display_state.armed,
            "captured": rt.display_state.captured,
            "total_photos": rt.display_state.total_photos,
            "photo_paths": list(rt.sm.photo_paths),
            "collage_path": rt.sm.collage_path,
            "download_url": rt.share.get("url"),
            "has_qr": bool(rt.share.get("qr")),
            "metrics": dict(rt.metrics),
            "use_demo": rt.use_demo,
            "use_camera": rt.use_camera,
            "consent": rt.consent,
            "operator_override": rt.operator_override,
            "stream_token": rt.stream_token,
        }


def get_qr_bytes() -> Optional[bytes]:
    rt = st.session_state["runtime"]
    with rt.lock:
        qr = rt.share.get("qr")
        return qr if isinstance(qr, bytes) else None


def get_collage_bytes() -> Optional[bytes]:
    rt = st.session_state["runtime"]

    with rt.lock:
        path = rt.share.get("collage_path")

    if path and os.path.exists(path):
        try:
            return Path(path).read_bytes()
        except Exception:
            return None

    return None


def render_control_panel() -> None:
    rt = st.session_state["runtime"]
    snap = get_ui_snapshot()
    state = snap["state"]

    st.subheader("Live controls")

    demo = st.checkbox("Demo simulation", value=bool(snap["use_demo"]), key="cb_demo")
    with rt.lock:
        rt.use_demo = bool(demo)

    cam = st.checkbox(
        "Use real webcam",
        value=bool(snap["use_camera"]),
        disabled=bool(rt.use_demo),
        key="cb_cam",
    )

    consent = st.checkbox(
        "Everyone in view knows the camera is on",
        value=bool(snap["consent"]),
        disabled=bool(rt.use_demo or not cam),
        key="cb_consent",
    )

    with rt.lock:
        rt.consent = bool(consent)
        rt.use_camera = bool(cam and consent and not rt.use_demo)

    if rt.use_camera and not MP_AVAILABLE:
        st.warning("MediaPipe is not available; gestures will not be detected from the real webcam.")

    override = st.checkbox("Operator override", value=bool(snap["operator_override"]), key="cb_override")
    with rt.lock:
        rt.operator_override = bool(override)

    code = st.text_input("Operator code", type="password", key="op_code")
    if code.strip().upper() == CONFIG["operator_code"]:
        with rt.lock:
            rt.operator_override = True
        st.success("Operator override enabled.")

    st.markdown("---")
    st.subheader("Touchless gesture simulator")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Wave", key="btn_wave"):
            enqueue_event(rt, "wave", "sim")

        if st.button("Swipe right", key="btn_swipe_right"):
            enqueue_event(rt, "swipe", "sim", params={"direction": "right"})

    with c2:
        if st.button("Open palm 1s", key="btn_arm"):
            enqueue_event(rt, "arm", "sim")

        if st.button("Swipe left", key="btn_swipe_left"):
            enqueue_event(rt, "swipe", "sim", params={"direction": "left"})

    st.subheader("Session")

    if st.button("Start / Next", key="btn_start_next"):
        if state == KioskState.ATTRACT.value:
            enqueue_event(rt, "wave")
        elif state == KioskState.GREETING.value:
            enqueue_event(rt, "start")
        elif state == KioskState.FRAME_SELECT.value:
            enqueue_event(rt, "confirm")
        elif state == KioskState.REVIEW.value:
            enqueue_event(rt, "share")
        elif state == KioskState.SHARE.value:
            enqueue_event(rt, "reset")
        else:
            enqueue_event(rt, "reset")

    b1, b2 = st.columns(2)

    with b1:
        if st.button("Retake", key="btn_retake"):
            enqueue_event(rt, "retake")

        if st.button("Reset", key="btn_reset"):
            enqueue_event(rt, "reset")

    with b2:
        if st.button("Download link", key="btn_download"):
            enqueue_event(rt, "share")

        if st.button("Fault", key="btn_fault"):
            enqueue_event(rt, "error", "ui", params={"reason": "manual fault"})

    if st.button("Delete all data", key="btn_delete_all"):
        enqueue_event(rt, "delete_all", "ui")

    st.markdown("---")

    m = snap["metrics"]
    cols = st.columns(4)
    cols[0].metric("State", state)
    cols[1].metric("FPS", f"{m.get('fps', 0.0):.1f}")
    cols[2].metric("Track ms", f"{m.get('tracker_latency_ms', 0.0):.1f}")
    cols[3].metric("Armed", "Yes" if snap["armed"] else "No")

    st.caption(snap["status"])

    if state == KioskState.ATTRACT.value:
        st.info("Attract mode: wave your hand, or press Wave, to start.")

    elif state == KioskState.GREETING.value:
        st.success("Greeting: get ready. Frame selection starts automatically.")

    elif state == KioskState.FRAME_SELECT.value:
        st.info("Swipe left/right to choose a frame, then hold an open palm for 1 second to start capture.")
        st.progress(
            (snap["template_index"] + 1) / len(FRAME_TEMPLATES),
            text=f"Frame {snap['template_index'] + 1}/{len(FRAME_TEMPLATES)}: {snap['template_name']}",
        )

    elif state == KioskState.COUNTDOWN.value:
        st.warning("Countdown and capture in progress. Stay in frame.")
        st.progress(
            min(1.0, snap["captured"] / max(1, snap["total_photos"])),
            text=f"Captured {snap['captured']}/{snap['total_photos']}",
        )

    elif state == KioskState.REVIEW.value:
        st.success("Review: choose Download link to create a local one-time QR, or Reset to discard.")

        paths = [p for p in snap["photo_paths"] if os.path.exists(p)]
        if paths:
            thcols = st.columns(min(3, len(paths)))
            for i, p in enumerate(paths):
                with thcols[i]:
                    st.image(p, caption=f"Photo {i + 1}", use_container_width=True)

    elif state == KioskState.SHARE.value:
        st.success("Share: scan the QR code on the same Wi-Fi or use the download button.")

        url = snap["download_url"]
        if url:
            cqr, cfile = st.columns(2)

            with cqr:
                qr = get_qr_bytes()
                if qr:
                    st.image(qr, caption="One-time local download QR")
                st.code(url, language="text")

            with cfile:
                data = get_collage_bytes()
                if data:
                    st.download_button(
                        "Download collage.jpg",
                        data=data,
                        file_name="collage.jpg",
                        mime="image/jpeg",
                    )
                else:
                    st.warning("File is no longer available.")
        else:
            st.warning("Download link not ready.")

    elif state == KioskState.RESET.value:
        st.error("Resetting and deleting local session files...")

    st.markdown("---")

    if st.button("Run self-tests", key="btn_tests"):
        rt.test_results = run_self_tests()

    with st.expander("Self-tests and evaluation"):
        res = rt.test_results

        if not res:
            st.write("Run self-tests to validate state, gestures, token expiry, deletion, and false starts.")
        else:
            st.metric("Unit-test accuracy", f"{res['accuracy_pct']:.1f}%")
            st.metric("Simulated completion", f"{res['completion_rate_pct']:.0f}%")
            st.metric("Average session", f"{res['avg_session_s']:.1f} s")
            st.metric("False starts", str(res["false_starts"]))

            for name, ok, msg in res["results"]:
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} **{name}**: {msg}")

    with st.expander("Privacy"):
        st.markdown(
            """
            - No network upload of photos.
            - Real camera is disabled unless you explicitly enable it and confirm consent.
            - Capture only happens after session start and countdown.
            - People in view are told the camera is on by the on-screen banner.
            - Delete-all-data removes local session media and one-time tokens.
            - Logs contain no images and no person identification.
            """
        )

    with st.expander("Known limits"):
        st.markdown(
            """
            - Gesture thresholds must be tuned on real recordings and lighting.
            - Demo mode uses synthetic landmarks, not real camera inference.
            - Real webcam mode requires local execution and OS camera permission.
            - QR download requires the phone and kiosk computer to be on the same LAN.
            - Auto-delete removes app-managed files; it does not guarantee forensic disk erasure.
            - MediaPipe legacy Hands is used for single-file offline compatibility.
            """
        )

    with st.expander("Licenses"):
        st.markdown(
            """
            - Streamlit: Apache-2.0  
            - OpenCV: Apache-2.0  
            - NumPy: BSD-3-Clause  
            - MediaPipe: Apache-2.0  
            - Pillow: HPND/MIT-style license  
            - qrcode: BSD-3-Clause  
            - Flask: BSD-3-Clause  
            - Python standard library modules: PSF/BSD-style licenses  
            """
        )


# =============================================================================
# STREAMLIT MAIN
# =============================================================================
st.set_page_config(
    page_title="BoothKiosk Touchless Photo Booth",
    layout="wide",
    initial_sidebar_state="collapsed",
)

rt = get_runtime()
ensure_workers(rt)
flask_base = ensure_flask(rt)
stream_url = f"{flask_base}/stream?t={rt.stream_token}"

st.title("📸 BoothKiosk: Touchless Photo Booth")
st.caption(
    "Local-only kiosk demo. No photos are uploaded. "
    "Demo simulation starts automatically and behaves like a live booth."
)

col_video, col_panel = st.columns([2, 1], gap="large")

with col_video:
    st.markdown(
        f"""
        <div style="
            width: 100%;
            max-width: 960px;
            aspect-ratio: 4 / 3;
            background: #000;
            overflow: hidden;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        ">
            <img
                src="{stream_url}"
                alt="BoothKiosk live preview"
                style="width: 100%; height: 100%; object-fit: cover; display: block;"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Smooth live preview is served locally as MJPEG. "
        "If the preview does not appear, check firewall settings and make sure the local Flask port is allowed."
    )

with col_panel:
    if hasattr(st, "fragment"):
        try:
            panel = st.fragment(render_control_panel, run_every=CONFIG["ui_refresh_s"])
        except Exception:
            panel = st.fragment(render_control_panel)
    else:
        panel = render_control_panel

    panel()