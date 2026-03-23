"""
notify.py — ntfy.sh notification helper.

Priority levels  (ntfy integer scale):
  5 = max      (crash / critical error)
  4 = high     (loop / pipeline completed)
  3 = default  (phase transitions)
  2 = low      (verbose progress)

Convenience API:
    notify_start(model_name, pipeline, total_images)  -> Timer
    notify_phase(model_name, phase, detail, timer)
    notify_loop(model_name, loop_idx, max_loops, map50, map50_95, timer)
    notify_done(model_name, best_map50, best_round, timer)
    notify_fail(model_name, phase, error, timer)
    notify(title, msg, ...)                            # general / legacy
"""

import requests
import socket
import traceback
import time
from datetime import datetime

TOPIC = "autodistill-train"
BASE  = f"https://ntfy.sh/{TOPIC}"
HOST  = socket.gethostname()


# ── Internal sender ───────────────────────────────────────────────────────────

def _send(title: str, msg: str, tags: str = "white_check_mark", priority: int = 3) -> bool:
    """Post one notification. Returns True on success, never raises."""
    try:
        resp = requests.post(
            BASE,
            data    = msg.encode("utf-8"),
            headers = {
                "Title":    title,
                "Tags":     tags,
                "Priority": str(priority),
            },
            timeout = 6,
        )
        return resp.ok
    except Exception:
        return False


# ── Timer ─────────────────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> str:
        secs = int(time.time() - self._start)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def reset(self):
        self._start = time.time()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Public API ────────────────────────────────────────────────────────────────

def notify(title: str, msg: str, tags: str = "white_check_mark", priority: int = 3) -> bool:
    """General-purpose send — backwards-compatible with original call sites."""
    return _send(title, msg, tags=tags, priority=priority)


def notify_start(model_name: str, pipeline: str, total_images: int) -> Timer:
    """
    Call once when a pipeline run begins.
    Returns a Timer for tracking elapsed time throughout the run.
    """
    timer = Timer()
    _send(
        title    = f"[{model_name}] Pipeline started",
        msg      = (
            f"Host     : {HOST}\n"
            f"Model    : {model_name}\n"
            f"Pipeline : {pipeline}\n"
            f"Images   : {total_images}\n"
            f"Started  : {_ts()}"
        ),
        tags     = "rocket",
        priority = 3,
    )
    return timer


def notify_phase(model_name: str, phase: str, detail: str = "", timer: Timer = None) -> None:
    """Lightweight update for sub-phase transitions (labelling, cold-start, etc.)."""
    elapsed = f"  [{timer.elapsed()}]" if timer else ""
    _send(
        title    = f"[{model_name}] {phase}",
        msg      = f"{detail}{elapsed}\n{_ts()}",
        tags     = "hourglass_flowing_sand",
        priority = 2,
    )


def notify_loop(
    model_name : str,
    loop_idx   : int,
    max_loops  : int,
    map50      : float,
    map50_95   : float,
    timer      : Timer = None,
) -> None:
    """Call after each active-learning loop with live mAP results."""
    elapsed     = timer.elapsed() if timer else "n/a"
    bar_filled  = round(loop_idx / max_loops * 10)
    progress    = "█" * bar_filled + "░" * (10 - bar_filled)
    _send(
        title    = f"[{model_name}] Loop {loop_idx}/{max_loops} done",
        msg      = (
            f"Progress : {progress} {loop_idx}/{max_loops}\n"
            f"mAP@50   : {map50 * 100:.2f}%\n"
            f"mAP@50-95: {map50_95 * 100:.2f}%\n"
            f"Elapsed  : {elapsed}\n"
            f"Time     : {_ts()}"
        ),
        tags     = "bar_chart",
        priority = 4,
    )


def notify_done(
    model_name : str,
    best_map50 : float,
    best_round : int,
    timer      : Timer = None,
) -> None:
    """Call when the full pipeline + evaluation finishes successfully."""
    elapsed = timer.elapsed() if timer else "n/a"
    _send(
        title    = f"[{model_name}] Run complete",
        msg      = (
            f"Best mAP@50 : {best_map50 * 100:.2f}%  (round {best_round})\n"
            f"Total time  : {elapsed}\n"
            f"Host        : {HOST}\n"
            f"Finished    : {_ts()}"
        ),
        tags     = "white_check_mark,trophy",
        priority = 4,
    )


def notify_fail(model_name: str, phase: str, error: Exception, timer: Timer = None) -> None:
    """Call in except blocks — captures phase, error type, and trimmed traceback."""
    elapsed    = timer.elapsed() if timer else "n/a"
    tb         = traceback.format_exc()
    tb_trimmed = tb[-800:] if len(tb) > 800 else tb
    _send(
        title    = f"[{model_name}] FAILED — {phase}",
        msg      = (
            f"Error   : {error}\n"
            f"Phase   : {phase}\n"
            f"Elapsed : {elapsed}\n"
            f"Host    : {HOST}\n\n"
            f"Traceback (tail):\n{tb_trimmed}"
        ),
        tags     = "x,rotating_light",
        priority = 5,
    )
