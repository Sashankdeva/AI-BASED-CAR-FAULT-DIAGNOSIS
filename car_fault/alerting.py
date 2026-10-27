"""Alert persistence — cuts false alarms by requiring a fault to repeat.

A single noisy reading should not light up the driver's dashboard. Per vehicle
we keep the last PERSIST_WINDOW raw detections and confirm an alert only when
the same fault appears PERSIST_MIN_HITS times. Safety-critical faults
(overheating, low oil pressure) skip the wait when the model is confident.

When readings carry timestamps the window is also bounded in time: detections
older than PERSIST_MAX_AGE_S are dropped, and the confirming hits must span at
least PERSIST_MIN_SPAN_S. Without timestamps (e.g. an offline CSV log with no
time column) only the reading count is used.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque

from .config import (
    IMMEDIATE_CONFIDENCE,
    IMMEDIATE_SEVERITY,
    PERSIST_MAX_AGE_S,
    PERSIST_MIN_HITS,
    PERSIST_MIN_SPAN_OVERRIDES_S,
    PERSIST_MIN_SPAN_S,
    PERSIST_WINDOW,
)


def untracked(result: dict) -> dict:
    """No vehicle id: persistence cannot be applied, the reading is judged alone."""
    result["alert"] = result["raw_alert"]
    result["persistence"] = {
        "window": PERSIST_WINDOW, "required": PERSIST_MIN_HITS, "hits": 0, "immediate": False,
        "status": "untracked",
        "reason": "No vehicle_id sent — this reading is judged on its own, without confirmation.",
    }
    return result


class AlertTracker:
    def __init__(self, window: int = PERSIST_WINDOW, min_hits: int = PERSIST_MIN_HITS,
                 max_age_s: float = PERSIST_MAX_AGE_S, min_span_s: float = PERSIST_MIN_SPAN_S,
                 max_vehicles: int = 10_000):
        self.window = window
        self.min_hits = min_hits
        self.max_age_s = max_age_s
        self.min_span_s = min_span_s
        self.max_vehicles = max_vehicles
        self._history: OrderedDict[str, tuple[deque, dict]] = OrderedDict()  # (window, episode starts)
        self._lock = threading.Lock()

    def reset(self, vehicle_id: str | None = None) -> None:
        with self._lock:
            if vehicle_id is None:
                self._history.clear()
            else:
                self._history.pop(vehicle_id, None)

    def update(self, vehicle_id: str, result: dict, severity: float,
               probability: float | None = None, at: float | None = None) -> dict:
        """Record a raw detection and set result['alert'] / result['persistence'].

        `probability` is the best evidence for this specific fault. With two faults
        at once the classifier splits its probability between them, so the caller
        passes the multi-label detector's (unsplit) probability as well.
        `at` is the reading time in Unix seconds; None means count-only.
        """
        probability = result["confidence"] if probability is None else probability
        key = result["fault"] if result["raw_alert"] else None
        with self._lock:
            hist, episodes = self._history.pop(vehicle_id, None) or (deque(maxlen=self.window), {})
            if at is not None:
                while hist and hist[0][0] is not None and hist[0][0] < at - self.max_age_s:
                    hist.popleft()
            previous_hits = sum(k == key for _, k in hist) if key else 0
            hist.append((at, key))
            # An episode starts when a fault (re)appears and ends once it has left
            # the window. Its duration, not the span of the last few hits, is what
            # must reach the minimum — at 1 Hz five readings only cover 4 seconds.
            if key and at is not None and (previous_hits == 0 or key not in episodes):
                episodes[key] = at
            for k in [k for k in episodes if all(k != h for _, h in hist)]:
                del episodes[k]
            self._history[vehicle_id] = (hist, episodes)  # most recently used goes last
            while len(self._history) > self.max_vehicles:
                self._history.popitem(last=False)
            hits = sum(k == key for _, k in hist) if key else 0
            seen = len(hist)
            started = episodes.get(key) if key else None

        min_span = PERSIST_MIN_SPAN_OVERRIDES_S.get(key, self.min_span_s)
        span = (at - started) if at is not None and started is not None else None
        long_enough = span is None or span >= min_span

        immediate = bool(key) and severity >= IMMEDIATE_SEVERITY and probability >= IMMEDIATE_CONFIDENCE
        confirmed = bool(key) and (immediate or (hits >= self.min_hits and long_enough))
        if not key:
            status, reason = "clear", "No fault in this reading."
        elif immediate:
            status, reason = "confirmed", "Safety-critical fault — alerted immediately."
        elif confirmed:
            status, reason = "confirmed", f"Seen in {hits} of the last {seen} readings."
        elif hits >= self.min_hits:
            status, reason = "pending", (f"Present for {span:.0f} s — alerts once it persists "
                                         f"for {min_span:g} s.")
        else:
            status, reason = "pending", f"Waiting for confirmation ({hits}/{self.min_hits} readings)."

        result["alert"] = confirmed
        result["persistence"] = {"window": self.window, "required": self.min_hits, "hits": hits,
                                 "immediate": immediate, "status": status, "reason": reason}
        return result
