"""
Global statistics tracker for LLM API calls.

Tracks total tokens consumed and total time spent across all API calls
(both MAS agent calls and memory-method calls).

Usage:
    from common.stats import stats
    stats.summary()        # print summary
    stats.reset()          # reset counters
    stats.to_dict()        # get as dict
"""
import threading
import time


class APIStats:
    """Thread-safe global API call statistics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock if hasattr(self, '_lock') else threading.Lock():
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_calls = 0
            self.total_time = 0.0
            self.total_failures = 0
            # Breakdown by source
            self.by_source: dict[str, dict] = {}

    def record(self, source: str, prompt_tokens: int, completion_tokens: int,
               elapsed: float, success: bool = True):
        """Record a single API call."""
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1
            self.total_time += elapsed
            if not success:
                self.total_failures += 1

            if source not in self.by_source:
                self.by_source[source] = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "calls": 0, "time": 0.0,
                }
            s = self.by_source[source]
            s["prompt_tokens"] += prompt_tokens
            s["completion_tokens"] += completion_tokens
            s["calls"] += 1
            s["time"] += elapsed

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total_tokens": self.total_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_calls": self.total_calls,
                "total_time_seconds": round(self.total_time, 2),
                "total_failures": self.total_failures,
                "by_source": {src: dict(values) for src, values in self.by_source.items()},
            }

    def summary(self) -> str:
        d = self.to_dict()
        lines = [
            f"Total API Calls:       {d['total_calls']}",
            f"Total Tokens:          {d['total_tokens']:,}",
            f"  Prompt Tokens:       {d['total_prompt_tokens']:,}",
            f"  Completion Tokens:   {d['total_completion_tokens']:,}",
            f"Total Time:            {d['total_time_seconds']:.1f}s",
            f"Failures:              {d['total_failures']}",
        ]
        if d["by_source"]:
            lines.append("Breakdown:")
            for src, s in d["by_source"].items():
                lines.append(
                    f"  {src:20s}  calls={s['calls']:4d}  "
                    f"tokens={s['prompt_tokens']+s['completion_tokens']:6,}  "
                    f"time={s['time']:.1f}s"
                )
        return "\n".join(lines)


# Global singleton
stats = APIStats()
