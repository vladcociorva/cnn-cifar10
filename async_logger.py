import json
import queue
import threading
import atexit
from pathlib import Path
from typing import Any, Dict

class AsyncLogger:
    _STOP = object()

    def __init__(self, path: str | Path = "metrics.jsonl", queue_size: int = 10_000):
        self._path = Path(path).expanduser()
        self._q: "queue.Queue[Dict[str, Any] | object]" = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def __call__(self, **metrics: Any) -> None:
        """Shorthand: logger(step=..., loss=...)"""
        self._q.put(metrics, block=False)

    def close(self) -> None:
        self._q.put(self._STOP)
        self._thread.join()

    def _writer(self):
        with self._path.open("w", encoding="utf-8") as f:
            while True:
                item = self._q.get()
                if item is self._STOP:
                    break
                f.write(json.dumps(item, allow_nan=False) + "\n")
                f.flush()
