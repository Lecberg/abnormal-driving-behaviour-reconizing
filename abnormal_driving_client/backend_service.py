from __future__ import annotations

import json
import sys
import threading
from typing import Any

from backend_runtime import BackendRuntime, to_jsonable


output_lock = threading.Lock()


def emit(message: dict[str, Any]) -> None:
    with output_lock:
        sys.stdout.write(json.dumps(to_jsonable(message), ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> int:
    runtime = BackendRuntime(emit)
    runtime.start()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"event": "error", "payload": {"message": f"Invalid JSON command: {exc}"}})
            continue
        runtime.handle_command(command)
        if command.get("command") == "shutdown":
            break
    runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
