"""Keep system/display awake only while a specified training worker is alive."""

import argparse
import ctypes
import json
import time
from pathlib import Path

import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
worker = psutil.Process(args.pid)
if not any("train_distillation.py" in part for part in worker.cmdline()):
    raise ValueError("Target is not the training worker")
created = worker.create_time()
if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000003):
    raise ctypes.WinError()
args.report.write_text(
    json.dumps(
        {
            "worker_pid": args.pid,
            "guard_pid": psutil.Process().pid,
            "system_required": True,
            "display_required": True,
            "scope": "until this training worker exits",
        },
        indent=2,
    )
    + "\n"
)
try:
    while worker.is_running() and worker.create_time() == created:
        time.sleep(5)
except psutil.NoSuchProcess:
    pass
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
