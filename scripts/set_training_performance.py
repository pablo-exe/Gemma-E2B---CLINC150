"""Disable Windows' inferred power throttling for an existing training worker.

This changes only the specified process; system power-plan settings are untouched.
"""

import argparse
import ctypes
import json
from ctypes import wintypes
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    if not any("train_distillation.py" in part for part in process.cmdline()):
        raise ValueError("Target process is not the distillation worker")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetProcessInformation.argtypes = kernel.GetProcessInformation.argtypes
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]

    class PowerState(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.DWORD),
            ("ControlMask", wintypes.DWORD),
            ("StateMask", wintypes.DWORD),
        ]

    handle = kernel.OpenProcess(0x0200 | 0x1000, False, args.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        before = PowerState(1, 0, 0)
        if not kernel.GetProcessInformation(handle, 4, ctypes.byref(before), ctypes.sizeof(before)):
            raise ctypes.WinError(ctypes.get_last_error())
        after = PowerState(1, 0x1 | 0x4, 0)
        if not kernel.SetProcessInformation(handle, 4, ctypes.byref(after), ctypes.sizeof(after)):
            after = PowerState(1, 0x1, 0)
            if not kernel.SetProcessInformation(
                handle, 4, ctypes.byref(after), ctypes.sizeof(after)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        previous_priority = process.nice()
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        report = {
            "worker_pid": args.pid,
            "before": {"control_mask": before.ControlMask, "state_mask": before.StateMask},
            "after": {"control_mask": after.ControlMask, "state_mask": after.StateMask},
            "previous_priority": int(previous_priority),
            "priority": int(process.nice()),
            "scope": "training process only",
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    main()
