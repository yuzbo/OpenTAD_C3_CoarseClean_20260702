#!/usr/bin/env python3
"""Report processes that Windows Restart Manager associates with one file."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys


CCH_RM_MAX_APP_NAME = 255
CCH_RM_MAX_SVC_NAME = 63
CCH_RM_SESSION_KEY = 32
ERROR_MORE_DATA = 234


class RM_UNIQUE_PROCESS(ctypes.Structure):
    _fields_ = [("dwProcessId", wintypes.DWORD), ("ProcessStartTime", wintypes.FILETIME)]


class RM_PROCESS_INFO(ctypes.Structure):
    _fields_ = [
        ("Process", RM_UNIQUE_PROCESS),
        ("strAppName", wintypes.WCHAR * (CCH_RM_MAX_APP_NAME + 1)),
        ("strServiceShortName", wintypes.WCHAR * (CCH_RM_MAX_SVC_NAME + 1)),
        ("ApplicationType", wintypes.DWORD),
        ("AppStatus", wintypes.ULONG),
        ("TSSessionId", wintypes.DWORD),
        ("bRestartable", wintypes.BOOL),
    ]


def main() -> int:
    target = str(Path(sys.argv[1]).resolve())
    restart_manager = ctypes.WinDLL("rstrtmgr")
    handle = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(CCH_RM_SESSION_KEY + 1)
    result = restart_manager.RmStartSession(ctypes.byref(handle), 0, key)
    if result:
        raise OSError(result, "RmStartSession")
    try:
        files = (wintypes.LPCWSTR * 1)(target)
        result = restart_manager.RmRegisterResources(handle, 1, files, 0, None, 0, None)
        if result:
            raise OSError(result, "RmRegisterResources")
        needed = wintypes.UINT()
        count = wintypes.UINT()
        reason = wintypes.DWORD()
        result = restart_manager.RmGetList(
            handle, ctypes.byref(needed), ctypes.byref(count), None, ctypes.byref(reason)
        )
        if result not in (0, ERROR_MORE_DATA):
            raise OSError(result, "RmGetList(size)")
        entries = []
        if needed.value:
            records = (RM_PROCESS_INFO * needed.value)()
            count.value = needed.value
            result = restart_manager.RmGetList(
                handle, ctypes.byref(needed), ctypes.byref(count), records, ctypes.byref(reason)
            )
            if result:
                raise OSError(result, "RmGetList(data)")
            for record in records[: count.value]:
                entries.append(
                    {
                        "pid": record.Process.dwProcessId,
                        "application": record.strAppName,
                        "service": record.strServiceShortName,
                        "restartable": bool(record.bRestartable),
                    }
                )
        print(json.dumps({"file": target, "processes": entries}, ensure_ascii=False))
        return 0
    finally:
        restart_manager.RmEndSession(handle)


if __name__ == "__main__":
    raise SystemExit(main())
