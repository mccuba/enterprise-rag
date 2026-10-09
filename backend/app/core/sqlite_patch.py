"""Ubuntu/Debian antiguos traen sqlite3 < 3.35; Chroma exige >= 3.35.

Parche oficial documentado por Chroma: usar pysqlite3-binary.
https://docs.trychroma.com/troubleshooting#sqlite
"""
from __future__ import annotations

import sys


def apply() -> None:
    try:
        import sqlite3

        parts = tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:3])
        if parts >= (3, 35, 0):
            return
    except Exception:
        pass

    try:
        import pysqlite3  # type: ignore

        sys.modules["sqlite3"] = pysqlite3
    except ImportError as exc:
        raise RuntimeError(
            "Chroma necesita sqlite3 >= 3.35. Instala: pip install pysqlite3-binary"
        ) from exc


apply()
