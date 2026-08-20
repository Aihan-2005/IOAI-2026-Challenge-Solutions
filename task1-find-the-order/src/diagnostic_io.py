from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.diagnostics import (
    DiagnosticsReport,
)


def write_diagnostics_report(
    *,
    report: DiagnosticsReport,
    output_path: str | Path,
) -> Path:
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    payload = asdict(
        report
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )

    return path