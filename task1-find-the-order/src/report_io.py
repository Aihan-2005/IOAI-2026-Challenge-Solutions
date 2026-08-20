from __future__ import annotations

import json
from pathlib import Path

from src.benchmark import (
    BenchmarkReport,
    report_to_dict,
)


def write_benchmark_report(
    *,
    report: BenchmarkReport,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            report_to_dict(
                report
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )

    return path