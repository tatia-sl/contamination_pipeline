#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "assessment" / "data" / "artifacts_manifest.json"
ARTEFACT_DIRS = ("runs", "outputs", "logs")


@dataclass
class ArtifactEntry:
    path: str
    category: str
    size_bytes: int


def collect_entries() -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    for dirname in ARTEFACT_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            entries.append(
                ArtifactEntry(
                    path=path.relative_to(ROOT).as_posix(),
                    category=dirname,
                    size_bytes=path.stat().st_size,
                )
            )
    return entries


def main() -> None:
    entries = collect_entries()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(entries),
        "categories": {dirname: sum(1 for entry in entries if entry.category == dirname) for dirname in ARTEFACT_DIRS},
        "files": [asdict(entry) for entry in entries],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
