from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DermBenchCase:
    id: str
    source: str
    split: str
    category: str
    diagnosis: str
    dermnet_image_path: Path
    cot: Path
    cot_sha256: str

    def load_reference(self) -> str:
        return self.cot.read_text(encoding="utf-8")

    def image_file(self, image_root: str | Path) -> Path:
        return Path(image_root) / self.dermnet_image_path

    def image_data_url(self, image_root: str | Path) -> str:
        image = self.image_file(image_root)
        media_type = mimetypes.guess_type(image.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> list[DermBenchCase]:
    manifest_path = Path(path)
    repo_root = manifest_path.parents[2]
    cases: list[DermBenchCase] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                cases.append(
                    DermBenchCase(
                        id=row["id"],
                        source=row["source"],
                        split=row["split"],
                        category=row["category"],
                        diagnosis=row["diagnosis"],
                        dermnet_image_path=Path(row["dermnet_image_path"]),
                        cot=repo_root / row["cot"],
                        cot_sha256=row["cot_sha256"],
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Malformed manifest row {line_no}: missing {exc}") from exc
    return cases


def iter_limited(cases: Iterable[DermBenchCase], limit: int | None) -> Iterable[DermBenchCase]:
    for index, case in enumerate(cases):
        if limit is not None and index >= limit:
            break
        yield case
