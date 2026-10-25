"""Create or check SHA-256 hashes for the shareable experiment folder."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST or "__pycache__" in path.parts:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="write a fresh manifest")
    args = parser.parse_args()
    if args.create:
        hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in included_files()}
        MANIFEST.write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Recorded {len(hashes)} files in {MANIFEST.name}")
        return 0
    hashes = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"MISSING {relative}")
        elif file_hash(path) != expected:
            failures.append(f"CHANGED {relative}")
    if failures:
        print("\n".join(failures[:20]))
        print(f"Failed: {len(failures)} of {len(hashes)} files")
        return 1
    print(f"Verified {len(hashes)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
