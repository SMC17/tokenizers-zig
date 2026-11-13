#!/usr/bin/env python3
"""Fetch only the three pinned public tokenizer JSON files; never model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing files without network access")
    args = parser.parse_args()
    pins = json.loads((ROOT / "tests/fixtures/real-model-pins.json").read_text())
    for pin in pins:
        path = (ROOT / pin["path"]).resolve()
        if not path.is_relative_to(ROOT / "tests/fixtures") or not path.name.endswith(".json"):
            raise RuntimeError("Fixture path outside the allowed directory")
        if path.exists():
            data = path.read_bytes()
        elif args.check:
            raise RuntimeError(f"Missing fixture: {path.name}")
        else:
            expected_url = f"https://huggingface.co/{pin['model']}/resolve/{pin['revision']}/tokenizer.json"
            if pin["url"] != expected_url or len(pin["revision"]) != 40:
                raise RuntimeError("Unexpected source identity")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(expected_url, timeout=30) as response:
                        data = response.read(5_000_001)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise RuntimeError(f"Fixture identity differs; refusing overwrite: {path.name}")
        json.loads(data)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                file.write(data)
            temporary.replace(path)
        print(f"FIXTURE_OK {pin['model']} {pin['sha256']}")


if __name__ == "__main__":
    main()
