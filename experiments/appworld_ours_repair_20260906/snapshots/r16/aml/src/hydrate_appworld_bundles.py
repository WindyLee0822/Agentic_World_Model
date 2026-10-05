#!/usr/bin/env python3
"""Replace AppWorld Git LFS pointers with the pinned encrypted bundle objects."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import urllib.request
from pathlib import Path

APPWORLD_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
APPWORLD_REPOSITORY = "StonyBrookNLP/appworld"
BUNDLE_NAMES = ("apps.bundle", "tests.bundle")
LFS_SPEC = "version https://git-lfs.github.com/spec/v1"


def _package_directory() -> Path:
    spec = importlib.util.find_spec("appworld")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("the appworld package is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def _parse_lfs_pointer(payload: bytes) -> tuple[str, int] | None:
    if not payload.startswith(LFS_SPEC.encode("ascii")):
        return None
    try:
        fields = dict(line.split(" ", 1) for line in payload.decode("ascii").splitlines())
        algorithm, digest = fields["oid"].split(":", 1)
        size = int(fields["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed Git LFS pointer") from exc
    if algorithm != "sha256" or len(digest) != 64:
        raise ValueError(f"unsupported Git LFS object ID: {fields['oid']}")
    return digest, size


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tts-appworld-aml"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def hydrate_bundle(bundle_path: Path) -> bool:
    """Hydrate one pointer file. Return whether the file was replaced."""
    pointer = _parse_lfs_pointer(bundle_path.read_bytes())
    if pointer is None:
        return False

    expected_digest, expected_size = pointer
    url = (
        "https://media.githubusercontent.com/media/"
        f"{APPWORLD_REPOSITORY}/{APPWORLD_COMMIT}/"
        f"src/appworld/.source/{bundle_path.name}"
    )
    payload = _download_bytes(url)
    actual_digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size or actual_digest != expected_digest:
        raise RuntimeError(
            f"downloaded {bundle_path.name} failed Git LFS verification: "
            f"expected {expected_size} bytes/{expected_digest}, "
            f"found {len(payload)} bytes/{actual_digest}"
        )

    temporary_path = bundle_path.with_name(bundle_path.name + ".tmp")
    try:
        temporary_path.write_bytes(payload)
        os.replace(temporary_path, bundle_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def hydrate_appworld_bundles(package_directory: Path | None = None) -> None:
    source_directory = (package_directory or _package_directory()) / ".source"
    for bundle_name in BUNDLE_NAMES:
        bundle_path = source_directory / bundle_name
        if not bundle_path.is_file():
            raise FileNotFoundError(f"AppWorld package bundle not found: {bundle_path}")
        status = "hydrated" if hydrate_bundle(bundle_path) else "ready"
        print(f"[appworld] {bundle_name}: {status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-directory",
        type=Path,
        help="AppWorld package directory; defaults to the installed package",
    )
    args = parser.parse_args()
    hydrate_appworld_bundles(args.package_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
