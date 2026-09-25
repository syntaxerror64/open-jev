"""Publish the exported artifacts as a GitHub release (stage 5, option 1).

Usage::

    python scripts/release.py --tag vX.Y.Z --asset <path> [--asset ...]
                              [--notes-file FILE] [--dry-run]

The repository has no CI (no ``.github/`` at all), so we picked the
minimal-invasiveness path: one script that shells out to the GitHub CLI.

``--dry-run`` prints the manifest as JSON to stdout and exits 0 **without any
network access and without requiring ``gh``**::

    {"tag": ..., "assets": [{"path", "sha256", "size_bytes"}, ...],
     "notes_file": ..., "command": "gh release create ..."}

That flag is what the offline tests use (risk O2: tokens/network are
never exercised by the suite). Without ``--dry-run`` the same manifest is
executed with ``gh release create``; a missing ``gh`` is reported with an
install hint instead of a traceback.

Only the standard library — the script must run even where torch is not
installed (it operates on already-exported files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

__all__ = ["build_manifest", "main"]

_GH_HINT = (
    "error: `gh` GitHub CLI not found on PATH — install it "
    "(https://cli.github.com/) or pass --dry-run to see the manifest offline"
)


def _release_argv(tag: str, assets: Sequence[str], notes_file: str | None) -> list[str]:
    argv = ["gh", "release", "create", tag]
    if notes_file:
        argv += ["--notes-file", str(notes_file)]
    argv += [str(Path(a)) for a in assets]
    return argv


def build_manifest(
    tag: str, assets: Sequence[str], notes_file: str | None
) -> dict[str, object]:
    """Assemble the release manifest; digest every asset now, before publishing.

    Each asset must exist — a release whose ``sha256`` cannot be computed
    would be unauditable, and the sidecar manifest (see
    ``scripts/export_checkpoint.py``) is worthless if the artifact it
    describes never made it into the manifest.
    """
    entries: list[dict[str, object]] = []
    for asset in assets:
        path = Path(asset)
        if not path.is_file():
            raise SystemExit(f"error: asset not found: {path}")
        data = path.read_bytes()
        entries.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    if notes_file is not None and not Path(notes_file).is_file():
        raise SystemExit(f"error: --notes-file not found: {notes_file}")
    return {
        "tag": tag,
        "assets": entries,
        "notes_file": notes_file,
        "command": shlex.join(_release_argv(tag, assets, notes_file)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/release.py",
        description="Create a GitHub release for the exported artifacts; "
        "use --dry-run for an offline JSON manifest.",
    )
    parser.add_argument("--tag", required=True, help="release tag, e.g. v0.1.0")
    parser.add_argument(
        "--asset",
        action="append",
        required=True,
        dest="assets",
        metavar="PATH",
        help="file to attach (repeat the flag for several); usually the "
        "artifact plus its .json manifest",
    )
    parser.add_argument(
        "--notes-file", default=None, metavar="FILE", help="release notes file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the JSON manifest and exit 0 without network or gh",
    )
    args = parser.parse_args(argv)

    manifest = build_manifest(args.tag, args.assets, args.notes_file)

    if args.dry_run:
        # stdout must stay a single JSON document — tests json.loads() it.
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    argv_gh = _release_argv(args.tag, args.assets, args.notes_file)
    if shutil.which(argv_gh[0]) is None:
        raise SystemExit(_GH_HINT)
    print(f"+ {shlex.join(argv_gh)}", file=sys.stderr)
    try:
        proc = subprocess.run(argv_gh)
    except OSError as exc:  # PATH race / not executable
        raise SystemExit(f"{_GH_HINT} ({exc})") from exc
    if proc.returncode != 0:
        print(
            f"error: `gh release create` failed with exit code {proc.returncode}",
            file=sys.stderr,
        )
        return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
