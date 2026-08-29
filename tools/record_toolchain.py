"""The external codes this package leans on, hashed where they stand.

MONKES, SPEC, stella and DESC are hand-built rather than pinned by a package manager, so
the records that depend on them otherwise assume whatever binary happened to be on the
node. This hashes each one that exists here and writes the roster the other records can
name.

    python tools/record_toolchain.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path("results/hardware/toolchain.json")

#: Tool name -> candidate paths, first match wins.
TOOLS: dict[str, tuple[str, ...]] = {
    "monkes": ("~/monkes/bin/main_monkes.x",),
    "spec": ("~/src/SPEC/build/build/bin/xspec",),
    "stella": ("~/src/stella/build/gnu/COMPILATION/stella",),
    "desc_python": ("~/.venv/bin/python",),
    "vmecpp": ("venv/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so",),
}


#: Shared objects a tool above loads at run time, as tool name -> soname. A build that
#: moves its core into a sibling library leaves the extension module a stub, and hashing
#: the stub then identifies the binding and not the solver; the loader also chooses
#: between `glibc-hwcaps` variants by CPU, so what the roster wants is the one that
#: resolves on this node.
LINKED: dict[str, str] = {"vmecpp": "libvmecpp_core.so"}


def linked(path: Path, soname: str) -> Path | None:
    """Where the dynamic loader finds ``soname`` for ``path``, or None if it does not."""
    try:
        report = subprocess.run(
            ["ldd", str(path)], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in report.splitlines():
        left, arrow, right = line.partition("=>")
        if not arrow or left.strip().rpartition("/")[2] != soname:
            continue
        target = right.strip().partition(" (")[0]
        if target:
            return Path(target)
    return None


def declared(path: Path) -> str:
    """``path`` written without a node or account name, as the roster states them."""
    for root, prefix in ((Path.cwd(), ""), (Path.home(), "~/")):
        try:
            return prefix + Path(path).relative_to(root).as_posix()
        except ValueError:
            continue
    return Path(path).name


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    # The sha256 identifies the build. The path is written as it is declared
    # above rather than expanded, and the node is not recorded, so the roster
    # carries no account or machine name into a committed record.
    record: dict = {"tools": {}}
    for name, candidates in TOOLS.items():
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.exists():
                record["tools"][name] = {
                    "path": candidate,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
                print(f"{name:12s} {record['tools'][name]['sha256'][:12]}  {candidate}")
                soname = LINKED.get(name)
                core = linked(path, soname) if soname else None
                if core is not None and core.exists():
                    key = f"{name}_core"
                    record["tools"][key] = {
                        "path": declared(core),
                        "sha256": digest(core),
                        "bytes": core.stat().st_size,
                    }
                    print(
                        f"{key:12s} {record['tools'][key]['sha256'][:12]}  "
                        f"{record['tools'][key]['path']}"
                    )
                break
        else:
            print(f"{name:12s} absent on this node")
    if not record["tools"]:
        # Run on a node carrying none of them this would replace the roster
        # with an empty one, so it writes nothing instead.
        print(f"no tool found here; {OUT} left as it stands")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
