"""Load behavioral harness texts from HARNESS_LIBRARY.md -- the single source of truth.

Harness text is NEVER hardcoded in training or evaluation code: every consumer resolves
a registry id through `resolve_harness` (or the `--harness-name` CLI flag via
`add_harness_argument` / `harness_from_args`) and receives the registered text together
with its sha256. The hash is verified against the text at load time; a mismatch raises
immediately, so registry/text drift fails loudly instead of silently distilling the
wrong harness.

Registry rules (HARNESS_LIBRARY.md header): hashes are over the exact UTF-8 bytes of
the text with no trailing newline; any wording change is a NEW entry with a new id and
hash, never an in-place edit.
"""

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "HARNESS_LIBRARY.md"

# Reserved --harness-name value meaning "no harness text" (the student condition).
BASE_HARNESS_NAME = "base"


@dataclass(frozen=True)
class Harness:
    harness_id: str
    text: str
    sha256: str
    status: str

    @property
    def short_name(self) -> str:
        """'VERIFY-001' -> 'verify' (unambiguous only while one entry per prefix exists)."""
        return self.harness_id.rsplit("-", 1)[0].lower()


def _extract_entry_field(section: str, field: str) -> Optional[str]:
    """Pull the value cell of a `| **field** | value |` row inside one registry section."""
    m = re.search(rf"^\|\s*\*\*{re.escape(field)}\*\*\s*\|(.+)\|\s*$", section, re.MULTILINE)
    if not m:
        return None
    cell = m.group(1).strip()
    if cell.startswith("`") and cell.endswith("`"):
        return cell[1:-1]
    # status rows look like: `candidate` (pre-registered seed)
    m = re.match(r"`([^`]+)`", cell)
    return m.group(1) if m else cell


def parse_registry(markdown: str) -> Dict[str, Harness]:
    """Parse `## Entry:` sections of HARNESS_LIBRARY.md, verifying every sha256."""
    entries: Dict[str, Harness] = {}
    sections = re.split(r"(?m)^## Entry:\s*", markdown)[1:]
    for section in sections:
        harness_id = section.splitlines()[0].strip()
        text = _extract_entry_field(section, "harness text")
        sha = _extract_entry_field(section, "sha256")
        status = _extract_entry_field(section, "status")
        if text is None or sha is None:
            raise ValueError(f"Registry entry {harness_id!r} is missing harness text or sha256")
        if status is None:
            raise ValueError(f"Registry entry {harness_id!r} is missing status")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != sha:
            raise ValueError(
                f"Harness registry hash mismatch for {harness_id!r}: registered {sha}, "
                f"computed {actual} over the text currently in HARNESS_LIBRARY.md. The registry "
                "text and hash have drifted; register a NEW entry instead of editing this one."
            )
        entries[harness_id] = Harness(harness_id=harness_id, text=text, sha256=sha, status=status)
    if not entries:
        raise ValueError("No '## Entry:' sections found in the harness registry")
    return entries


def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, Harness]:
    return parse_registry(path.read_text(encoding="utf-8"))


def resolve_harness(name: Optional[str], registry: Optional[Dict[str, Harness]] = None) -> Optional[Harness]:
    """Resolve a --harness-name value to a registry entry.

    Accepts 'base' (no harness text -> the student condition), a full registry id
    ('VERIFY-001', case-insensitive), or an unambiguous short form ('verify').
    Raises ValueError, listing the available ids, when the name is unknown or ambiguous.
    """
    if name is None:
        return None
    name = name.strip()
    if name == "" or name.lower() == BASE_HARNESS_NAME:
        return None
    registry = registry if registry is not None else load_registry()
    by_id = {h.harness_id.lower(): h for h in registry.values()}
    if name.lower() in by_id:
        return by_id[name.lower()]
    short = name.lower()
    matches = [h for h in registry.values() if h.short_name == short]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        ids = ", ".join(sorted(h.harness_id for h in matches))
        raise ValueError(f"Ambiguous harness name {name!r}; matching entries: {ids}. Use the full registry id.")
    short_names = sorted({h.short_name for h in registry.values()})
    raise ValueError(
        f"Unknown harness name {name!r}. Available: '{BASE_HARNESS_NAME}', registry ids "
        f"{sorted(registry)} (short names: {short_names})."
    )


def add_harness_argument(parser: argparse.ArgumentParser, default: str = BASE_HARNESS_NAME) -> None:
    """Add the shared --harness-name flag so every harness-consuming CLI selects
    texts through the registry rather than hardcoded strings."""
    parser.add_argument(
        "--harness-name",
        type=str,
        default=default,
        help="Harness registry id or short name ('verify', 'decompose', 'VERIFY-001', ...); "
             f"'{BASE_HARNESS_NAME}' selects no harness text (the student condition). "
             "Texts are loaded from HARNESS_LIBRARY.md, never hardcoded.",
    )


def harness_from_args(args: argparse.Namespace, registry: Optional[Dict[str, Harness]] = None) -> Optional[Harness]:
    return resolve_harness(getattr(args, "harness_name", None), registry)
