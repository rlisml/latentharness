"""Registry tests: loading, hash integrity, name resolution, no hardcoding."""

import hashlib
import unittest
from pathlib import Path

from harness.harness_text import (
    BASE_HARNESS_NAME,
    REGISTRY_PATH,
    Harness,
    load_registry,
    resolve_harness,
)

HARNESS_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "harness"


class TestHarnessRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()

    def test_seed_entries_present(self):
        self.assertIn("VERIFY-001", self.registry)
        self.assertIn("DECOMPOSE-001", self.registry)

    def test_registry_hash_matches_text(self):
        for h in self.registry.values():
            computed = hashlib.sha256(h.text.encode("utf-8")).hexdigest()
            self.assertEqual(computed, h.sha256, f"hash mismatch for {h.harness_id}")

    def test_registry_hash_is_over_exact_bytes_no_trailing_newline(self):
        # Registry rule: hash over exact UTF-8 bytes, no trailing newline.
        for h in self.registry.values():
            self.assertFalse(h.text.endswith("\n"), h.harness_id)
            with_hash = hashlib.sha256((h.text + "\n").encode("utf-8")).hexdigest()
            self.assertNotEqual(with_hash, h.sha256, h.harness_id)

    def test_texts_nonempty_and_statuses_valid(self):
        valid = {"candidate", "screened-in", "screened-out", "harmful-control", "distilled"}
        for h in self.registry.values():
            self.assertTrue(h.text.strip())
            self.assertIn(h.status, valid)

    def test_resolve_base_returns_none(self):
        self.assertIsNone(resolve_harness("base"))
        self.assertIsNone(resolve_harness("BASE"))
        self.assertIsNone(resolve_harness(""))
        self.assertIsNone(resolve_harness(None))

    def test_resolve_short_and_full_ids(self):
        self.assertEqual(resolve_harness("verify").harness_id, "VERIFY-001")
        self.assertEqual(resolve_harness("VERIFY").harness_id, "VERIFY-001")
        self.assertEqual(resolve_harness("verify-001").harness_id, "VERIFY-001")
        # E4 (2026-09-06): a family with >1 entry (DECOMPOSE-002/003) makes the
        # bare short form ambiguous by design — the full id still resolves
        self.assertEqual(resolve_harness("DECOMPOSE-001").harness_id, "DECOMPOSE-001")
        with self.assertRaises(ValueError) as ctx:
            resolve_harness("decompose")
        self.assertIn("Ambiguous", str(ctx.exception))

    def test_resolve_uses_provided_registry(self):
        fake = {
            "VERIFY-001": Harness("VERIFY-001", "text one", "0" * 64, "candidate"),
            "VERIFY-002": Harness("VERIFY-002", "text two", "1" * 64, "candidate"),
        }
        self.assertEqual(resolve_harness("VERIFY-002", fake).harness_id, "VERIFY-002")
        with self.assertRaises(ValueError) as ctx:
            resolve_harness("verify", fake)
        self.assertIn("Ambiguous", str(ctx.exception))

    def test_resolve_unknown_raises_with_available_ids(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_harness("no-such-harness")
        msg = str(ctx.exception)
        self.assertIn("VERIFY-001", msg)
        self.assertIn(BASE_HARNESS_NAME, msg)

    def test_harness_text_not_hardcoded_in_code(self):
        # Task constraint: training/evaluation code must load texts from the
        # registry, never hardcode them.
        for path in HARNESS_PACKAGE_DIR.glob("*.py"):
            src = path.read_text(encoding="utf-8")
            for h in self.registry.values():
                self.assertNotIn(h.text, src, f"{path.name} hardcodes {h.harness_id}")

    def test_registry_file_is_the_declared_source(self):
        self.assertEqual(REGISTRY_PATH.name, "HARNESS_LIBRARY.md")
        self.assertTrue(REGISTRY_PATH.exists())


if __name__ == "__main__":
    unittest.main()
