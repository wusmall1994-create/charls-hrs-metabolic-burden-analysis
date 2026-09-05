from __future__ import annotations

import py_compile
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
IGNORED_PARTS = {".git", ".venv", "__pycache__", "data", "results", "outputs", "tables", "figures"}


class CodeReleaseTests(unittest.TestCase):
    def test_python_sources_compile(self) -> None:
        for path in SOURCE.glob("*.py"):
            with self.subTest(path=path.name):
                py_compile.compile(path, doraise=True)

    def test_no_prohibited_artifact_types(self) -> None:
        prohibited = {
            ".dta", ".sav", ".sas7bdat", ".csv", ".tsv", ".xlsx", ".xls",
            ".parquet", ".feather", ".rds", ".rdata", ".doc", ".docx", ".pdf",
            ".png", ".tif", ".tiff",
        }
        found = [
            p.relative_to(ROOT)
            for p in ROOT.rglob("*")
            if p.is_file()
            and not IGNORED_PARTS.intersection(p.parts)
            and p.suffix.lower() in prohibited
        ]
        self.assertEqual(found, [])

    def test_no_local_paths_or_contact_details(self) -> None:
        patterns = {
            "Windows absolute path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"),
            "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
            "Chinese mobile number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
            "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
            "generic secret assignment": re.compile(r"(?i)(token|password|secret)\s*[:=]\s*['\"][^'\"]+['\"]"),
        }
        violations = []
        for path in ROOT.rglob("*"):
            if (
                not path.is_file()
                or IGNORED_PARTS.intersection(path.parts)
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in patterns.items():
                if pattern.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: {label}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
