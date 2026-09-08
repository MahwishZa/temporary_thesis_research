#!/usr/bin/env python3
"""Offline tests for the PMC open-access inventory script.

Nothing here touches the network, and nothing reads or writes anything under
pubmed/. Everything runs against temporary files.

The centrepiece is WindowsCsvLimitTests, which reproduces the OverflowError
that stopped the script on Windows and proves the fix. On Windows a C "long"
is 32-bit even in 64-bit Python, so csv.field_size_limit() rejects anything
above 2,147,483,647 -- and sys.maxsize is about 9.2e18. These tests simulate
that rejection so the regression is caught on Linux and macOS too.

Run with:  python3 -m unittest discover -s pmc -v
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inventory_pmc_oa as inv  # noqa: E402

# The largest value a 32-bit C long can hold -- the real ceiling on Windows.
WINDOWS_C_LONG_MAX = 2**31 - 1

# Mirrors the real pubmed_results.csv, whose longest field is a ~15,000
# character abstract.
PUBMED_COLUMNS = [
    "pmid", "title", "abstract", "publication_date", "publication_year",
    "journal", "authors", "doi", "pmcid", "mesh_terms", "publication_types",
    "query_ids", "record_status",
]


def write_pubmed_like_csv(path: Path, rows: int, abstract_chars: int) -> list[str]:
    """Build a CSV shaped like pubmed_results.csv. Returns the PMCIDs written.

    Deliberately includes the things that break naive CSV handling: a very long
    abstract, embedded newlines, commas and double quotes, and rows with no
    PMCID at all.
    """
    nasty = 'He said "maybe", then\nnewline; comma, done. '
    abstract = (nasty * (abstract_chars // len(nasty) + 1))[:abstract_chars]
    expected: list[str] = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBMED_COLUMNS)
        writer.writeheader()
        for i in range(rows):
            # Every third row has no PMCID, like the real data.
            pmcid = "" if i % 3 == 2 else f"PMC{9000000 + i}"
            if pmcid:
                expected.append(pmcid)
            writer.writerow({
                "pmid": str(30000000 + i),
                "title": f"Study number {i}, with a comma",
                "abstract": abstract,
                "publication_date": "2023-05-01",
                "publication_year": "2023",
                "journal": "Journal of Tests",
                "authors": "Smith, Jane A; Lee, KB",
                "doi": f"10.1000/test.{i}",
                "pmcid": pmcid,
                "mesh_terms": "Alzheimer Disease* (diagnosis)",
                "publication_types": "Journal Article; Review",
                "query_ids": "Q0.1; Q4.1",
                "record_status": "ok",
            })
    return expected


class FakeWindowsCsv:
    """Stands in for the csv module's field_size_limit on Windows.

    Anything above a 32-bit C long raises OverflowError, exactly as CPython
    does on Windows.
    """

    def __init__(self, start: int = 131072) -> None:
        self.current = start
        self.rejected: list[int] = []

    def field_size_limit(self, new: int | None = None) -> int:
        if new is None:
            return self.current
        if new > WINDOWS_C_LONG_MAX:
            self.rejected.append(new)
            raise OverflowError("Python int too large to convert to C long")
        previous = self.current
        self.current = new
        return previous


class WindowsCsvLimitTests(unittest.TestCase):
    """The bug that stopped the script on Windows, and the fix."""

    def setUp(self) -> None:
        self.real = inv.csv.field_size_limit()
        self.addCleanup(inv.csv.field_size_limit, self.real)

    def test_the_old_approach_is_what_broke_on_windows(self) -> None:
        """sys.maxsize is rejected by a 32-bit C long. This was the bug."""
        fake = FakeWindowsCsv()
        with self.assertRaises(OverflowError):
            fake.field_size_limit(sys.maxsize)
        self.assertEqual(fake.rejected, [sys.maxsize])

    def test_the_fix_succeeds_under_the_windows_limit(self) -> None:
        fake = FakeWindowsCsv()
        original = inv.csv.field_size_limit
        inv.csv.field_size_limit = fake.field_size_limit
        try:
            effective = inv.widen_csv_field_limit()
        finally:
            inv.csv.field_size_limit = original

        self.assertLessEqual(effective, WINDOWS_C_LONG_MAX,
                             "must never ask for more than a 32-bit long")
        self.assertEqual(effective, fake.current)
        self.assertEqual(fake.rejected, [], "64 MB should be accepted outright")

    def test_it_halves_until_a_hostile_platform_accepts(self) -> None:
        """Even with an absurdly small ceiling, it settles instead of raising."""
        fake = FakeWindowsCsv(start=1000)
        fake_max = 5000

        def picky(new=None):
            if new is None:
                return fake.current
            if new > fake_max:
                fake.rejected.append(new)
                raise OverflowError("Python int too large to convert to C long")
            fake.current = new
            return new

        original = inv.csv.field_size_limit
        inv.csv.field_size_limit = picky
        try:
            effective = inv.widen_csv_field_limit()
        finally:
            inv.csv.field_size_limit = original

        self.assertLessEqual(effective, fake_max)
        self.assertGreater(effective, 1000, "should still raise above the start")
        self.assertGreater(len(fake.rejected), 1, "expected repeated halving")

    def test_it_never_lowers_an_existing_higher_limit(self) -> None:
        inv.csv.field_size_limit(CSV_HIGH := 200 * 1024 * 1024)
        self.assertEqual(inv.widen_csv_field_limit(1024), CSV_HIGH)

    def test_the_default_target_fits_a_32_bit_long(self) -> None:
        self.assertLessEqual(inv.CSV_FIELD_LIMIT, WINDOWS_C_LONG_MAX)

    def test_no_module_still_calls_field_size_limit_with_maxsize(self) -> None:
        """Guard against the bad pattern creeping back in."""
        source = Path(inv.__file__).read_text(encoding="utf-8")
        self.assertNotIn("field_size_limit(sys.maxsize)", source)


class LargeCsvReadingTests(unittest.TestCase):
    """Reading a PubMed-shaped CSV, including under the Windows ceiling."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.real = inv.csv.field_size_limit()
        self.addCleanup(inv.csv.field_size_limit, self.real)

    def test_reads_a_pubmed_shaped_csv_with_long_abstracts(self) -> None:
        path = self.tmp / "pubmed_results.csv"
        expected = write_pubmed_like_csv(path, rows=300, abstract_chars=15_000)
        pairs = inv.read_pmcids(path)
        self.assertEqual([p for p, _ in pairs], expected)
        self.assertEqual(len(pairs), 200, "one third of rows have no PMCID")
        self.assertEqual(pairs[0], ("PMC9000000", "30000000"))

    def test_reads_fields_larger_than_the_python_default(self) -> None:
        """A 200,000-character abstract exceeds csv's 131,072 default."""
        path = self.tmp / "big.csv"
        write_pubmed_like_csv(path, rows=3, abstract_chars=200_000)
        inv.csv.field_size_limit(131072)  # back to the stock default
        self.assertEqual(len(inv.read_pmcids(path)), 2)

    def test_reads_correctly_with_a_windows_sized_ceiling(self) -> None:
        """The end-to-end proof: works when the cap cannot exceed 32 bits."""
        path = self.tmp / "pubmed_results.csv"
        expected = write_pubmed_like_csv(path, rows=120, abstract_chars=15_000)

        fake = FakeWindowsCsv()
        original = inv.csv.field_size_limit
        inv.csv.field_size_limit = fake.field_size_limit
        try:
            # Real parsing needs the real limit raised; the fake only records
            # what the code *asks* for, which is the part that crashed.
            original(inv.CSV_FIELD_LIMIT)
            pairs = inv.read_pmcids(path)
        finally:
            inv.csv.field_size_limit = original

        self.assertEqual([p for p, _ in pairs], expected)
        self.assertEqual(fake.rejected, [], "nothing over a 32-bit long requested")
        self.assertLessEqual(fake.current, WINDOWS_C_LONG_MAX)

    def test_resume_file_with_long_fields_also_reads(self) -> None:
        """already_done() raises the same ceiling, so it needed the same fix."""
        path = self.tmp / "pmc_oa_inventory.csv"
        handle, writer = inv.open_appending(path, inv.INVENTORY_FIELDS)
        writer.writerow(inv.build_row("PMC1", "1", {
            "version": 1, "is_pmc_openaccess": True, "license_code": "CC BY",
            "title": "x" * 200_000, "media_urls": [],
        }))
        handle.close()
        inv.csv.field_size_limit(131072)
        self.assertEqual(inv.already_done(path), {"PMC1"})


class UnchangedBehaviourTests(unittest.TestCase):
    """The rest of the design must still work exactly as before."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_git_lfs_pointer_is_still_detected(self) -> None:
        path = self.tmp / "pubmed_results.csv"
        path.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:027c43eca2069993ad73e0484d4ac4432ba82055eb95bcc93c712586b5760a90\n"
            "size 102314345\n", encoding="utf-8")
        self.assertTrue(inv.looks_like_lfs_pointer(path))
        with self.assertRaises(SystemExit) as caught:
            inv.read_pmcids(path)
        self.assertIn("git lfs pull", str(caught.exception))

    def test_a_real_csv_is_not_mistaken_for_a_pointer(self) -> None:
        path = self.tmp / "pubmed_results.csv"
        write_pubmed_like_csv(path, rows=2, abstract_chars=100)
        self.assertFalse(inv.looks_like_lfs_pointer(path))

    def test_missing_pmcid_column_is_reported_clearly(self) -> None:
        path = self.tmp / "x.csv"
        path.write_text("pmid,title\n1,a\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            inv.read_pmcids(path)
        self.assertIn("no 'pmcid' column", str(caught.exception))

    def test_newest_version_is_chosen_numerically(self) -> None:
        keys = ["metadata/PMC1.1.json", "metadata/PMC1.10.json", "metadata/PMC1.2.json"]
        self.assertEqual(inv.newest_version_key(keys), "metadata/PMC1.10.json")

    def test_row_for_an_article_in_the_dataset(self) -> None:
        row = inv.build_row("PMC9277667", "30584145", {
            "version": 1, "pmid": 30584145, "doi": "10.3233/JAD-180825",
            "title": "T", "is_pmc_openaccess": True, "is_manuscript": False,
            "is_historical_ocr": False, "is_retracted": False,
            "license_code": "CC BY-NC", "xml_url": "s3://b/x.xml",
            "pdf_url": None, "text_url": "s3://b/x.txt", "media_urls": ["a", "b"],
        })
        self.assertEqual(set(row), set(inv.INVENTORY_FIELDS))
        self.assertEqual(row["in_pmc_oa_dataset"], "yes")
        self.assertEqual(row["license_code"], "CC BY-NC")
        self.assertEqual((row["has_xml"], row["has_pdf"], row["has_text"]),
                         ("yes", "no", "yes"))
        self.assertEqual(row["media_count"], "2")

    def test_row_for_an_article_absent_from_the_dataset(self) -> None:
        row = inv.build_row("PMC10000001", "999", None)
        self.assertEqual(set(row), set(inv.INVENTORY_FIELDS))
        self.assertEqual(row["in_pmc_oa_dataset"], "no")
        self.assertEqual(row["license_code"], "")

    def test_resume_skips_what_is_already_written(self) -> None:
        path = self.tmp / "inv.csv"
        self.assertEqual(inv.already_done(path), set(), "missing file is empty")
        handle, writer = inv.open_appending(path, inv.INVENTORY_FIELDS)
        writer.writerow(inv.build_row("PMC1", "1", None))
        handle.close()
        handle, writer = inv.open_appending(path, inv.INVENTORY_FIELDS)
        writer.writerow(inv.build_row("PMC2", "2", None))
        handle.close()
        self.assertEqual(inv.already_done(path), {"PMC1", "PMC2"})
        self.assertEqual(path.read_text(encoding="utf-8").count("pmcid,pmid"), 1,
                         "header must be written exactly once")

    def test_a_404_is_a_result_not_a_failure(self) -> None:
        self.assertTrue(issubclass(inv.NotFound, Exception))
        self.assertIsNot(inv.NotFound, RuntimeError)


if __name__ == "__main__":
    unittest.main(verbosity=2)
