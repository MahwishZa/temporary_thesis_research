#!/usr/bin/env python3
"""Offline tests for the PMC XML downloader.

No test here touches the network: urlopen is replaced by a fake that serves
bytes from an in-memory table and can be told to fail in specific ways. No real
PMC article is ever requested, and nothing under pubmed/ is read or written.

Run with:  python3 -m unittest discover -s pmc -v
"""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_pmc_xml as dl  # noqa: E402

XML_A = b"<article><front><article-title>Alzheimer's diagnosis</article-title></front></article>"
XML_B = b"<article><front><article-title>Second paper</article-title></front></article>"
MD5_A = hashlib.md5(XML_A).hexdigest()
MD5_B = hashlib.md5(XML_B).hexdigest()

INVENTORY_COLUMNS = [
    "pmcid", "pmid", "in_pmc_oa_dataset", "is_pmc_openaccess", "license_code",
    "is_manuscript", "is_historical_ocr", "is_retracted", "version",
    "pmid_from_pmc", "doi_from_pmc", "title_from_pmc", "has_xml", "has_pdf",
    "has_text", "media_count", "xml_url", "pdf_url", "text_url", "checked_at_utc",
]


def s3_url(pmcid: str, md5: str, version: int = 1) -> str:
    return (f"s3://pmc-oa-opendata/{pmcid}.{version}/{pmcid}.{version}.xml?md5={md5}")


def inventory_row(pmcid, md5, *, has_xml="yes", pmid="111", licence="CC BY",
                  retracted="no", manuscript="no", xml_url=None, version=1):
    return {
        "pmcid": pmcid, "pmid": pmid, "in_pmc_oa_dataset": "yes",
        "is_pmc_openaccess": "yes", "license_code": licence,
        "is_manuscript": manuscript, "is_historical_ocr": "no",
        "is_retracted": retracted, "version": str(version),
        "pmid_from_pmc": pmid, "doi_from_pmc": f"10.1000/{pmcid}",
        "title_from_pmc": f"Title for {pmcid}", "has_xml": has_xml,
        "has_pdf": "yes", "has_text": "yes", "media_count": "0",
        "xml_url": s3_url(pmcid, md5, version) if xml_url is None else xml_url,
        "pdf_url": f"s3://pmc-oa-opendata/{pmcid}.1/{pmcid}.1.pdf?md5={'a'*32}",
        "text_url": f"s3://pmc-oa-opendata/{pmcid}.1/{pmcid}.1.txt?md5={'b'*32}",
        "checked_at_utc": "2026-08-31T09:00:00+00:00",
    }


def write_inventory(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class FakeHTTP:
    """Replacement for urllib.request.urlopen. Records every URL requested."""

    def __init__(self, bodies: dict[str, bytes], *, failures: dict[str, list] | None = None):
        self.bodies = bodies
        self.failures = failures or {}     # url -> list of outcomes, consumed in order
        self.requested: list[str] = []

    def __call__(self, request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        self.requested.append(url)
        queue = self.failures.get(url)
        if queue:
            outcome = queue.pop(0)
            if outcome is not None:
                raise outcome
        if url not in self.bodies:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return _Response(self.bodies[url])


class _Response:
    def __init__(self, payload: bytes):
        self._buffer = io.BytesIO(payload)

    def read(self, size=-1):
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DownloaderTestCase(unittest.TestCase):
    """Shared scaffolding: a temp workspace and a patched urlopen."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.inventory = self.tmp / "pmc_oa_inventory.csv"
        self.out = self.tmp / "fulltext"
        self._real_urlopen = dl.urllib.request.urlopen
        self._real_sleep = dl.time.sleep
        dl.time.sleep = lambda _s: None          # no real delays in tests
        self.addCleanup(self.restore)

    def restore(self) -> None:
        dl.urllib.request.urlopen = self._real_urlopen
        dl.time.sleep = self._real_sleep

    def serve(self, bodies, failures=None) -> FakeHTTP:
        fake = FakeHTTP(bodies, failures=failures)
        dl.urllib.request.urlopen = fake
        return fake

    def args(self, **overrides):
        base = dict(inventory=self.inventory, output_dir=self.out, limit=None,
                    sleep=0.0, timeout=5, max_attempts=3, list_only=False)
        base.update(overrides)
        return type("Args", (), base)()

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def manifest_rows(self):
        return self._read_csv(self.out / "manifest.csv")

    def failure_rows(self):
        return self._read_csv(self.out / "failures.csv")


# ---------------------------------------------------------------------------


class InventoryReadingTests(DownloaderTestCase):
    def test_selects_only_has_xml_rows(self) -> None:
        write_inventory(self.inventory, [
            inventory_row("PMC1", MD5_A),
            inventory_row("PMC2", MD5_B, has_xml=""),      # not in the dataset
            inventory_row("PMC3", MD5_A, has_xml="no"),
            inventory_row("PMC4", MD5_B),
        ])
        candidates, malformed = dl.read_candidates(self.inventory)
        self.assertEqual([c["pmcid"] for c in candidates], ["PMC1", "PMC4"])
        self.assertEqual(malformed, [])

    def test_malformed_rows_are_reported_not_dropped(self) -> None:
        write_inventory(self.inventory, [
            inventory_row("PMC1", MD5_A),
            inventory_row("PMC2", MD5_A, xml_url="https://evil.example/x.xml"),
            inventory_row("PMC3", MD5_A, xml_url=""),
            inventory_row("", MD5_A),                       # blank pmcid
        ])
        candidates, malformed = dl.read_candidates(self.inventory)
        self.assertEqual([c["pmcid"] for c in candidates], ["PMC1"])
        self.assertEqual(len(malformed), 3)
        self.assertIn("blank pmcid", [m["_reason"] for m in malformed])

    def test_missing_columns_are_refused(self) -> None:
        self.inventory.write_text("pmcid,pmid\nPMC1,2\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            dl.read_candidates(self.inventory)
        self.assertIn("missing columns", str(caught.exception))

    def test_lfs_pointer_is_refused(self) -> None:
        self.inventory.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n",
            encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            dl.read_candidates(self.inventory)
        self.assertIn("git lfs pull", str(caught.exception))

    def test_field_size_limit_is_windows_safe(self) -> None:
        self.assertLessEqual(dl.widen_csv_field_limit(), 2**31 - 1)
        self.assertNotIn("field_size_limit(sys.maxsize)",
                         Path(dl.__file__).read_text(encoding="utf-8"))


class UrlConversionTests(unittest.TestCase):
    def test_s3_url_becomes_https_and_md5(self) -> None:
        url, md5 = dl.split_xml_url(
            "s3://pmc-oa-opendata/PMC9277667.1/PMC9277667.1.xml?md5=" + "0" * 32)
        self.assertEqual(
            url, "https://pmc-oa-opendata.s3.amazonaws.com/PMC9277667.1/PMC9277667.1.xml")
        self.assertEqual(md5, "0" * 32)

    def test_multi_digit_versions_are_handled(self) -> None:
        url, _ = dl.split_xml_url(
            "s3://pmc-oa-opendata/PMC11167622.319/PMC11167622.319.xml?md5=" + "f" * 32)
        self.assertTrue(url.endswith("PMC11167622.319/PMC11167622.319.xml"))

    def test_bad_urls_are_rejected(self) -> None:
        for bad in ["", "https://example.com/x.xml", "s3://other-bucket/PMC1.1/PMC1.1.xml?md5=" + "0"*32,
                    "s3://pmc-oa-opendata/PMC1.1/PMC1.1.xml", "s3://pmc-oa-opendata/PMC1.1/PMC1.1.pdf?md5=" + "0"*32]:
            with self.assertRaises(ValueError, msg=bad):
                dl.split_xml_url(bad)


class SafetyTests(DownloaderTestCase):
    def test_refuses_to_write_under_pubmed(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            dl.assert_safe_output_dir(dl.REPO_ROOT / "pubmed" / "x", self.inventory)
        self.assertIn("never write under pubmed", str(caught.exception))

    def test_refuses_to_write_beside_the_inventory(self) -> None:
        with self.assertRaises(SystemExit):
            dl.assert_safe_output_dir(self.inventory.parent, self.inventory)

    def test_source_never_references_pubmed_data_files(self) -> None:
        source = Path(dl.__file__).read_text(encoding="utf-8")
        for name in ("pubmed_results.csv", "pubmed_results.json", "search_log.csv",
                     "search_queries.txt"):
            self.assertNotIn(name, source,
                             f"downloader must never reference {name}")

    def test_a_real_run_writes_nothing_outside_the_output_dir(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        self.serve({url: XML_A})
        before = {p: p.stat().st_mtime_ns for p in self.tmp.rglob("*") if p.is_file()}

        dl.run(self.args())

        for path, mtime in before.items():
            self.assertTrue(path.exists(), f"{path} was deleted")
            self.assertEqual(path.stat().st_mtime_ns, mtime, f"{path} was modified")
        written = {p for p in self.tmp.rglob("*") if p.is_file()} - set(before)
        self.assertTrue(all(self.out in p.parents for p in written),
                        f"wrote outside the output directory: {written}")

    def test_only_xml_urls_are_ever_requested(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        fake = self.serve({url: XML_A})
        dl.run(self.args())
        self.assertEqual(fake.requested, [url])
        self.assertTrue(all(u.endswith(".xml") for u in fake.requested),
                        "no PDF, text or media URL may ever be requested")

    def test_inventory_file_is_not_modified(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        before = self.inventory.read_bytes()
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        self.serve({url: XML_A})
        dl.run(self.args())
        self.assertEqual(self.inventory.read_bytes(), before)


class DownloadTests(DownloaderTestCase):
    def test_successful_download_is_verified_and_recorded(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A, retracted="no")])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        self.serve({url: XML_A})

        self.assertEqual(dl.run(self.args()), 0)

        saved = self.out / "xml" / "PMC1.xml"
        self.assertTrue(saved.exists())
        self.assertEqual(saved.read_bytes(), XML_A)
        row = self.manifest_rows()[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["expected_md5"], MD5_A)
        self.assertEqual(row["actual_md5"], MD5_A)
        self.assertEqual(row["md5_verified"], "yes")
        self.assertEqual(row["bytes"], str(len(XML_A)))
        self.assertEqual(row["filename"], "PMC1.xml")

    def test_metadata_is_copied_from_the_inventory_unchanged(self) -> None:
        write_inventory(self.inventory, [
            inventory_row("PMC1", MD5_A, licence="CC BY-NC-ND", retracted="yes",
                          manuscript="yes", pmid="98765", version=3)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A, version=3))
        self.serve({url: XML_A})
        dl.run(self.args())
        row = self.manifest_rows()[0]
        self.assertEqual(row["license_code"], "CC BY-NC-ND")
        self.assertEqual(row["is_retracted"], "yes", "retraction status must survive")
        self.assertEqual(row["is_manuscript"], "yes")
        self.assertEqual(row["pmid"], "98765")
        self.assertEqual(row["version"], "3")
        self.assertEqual(row["doi"], "10.1000/PMC1")
        self.assertEqual(row["source_xml_url"], s3_url("PMC1", MD5_A, version=3))
        self.assertEqual(row["resolved_url"], url)

    def test_retracted_records_are_downloaded_not_silently_dropped(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A, retracted="yes")])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        self.serve({url: XML_A})
        dl.run(self.args())
        rows = self.manifest_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["is_retracted"], "yes")

    def test_no_part_file_survives_a_success(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        self.serve({url: XML_A})
        dl.run(self.args())
        self.assertEqual(list((self.out / "xml").glob("*.part")), [])

    def test_limit_restricts_the_run(self) -> None:
        rows = [inventory_row(f"PMC{i}", MD5_A) for i in range(1, 6)]
        write_inventory(self.inventory, rows)
        bodies = {dl.split_xml_url(r["xml_url"])[0]: XML_A for r in rows}
        fake = self.serve(bodies)
        dl.run(self.args(limit=2))
        self.assertEqual(len(fake.requested), 2)
        self.assertEqual(len(self.manifest_rows()), 2)

    def test_list_only_makes_no_requests(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        fake = self.serve({})
        self.assertEqual(dl.run(self.args(list_only=True)), 0)
        self.assertEqual(fake.requested, [], "--list-only must not touch the network")
        self.assertFalse((self.out / "xml").exists())


class IntegrityAndResumeTests(DownloaderTestCase):
    def test_hash_mismatch_is_a_failure_and_no_file_is_kept(self) -> None:
        # Inventory promises MD5_A but the server returns different bytes.
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        self.serve({url: b"corrupted payload"})

        self.assertEqual(dl.run(self.args()), 1)
        self.assertFalse((self.out / "xml" / "PMC1.xml").exists(),
                         "a file failing verification must never be kept")
        self.assertEqual(list((self.out / "xml").glob("*.part")), [])
        row = self.manifest_rows()[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("md5 mismatch", row["error"])
        self.assertEqual(row["md5_verified"], "no")
        self.assertEqual(self.failure_rows()[0]["pmcid"], "PMC1")

    def test_existing_verified_file_is_skipped_without_a_request(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        target = self.out / "xml" / "PMC1.xml"
        target.parent.mkdir(parents=True)
        target.write_bytes(XML_A)
        fake = self.serve({})                       # any request would 404

        self.assertEqual(dl.run(self.args()), 0)
        self.assertEqual(fake.requested, [], "a verified file must not be re-downloaded")
        self.assertEqual(self.manifest_rows()[0]["status"], "verified_existing")

    def test_existing_corrupt_file_is_detected_and_replaced(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        target = self.out / "xml" / "PMC1.xml"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"truncated half a file")
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        fake = self.serve({url: XML_A})

        dl.run(self.args())
        self.assertEqual(fake.requested, [url], "a bad file must trigger a re-download")
        self.assertEqual(target.read_bytes(), XML_A)
        self.assertEqual(self.manifest_rows()[0]["status"], "ok")

    def test_a_second_run_resumes_and_does_not_refetch(self) -> None:
        rows = [inventory_row(f"PMC{i}", MD5_A) for i in (1, 2)]
        write_inventory(self.inventory, rows)
        bodies = {dl.split_xml_url(r["xml_url"])[0]: XML_A for r in rows}

        fake1 = self.serve(bodies)
        dl.run(self.args())
        self.assertEqual(len(fake1.requested), 2)

        fake2 = self.serve(bodies)
        dl.run(self.args())
        self.assertEqual(fake2.requested, [], "second run must download nothing")
        self.assertEqual(len(self.manifest_rows()), 2, "manifest stays one row per article")

    def test_failures_are_retried_on_the_next_run(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))

        self.serve({})                              # first run: 404
        self.assertEqual(dl.run(self.args()), 1)
        self.assertEqual(self.manifest_rows()[0]["status"], "failed_permanent")

        fake = self.serve({url: XML_A})             # second run: now available
        self.assertEqual(dl.run(self.args()), 0)
        self.assertEqual(fake.requested, [url], "a failed article must be retried")
        rows = self.manifest_rows()
        self.assertEqual(len(rows), 1, "consolidation leaves one row per article")
        self.assertEqual(rows[0]["status"], "ok")

    def test_manifest_history_survives_an_interrupted_run(self) -> None:
        # Simulate an append-only manifest with two states for one article.
        self.out.mkdir(parents=True)
        handle, writer = dl.open_appending(self.out / "manifest.csv", dl.MANIFEST_FIELDS)
        writer.writerow({"pmcid": "PMC1", "status": "failed", "attempts": "3"})
        writer.writerow({"pmcid": "PMC1", "status": "ok", "attempts": "4"})
        handle.close()
        state = dl.load_manifest(self.out / "manifest.csv")
        self.assertEqual(state["PMC1"]["status"], "ok", "later rows must win")
        dl.consolidate_manifest(self.out / "manifest.csv", dl.MANIFEST_FIELDS)
        self.assertEqual(len(self.manifest_rows()), 1)
        self.assertEqual(self.manifest_rows()[0]["status"], "ok")


class RetryTests(DownloaderTestCase):
    def test_transient_error_is_retried_then_succeeds(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        boom = urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
        fake = self.serve({url: XML_A}, failures={url: [boom, None]})

        self.assertEqual(dl.run(self.args()), 0)
        self.assertEqual(len(fake.requested), 2, "one failure then one success")
        self.assertEqual(self.manifest_rows()[0]["status"], "ok")

    def test_transient_error_gives_up_after_max_attempts(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        boom = [urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None) for _ in range(5)]
        fake = self.serve({url: XML_A}, failures={url: boom})

        self.assertEqual(dl.run(self.args(max_attempts=3)), 1)
        self.assertEqual(len(fake.requested), 3, "bounded by --max-attempts")
        row = self.manifest_rows()[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["http_status"], "429")

    def test_permanent_error_is_not_retried(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        fake = self.serve({})                        # every request 404s

        self.assertEqual(dl.run(self.args(max_attempts=4)), 1)
        self.assertEqual(len(fake.requested), 1, "a 404 must not be retried")
        row = self.manifest_rows()[0]
        self.assertEqual(row["status"], "failed_permanent")
        self.assertEqual(row["http_status"], "404")

    def test_network_error_is_treated_as_transient(self) -> None:
        write_inventory(self.inventory, [inventory_row("PMC1", MD5_A)])
        url, _ = dl.split_xml_url(s3_url("PMC1", MD5_A))
        fake = self.serve({url: XML_A}, failures={url: [urllib.error.URLError("reset"), None]})
        self.assertEqual(dl.run(self.args()), 0)
        self.assertEqual(len(fake.requested), 2)

    def test_one_failure_does_not_stop_the_run(self) -> None:
        rows = [inventory_row("PMC1", MD5_A), inventory_row("PMC2", MD5_B)]
        write_inventory(self.inventory, rows)
        url_a, _ = dl.split_xml_url(rows[0]["xml_url"])
        url_b, _ = dl.split_xml_url(rows[1]["xml_url"])
        self.serve({url_b: XML_B})                  # PMC1 will 404

        dl.run(self.args())
        by_id = {r["pmcid"]: r for r in self.manifest_rows()}
        self.assertEqual(by_id["PMC1"]["status"], "failed_permanent")
        self.assertEqual(by_id["PMC2"]["status"], "ok")
        self.assertTrue((self.out / "xml" / "PMC2.xml").exists())


class InterruptionTests(DownloaderTestCase):
    def test_ctrl_c_stops_cleanly_and_keeps_progress(self) -> None:
        rows = [inventory_row(f"PMC{i}", MD5_A) for i in (1, 2, 3)]
        write_inventory(self.inventory, rows)
        bodies = {dl.split_xml_url(r["xml_url"])[0]: XML_A for r in rows}
        fake = FakeHTTP(bodies)

        calls = {"n": 0}
        def interrupting(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            return fake(request, timeout)
        dl.urllib.request.urlopen = interrupting

        code = dl.run(self.args())
        self.assertEqual(code, 130, "Ctrl+C exits with 130")
        rows_out = self.manifest_rows()
        self.assertEqual(len(rows_out), 2, "work completed before the interrupt is kept")
        self.assertTrue(all(r["status"] == "ok" for r in rows_out))
        self.assertEqual(list((self.out / "xml").glob("*.part")), [],
                         "no partial file may be left behind")


if __name__ == "__main__":
    unittest.main(verbosity=2)
