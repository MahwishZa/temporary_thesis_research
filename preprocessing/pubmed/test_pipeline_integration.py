#!/usr/bin/env python3
"""End-to-end tests of the acquisition pipeline against a fake E-utilities server.

The unit tests in test_fetch_pubmed.py cover parsing and output. These cover the
orchestration that only shows up under real traffic: paging a search past
ESearch's UID ceiling by splitting the publication-date window, batching and
bisecting EFetch, carrying query provenance through deduplication, surviving a
query that fails outright, and retrying transient HTTP errors.

Run with:  python3 -m unittest discover -s pubmed -v
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
import urllib.error
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import fetch_pubmed as fp

WINDOW = (date(2021, 8, 30), date(2026, 8, 30))


def article_xml(pmid: str, published: date) -> str:
    return f"""
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">{pmid}</PMID>
      <Article>
        <Journal>
          <Title>Fake Journal</Title>
          <JournalIssue><PubDate>
            <Year>{published.year}</Year><Month>{published.month:02d}</Month>
            <Day>{published.day:02d}</Day>
          </PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Record {pmid}</ArticleTitle>
        <Abstract><AbstractText>Abstract for {pmid}.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Author</LastName>
          <ForeName>{pmid}</ForeName></Author></AuthorList>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList>
      <ArticleId IdType="pubmed">{pmid}</ArticleId>
      <ArticleId IdType="doi">10.1000/fake.{pmid}</ArticleId>
    </ArticleIdList></PubmedData>
  </PubmedArticle>"""


class FakeEutils:
    """Minimal stand-in for ESearch/EFetch over an in-memory corpus."""

    def __init__(
        self,
        corpus: dict[str, date],
        query_hits: dict[str, list[str]],
        *,
        efetch_poison: set[str] | None = None,
        esearch_fails: set[str] | None = None,
    ) -> None:
        self.corpus = corpus
        self.query_hits = query_hits
        self.efetch_poison = efetch_poison or set()
        self.esearch_fails = esearch_fails or set()
        self.esearch_calls: list[tuple[str, int]] = []
        self.efetch_calls: list[list[str]] = []

    def marker(self, term: str) -> str:
        for name in self.query_hits:
            if name in term:
                return name
        raise AssertionError(f"fake server saw an unrecognised term: {term!r}")

    def __call__(self, url: str, params: dict) -> bytes:
        if url == fp.ESEARCH_URL:
            return self.esearch(params)
        if url == fp.EFETCH_URL:
            return self.efetch(params)
        raise AssertionError(f"unexpected URL {url}")

    def esearch(self, params: dict) -> bytes:
        term = params["term"]
        name = self.marker(term)
        if name in self.esearch_fails:
            raise fp.EutilsError("HTTP 400 from NCBI: b'bad request'")

        match = fp.DATE_FILTER_RE.search(term)
        assert match, f"every search must carry the date window: {term!r}"
        start = fp.parse_date_ymd(match.group(1))
        end = fp.parse_date_ymd(match.group(2))

        hits = [pmid for pmid in self.query_hits[name]
                if start <= self.corpus[pmid] <= end]
        retmax = int(params["retmax"])
        self.esearch_calls.append((f"{start}:{end}", retmax))
        return json.dumps(
            {
                "esearchresult": {
                    "count": str(len(hits)),
                    "retmax": str(retmax),
                    "retstart": "0",
                    "idlist": hits[:retmax],
                }
            }
        ).encode()

    def efetch(self, params: dict) -> bytes:
        pmids = params["id"].split(",")
        self.efetch_calls.append(pmids)
        if self.efetch_poison & set(pmids):
            return b"<PubmedArticleSet><PubmedArticle>truncated"
        body = "".join(article_xml(p, self.corpus[p]) for p in pmids)
        return f"<?xml version='1.0'?><PubmedArticleSet>{body}</PubmedArticleSet>".encode()


def spread_dates(pmids: list[str], start: date, step_days: int) -> dict[str, date]:
    return {pmid: start + timedelta(days=index * step_days)
            for index, pmid in enumerate(pmids)}


def make_query(query_id: str, marker: str) -> dict[str, str]:
    return {"id": query_id, "label": f"fake {query_id}", "base_term": marker,
            "term_as_written": marker}


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self._real_request = fp.eutils_request
        self._real_cap = fp.ESEARCH_ID_CAP
        self._real_batch = fp.EFETCH_BATCH_SIZE
        self.addCleanup(self.restore)

    def restore(self) -> None:
        fp.eutils_request = self._real_request
        fp.ESEARCH_ID_CAP = self._real_cap
        fp.EFETCH_BATCH_SIZE = self._real_batch

    def install(self, server: FakeEutils) -> None:
        fp.eutils_request = server

    def run_pipeline(self, queries, pmid_limit=None):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = fp.run_queries(queries, WINDOW, fp.RecordCache(None), pmid_limit)
        return result

    # -- searching and pagination -------------------------------------------

    def test_search_below_the_cap_takes_a_single_pass(self) -> None:
        pmids = [str(40000000 + i) for i in range(12)]
        server = FakeEutils(spread_dates(pmids, date(2022, 1, 1), 30), {"ALPHA": pmids})
        self.install(server)

        warnings: list[str] = []
        count, found = fp.search_window(*("ALPHA", *WINDOW), warnings)

        self.assertEqual(count, 12)
        self.assertEqual(found, pmids)
        self.assertEqual(warnings, [])
        # One count probe plus one retrieval, no window splitting.
        self.assertEqual(len(server.esearch_calls), 2)

    def test_search_above_the_cap_splits_the_date_window(self) -> None:
        pmids = [str(41000000 + i) for i in range(25)]
        server = FakeEutils(spread_dates(pmids, date(2021, 9, 1), 60), {"ALPHA": pmids})
        self.install(server)
        fp.ESEARCH_ID_CAP = 8  # force several levels of splitting

        warnings: list[str] = []
        count, found = fp.search_window(*("ALPHA", *WINDOW), warnings)

        self.assertEqual(count, 25, "the reported total must be the unsplit count")
        self.assertEqual(sorted(found), sorted(pmids), "every PMID must survive splitting")
        self.assertEqual(len(found), len(set(found)), "splitting must not duplicate PMIDs")
        self.assertGreater(len(server.esearch_calls), 2, "expected the window to be split")
        self.assertEqual(warnings, [])

        # The windows actually retrieved from (retmax > 0) must not overlap, or a
        # record could be counted twice across sub-windows.
        leaves = sorted(label for label, retmax in server.esearch_calls if retmax)
        spans = [tuple(date.fromisoformat(p) for p in label.split(":")) for label in leaves]
        for (_, earlier_end), (later_start, _) in zip(spans, spans[1:]):
            self.assertLess(earlier_end, later_start, f"overlapping sub-windows: {leaves}")

    def test_split_windows_never_escape_the_approved_range(self) -> None:
        pmids = [str(42000000 + i) for i in range(20)]
        server = FakeEutils(spread_dates(pmids, date(2021, 9, 1), 80), {"ALPHA": pmids})
        self.install(server)
        fp.ESEARCH_ID_CAP = 4

        fp.search_window(*("ALPHA", *WINDOW), [])
        for label, _retmax in server.esearch_calls:
            start, end = (date.fromisoformat(part) for part in label.split(":"))
            self.assertGreaterEqual(start, WINDOW[0], label)
            self.assertLessEqual(end, WINDOW[1], label)

    def test_empty_result_set_is_not_an_error(self) -> None:
        self.install(FakeEutils({}, {"ALPHA": []}))
        warnings: list[str] = []
        self.assertEqual(fp.search_window(*("ALPHA", *WINDOW), warnings), (0, []))
        self.assertEqual(warnings, [])

    # -- fetching ------------------------------------------------------------

    def test_efetch_batches_and_keeps_every_record(self) -> None:
        pmids = [str(43000000 + i) for i in range(7)]
        server = FakeEutils(spread_dates(pmids, date(2022, 5, 1), 20), {"ALPHA": pmids})
        self.install(server)
        fp.EFETCH_BATCH_SIZE = 3

        errors: list[str] = []
        with contextlib.redirect_stdout(io.StringIO()):
            records = fp.fetch_records(pmids, fp.RecordCache(None), errors, "ALPHA")

        self.assertEqual([r["pmid"] for r in records], pmids)
        self.assertEqual([len(batch) for batch in server.efetch_calls], [3, 3, 1])
        self.assertEqual(errors, [])
        self.assertTrue(all(r["record_status"] == "ok" for r in records))

    def test_one_unparseable_record_does_not_lose_its_batch(self) -> None:
        pmids = [str(44000000 + i) for i in range(6)]
        poison = pmids[3]
        server = FakeEutils(
            spread_dates(pmids, date(2023, 1, 1), 10),
            {"ALPHA": pmids},
            efetch_poison={poison},
        )
        self.install(server)
        fp.EFETCH_BATCH_SIZE = 6

        errors: list[str] = []
        with contextlib.redirect_stdout(io.StringIO()):
            records = fp.fetch_records(pmids, fp.RecordCache(None), errors, "ALPHA")

        by_pmid = {r["pmid"]: r for r in records}
        self.assertEqual(sorted(by_pmid), sorted(pmids), "no PMID may be dropped")
        for pmid in pmids:
            expected = "metadata_unavailable" if pmid == poison else "ok"
            self.assertEqual(by_pmid[pmid]["record_status"], expected, pmid)
        self.assertEqual(len(errors), 1)
        self.assertIn(poison, errors[0])

    def test_the_cache_prevents_a_refetch(self) -> None:
        pmids = [str(45000000 + i) for i in range(4)]
        server = FakeEutils(spread_dates(pmids, date(2024, 1, 1), 5), {"ALPHA": pmids})
        self.install(server)
        cache = fp.RecordCache(self.tmp)

        with contextlib.redirect_stdout(io.StringIO()):
            fp.fetch_records(pmids, cache, [], "ALPHA")
            first_pass = len(server.efetch_calls)
            fp.fetch_records(pmids, fp.RecordCache(self.tmp), [], "ALPHA")

        self.assertEqual(len(server.efetch_calls), first_pass, "second pass refetched")
        self.assertEqual(len(fp.RecordCache(self.tmp).records), 4)

    # -- whole-run behaviour -------------------------------------------------

    def test_overlapping_queries_dedupe_and_keep_both_query_ids(self) -> None:
        alpha = [str(46000000 + i) for i in range(10)]
        beta = alpha[6:] + [str(46000100 + i) for i in range(4)]
        corpus = {**spread_dates(alpha, date(2022, 1, 1), 40),
                  **spread_dates(beta[4:], date(2025, 1, 1), 20)}
        self.install(FakeEutils(corpus, {"ALPHA": alpha, "BETA": beta}))

        records, log_rows, per_query, total_hits, errors = self.run_pipeline(
            [make_query("Q4.1", "ALPHA"), make_query("Q13.1", "BETA")]
        )

        self.assertEqual(errors, [])
        self.assertEqual(total_hits, len(alpha) + len(beta), "hits count duplicates")
        self.assertEqual(len(records), 14, "unique PMIDs after dedup")

        by_pmid = {r["pmid"]: r for r in records}
        for pmid in alpha[:6]:
            self.assertEqual(by_pmid[pmid]["query_ids"], ["Q4.1"])
        for pmid in alpha[6:]:
            self.assertEqual(by_pmid[pmid]["query_ids"], ["Q4.1", "Q13.1"],
                             "an overlapping record must keep both query IDs")
        self.assertEqual(per_query["Q4.1"], alpha)
        self.assertEqual(int(log_rows[1]["new_unique_pmids"]), 4)

    def test_a_failing_query_is_logged_and_the_run_continues(self) -> None:
        alpha = [str(47000000 + i) for i in range(5)]
        self.install(
            FakeEutils(spread_dates(alpha, date(2023, 3, 1), 15),
                       {"ALPHA": alpha, "BROKEN": []},
                       esearch_fails={"BROKEN"})
        )

        records, log_rows, per_query, _hits, errors = self.run_pipeline(
            [make_query("Q5.1", "BROKEN"), make_query("Q4.1", "ALPHA")]
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("Q5.1", errors[0])
        self.assertNotIn("Q5.1", per_query, "a failed query contributes no provenance")
        self.assertEqual(len(records), 5, "the healthy query still ran")
        self.assertIn("HTTP 400", log_rows[0]["errors_warnings"])
        self.assertEqual(log_rows[0]["retrieved_pmid_count"], "0")

    def test_limit_caps_retrieval_and_says_so(self) -> None:
        alpha = [str(48000000 + i) for i in range(10)]
        self.install(FakeEutils(spread_dates(alpha, date(2022, 6, 1), 30), {"ALPHA": alpha}))

        records, log_rows, _pq, _hits, _errors = self.run_pipeline(
            [make_query("Q0.2", "ALPHA")], pmid_limit=3
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(log_rows[0]["result_count"], "10", "the true hit count is logged")
        self.assertEqual(log_rows[0]["retrieved_pmid_count"], "3")
        self.assertIn("test mode", log_rows[0]["errors_warnings"])

    def test_full_run_writes_files_that_pass_verification(self) -> None:
        alpha = [str(49000000 + i) for i in range(9)]
        beta = alpha[5:] + [str(49000100 + i) for i in range(3)]
        corpus = {**spread_dates(alpha, date(2021, 10, 1), 100),
                  **spread_dates(beta[4:], date(2024, 2, 1), 30)}
        self.install(FakeEutils(corpus, {"ALPHA": alpha, "BETA": beta}))
        fp.ESEARCH_ID_CAP = 3  # exercise splitting inside the full run
        fp.EFETCH_BATCH_SIZE = 2

        records, log_rows, per_query, _hits, errors = self.run_pipeline(
            [make_query("Q4.1", "ALPHA"), make_query("Q13.1", "BETA")]
        )
        self.assertEqual(errors, [])

        csv_path = self.tmp / "pubmed_results.csv"
        json_path = self.tmp / "pubmed_results.json"
        fp.write_csv(csv_path, records)
        fp.write_json(json_path, records)
        fp.write_log(self.tmp / "search_log.csv", log_rows)

        problems, stats = fp.verify_outputs(
            csv_path, json_path, records, per_query, WINDOW
        )
        self.assertEqual(problems, [])
        self.assertEqual(stats["unique_pmids"], 12)
        self.assertEqual(stats["undated"], [])
        self.assertEqual(stats["out_of_range"], [])
        self.assertEqual([r["pmid"] for r in records],
                         sorted((r["pmid"] for r in records), key=int),
                         "records must be written in PMID order")


class RetryTests(unittest.TestCase):
    """eutils_request must retry the transient failures and surface the rest."""

    def setUp(self) -> None:
        self.slept: list[float] = []
        self._real_sleep = fp.time.sleep
        self._real_urlopen = fp.urllib.request.urlopen
        self._real_interval = fp.LIMITER.min_interval
        # Silence the rate limiter so `slept` records only retry backoff.
        fp.LIMITER.min_interval = 0.0
        fp.time.sleep = self.slept.append
        self.addCleanup(self.restore)

    def restore(self) -> None:
        fp.time.sleep = self._real_sleep
        fp.urllib.request.urlopen = self._real_urlopen
        fp.LIMITER.min_interval = self._real_interval

    def install_responses(self, responses) -> None:
        self.remaining = list(responses)

        class Response:
            def __init__(self, body): self.body = body
            def read(self): return self.body
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        def fake_urlopen(request, timeout=None):
            outcome = self.remaining.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return Response(outcome)

        fp.urllib.request.urlopen = fake_urlopen

    @staticmethod
    def http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            "https://example.invalid", code, "err", headers or {}, None
        )

    def test_a_rate_limit_response_is_retried(self) -> None:
        self.install_responses([self.http_error(429), b"ok"])
        self.assertEqual(fp.eutils_request(fp.ESEARCH_URL, {"term": "x"}), b"ok")
        self.assertEqual(len(self.slept), 1)

    def test_retry_after_header_is_honoured(self) -> None:
        self.install_responses([self.http_error(429, {"Retry-After": "42"}), b"ok"])
        fp.eutils_request(fp.ESEARCH_URL, {"term": "x"})
        self.assertEqual(self.slept, [42.0])

    def test_server_errors_are_retried_then_raised(self) -> None:
        self.install_responses([self.http_error(503)] * fp.MAX_RETRIES)
        with self.assertRaises(fp.EutilsError) as caught:
            fp.eutils_request(fp.ESEARCH_URL, {"term": "x"})
        self.assertIn("503", str(caught.exception))
        self.assertEqual(len(self.slept), fp.MAX_RETRIES - 1)

    def test_a_client_error_fails_immediately(self) -> None:
        self.install_responses([self.http_error(400), b"unreached"])
        with self.assertRaises(fp.EutilsError):
            fp.eutils_request(fp.ESEARCH_URL, {"term": "x"})
        self.assertEqual(self.slept, [], "a 400 must not be retried")

    def test_network_errors_are_retried(self) -> None:
        self.install_responses([urllib.error.URLError("reset"), b"ok"])
        self.assertEqual(fp.eutils_request(fp.ESEARCH_URL, {"term": "x"}), b"ok")
        self.assertEqual(len(self.slept), 1)

    def test_parameters_are_form_encoded_utf8(self) -> None:
        captured = {}

        class Response:
            def read(self): return b"ok"
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        def fake_urlopen(request, timeout=None):
            captured["body"] = request.data
            return Response()

        fp.urllib.request.urlopen = fake_urlopen
        fp.eutils_request(fp.ESEARCH_URL, {"term": '"Aβ42"[tiab]', "empty": ""})
        decoded = urllib.parse.parse_qs(captured["body"].decode())
        self.assertEqual(decoded["term"], ['"Aβ42"[tiab]'])
        self.assertNotIn("empty", decoded, "blank parameters must be dropped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
