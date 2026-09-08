#!/usr/bin/env python3
"""Offline tests for the PubMed acquisition pipeline.

These exercise everything that does not need network access: query-file parsing,
PubMed XML parsing, date normalisation, deduplication with query provenance,
backup-before-overwrite, CSV/JSON generation, and the verification checks.

Run with:  python3 -m unittest discover -s pubmed -v
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import fetch_pubmed as fp


# A PubmedArticleSet covering the shapes the pipeline has to survive:
#   32000001 complete journal article (structured abstract, MeSH, DOI, PMCID)
#   32000002 no abstract, no DOI, no PMCID, month-name date, collective author
#   32000003 MedlineDate only ("2022 Nov-Dec"), ELocationID DOI, no MeSH
#   32000004 book chapter (PubmedBookArticle)
SAMPLE_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE">
      <PMID Version="1">32000001</PMID>
      <Article PubModel="Print">
        <Journal>
          <Title>Journal of Clinical Reasoning</Title>
          <ISOAbbreviation>J Clin Reason</ISOAbbreviation>
          <JournalIssue>
            <PubDate><Year>2023</Year><Month>Mar</Month><Day>15</Day></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Diagnostic reasoning in <i>early</i> Alzheimer disease.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Clinicians vary widely.</AbstractText>
          <AbstractText Label="RESULTS">Accuracy was 78%.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>Jane A</ForeName></Author>
          <Author><LastName>Lee</LastName><Initials>KB</Initials></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName MajorTopicYN="N">Alzheimer Disease</DescriptorName>
          <QualifierName MajorTopicYN="Y">diagnosis</QualifierName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName MajorTopicYN="Y">Clinical Decision-Making</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">32000001</ArticleId>
        <ArticleId IdType="doi">10.1000/jcr.2023.001</ArticleId>
        <ArticleId IdType="pmc">PMC9000001</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation Status="PubMed-not-MEDLINE">
      <PMID Version="1">32000002</PMID>
      <Article PubModel="Electronic">
        <Journal>
          <Title>Dementia Diagnostics</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year><Month>07</Month></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>A registry note without an abstract.</ArticleTitle>
        <AuthorList>
          <Author><CollectiveName>The ADRD Study Group</CollectiveName></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Letter</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">32000002</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation Status="MEDLINE">
      <PMID Version="2">32000003</PMID>
      <Article PubModel="Print-Electronic">
        <Journal>
          <Title>Neurology of Aging</Title>
          <JournalIssue>
            <PubDate><MedlineDate>2022 Nov-Dec</MedlineDate></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Differential diagnosis of atypical presentations.</ArticleTitle>
        <Abstract><AbstractText>Unlabelled abstract body.</AbstractText></Abstract>
        <ELocationID EIdType="doi" ValidYN="Y">10.1000/noa.2022.777</ELocationID>
        <AuthorList>
          <Author><LastName>Okafor</LastName><ForeName>N</ForeName></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">32000003</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>

  <PubmedBookArticle>
    <BookDocument>
      <PMID Version="1">32000004</PMID>
      <ArticleIdList>
        <ArticleId IdType="pubmed">32000004</ArticleId>
        <ArticleId IdType="doi">10.1000/book.2021.42</ArticleId>
      </ArticleIdList>
      <Book>
        <PubDate><Year>2021</Year><Month>Sep</Month><Day>02</Day></PubDate>
        <BookTitle book="statpearls">StatPearls</BookTitle>
      </Book>
      <ArticleTitle>Alzheimer Disease Evaluation</ArticleTitle>
      <AuthorList>
        <Author><LastName>Rivera</LastName><ForeName>Ana</ForeName></Author>
      </AuthorList>
      <PublicationTypeList>
        <PublicationType>Study Guide</PublicationType>
      </PublicationTypeList>
      <Abstract><AbstractText>Chapter level abstract.</AbstractText></Abstract>
    </BookDocument>
  </PubmedBookArticle>
</PubmedArticleSet>
"""


class QueryFileTests(unittest.TestCase):
    """The real search_queries.txt is the fixture; it must never be modified."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.start, cls.end, cls.queries = fp.load_queries_file(fp.QUERIES_PATH)
        cls.by_id = {q["id"]: q for q in cls.queries}

    def test_reads_the_approved_window(self) -> None:
        self.assertEqual(self.start, date(2021, 8, 30))
        self.assertEqual(self.end, date(2026, 8, 30))

    def test_every_approved_query_is_present_and_searchable(self) -> None:
        for query_id in fp.APPROVED_QUERY_IDS:
            self.assertIn(query_id, self.by_id, f"{query_id} missing from the file")
            self.assertTrue(self.by_id[query_id]["base_term"], f"{query_id} is empty")

    def test_multi_line_blocks_join_into_one_term(self) -> None:
        term = self.by_id["Q0.2"]["base_term"]
        self.assertIn('"Alzheimer Disease"[Mesh]', term)
        self.assertIn('"clinical reasoning"[tiab]', term)
        self.assertIn(") AND (", term)
        self.assertNotIn("\n", term)

    def test_date_filter_is_stripped_then_reapplied_unchanged(self) -> None:
        # Q0.1 ships with the window inline; Q4.1 does not. Both must end up
        # scoped to exactly the approved window, with no dangling AND.
        for query_id in ("Q0.1", "Q4.1"):
            base = self.by_id[query_id]["base_term"]
            self.assertNotIn("Date - Publication", base)
            self.assertFalse(base.upper().endswith("AND"), query_id)
            scoped = fp.scoped_term(base, self.start, self.end)
            self.assertEqual(scoped.count("[Date - Publication]"), 2)
            self.assertIn('"2021/08/30"[Date - Publication]', scoped)
            self.assertIn('"2026/08/30"[Date - Publication]', scoped)

    def test_bare_building_blocks_are_excluded_from_the_approved_set(self) -> None:
        for query_id in ("Q1.1", "Q2.1", "Q3.1", "Q3.5", "Q12.3"):
            self.assertNotIn(query_id, fp.APPROVED_QUERY_IDS)

    def test_select_queries_rejects_unknown_ids(self) -> None:
        with self.assertRaises(ValueError):
            fp.select_queries(self.queries, ["Q99.9"])

    def test_conflicting_windows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.txt"
            path.write_text(
                '# Q1.1 a\n("2021/01/01"[Date - Publication] : "2026/01/01"[Date - Publication])\n'
                '# Q1.2 b\n("2020/01/01"[Date - Publication] : "2026/01/01"[Date - Publication])\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                fp.load_queries_file(path)


class XmlParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = fp.parse_pubmed_xml(SAMPLE_XML)
        cls.by_pmid = {r["pmid"]: r for r in cls.records}

    def test_all_four_record_shapes_parse(self) -> None:
        self.assertEqual(
            sorted(self.by_pmid), ["32000001", "32000002", "32000003", "32000004"]
        )

    def test_full_journal_article(self) -> None:
        record = self.by_pmid["32000001"]
        self.assertEqual(record["title"], "Diagnostic reasoning in early Alzheimer disease.")
        self.assertEqual(record["abstract"],
                         "BACKGROUND: Clinicians vary widely.\nRESULTS: Accuracy was 78%.")
        self.assertEqual(record["publication_date"], "2023-03-15")
        self.assertEqual(record["publication_year"], "2023")
        self.assertEqual(record["journal"], "Journal of Clinical Reasoning")
        self.assertEqual(record["authors"], ["Smith, Jane A", "Lee, KB"])
        self.assertEqual(record["doi"], "10.1000/jcr.2023.001")
        self.assertEqual(record["pmcid"], "PMC9000001")
        self.assertEqual(record["mesh_terms"],
                         ["Alzheimer Disease (diagnosis*)", "Clinical Decision-Making*"])
        self.assertEqual(record["publication_types"], ["Journal Article", "Review"])
        self.assertEqual(record["record_status"], "ok")

    def test_record_without_abstract_doi_or_pmcid_is_kept(self) -> None:
        record = self.by_pmid["32000002"]
        self.assertEqual(record["abstract"], "")
        self.assertEqual(record["doi"], "")
        self.assertEqual(record["pmcid"], "")
        self.assertEqual(record["publication_date"], "2024-07")
        self.assertEqual(record["authors"], ["The ADRD Study Group"])
        self.assertEqual(record["mesh_terms"], [])

    def test_medline_date_and_elocation_doi(self) -> None:
        record = self.by_pmid["32000003"]
        self.assertEqual(record["publication_date"], "2022-11")
        self.assertEqual(record["doi"], "10.1000/noa.2022.777")

    def test_book_chapter(self) -> None:
        record = self.by_pmid["32000004"]
        self.assertEqual(record["title"], "Alzheimer Disease Evaluation")
        self.assertEqual(record["journal"], "StatPearls")
        self.assertEqual(record["publication_date"], "2021-09-02")
        self.assertEqual(record["doi"], "10.1000/book.2021.42")
        self.assertEqual(record["publication_types"], ["Study Guide"])

    def test_pmcid_without_prefix_is_normalised(self) -> None:
        xml = SAMPLE_XML.replace(b'IdType="pmc">PMC9000001', b'IdType="pmc">9000001')
        record = {r["pmid"]: r for r in fp.parse_pubmed_xml(xml)}["32000001"]
        self.assertEqual(record["pmcid"], "PMC9000001")

    def test_records_without_a_pmid_are_skipped(self) -> None:
        xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
        <Article><ArticleTitle>No identifier</ArticleTitle></Article>
        </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
        self.assertEqual(fp.parse_pubmed_xml(xml), [])


class DateHelperTests(unittest.TestCase):
    def test_medline_date_shapes(self) -> None:
        cases = {
            "2021 Nov-Dec": "2021-11",
            "2022 Winter": "2022",
            "2021-2022": "2021",
            "2023 Jan 15": "2023-01-15",
            "": "",
            "no year here": "",
        }
        for raw, expected in cases.items():
            self.assertEqual(fp.parse_medline_date(raw), expected, raw)

    def test_compose_date_rejects_bad_parts(self) -> None:
        self.assertEqual(fp.compose_date("2023", "13", "05"), "2023")
        self.assertEqual(fp.compose_date("2023", "Mar", "99"), "2023-03")
        self.assertEqual(fp.compose_date("", "Mar", "01"), "")

    def test_date_bounds_widen_for_partial_dates(self) -> None:
        self.assertEqual(fp.date_bounds("2023"), (date(2023, 1, 1), date(2023, 12, 31)))
        self.assertEqual(fp.date_bounds("2023-02"), (date(2023, 2, 1), date(2023, 2, 28)))
        self.assertEqual(fp.date_bounds("2023-12"), (date(2023, 12, 1), date(2023, 12, 31)))
        self.assertEqual(fp.date_bounds("2023-03-15"), (date(2023, 3, 15), date(2023, 3, 15)))
        self.assertIsNone(fp.date_bounds("not-a-date"))


class MergeTests(unittest.TestCase):
    def test_provenance_accumulates_across_queries(self) -> None:
        store: dict[str, dict] = {}
        record = fp.parse_pubmed_xml(SAMPLE_XML)[0]
        self.assertTrue(fp.merge_record(store, dict(record), "Q4.1"))
        self.assertFalse(fp.merge_record(store, dict(record), "Q0.1"))
        self.assertFalse(fp.merge_record(store, dict(record), "Q4.1"))
        self.assertEqual(store[record["pmid"]]["query_ids"], ["Q4.1", "Q0.1"])

    def test_a_later_richer_record_fills_gaps_but_never_overwrites(self) -> None:
        store: dict[str, dict] = {}
        fp.merge_record(store, fp.empty_record("32000001"), "Q4.1")
        self.assertEqual(store["32000001"]["record_status"], "metadata_unavailable")

        full = {r["pmid"]: r for r in fp.parse_pubmed_xml(SAMPLE_XML)}["32000001"]
        fp.merge_record(store, dict(full), "Q0.1")
        merged = store["32000001"]
        self.assertEqual(merged["record_status"], "ok")
        self.assertEqual(merged["title"], full["title"])
        self.assertEqual(merged["query_ids"], ["Q4.1", "Q0.1"])

        fp.merge_record(store, {**full, "title": "Different title"}, "Q5.1")
        self.assertEqual(store["32000001"]["title"], full["title"])

    def test_query_ids_sort_numerically_not_lexically(self) -> None:
        ids = ["Q13.1", "Q0.2", "Q4.10", "Q4.2"]
        self.assertEqual(
            sorted(ids, key=fp.query_sort_key), ["Q0.2", "Q4.2", "Q4.10", "Q13.1"]
        )


class OutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.csv_path = self.tmp / "pubmed_results.csv"
        self.json_path = self.tmp / "pubmed_results.json"
        self.records = []
        for index, record in enumerate(fp.parse_pubmed_xml(SAMPLE_XML)):
            record = dict(record)
            record["query_ids"] = ["Q0.1", "Q4.1"] if index % 2 == 0 else ["Q13.1"]
            self.records.append(record)

    def write_both(self) -> None:
        fp.write_csv(self.csv_path, self.records)
        fp.write_json(self.json_path, self.records)

    def test_csv_and_json_round_trip_identically(self) -> None:
        self.write_both()
        with self.csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))

        self.assertEqual(list(rows[0]), fp.CSV_FIELDS)
        self.assertEqual(len(rows), len(payload), "CSV and JSON row counts differ")
        for row, item in zip(rows, payload):
            self.assertEqual(row["pmid"], item["pmid"])
            self.assertEqual(row["authors"], "; ".join(item["authors"]))
            self.assertEqual(row["query_ids"], "; ".join(item["query_ids"]))
            self.assertEqual(row["abstract"], item["abstract"])

    def test_embedded_newlines_and_commas_survive_the_csv(self) -> None:
        self.write_both()
        with self.csv_path.open(encoding="utf-8", newline="") as handle:
            rows = {row["pmid"]: row for row in csv.DictReader(handle)}
        self.assertIn("\n", rows["32000001"]["abstract"])
        self.assertEqual(rows["32000001"]["authors"], "Smith, Jane A; Lee, KB")

    def test_json_is_a_valid_array_when_empty(self) -> None:
        fp.write_json(self.json_path, [])
        self.assertEqual(json.loads(self.json_path.read_text(encoding="utf-8")), [])

    def test_verification_passes_on_a_clean_run(self) -> None:
        self.write_both()
        per_query = {
            "Q0.1": ["32000001", "32000003"],
            "Q4.1": ["32000001", "32000003"],
            "Q13.1": ["32000002", "32000004"],
        }
        problems, stats = fp.verify_outputs(
            self.csv_path, self.json_path, self.records, per_query,
            (date(2021, 8, 30), date(2026, 8, 30)),
        )
        self.assertEqual(problems, [])
        self.assertEqual(stats["unique_pmids"], 4)
        self.assertEqual(stats["out_of_range"], [])

    def test_verification_catches_a_duplicate_pmid(self) -> None:
        self.records.append(dict(self.records[0]))
        self.write_both()
        problems, _stats = fp.verify_outputs(
            self.csv_path, self.json_path, self.records, {},
            (date(2021, 8, 30), date(2026, 8, 30)),
        )
        self.assertTrue(any("duplicate" in p for p in problems), problems)

    def test_verification_catches_csv_json_divergence(self) -> None:
        self.write_both()
        fp.write_json(self.json_path, self.records[:-1])
        problems, _stats = fp.verify_outputs(
            self.csv_path, self.json_path, self.records, {},
            (date(2021, 8, 30), date(2026, 8, 30)),
        )
        self.assertTrue(any("CSV and JSON differ" in p for p in problems), problems)

    def test_verification_catches_lost_and_unattributed_records(self) -> None:
        self.write_both()
        per_query = {"Q0.1": ["32000001", "99999999"]}
        problems, _stats = fp.verify_outputs(
            self.csv_path, self.json_path, self.records, per_query,
            (date(2021, 8, 30), date(2026, 8, 30)),
        )
        self.assertTrue(any("missing from the output" in p for p in problems), problems)
        self.assertTrue(any("provenance mismatch" in p for p in problems), problems)

    def test_verification_flags_dates_outside_the_window(self) -> None:
        self.records[0]["publication_date"] = "2019-05-01"
        self.write_both()
        _problems, stats = fp.verify_outputs(
            self.csv_path, self.json_path, self.records, {},
            (date(2021, 8, 30), date(2026, 8, 30)),
        )
        self.assertEqual(stats["out_of_range"], ["32000001"])


class BackupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_empty_outputs_are_not_backed_up(self) -> None:
        # Exactly the state of the repository files before the first real run.
        csv_path = self.tmp / "pubmed_results.csv"
        json_path = self.tmp / "pubmed_results.json"
        csv_path.write_text("pmid,title,authors,journal,year,doi,abstract,url\n",
                            encoding="utf-8")
        json_path.write_text("[]", encoding="utf-8")
        self.assertFalse(fp.has_data(csv_path))
        self.assertFalse(fp.has_data(json_path))
        self.assertEqual(fp.backup_existing([csv_path, json_path], "STAMP"), [])

    def test_populated_outputs_are_copied_before_being_replaced(self) -> None:
        csv_path = self.tmp / "pubmed_results.csv"
        json_path = self.tmp / "pubmed_results.json"
        csv_path.write_text("pmid,title\n123,Existing work\n", encoding="utf-8")
        json_path.write_text('[{"pmid": "123"}]', encoding="utf-8")

        made = fp.backup_existing([csv_path, json_path], "20260830T120000Z")
        self.assertEqual(
            sorted(p.name for p in made),
            ["pubmed_results_backup_20260830T120000Z.csv",
             "pubmed_results_backup_20260830T120000Z.json"],
        )
        self.assertIn("Existing work", made[0].read_text(encoding="utf-8"))

        fp.write_csv(csv_path, [])
        self.assertIn("Existing work", made[0].read_text(encoding="utf-8"))

    def test_missing_files_are_ignored(self) -> None:
        self.assertFalse(fp.has_data(self.tmp / "nope.csv"))
        self.assertEqual(fp.backup_existing([self.tmp / "nope.csv"], "STAMP"), [])

    def test_unreadable_but_non_empty_file_is_preserved(self) -> None:
        broken = self.tmp / "pubmed_results.json"
        broken.write_text("{not json", encoding="utf-8")
        self.assertTrue(fp.has_data(broken))


class NetworkFreeCliTests(unittest.TestCase):
    def test_list_queries_resolves_every_approved_query(self) -> None:
        self.assertEqual(fp.main(["--list-queries"]), 0)

    def test_approved_set_has_no_duplicates(self) -> None:
        self.assertEqual(len(fp.APPROVED_QUERY_IDS), len(set(fp.APPROVED_QUERY_IDS)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
