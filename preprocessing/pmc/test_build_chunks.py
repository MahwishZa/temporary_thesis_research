#!/usr/bin/env python3
"""Tests for build_chunks.py -- the retrieval-ready chunk layer.

Run from the pmc/ directory:
    cd pmc && python3 -m unittest test_build_chunks

Offline: synthetic inputs only, no corpus files and no network.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_chunks as bc


# --------------------------------------------------------------------------- windowing
class Windowing(unittest.TestCase):
    def test_short_text_is_one_window(self):
        out = bc.window_words("a b c", 256, 32)
        self.assertEqual(out, [(0, "a b c")])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(bc.window_words("   ", 256, 32), [])

    def test_windows_overlap_by_the_configured_amount(self):
        text = " ".join(str(i) for i in range(300))
        out = bc.window_words(text, 100, 20)
        self.assertEqual(out[0][0], 0)
        self.assertEqual(out[1][0], 80)          # step = 100 - 20
        first_tail = out[0][1].split()[-20:]
        second_head = out[1][1].split()[:20]
        self.assertEqual(first_tail, second_head)

    def test_window_never_exceeds_size(self):
        text = " ".join(str(i) for i in range(1000))
        for _, piece in bc.window_words(text, 256, 32):
            self.assertLessEqual(len(piece.split()), 256)

    def test_full_coverage_of_source_words(self):
        text = " ".join(str(i) for i in range(500))
        covered = set()
        for start, piece in bc.window_words(text, 100, 20):
            covered.update(range(start, start + len(piece.split())))
        self.assertEqual(covered, set(range(500)))

    def test_tiny_trailing_window_is_dropped_not_padded(self):
        # trailing remainder below MIN_CHUNK_WORDS is already inside the overlap
        text = " ".join(str(i) for i in range(105))
        out = bc.window_words(text, 100, 20)
        self.assertTrue(all(len(p.split()) >= bc.MIN_CHUNK_WORDS for _, p in out[1:]))

    def test_overlap_must_be_smaller_than_window(self):
        with self.assertRaises(ValueError):
            bc.window_words(" ".join("x" * 300), 10, 10)

    def test_windowing_is_deterministic(self):
        text = " ".join(str(i) for i in range(700))
        self.assertEqual(bc.window_words(text, 256, 32), bc.window_words(text, 256, 32))


class EmbedComposition(unittest.TestCase):
    def test_title_and_heading_are_prepended_not_stored_in_text(self):
        s = bc.compose_embed_text("Title", "Methods", "body words")
        self.assertEqual(s, "Title | Methods | body words")

    def test_missing_parts_are_skipped(self):
        self.assertEqual(bc.compose_embed_text("", "", "only text"), "only text")


# --------------------------------------------------------------------------- fixtures
def parsed_doc(pmcid="PMC1", body_words=600, abstract="An abstract here."):
    words = " ".join(f"w{i}" for i in range(body_words))
    return {
        "pmcid": pmcid, "pmcid_versioned": f"{pmcid}.1", "pmid": "111",
        "doi": "10.1/x", "title": "A Title", "journal": "J",
        "abstract": {"text": abstract, "is_structured": False, "sections": []},
        "sections": [{
            "section_id": f"{pmcid}.1#s1", "path": [1], "depth": 1,
            "title_raw": "Introduction", "imrad": "introduction",
            "paragraphs": [{"paragraph_id": f"{pmcid}.1#s1.p1", "text": words}],
            "subsections": [],
        }],
        "provenance": {"xml_md5": "abc123"},
    }


def meta(**kw):
    base = dict(document_id="PMC1.1", pmcid="PMC1", pmcid_versioned="PMC1.1", pmid="111",
                doi="10.1/x", title="A Title", journal="J",
                source_category="pmc-fulltext", eligibility_status="eligible",
                fulltext_eligible="yes", document_type="Journal Article",
                canonical_date="2024-08", date_precision="month", date_source="jats:epub",
                split_june_2024="post", authority_tier_label="", guideline_family="",
                organization="", in_currency_pack="no", claim_class="",
                license_code="CC BY", license_band="open", retracted="no", flags="",
                source_xml_md5="abc123")
    base.update(kw)
    return base


# --------------------------------------------------------------------------- chunking
class DocumentChunking(unittest.TestCase):
    def test_abstract_and_body_both_chunked(self):
        recs = bc.chunks_for_document(meta(), parsed_doc(), "An abstract here.", 256, 32)
        locs = {r["location"] for r in recs}
        self.assertEqual(locs, {"abstract", "body"})

    def test_body_skipped_when_not_fulltext_eligible(self):
        recs = bc.chunks_for_document(meta(fulltext_eligible="no"), parsed_doc(),
                                      "An abstract here.", 256, 32)
        self.assertEqual({r["location"] for r in recs}, {"abstract"})

    def test_no_text_is_ever_fabricated(self):
        recs = bc.chunks_for_document(meta(fulltext_eligible="no"), None, "", 256, 32)
        self.assertEqual(recs, [])

    def test_chunk_ids_unique_and_traceable(self):
        recs = bc.chunks_for_document(meta(), parsed_doc(), "An abstract here.", 100, 20)
        ids = [r["chunk_id"] for r in recs]
        self.assertEqual(len(ids), len(set(ids)))
        for r in recs:
            self.assertTrue(r["chunk_id"].startswith("PMC1.1#"))
            self.assertEqual(r["document_id"], "PMC1.1")

    def test_chunks_never_mix_documents(self):
        a = bc.chunks_for_document(meta(), parsed_doc("PMC1"), "abs a", 100, 20)
        b = bc.chunks_for_document(meta(document_id="PMC2.1", pmcid="PMC2"),
                                   parsed_doc("PMC2"), "abs b", 100, 20)
        self.assertTrue(all(r["document_id"] == "PMC1.1" for r in a))
        self.assertTrue(all(r["document_id"] == "PMC2.1" for r in b))
        self.assertFalse({r["chunk_id"] for r in a} & {r["chunk_id"] for r in b})

    def test_body_chunks_carry_section_provenance(self):
        recs = [r for r in bc.chunks_for_document(meta(), parsed_doc(), "a", 100, 20)
                if r["location"] == "body"]
        for r in recs:
            self.assertEqual(r["section_id"], "s1")
            self.assertEqual(r["section_heading"], "Introduction")
            self.assertEqual(r["imrad"], "introduction")
            self.assertEqual(r["paragraph_id_first"], "PMC1.1#s1.p1")

    def test_thesis_critical_metadata_on_every_chunk(self):
        recs = bc.chunks_for_document(meta(), parsed_doc(), "abs", 100, 20)
        required = ["canonical_date", "date_precision", "date_source", "split_june_2024",
                    "source_category", "eligibility_status", "license_code",
                    "authority_tier_label", "in_currency_pack", "retracted",
                    "pmid", "pmcid", "doi", "title", "source_xml_md5"]
        for r in recs:
            for f in required:
                self.assertIn(f, r, f"{f} must survive onto every chunk")

    def test_windows_do_not_cross_section_boundaries(self):
        doc = parsed_doc()
        doc["sections"].append({
            "section_id": "PMC1.1#s2", "path": [2], "depth": 1, "title_raw": "Methods",
            "imrad": "methods",
            "paragraphs": [{"paragraph_id": "PMC1.1#s2.p1", "text": " ".join(f"m{i}" for i in range(50))}],
            "subsections": [],
        })
        recs = bc.chunks_for_document(meta(), doc, "abs", 100, 20)
        for r in recs:
            if r["section_id"] == "s2":
                self.assertNotIn("w0 ", r["text"])   # no leakage from section 1

    def test_subsections_are_walked(self):
        doc = parsed_doc()
        doc["sections"][0]["subsections"] = [{
            "section_id": "PMC1.1#s1.1", "path": [1, 1], "depth": 2, "title_raw": "Sub",
            "imrad": "introduction",
            "paragraphs": [{"paragraph_id": "PMC1.1#s1.1.p1", "text": "sub text here"}],
            "subsections": [],
        }]
        recs = bc.chunks_for_document(meta(), doc, "", 100, 20)
        self.assertIn("s1.1", {r["section_id"] for r in recs})


# --------------------------------------------------------------------------- end to end
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "meta").mkdir()
        arts = self.tmp / "articles.jsonl"
        with arts.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(parsed_doc("PMC1")) + "\n")
        self.articles = arts
        # frozen overlays
        (self.tmp / "meta" / "corpus_policy.csv").write_text(
            "pmcid,pmid,source_category,eligibility_status,fulltext_eligible,retracted,"
            "is_manuscript,license_code,license_band,flags,eligibility_reason\n"
            "PMC1,111,pmc-fulltext,eligible,yes,no,no,CC BY,open,,ok\n"
            "PMC2,222,pmc-fulltext,excluded,yes,no,no,CC BY,open,erratum,notice\n"
            ",333,pubmed-abstract,eligible,no,no,no,,none,,ok\n", encoding="utf-8")
        (self.tmp / "meta" / "canonical_dates.csv").write_text(
            "pmcid,pmid,canonical_date,canonical_date_precision,canonical_date_type,"
            "date_source,recovered_month,recovered_month_source,split_june_2024\n"
            "PMC1,111,2024-08,month,epub,jats:epub,,,post\n"
            ",333,2022,year,,pubmed,,,pre\n", encoding="utf-8")
        (self.tmp / "meta" / "cpg_registry.csv").write_text("pmcid\n", encoding="utf-8")
        (self.tmp / "meta" / "currency_pack.csv").write_text("pmcid\n", encoding="utf-8")
        (self.tmp / "pubmed.csv").write_text(
            "pmid,title,abstract,publication_date,journal,doi,pmcid,publication_types\n"
            "111,A Title,An abstract here.,2024-08,J,10.1/x,PMC1,Journal Article\n"
            "333,Abstract Only,Just an abstract.,2022,J2,10.2/y,,Journal Article\n",
            encoding="utf-8")

    def run_build(self, out=None):
        return bc.build(self.articles, self.tmp / "missing.json", self.tmp / "pubmed.csv",
                        self.tmp / "meta", out or (self.tmp / "out"), 100, 20)

    def load(self, out=None):
        p = (out or (self.tmp / "out")) / "chunks.jsonl"
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]

    def test_excluded_records_are_not_chunked(self):
        self.run_build()
        self.assertNotIn("PMC2", {c["pmcid"] for c in self.load()})

    def test_abstract_only_layer_is_chunked(self):
        self.run_build()
        rows = [c for c in self.load() if c["source_category"] == "pubmed-abstract"]
        self.assertTrue(rows)
        self.assertTrue(all(c["location"] == "abstract" for c in rows))
        self.assertEqual(rows[0]["document_id"], "PMID333")

    def test_no_orphan_chunks_and_valid_doc_mapping(self):
        self.run_build()
        chunks = self.load()
        for c in chunks:
            self.assertTrue(c["document_id"])
            self.assertTrue(c["chunk_id"].startswith(c["document_id"] + "#"))

    def test_chunk_ids_globally_unique(self):
        self.run_build()
        ids = [c["chunk_id"] for c in self.load()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_deterministic_across_runs(self):
        a = self.run_build(self.tmp / "o1")
        b = self.run_build(self.tmp / "o2")
        self.assertEqual(a, b)
        self.assertEqual((self.tmp / "o1" / "chunks.jsonl").read_bytes(),
                         (self.tmp / "o2" / "chunks.jsonl").read_bytes())

    def test_canonical_date_and_category_preserved(self):
        self.run_build()
        for c in self.load():
            if c["pmcid"] == "PMC1":
                self.assertEqual(c["canonical_date"], "2024-08")
                self.assertEqual(c["date_precision"], "month")
                self.assertEqual(c["split_june_2024"], "post")
                self.assertEqual(c["source_category"], "pmc-fulltext")

    def test_year_only_precision_is_not_upgraded(self):
        self.run_build()
        rows = [c for c in self.load() if c["pmid"] == "333"]
        self.assertTrue(rows)
        self.assertEqual(rows[0]["canonical_date"], "2022")
        self.assertEqual(rows[0]["date_precision"], "year")

    def test_exact_duplicates_flagged_not_deleted(self):
        stats = self.run_build()
        chunks = self.load()
        self.assertEqual(stats["chunks"], len(chunks))   # nothing dropped
        for c in chunks:
            if c["duplicate_of"]:
                self.assertNotEqual(c["duplicate_of"], c["chunk_id"])

    def test_stats_file_written(self):
        self.run_build()
        s = json.loads((self.tmp / "out" / "chunk_stats.json").read_text(encoding="utf-8"))
        self.assertIn("chunks", s)
        self.assertEqual(s["window_words"], 100)

    def test_lfs_pointer_is_rejected_not_silently_empty(self):
        p = self.tmp / "pointer.csv"
        p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 1\n",
                     encoding="utf-8")
        with self.assertRaises(SystemExit):
            bc.load_pubmed(p)


if __name__ == "__main__":
    unittest.main()
