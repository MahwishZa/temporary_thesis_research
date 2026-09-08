#!/usr/bin/env python3
"""Tests for build_corpus_metadata.py (M1-M4).

Run from the pmc/ directory:
    cd pmc && python3 -m unittest test_build_corpus_metadata

These test the pure derivation logic (dates, eligibility, tiers, splits) with
synthetic inputs. No network, no corpus files. The Cochrane fetch itself is not
exercised here (it is a one-off network ingestion); its provenance recording is
tested through build_currency_pack with an injected parsed record.
"""

from __future__ import annotations

import unittest

import build_corpus_metadata as b


class Precision(unittest.TestCase):
    def test_day_month_year_blank(self):
        self.assertEqual(b.precision_of("2024-06-15"), "day")
        self.assertEqual(b.precision_of("2024-06"), "month")
        self.assertEqual(b.precision_of("2024"), "year")
        self.assertEqual(b.precision_of(""), "")
        self.assertEqual(b.precision_of("garbage"), "")


class CanonicalSelection(unittest.TestCase):
    def test_epub_wins_over_ppub_and_collection(self):
        d, p, t = b.canonical_from_dates_all(
            {"epub": "2024-03-01", "ppub": "2024-08", "collection": "2024"})
        self.assertEqual((d, p, t), ("2024-03-01", "day", "epub"))

    def test_preprint_is_excluded(self):
        # A preprint date years earlier must never become canonical.
        d, p, t = b.canonical_from_dates_all(
            {"preprint": "2019-01-01", "pub": "2023-05-10"})
        self.assertEqual((d, t), ("2023-05-10", "pub"))

    def test_falls_through_to_collection(self):
        d, p, t = b.canonical_from_dates_all({"collection": "2022"})
        self.assertEqual((d, p, t), ("2022", "year", "collection"))

    def test_empty(self):
        self.assertEqual(b.canonical_from_dates_all({}), ("", "", ""))

    def test_unknown_type_used_only_as_last_resort(self):
        d, p, t = b.canonical_from_dates_all({"weird": "2020-01-01"})
        self.assertEqual((d, t), ("2020-01-01", "weird"))


class SplitBoundary(unittest.TestCase):
    def test_boundary_is_june_2024(self):
        self.assertEqual(b.split_side("2024-05"), "pre")
        self.assertEqual(b.split_side("2024-06"), "post")
        self.assertEqual(b.split_side("2024-07"), "post")
        self.assertEqual(b.split_side("2021-01"), "pre")


class MonthRecovery(unittest.TestCase):
    def test_recovers_only_real_2024_month(self):
        self.assertEqual(b.recover_month(["2024", "2024-09-01"]), "2024-09")
        self.assertEqual(b.recover_month(["2023-12", "2024-03"]), "2024-03")

    def test_no_recovery_when_only_year(self):
        self.assertEqual(b.recover_month(["2024", "2024"]), "")

    def test_does_not_recover_other_years(self):
        self.assertEqual(b.recover_month(["2025-01", "2023-06"]), "")


class M3Rows(unittest.TestCase):
    def test_jats_primary_when_article_present(self):
        art = {"publication_dates_all": {"epub": "2023-02-10", "ppub": "2023"}}
        pm = {"publication_date": "2023-05"}
        row = b.m3_row("PMC1", "1", art, pm)
        self.assertEqual(row["canonical_date"], "2023-02-10")
        self.assertEqual(row["date_source"], "jats:epub")
        self.assertEqual(row["split_june_2024"], "pre")

    def test_pubmed_fallback_when_no_article(self):
        row = b.m3_row("PMC2", "2", None, {"publication_date": "2025-01-01"})
        self.assertEqual(row["canonical_date"], "2025-01-01")
        self.assertEqual(row["date_source"], "pubmed")
        self.assertEqual(row["split_june_2024"], "post")

    def test_year_only_2023_is_pre_2025_is_post(self):
        self.assertEqual(b.m3_row("P", "1", None, {"publication_date": "2023"})["split_june_2024"], "pre")
        self.assertEqual(b.m3_row("P", "1", None, {"publication_date": "2025"})["split_june_2024"], "post")

    def test_year_only_2024_is_unknown_without_recovery(self):
        row = b.m3_row("P", "1", None, {"publication_date": "2024"})
        self.assertEqual(row["split_june_2024"], "unknown")
        self.assertEqual(row["canonical_date_precision"], "year")  # precision NOT invented
        self.assertEqual(row["recovered_month"], "")

    def test_year_only_2024_recovered_from_jats(self):
        art = {"publication_dates_all": {"collection": "2024", "epub": "2024-09-15"}}
        # canonical is epub here, so it is already month/day precise -> exercise
        # recovery via a case where canonical is the year-only collection:
        art2 = {"publication_dates_all": {"collection": "2024"}}
        pm = {"publication_date": "2024-09"}
        row = b.m3_row("P", "1", art2, pm)
        self.assertEqual(row["canonical_date"], "2024")           # precision preserved
        self.assertEqual(row["canonical_date_precision"], "year")
        self.assertEqual(row["recovered_month"], "2024-09")       # recovered for the split only
        self.assertEqual(row["recovered_month_source"], "pubmed")
        self.assertEqual(row["split_june_2024"], "post")

    def test_no_date_at_all_is_unknown(self):
        row = b.m3_row("P", "1", None, {})
        self.assertEqual(row["canonical_date"], "")
        self.assertEqual(row["split_june_2024"], "unknown")


class LicenseBands(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(b.license_band("CC BY"), "open")
        self.assertEqual(b.license_band("CC0"), "open")
        self.assertEqual(b.license_band("CC BY-NC"), "cc-restrictive")
        self.assertEqual(b.license_band("CC BY-NC-ND"), "cc-restrictive")
        self.assertEqual(b.license_band("TDM"), "tdm")
        self.assertEqual(b.license_band(""), "none")


def mf(**kw):
    base = {"pmcid": "PMC9", "pmid": "9", "doi": "10.1/x", "license_code": "CC BY",
            "is_retracted": "no", "is_manuscript": "no", "status": "ok"}
    base.update(kw)
    return base


def pm(types="Journal Article", **kw):
    base = {"publication_types": types, "publication_date": "2023-01-01", "doi": "10.1/x"}
    base.update(kw)
    return base


class M4Eligibility(unittest.TestCase):
    def test_plain_research_is_eligible_fulltext(self):
        r = b.m4_row("PMC9", "9", mf(), pm(), in_fulltext=True, is_currency=False,
                     stub=False, no_body=False, doi_disagree=False, license_disagree=False)
        self.assertEqual(r["source_category"], "pmc-fulltext")
        self.assertEqual(r["eligibility_status"], "eligible")
        self.assertEqual(r["fulltext_eligible"], "yes")

    def test_retracted_stays_eligible_but_flagged(self):
        r = b.m4_row("PMC9", "9", mf(is_retracted="yes"), pm(), in_fulltext=True,
                     is_currency=False, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertEqual(r["eligibility_status"], "eligible")   # gamma rejects at admission, not here
        self.assertIn("retracted", r["flags"])

    def test_erratum_is_excluded(self):
        r = b.m4_row("PMC9", "9", mf(), pm(types="Published Erratum"), in_fulltext=True,
                     is_currency=False, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertEqual(r["eligibility_status"], "excluded")
        self.assertIn("erratum", r["flags"])

    def test_retraction_notice_is_excluded(self):
        r = b.m4_row("PMC9", "9", mf(), pm(types="Retraction of Publication"),
                     in_fulltext=True, is_currency=False, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertEqual(r["eligibility_status"], "excluded")
        self.assertIn("retraction-notice", r["flags"])

    def test_no_license_is_manual_review(self):
        r = b.m4_row("PMC9", "9", mf(license_code=""), pm(), in_fulltext=True,
                     is_currency=False, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertEqual(r["eligibility_status"], "manual-review")
        self.assertIn("no-license", r["flags"])

    def test_doi_disagreement_is_manual_review(self):
        r = b.m4_row("PMC9", "9", mf(), pm(), in_fulltext=True, is_currency=False,
                     stub=False, no_body=False, doi_disagree=True, license_disagree=False)
        self.assertEqual(r["eligibility_status"], "manual-review")
        self.assertIn("doi-disagreement", r["flags"])

    def test_preprint_eligible_but_flagged(self):
        r = b.m4_row("PMC9", "9", mf(license_code="CC BY"), pm(types="Preprint"),
                     in_fulltext=True, is_currency=False, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertEqual(r["eligibility_status"], "eligible")
        self.assertIn("preprint", r["flags"])

    def test_editorial_flagged_opinion(self):
        r = b.m4_row("PMC9", "9", mf(), pm(types="Editorial"), in_fulltext=True,
                     is_currency=False, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertIn("opinion", r["flags"])

    def test_stub_is_not_fulltext_eligible(self):
        r = b.m4_row("PMC9", "9", mf(), pm(), in_fulltext=True, is_currency=False,
                     stub=True, no_body=False, doi_disagree=False, license_disagree=False)
        self.assertEqual(r["fulltext_eligible"], "no")
        self.assertEqual(r["eligibility_status"], "eligible")  # abstract-level still eligible
        self.assertIn("stub", r["flags"])

    def test_no_body_is_not_fulltext_eligible(self):
        r = b.m4_row("PMC9", "9", mf(), pm(), in_fulltext=True, is_currency=False,
                     stub=False, no_body=True, doi_disagree=False, license_disagree=False)
        self.assertEqual(r["fulltext_eligible"], "no")
        self.assertIn("no-body", r["flags"])

    def test_pubmed_abstract_layer_category(self):
        r = b.m4_row("", "9", None, pm(), in_fulltext=False, is_currency=False,
                     stub=False, no_body=False, doi_disagree=False, license_disagree=False)
        self.assertEqual(r["source_category"], "pubmed-abstract")
        self.assertEqual(r["fulltext_eligible"], "no")

    def test_cochrane_currency_is_externally_added(self):
        r = b.m4_row(b.COCHRANE_PMCID, "41985900", mf(pmcid=b.COCHRANE_PMCID,
                     license_code="CC BY-NC", status="failed"),
                     pm(types="Systematic Review;Meta-Analysis"),
                     in_fulltext=False, is_currency=True, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertEqual(r["source_category"], "currency-pack")
        self.assertEqual(r["eligibility_status"], "externally-added")
        self.assertIn("currency-pack", r["flags"])
        # ingested full text, though not in the downloaded OA corpus:
        self.assertEqual(r["fulltext_eligible"], "yes")

    def test_in_corpus_currency_anchor_is_eligible(self):
        r = b.m4_row("PMC11350039", "38934362", mf(pmcid="PMC11350039",
                     license_code="CC BY-NC-ND"), pm(), in_fulltext=True,
                     is_currency=True, stub=False, no_body=False,
                     doi_disagree=False, license_disagree=False)
        self.assertEqual(r["eligibility_status"], "eligible")
        self.assertIn("currency-pack", r["flags"])


class CurrencyPack(unittest.TestCase):
    def test_anchor_set_is_small_and_contains_cochrane(self):
        ids = [a["pmcid"] for a in b.CURRENCY_ANCHORS]
        self.assertEqual(len(ids), len(set(ids)), "no duplicate anchors")
        self.assertIn(b.COCHRANE_PMCID, ids)
        self.assertLessEqual(len(ids), 12, "currency pack stays small, not a literature search")
        coch = [a for a in b.CURRENCY_ANCHORS if a["pmcid"] == b.COCHRANE_PMCID][0]
        self.assertEqual(coch["acquisition"], "manual-oa-fetch")

    def test_covers_the_required_currency_axes(self):
        classes = " ".join(a["claim_class"] for a in b.CURRENCY_ANCHORS)
        for axis in ("criteria", "eligibility", "aria", "plasma-biomarkers",
                     "amyloid-imaging", "contested"):
            self.assertIn(axis, classes, f"currency pack must cover {axis}")

    def test_build_rows_marks_in_corpus_vs_manual(self):
        manifest = {a["pmcid"]: mf(pmcid=a["pmcid"], pmid=a["pmid"], status="ok",
                                   title=a["pmcid"])
                    for a in b.CURRENCY_ANCHORS}
        manifest[b.COCHRANE_PMCID] = mf(pmcid=b.COCHRANE_PMCID, pmid="41985900",
                                        license_code="CC BY-NC", status="failed",
                                        title="Cochrane CD016297")
        pubmed = {a["pmid"]: pm(publication_date="2025-01") for a in b.CURRENCY_ANCHORS}
        cochrane = {"pmcid": b.COCHRANE_PMCID, "staged": True,
                    "observed_md5": "abc123",
                    "record": {"publication_dates_all": {"epub": "2026-04-16"}}}
        rows = b.build_currency_pack(manifest, pubmed, {}, cochrane)
        self.assertEqual(len(rows), len(b.CURRENCY_ANCHORS))
        by = {r["pmcid"]: r for r in rows}
        self.assertEqual(by["PMC11350039"]["in_raw_corpus"], "yes")
        self.assertEqual(by[b.COCHRANE_PMCID]["in_raw_corpus"], "no")
        self.assertEqual(by[b.COCHRANE_PMCID]["full_text_available"], "yes")
        self.assertEqual(by[b.COCHRANE_PMCID]["observed_md5"], "abc123")
        self.assertEqual(by[b.COCHRANE_PMCID]["canonical_date"], "2026-04-16")

    def test_dual_role_documents_are_flagged_not_duplicated(self):
        # A currency anchor that is also CPG is flagged, never emitted twice.
        manifest = {a["pmcid"]: mf(pmcid=a["pmcid"], pmid=a["pmid"], status="ok")
                    for a in b.CURRENCY_ANCHORS}
        pubmed = {a["pmid"]: pm() for a in b.CURRENCY_ANCHORS}
        rows = b.build_currency_pack(manifest, pubmed, {}, {"staged": False})
        dual = [r for r in rows if r["in_cpg_layer"] == "yes"]
        self.assertTrue(dual, "some anchors also serve the CPG layer")
        self.assertEqual(len({r["pmcid"] for r in rows}), len(rows), "no duplicates")


class CpgLayer(unittest.TestCase):
    def test_curated_layer_is_small_and_verified(self):
        ids = [c["pmcid"] for c in b.CURATED_CPG]
        self.assertEqual(len(ids), len(set(ids)), "no duplicate CPG entries")
        # Defensible curated layer, not a ~10^3 bulk corpus.
        self.assertGreaterEqual(len(ids), 15)
        self.assertLessEqual(len(ids), 60)

    def test_cochrane_is_never_classified_as_cpg(self):
        self.assertNotIn(b.COCHRANE_PMCID, {c["pmcid"] for c in b.CURATED_CPG})

    def test_authority_tier_labels_are_labels_not_ordering(self):
        tiers = {c["tier"] for c in b.CURATED_CPG}
        self.assertTrue(tiers <= {b.TIER_CPG, b.TIER_AUC, b.TIER_CONSENSUS, b.TIER_CRITERIA})
        # No numeric rank/weight is attached anywhere -- authority is a tested variable.
        for c in b.CURATED_CPG:
            self.assertNotIn("rank", c)
            self.assertNotIn("weight", c)

    def test_claim_class_coverage(self):
        blob = " ".join(c["claim_classes"] for c in b.CURATED_CPG)
        for axis in ("criteria", "plasma", "amyloid-imaging", "aria",
                     "eligibility", "differential-diagnosis", "pharmacotherapy"):
            self.assertIn(axis, blob, f"CPG layer must cover {axis}")

    def test_build_layer_joins_verified_metadata(self):
        c0 = b.CURATED_CPG[0]
        manifest = {c0["pmcid"]: mf(pmcid=c0["pmcid"], pmid="1", status="ok",
                                    license_code="CC BY-NC-ND",
                                    title="Revised criteria")}
        manifest[c0["pmcid"]]["actual_md5"] = "deadbeef"
        manifest[c0["pmcid"]]["downloaded_at_utc"] = "2026-09-01T00:00:00+00:00"
        pubmed = {"1": pm(publication_date="2024-08")}
        rows = b.build_cpg_layer(manifest, pubmed, {})
        row = [r for r in rows if r.get("pmcid") == c0["pmcid"]][0]
        self.assertEqual(row["source_category"], "cpg")
        self.assertEqual(row["issue_date"], "2024-08")
        self.assertEqual(row["date_precision"], "month")   # precision preserved
        self.assertEqual(row["license_code"], "CC BY-NC-ND")
        self.assertEqual(row["content_hash_md5"], "deadbeef")
        self.assertEqual(row["authority_tier_label"], c0["tier"])

    def test_missing_manifest_record_is_reported_not_invented(self):
        rows = b.build_cpg_layer({}, {}, {})
        self.assertTrue(all(r.get("status") == "MISSING-from-manifest" for r in rows))

    def test_external_targets_assert_no_dates_or_licences(self):
        for t in b.build_cpg_external_targets():
            self.assertEqual(t["status"], "manual-acquisition-required")
            self.assertNotIn("issue_date", t)
            self.assertNotIn("license_code", t)


if __name__ == "__main__":
    unittest.main()
