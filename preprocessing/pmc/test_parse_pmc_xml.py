#!/usr/bin/env python3
"""Offline tests for parse_pmc_xml.py.

Every fixture is synthetic JATS built in-memory. No file in pmc/fulltext/xml/
is read, and nothing touches the network.

    python3 -m unittest pmc.test_parse_pmc_xml -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import parse_pmc_xml as pp


def article(front: str = "", body: str = "", back: str = "", floats: str = "",
            extra: str = "") -> str:
    return (
        '<article article-type="research-article" dtd-version="1.4">'
        f"<front><journal-meta><journal-title-group><journal-title>Test Journal"
        f"</journal-title></journal-title-group></journal-meta>"
        f"<article-meta>{front}</article-meta></front>"
        f"<body>{body}</body>"
        + (f"<back>{back}</back>" if back else "")
        + (f"<floats-group>{floats}</floats-group>" if floats else "")
        + extra + "</article>"
    )


IDS = (
    '<article-id pub-id-type="pmcid">PMC123</article-id>'
    '<article-id pub-id-type="pmcid-ver">PMC123.1</article-id>'
    '<article-id pub-id-type="pmid">999</article-id>'
    '<article-id pub-id-type="doi">10.1/xyz</article-id>'
)
TITLE = "<title-group><article-title>A Study</article-title></title-group>"
ABSTRACT = "<abstract><p>An abstract with enough words.</p></abstract>"


def long_body(words: int = 400) -> str:
    return f"<sec><title>Introduction</title><p>{' '.join(['word'] * words)}</p></sec>"


def parse(xml: str, row: dict[str, str] | None = None, threshold: int = 250) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "PMC123.xml"
        path.write_text(xml, encoding="utf-8")
        return pp.parse_article(path, row or {}, threshold)


class TextFlattening(unittest.TestCase):
    def test_bibliographic_xref_becomes_ref_placeholder(self):
        el = ET.fromstring('<p>Shown before<xref ref-type="bibr" rid="b1">12</xref> here.</p>')
        self.assertEqual(pp.flat_text(el), "Shown before[REF] here.")

    def test_non_bibliographic_xref_keeps_its_text(self):
        el = ET.fromstring('<p>See <xref ref-type="fig" rid="f1">Fig. 1</xref>.</p>')
        self.assertEqual(pp.flat_text(el), "See Fig. 1.")

    def test_figure_caption_is_never_spliced_into_prose(self):
        el = ET.fromstring(
            "<p>Before<fig><caption><p>CAPTION TEXT</p></caption></fig>after.</p>"
        )
        text = pp.flat_text(el)
        self.assertNotIn("CAPTION", text)
        self.assertEqual(text, "Beforeafter.")

    def test_inline_markup_is_unwrapped_and_counted(self):
        from collections import Counter
        stripped: Counter = Counter()
        el = ET.fromstring("<p>The <italic>APOE</italic> gene<sup>4</sup>.</p>")
        self.assertEqual(pp.flat_text(el, stripped), "The APOE gene4.")
        self.assertEqual(stripped["italic"], 1)
        self.assertEqual(stripped["sup"], 1)

    def test_tail_text_after_skipped_element_is_kept(self):
        el = ET.fromstring("<p>a<table-wrap><label>T1</label></table-wrap>b</p>")
        self.assertEqual(pp.flat_text(el), "ab")


class Sections(unittest.TestCase):
    def test_nested_sections_are_recursive_with_paths(self):
        body = (
            "<sec><title>Methods</title><p>Top level.</p>"
            "<sec><title>Participants</title><p>Nested one.</p>"
            "<sec><title>Screening</title><p>Nested two.</p></sec></sec></sec>"
        )
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        top = record["sections"][0]
        self.assertEqual(top["depth"], 1)
        self.assertEqual(top["subsections"][0]["depth"], 2)
        deepest = top["subsections"][0]["subsections"][0]
        self.assertEqual(deepest["depth"], 3)
        self.assertEqual(deepest["section_id"], "PMC123.1#s1.1.1")
        self.assertEqual(record["section_count"], 3)

    def test_paragraph_id_uses_versioned_pmcid_and_path_without_md5(self):
        body = "<sec><title>Methods</title><sec><title>Sub</title><p>a</p><p>b</p></sec></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        sub = record["sections"][0]["subsections"][0]
        self.assertEqual(sub["paragraphs"][1]["paragraph_id"], "PMC123.1#s1.1.p2")
        for paragraph in sub["paragraphs"]:
            self.assertNotIn("#s1.1.p1:", paragraph["paragraph_id"])
            self.assertEqual(paragraph["paragraph_id"].count("#"), 1)

    def test_classification_prefers_title_over_sec_type(self):
        body = '<sec sec-type="results"><title>Materials and Methods</title><p>x</p></sec>'
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        section = record["sections"][0]
        self.assertEqual(section["imrad"], "methods")
        self.assertEqual(section["imrad_source"], "title_match")
        self.assertEqual(section["sec_type_attr"], "results")

    def test_sec_type_used_when_title_is_unrecognised(self):
        body = '<sec sec-type="intro"><title>Setting the Scene</title><p>x</p></sec>'
        section = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)["sections"][0]
        self.assertEqual(section["imrad"], "introduction")
        self.assertEqual(section["imrad_source"], "sec_type")

    def test_supplementary_and_coi_sections_marked_not_content(self):
        body = (
            "<sec><title>Introduction</title><p>x</p></sec>"
            '<sec sec-type="supplementary-material"><title>Supplementary Material</title><p>y</p></sec>'
            "<sec><title>CONFLICT OF INTEREST STATEMENT</title><p>z</p></sec>"
        )
        sections = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)["sections"]
        self.assertTrue(sections[0]["is_content"])
        self.assertFalse(sections[1]["is_content"])
        self.assertFalse(sections[2]["is_content"])

    def test_section_title_path_is_denormalised_onto_paragraphs(self):
        body = "<sec><title>Methods</title><sec><title>Participants</title><p>x</p></sec></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        paragraph = record["sections"][0]["subsections"][0]["paragraphs"][0]
        self.assertEqual(paragraph["section_title_path"], ["Methods", "Participants"])

    def test_paragraphs_directly_on_body_are_captured_and_flagged(self):
        record = parse(article(IDS + TITLE + ABSTRACT, "<p>Loose paragraph.</p>"), threshold=0)
        self.assertIn("flat_sections", record["qc"]["flags"])
        self.assertEqual(record["paragraph_count"], 1)


class Authors(unittest.TestCase):
    def test_editors_are_excluded_and_flagged(self):
        front = IDS + TITLE + ABSTRACT + (
            '<contrib-group><contrib contrib-type="editor"><name>'
            "<surname>Muacevic</surname><given-names>Alexander</given-names></name></contrib>"
            "</contrib-group>"
            '<contrib-group><contrib contrib-type="author"><name>'
            "<surname>Charron</surname><given-names>Lily</given-names></name></contrib>"
            "</contrib-group>"
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["author_count"], 1)
        self.assertEqual(record["authors"][0]["surname"], "Charron")
        self.assertIn("editors_present", record["qc"]["flags"])

    def test_collab_authors_are_supported(self):
        front = IDS + TITLE + ABSTRACT + (
            '<contrib-group><contrib contrib-type="author">'
            "<collab>The ADNI Consortium</collab></contrib></contrib-group>"
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["authors"][0]["collab"], "The ADNI Consortium")
        self.assertEqual(record["authors"][0]["surname"], "")

    def test_affiliation_links_and_orcid_preserved(self):
        front = IDS + TITLE + ABSTRACT + (
            '<contrib-group><contrib contrib-type="author">'
            '<contrib-id contrib-id-type="orcid">https://orcid.org/0000-0003-2851-9763</contrib-id>'
            "<name><surname>Counts</surname><given-names>Scott</given-names></name>"
            '<xref ref-type="aff" rid="Aff1"/><xref ref-type="aff" rid="Aff2"/>'
            "</contrib></contrib-group>"
            '<aff id="Aff1"><label>1</label>Michigan State University</aff>'
            '<aff id="Aff2">Second Institute</aff>'
        )
        record = parse(article(front, long_body()))
        author = record["authors"][0]
        self.assertEqual(author["affiliation_ids"], ["Aff1", "Aff2"])
        self.assertEqual(author["orcid"], "0000-0003-2851-9763")
        self.assertEqual(record["affiliations"][0]["text"], "Michigan State University")


class Abstracts(unittest.TestCase):
    def test_scientific_abstract_chosen_over_plain_language_summary(self):
        front = IDS + TITLE + (
            "<abstract><sec><title>Background</title><p>Real background.</p></sec>"
            "<sec><title>Methods</title><p>Real methods.</p></sec></abstract>"
            '<abstract abstract-type="plain-language-summary"><p>Lay summary.</p></abstract>'
        )
        record = parse(article(front, long_body()))
        self.assertIn("Real background.", record["abstract"]["text"])
        self.assertNotIn("Lay summary", record["abstract"]["text"])
        self.assertTrue(record["abstract"]["is_structured"])
        self.assertEqual(record["abstract"]["sections"][0]["label"], "Background")
        self.assertEqual(record["abstract_other"][0]["type"], "plain-language-summary")
        self.assertIn("multiple_abstracts", record["qc"]["flags"])

    def test_unstructured_abstract(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()))
        self.assertFalse(record["abstract"]["is_structured"])
        self.assertEqual(record["abstract"]["sections"], [])

    def test_empty_preferred_abstract_falls_through_to_one_with_text(self):
        front = IDS + TITLE + (
            "<abstract></abstract>"
            '<abstract abstract-type="plain-language-summary"><p>Lay summary.</p></abstract>'
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["abstract"]["text"], "Lay summary.")
        self.assertEqual(record["qc"]["status"], "ok")
        self.assertEqual(record["abstract_other"][0]["type"], "untyped")
        self.assertEqual(record["abstract_other"][0]["text"], "")
        self.assertIn("multiple_abstracts", record["qc"]["flags"])

    def test_textless_abstracts_remain_no_abstract(self):
        front = IDS + TITLE + (
            "<abstract></abstract>"
            '<abstract abstract-type="graphical"><fig/></abstract>'
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["abstract"]["text"], "")
        self.assertEqual(record["qc"]["status"], "no_abstract")
        self.assertIn("multiple_abstracts", record["qc"]["flags"])


class Dates(unittest.TestCase):
    def test_epub_beats_ppub_and_precision_recorded(self):
        front = IDS + TITLE + ABSTRACT + (
            '<pub-date pub-type="ppub"><year>2025</year><month>1</month></pub-date>'
            '<pub-date pub-type="epub"><year>2024</year><month>11</month><day>7</day></pub-date>'
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["publication_date"], "2024-11-07")
        self.assertEqual(record["publication_date_type"], "epub")
        self.assertEqual(record["publication_date_precision"], "day")
        self.assertEqual(record["publication_dates_all"]["ppub"], "2025-01")

    def test_preprint_date_never_wins_over_pub(self):
        front = IDS + TITLE + ABSTRACT + (
            '<pub-date pub-type="preprint"><year>2018</year><month>12</month><day>20</day></pub-date>'
            '<pub-date pub-type="pub"><year>2022</year><month>5</month><day>3</day></pub-date>'
        )
        record = parse(article(front, long_body()))
        self.assertEqual(record["publication_date"], "2022-05-03")
        self.assertEqual(record["publication_date_type"], "pub")

    def test_year_only_date_flagged_partial(self):
        front = IDS + TITLE + ABSTRACT + '<pub-date pub-type="collection"><year>2022</year></pub-date>'
        record = parse(article(front, long_body()))
        self.assertEqual(record["publication_date"], "2022")
        self.assertEqual(record["publication_date_precision"], "year")
        self.assertIn("partial_date", record["qc"]["flags"])

    def test_named_month_is_understood(self):
        front = IDS + TITLE + ABSTRACT + (
            '<pub-date pub-type="epub"><year>2023</year><month>Mar</month><day>4</day></pub-date>'
        )
        self.assertEqual(parse(article(front, long_body()))["publication_date"], "2023-03-04")


class Licences(unittest.TestCase):
    def test_ali_license_ref_is_read(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' specific-use="textmining" content-type="ccbynclicense">'
            "https://creativecommons.org/licenses/by-nc/4.0/</ali:license_ref>"
            "<license-p>Open access under CC BY-NC.</license-p></license></permissions>"
        )
        prov = parse(article(front, long_body()), {"license_code": "CC BY-NC"})["provenance"]
        self.assertEqual(prov["license_code_xml"], "CC BY-NC")
        self.assertEqual(prov["license_content_type_xml"], "ccbynclicense")
        self.assertIn("creativecommons.org", prov["license_ref_xml"])

    def test_nc_nd_not_confused_with_nc(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbyncndlicense">https://creativecommons.org/licenses/by-nc-nd/4.0/'
            "</ali:license_ref></license></permissions>"
        )
        self.assertEqual(
            parse(article(front, long_body()))["provenance"]["license_code_xml"], "CC BY-NC-ND")

    def test_text_mining_prose_classified_as_tdm(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license><license-p>This file is available for text mining."
            "</license-p></license></permissions>"
        )
        self.assertEqual(
            parse(article(front, long_body()))["provenance"]["license_code_xml"], "TDM")

    def test_manifest_code_kept_separately_and_disagreement_flagged(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/'
            "</ali:license_ref></license></permissions>"
        )
        record = parse(article(front, long_body()), {"license_code": "CC BY-NC"})
        self.assertEqual(record["provenance"]["license_code_manifest"], "CC BY-NC")
        self.assertEqual(record["provenance"]["license_code_xml"], "CC BY")
        self.assertIn("license_disagreement", record["qc"]["flags"])

    def test_blank_manifest_licence_recovered_from_xml_is_flagged(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/'
            "</ali:license_ref></license></permissions>"
        )
        record = parse(article(front, long_body()), {"license_code": ""})
        self.assertIn("license_recovered_from_xml", record["qc"]["flags"])

    def test_later_license_with_ali_ref_is_preferred_over_unusable_first(self):
        front = IDS + TITLE + ABSTRACT + (
            "<permissions>"
            "<license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/">'
            "http://www.springer.com/gb/open-access/authors-rights/aam-terms-v1"
            "</ali:license_ref>"
            "<license-p>Author accepted manuscript terms.</license-p>"
            "</license>"
            "<license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/'
            "</ali:license_ref>"
            "</license>"
            "</permissions>"
        )
        prov = parse(article(front, long_body()), {"license_code": "CC BY"})["provenance"]
        self.assertEqual(prov["license_code_xml"], "CC BY")
        self.assertEqual(prov["license_content_type_xml"], "ccbylicense")
        self.assertIn("creativecommons.org", prov["license_ref_xml"])


class FiguresTablesAndFloats(unittest.TestCase):
    def test_floats_group_figures_and_tables_are_counted(self):
        floats = "<fig id='f1'/><fig id='f2'/><table-wrap id='t1'/>"
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(), floats=floats))
        self.assertEqual(record["figure_count"], 2)
        self.assertEqual(record["table_count"], 1)
        self.assertIn("floats_group_present", record["qc"]["flags"])

    def test_inline_body_figures_are_counted(self):
        body = "<sec><title>Results</title><p>text</p><fig id='f1'/><table-wrap id='t1'/></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body), threshold=0)
        self.assertEqual(record["figure_count"], 1)
        self.assertEqual(record["table_count"], 1)

    def test_references_counted_from_back(self):
        back = "<ref-list><ref id='r1'/><ref id='r2'/><ref id='r3'/></ref-list>"
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(), back=back))
        self.assertEqual(record["reference_count"], 3)
        self.assertNotIn("no_references", record["qc"]["flags"])

    def test_references_counted_from_all_ref_lists_under_back(self):
        back = (
            "<ref-list><title>Notes</title></ref-list>"
            "<app-group><app><ref-list>"
            "<ref id='r1'/><ref id='r2'/>"
            "</ref-list></app></app-group>"
            "<ref-list><ref id='r3'/></ref-list>"
        )
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(), back=back))
        self.assertEqual(record["reference_count"], 3)
        self.assertNotIn("no_references", record["qc"]["flags"])

    def test_nested_ref_lists_are_not_double_counted(self):
        back = (
            "<ref-list>"
            "<ref id='r1'/>"
            "<ref-list><ref id='r2'/><ref id='r3'/></ref-list>"
            "</ref-list>"
        )
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(), back=back))
        self.assertEqual(record["reference_count"], 3)
        self.assertNotIn("no_references", record["qc"]["flags"])


class QualityControl(unittest.TestCase):
    def test_short_body_is_stub_but_record_is_kept_whole(self):
        body = "<sec><title>Full Text Availability</title><p>The license terms do not permit archiving.</p></sec>"
        record = parse(article(IDS + TITLE + ABSTRACT, body))
        self.assertEqual(record["qc"]["status"], "stub")
        self.assertEqual(record["pmcid"], "PMC123")
        self.assertEqual(record["title"], "A Study")
        self.assertTrue(record["sections"], "stub records must keep their sections")

    def test_body_just_above_threshold_is_ok(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(260)))
        self.assertEqual(record["qc"]["status"], "ok")

    def test_missing_abstract_reported(self):
        record = parse(article(IDS + TITLE, long_body()))
        self.assertEqual(record["qc"]["status"], "no_abstract")

    def test_malformed_xml_recorded_not_raised(self):
        record = parse("<article><front>unclosed")
        self.assertEqual(record["qc"]["status"], "parse_error")
        self.assertIn("xml_parse_error", record["qc"]["flags"])
        self.assertEqual(record["pmcid"], "PMC123")

    def test_missing_identifiers_flagged(self):
        front = '<article-id pub-id-type="pmcid">PMC123</article-id>' + TITLE + ABSTRACT
        record = parse(article(front, long_body()))
        self.assertIn("no_pmid", record["qc"]["flags"])
        self.assertIn("no_doi", record["qc"]["flags"])


class Provenance(unittest.TestCase):
    def test_manifest_fields_are_joined_through(self):
        row = {
            "actual_md5": "abc123", "bytes": "107631", "version": "1",
            "is_manuscript": "no", "is_retracted": "yes", "license_code": "CC BY",
            "status": "ok", "downloaded_at_utc": "2026-09-01T22:14:03+00:00",
            "source_xml_url": "s3://pmc-oa-opendata/PMC123.1/PMC123.1.xml?md5=abc123",
        }
        prov = parse(article(IDS + TITLE + ABSTRACT, long_body()), row)["provenance"]
        self.assertEqual(prov["xml_md5"], "abc123")
        self.assertEqual(prov["pmc_version"], "1")
        self.assertEqual(prov["is_retracted"], "yes")
        self.assertEqual(prov["license_code_manifest"], "CC BY")
        self.assertEqual(prov["downloaded_at_utc"], "2026-09-01T22:14:03+00:00")

    def test_retracted_article_is_kept_not_dropped(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()), {"is_retracted": "yes"})
        self.assertEqual(record["qc"]["status"], "ok")
        self.assertEqual(record["provenance"]["is_retracted"], "yes")

    def test_parser_version_and_timestamp_present(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()))
        self.assertEqual(record["qc"]["parser_version"], pp.PARSER_VERSION)
        self.assertTrue(record["qc"]["parsed_at_utc"].endswith("+00:00"))
        self.assertEqual(record["schema_version"], pp.SCHEMA_VERSION)


class OutputSafety(unittest.TestCase):
    def test_refuses_to_write_under_pubmed(self):
        with self.assertRaises(SystemExit):
            pp.assert_safe_output(pp.REPO_ROOT / "pubmed" / "x.jsonl", pp.DEFAULT_XML_DIR)

    def test_refuses_to_write_inside_the_xml_directory(self):
        with self.assertRaises(SystemExit):
            pp.assert_safe_output(pp.DEFAULT_XML_DIR / "out.jsonl", pp.DEFAULT_XML_DIR)

    def test_record_is_json_serialisable(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body()))
        self.assertIsInstance(json.dumps(record, ensure_ascii=False), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# Reference lists nested in article content, and nested documents excluded
# ---------------------------------------------------------------------------


SUB_ARTICLE_WITH_REFS = (
    "<sub-article article-type='peer-review'><front-stub>"
    "<title-group><article-title>Reviewer report</article-title></title-group>"
    "</front-stub><body><sec><title>Review</title><p>Review text.</p></sec></body>"
    "<back><ref-list><ref id='sr1'/><ref id='sr2'/><ref id='sr3'/></ref-list></back>"
    "</sub-article>"
)
RESPONSE_WITH_REFS = (
    "<response response-type='reply'><front-stub>"
    "<title-group><article-title>Author reply</article-title></title-group>"
    "</front-stub>"
    "<back><ref-list><ref id='pr1'/><ref id='pr2'/></ref-list></back></response>"
)


class ReferenceListLocation(unittest.TestCase):
    """Where the bibliography sits varies by publisher; all of it is the article's."""

    def count(self, body: str = "", back: str = "", extra: str = "") -> int:
        record = parse(article(IDS + TITLE + ABSTRACT,
                               body or long_body(), back=back, extra=extra))
        return record["reference_count"]

    def test_back_ref_list(self):
        self.assertEqual(self.count(back="<ref-list><ref id='r1'/><ref id='r2'/></ref-list>"), 2)

    def test_back_sec_ref_list(self):
        self.assertEqual(
            self.count(back="<sec><title>References</title>"
                            "<ref-list><ref id='r1'/><ref id='r2'/></ref-list></sec>"), 2)

    def test_back_app_group_app_ref_list(self):
        self.assertEqual(
            self.count(back="<app-group><app id='a1'><title>Appendix</title>"
                            "<ref-list><ref id='r1'/><ref id='r2'/><ref id='r3'/>"
                            "</ref-list></app></app-group>"), 3)

    def test_body_ref_list(self):
        self.assertEqual(
            self.count(body=long_body() + "<ref-list><ref id='r1'/><ref id='r2'/></ref-list>"), 2)

    def test_body_sec_ref_list(self):
        """PMC9545113: the bibliography lives inside a body section."""
        body = ("<sec><title>Introduction</title><p>%s</p>"
                "<ref-list><ref id='r1'/><ref id='r2'/><ref id='r3'/></ref-list></sec>"
                % " ".join(["word"] * 400))
        self.assertEqual(self.count(body=body), 3)

    def test_body_and_back_ref_lists_are_summed(self):
        body = long_body() + "<ref-list><ref id='b1'/></ref-list>"
        self.assertEqual(
            self.count(body=body, back="<ref-list><ref id='r1'/><ref id='r2'/></ref-list>"), 3)

    def test_body_ref_list_clears_the_no_references_flag(self):
        body = ("<sec><title>Introduction</title><p>%s</p>"
                "<ref-list><ref id='r1'/></ref-list></sec>" % " ".join(["word"] * 400))
        record = parse(article(IDS + TITLE + ABSTRACT, body))
        self.assertEqual(record["reference_count"], 1)
        self.assertNotIn("no_references", record["qc"]["flags"])


class NestedDocumentReferencesExcluded(unittest.TestCase):
    """A peer review's or reply's citations are not the article's."""

    def count(self, body: str = "", back: str = "", extra: str = "") -> int:
        record = parse(article(IDS + TITLE + ABSTRACT,
                               body or long_body(), back=back, extra=extra))
        return record["reference_count"]

    def test_sub_article_references_are_excluded(self):
        """PMC11064958: refs exist only inside a peer-review sub-article."""
        self.assertEqual(self.count(extra=SUB_ARTICLE_WITH_REFS), 0)

    def test_response_references_are_excluded(self):
        self.assertEqual(self.count(extra=RESPONSE_WITH_REFS), 0)

    def test_sub_article_refs_leave_the_no_references_flag_set(self):
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(),
                               extra=SUB_ARTICLE_WITH_REFS))
        self.assertEqual(record["reference_count"], 0)
        self.assertIn("no_references", record["qc"]["flags"])

    def test_mixed_article_and_sub_article_refs_count_only_the_article(self):
        self.assertEqual(
            self.count(back="<ref-list><ref id='a1'/><ref id='a2'/></ref-list>",
                       extra=SUB_ARTICLE_WITH_REFS), 2)

    def test_mixed_body_refs_and_response_refs_count_only_the_article(self):
        body = ("<sec><title>Introduction</title><p>%s</p>"
                "<ref-list><ref id='a1'/></ref-list></sec>" % " ".join(["word"] * 400))
        self.assertEqual(self.count(body=body, extra=RESPONSE_WITH_REFS), 1)

    def test_sub_article_body_does_not_become_article_sections(self):
        """The exclusion must not disturb section parsing."""
        record = parse(article(IDS + TITLE + ABSTRACT, long_body(),
                               extra=SUB_ARTICLE_WITH_REFS))
        self.assertEqual([s["title_raw"] for s in record["sections"]], ["Introduction"])


class Cc0LicenceNormalisation(unittest.TestCase):
    def licence_of(self, permissions: str) -> str:
        record = parse(article(IDS + TITLE + ABSTRACT + permissions, long_body()),
                       {"license_code": "CC0"})
        return record["provenance"]["license_code_xml"]

    def test_publicdomain_mark_url_is_normalised(self):
        """PMC11135165: Public Domain Mark with a cc0license content-type."""
        self.assertEqual(self.licence_of(
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="cc0license">https://creativecommons.org/publicdomain/mark/1.0/'
            "</ali:license_ref></license></permissions>"), "CC0")

    def test_publicdomain_mark_agrees_with_the_manifest_code(self):
        """The code must match PMC's own, or the comparison manufactures a defect.

        license_code_xml is compared for string equality against the manifest's
        license_code, and PMC records PMC11135165 as CC0. A separate PDM value
        would fail that comparison and raise a false license_disagreement, so
        this pins the contract rather than the implementation. The exact URL
        stays available in license_ref_xml for anyone who needs the distinction.
        """
        permissions = (
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="cc0license">https://creativecommons.org/publicdomain/mark/1.0/'
            "</ali:license_ref></license></permissions>"
        )
        record = parse(article(IDS + TITLE + ABSTRACT + permissions, long_body()),
                       {"license_code": "CC0"})
        self.assertEqual(record["provenance"]["license_code_xml"], "CC0")
        self.assertNotIn("license_disagreement", record["qc"]["flags"])
        self.assertNotIn("license_absent_in_xml", record["qc"]["flags"])
        self.assertEqual(record["provenance"]["license_ref_xml"],
                         "https://creativecommons.org/publicdomain/mark/1.0/")

    def test_cc0license_content_type_alone_is_recognised(self):
        self.assertEqual(self.licence_of(
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="cc0license">https://example.org/terms</ali:license_ref>'
            "</license></permissions>"), "CC0")

    def test_publicdomain_zero_still_normalises(self):
        self.assertEqual(self.licence_of(
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/">'
            "https://creativecommons.org/publicdomain/zero/1.0/</ali:license_ref>"
            "</license></permissions>"), "CC0")

    def test_cc_by_normalisation_is_unaffected(self):
        self.assertEqual(self.licence_of(
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/'
            "</ali:license_ref></license></permissions>"), "CC BY")

    def test_cc_by_nc_nd_normalisation_is_unaffected(self):
        self.assertEqual(self.licence_of(
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/"'
            ' content-type="ccbyncndlicense">https://creativecommons.org/licenses/by-nc-nd/4.0/'
            "</ali:license_ref></license></permissions>"), "CC BY-NC-ND")

    def test_public_domain_prose_alone_is_not_normalised(self):
        """Prose is never part of the haystack; only identifiers are."""
        self.assertEqual(self.licence_of(
            "<permissions><license><license-p>All material appearing in this journal "
            "is in the public domain and may be reproduced without permission."
            "</license-p></license></permissions>"), "")

    def test_unrelated_publisher_reference_is_not_normalised(self):
        self.assertEqual(self.licence_of(
            "<permissions><license>"
            '<ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/">'
            "http://www.springer.com/gb/open-access/authors-rights/aam-terms-v1"
            "</ali:license_ref><license-p>Terms of use and reuse: academic research "
            "for non-commercial purposes.</license-p></license></permissions>"), "")
