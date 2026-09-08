#!/usr/bin/env python3
"""Tests for the thesis architecture layer.

Run from the repository root:
    python3 -m unittest discover -s thesis/tests -t .
    (or: python3 -m pytest thesis/tests)

Offline: synthetic fixtures, a hash encoder, no models, no network, no GPU, and
the production corpus and index are never touched.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from thesis.conditions.base import (  # noqa: E402
    AdmittedEvidence, available_conditions, build_condition,
)
from thesis.config import ThesisConfig, load_config, merge_overrides  # noqa: E402
from thesis.corpus import CorpusError, CorpusHandle, normalise_chunk  # noqa: E402
from thesis.evaluation import compare, evaluate, retrieval_report  # noqa: E402
from thesis.pipeline import run_pipeline  # noqa: E402
from thesis.provenance import (  # noqa: E402
    CARRIED_TEMPORAL_FIELDS, CorpusStamp, RunRecord, check_evidence_provenance,
    temporal_fields_carried,
)
from thesis.queries import in_memory_query_set, load_query_set, normalise_query  # noqa: E402
from thesis.recency import (  # noqa: E402
    NullTemporalPolicy, TemporalPolicyError, available_policies, build_policy,
)
from thesis.retrieval import RetrievalService  # noqa: E402
from thesis.smoke import HashEncoder, build_fixture, smoke_config  # noqa: E402


class FixtureCase(unittest.TestCase):
    """Builds a miniature corpus + index in a temp dir."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="thesis-test-")
        self.fixture = build_fixture(self.root)
        self.config = smoke_config(self.root, self.fixture)
        self.queries = in_memory_query_set(
            [{"query_id": "T1", "query": "amyloid and cognition"},
             {"query_id": "T2", "query": "anti-amyloid therapy evidence"}], name="unit"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def service(self, config=None) -> RetrievalService:
        return RetrievalService(config or self.config, encoder=HashEncoder())


# ---------------------------------------------------------------- config ---
class TestConfig(unittest.TestCase):
    def test_defaults_are_the_control_arm(self):
        config = ThesisConfig()
        self.assertEqual(config.condition.name, "baseline")
        self.assertEqual(config.recency.policy, "none")

    def test_shipped_configs_load(self):
        directory = os.path.join(REPO_ROOT, "configs", "thesis")
        found = 0
        for dirpath, _, filenames in os.walk(directory):
            for filename in sorted(filenames):
                if filename.endswith(".yaml"):
                    load_config(os.path.join(dirpath, filename))
                    found += 1
        self.assertGreaterEqual(found, 4)

    def test_base_chain_is_inherited(self):
        config = load_config(os.path.join(REPO_ROOT, "configs/thesis/conditions/rag2.yaml"))
        self.assertEqual(config.condition.name, "rag2")
        self.assertEqual(config.rag2_config, "rag2/configs/thesis_corpus.yaml")  # from the base
        self.assertEqual(config.corpus.source_categories[0], "pubmed-abstract")

    def test_unknown_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            load_config(None, {"corpus": {"not_a_key": 1}})

    def test_overrides_use_the_rag2_parser(self):
        config = load_config(None, merge_overrides(
            ["retrieval.final_top_k=16", "condition.name=rag2"]))
        self.assertEqual(config.retrieval.final_top_k, 16)
        self.assertEqual(config.condition.name, "rag2")

    def test_rag2_config_is_referenced_not_duplicated(self):
        """Reproduction-critical settings must have exactly one definition."""
        config = ThesisConfig()
        rag2 = config.load_rag2_config()
        self.assertIsNotNone(rag2)
        self.assertEqual(rag2.retrieval.query_encoder, "ncbi/MedCPT-Query-Encoder")
        # None of RAG2's method-level keys are redeclared at the thesis level.
        thesis_keys = set(config.to_dict())
        for owned_by_rag2 in ("filter", "llm", "prompts", "generation", "filter_training"):
            self.assertNotIn(owned_by_rag2, thesis_keys)

    def test_fingerprint_changes_with_config(self):
        base = ThesisConfig().fingerprint()
        self.assertNotEqual(base, load_config(None, {"seed": 7}).fingerprint())


# ---------------------------------------------------------------- corpus ---
class TestCorpus(FixtureCase):
    def test_opens_and_verifies_digest(self):
        handle = CorpusHandle.open(self.config)
        self.assertTrue(handle.digest_verified)
        self.assertEqual(handle.stamp().chunk_digest, self.fixture["digest"])

    def test_digest_mismatch_is_refused(self):
        config = smoke_config(self.root, {**self.fixture, "digest": "0" * 64})
        with self.assertRaises(CorpusError) as ctx:
            CorpusHandle.open(config)
        self.assertIn("digest mismatch", str(ctx.exception))

    def test_chunk_count_mismatch_is_refused(self):
        self.config.corpus.expected_chunk_count = 999
        with self.assertRaises(CorpusError):
            CorpusHandle.open(self.config)

    def test_stub_index_refused_when_production_required(self):
        self.config.corpus.require_production_index = True
        with self.assertRaises(CorpusError) as ctx:
            CorpusHandle.open(self.config)
        self.assertIn("stub encoder", str(ctx.exception))

    def test_unverified_digest_is_recorded_as_unverified(self):
        self.config.corpus.expected_chunk_digest = ""
        handle = CorpusHandle.open(self.config)
        self.assertFalse(handle.digest_verified)

    def test_normalise_chunk_preserves_unknown_fields(self):
        record = normalise_chunk({"chunk_id": "c1", "text": "t", "novel_field": 42})
        self.assertEqual(record["extra"]["novel_field"], 42)

    def test_iter_chunks_streams_records(self):
        handle = CorpusHandle.open(self.config)
        chunks = list(handle.iter_chunks(limit=3))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(c["chunk_id"] for c in chunks))


# --------------------------------------------------------------- queries ---
class TestQueries(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "q.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"query_id": "D", "query": "a"}) + "\n")
                handle.write(json.dumps({"query_id": "D", "query": "b"}) + "\n")
            with self.assertRaises(ValueError):
                load_query_set(path)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_digest_identifies_the_set(self):
        one = in_memory_query_set([{"query_id": "A", "query": "x"}])
        same = in_memory_query_set([{"query_id": "A", "query": "x"}])
        other = in_memory_query_set([{"query_id": "A", "query": "y"}])
        self.assertEqual(one.digest, same.digest)
        self.assertNotEqual(one.digest, other.digest)

    def test_normalisation_is_whitespace_only(self):
        query = in_memory_query_set([{"query_id": "A", "query": "  a   b  "}]).queries[0]
        self.assertEqual(normalise_query(query), "a b")

    def test_empty_query_text_rejected(self):
        with self.assertRaises(ValueError):
            in_memory_query_set([{"query_id": "A", "query": "   "}])


# --------------------------------------------------------------- recency ---
class TestRecencyBoundary(unittest.TestCase):
    def test_null_policy_is_the_identity_and_reads_nothing(self):
        policy = build_policy("none")
        self.assertIsInstance(policy, NullTemporalPolicy)
        self.assertFalse(policy.reads_dates)
        candidates = [{"chunk_id": "c1", "canonical_date": "2019-01-01"}]
        self.assertEqual(policy.apply(candidates), candidates)

    def test_planned_policies_are_declared_but_refuse_to_run(self):
        for name in ("recency_weighted", "currency_three_state", "supersession"):
            policy = build_policy(name)
            self.assertFalse(policy.describe()["implemented"])
            with self.assertRaises(TemporalPolicyError):
                policy.apply([{"chunk_id": "c1"}])

    def test_unknown_policy_rejected(self):
        with self.assertRaises(TemporalPolicyError):
            build_policy("invented_policy")

    def test_registry_lists_the_control_and_the_planned(self):
        self.assertIn("none", available_policies())
        self.assertIn("recency_weighted", available_policies())


# ------------------------------------------------------------ conditions ---
class TestConditions(FixtureCase):
    def test_registry(self):
        self.assertEqual(available_conditions(), ["baseline", "rag2", "recency"])

    def test_baseline_admits_top_k_and_generates_nothing(self):
        service = self.service()
        condition = build_condition(self.config)
        retrieved = service.retrieve(self.queries.queries[0])
        result = condition.run(self.queries.queries[0], retrieved)
        self.assertEqual(len(result.admitted), self.config.retrieval.final_top_k)
        self.assertEqual(result.rejected, [])
        self.assertEqual(result.answer, "")

    def test_recency_condition_refuses_the_null_policy(self):
        config = smoke_config(self.root, self.fixture, condition="recency")
        with self.assertRaises(TemporalPolicyError) as ctx:
            build_condition(config)
        self.assertIn("control arm", str(ctx.exception))

    def test_recency_condition_wraps_an_inner_arm(self):
        config = smoke_config(self.root, self.fixture, condition="recency")
        config.recency.policy = "recency_weighted"
        config.recency.options = {"inner": "baseline"}
        condition = build_condition(config)
        self.assertEqual(condition.describe()["inner_condition"], "baseline")
        # and it still refuses to run, because the policy is not implemented
        service = self.service()
        retrieved = service.retrieve(self.queries.queries[0])
        with self.assertRaises(TemporalPolicyError):
            condition.run(self.queries.queries[0], retrieved)

    def test_unknown_condition_rejected(self):
        self.config.condition.name = "not_a_condition"
        with self.assertRaises(KeyError):
            build_condition(self.config)

    def test_admitted_evidence_carries_provenance(self):
        service = self.service()
        retrieved = service.retrieve(self.queries.queries[0])
        result = build_condition(self.config).run(self.queries.queries[0], retrieved)
        evidence = result.admitted[0]
        self.assertTrue(evidence.chunk_id)
        self.assertTrue(evidence.document_id)
        self.assertIn("canonical_date", evidence.metadata)


# ------------------------------------------------------------- retrieval ---
class TestRetrieval(FixtureCase):
    def test_balanced_across_categories(self):
        service = self.service()
        retrieved = service.retrieve(self.queries.queries[0])
        counts = {}
        for candidate in retrieved.candidates:
            counts[candidate["source_category"]] = counts.get(candidate["source_category"], 0) + 1
        self.assertEqual(set(counts), set(self.config.corpus.source_categories))
        self.assertEqual(len(set(counts.values())), 1, f"quota not equal: {counts}")

    def test_deterministic(self):
        first = self.service().retrieve(self.queries.queries[0])
        second = self.service().retrieve(self.queries.queries[0])
        self.assertEqual(first.candidate_digest, second.candidate_digest)

    def test_fingerprint_ignores_final_top_k_but_tracks_depth(self):
        base = self.service().fingerprint()
        deeper = smoke_config(self.root, self.fixture)
        deeper.retrieval.final_top_k = 99
        self.assertEqual(self.service(deeper).fingerprint(), base)
        deeper.retrieval.per_category = 99
        self.assertNotEqual(self.service(deeper).fingerprint(), base)

    def test_save_and_replay_preserve_the_population(self):
        service = self.service()
        retrieved = service.retrieve(self.queries.queries[0])
        directory = os.path.join(self.root, "cands")
        service.save(retrieved, directory)

        replay_config = smoke_config(self.root, self.fixture)
        replay_config.retrieval.replay_from = directory
        replayed = self.service(replay_config).retrieve(self.queries.queries[0])
        self.assertEqual(replayed.candidate_digest, retrieved.candidate_digest)
        self.assertTrue(replayed.replayed_from)

    def test_saved_candidates_record_index_identity(self):
        service = self.service()
        retrieved = service.retrieve(self.queries.queries[0])
        path = service.save(retrieved, os.path.join(self.root, "cands2"))
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertIn("index", payload)
        self.assertEqual(payload["params"]["query_encoder"], "ncbi/MedCPT-Query-Encoder")

    def test_uses_the_query_encoder_not_the_article_encoder(self):
        """MedCPT is asymmetric; encoding a query with the article encoder is wrong."""
        service = self.service()
        self.assertEqual(service.models.query_encoder, "ncbi/MedCPT-Query-Encoder")
        self.assertEqual(service.models.article_encoder, "ncbi/MedCPT-Article-Encoder")
        self.assertNotEqual(service.models.query_encoder, service.models.article_encoder)


# ------------------------------------------------------------ provenance ---
class TestProvenance(FixtureCase):
    def test_missing_identity_is_reported(self):
        problems = check_evidence_provenance([{"chunk_id": "c1"}])
        self.assertTrue(any("document_id" in p for p in problems))

    def test_temporal_carriage_detected(self):
        complete = [{f: "x" for f in CARRIED_TEMPORAL_FIELDS}]
        self.assertTrue(temporal_fields_carried(complete)["complete"])
        partial = [{"canonical_date": "2020-01-01"}]
        report = temporal_fields_carried(partial)
        self.assertFalse(report["complete"])
        self.assertIn("date_precision", report["missing"])

    def test_unverified_run_is_not_reportable(self):
        record = RunRecord(run_name="r", condition="baseline",
                           corpus=CorpusStamp(digest_verified=False))
        reportable, reasons = record.is_reportable()
        self.assertFalse(reportable)
        self.assertTrue(any("digest" in r for r in reasons))

    def test_stub_index_is_not_reportable(self):
        record = RunRecord(run_name="r", condition="baseline",
                           corpus=CorpusStamp(digest_verified=True, production_index=False))
        self.assertFalse(record.is_reportable()[0])

    def test_run_record_round_trips(self):
        record = RunRecord(run_name="r", condition="baseline", corpus=CorpusStamp())
        path = record.write(os.path.join(self.root, "rec"))
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in ("run_name", "condition", "corpus", "models", "git", "environment", "config"):
            self.assertIn(key, payload)


# -------------------------------------------------------------- pipeline ---
class TestPipeline(FixtureCase):
    def run_once(self, condition="baseline", **overrides):
        config = smoke_config(self.root, self.fixture, condition=condition)
        for key, value in overrides.items():
            setattr(config.retrieval, key, value)
        return run_pipeline(config, query_set=self.queries,
                            retrieval=self.service(config), write=False)

    def test_end_to_end_structure(self):
        outcome = self.run_once()
        self.assertEqual(len(outcome.results), len(self.queries))
        self.assertIn("retrieval", outcome.report)
        self.assertIsNotNone(outcome.record)

    def test_records_temporal_carriage(self):
        outcome = self.run_once()
        self.assertTrue(outcome.record.retrieval["temporal_fields_carried"]["complete"])

    def test_not_reportable_on_a_stub_index(self):
        outcome = self.run_once()
        self.assertFalse(outcome.reportable)

    def test_writes_expected_outputs(self):
        config = smoke_config(self.root, self.fixture)
        outcome = run_pipeline(config, query_set=self.queries,
                               retrieval=self.service(config), write=True)
        for filename in ("results.jsonl", "report.json", "run_record.json"):
            self.assertTrue(os.path.exists(os.path.join(outcome.output_dir, filename)), filename)

    def test_two_arms_see_the_same_evidence_population(self):
        """The controlled-comparison guarantee, checked rather than asserted."""
        first = self.run_once()
        second = self.run_once()
        self.assertEqual(
            [r.candidate_digest for r in first.results],
            [r.candidate_digest for r in second.results],
        )
        report = compare({"a": {"candidate_digest": first.report["candidate_digest"]},
                          "b": {"candidate_digest": second.report["candidate_digest"]}})
        self.assertTrue(report["same_evidence_population"])

    def test_compare_warns_when_populations_differ(self):
        report = compare({"a": {"candidate_digest": "X"}, "b": {"candidate_digest": "Y"}})
        self.assertFalse(report["same_evidence_population"])
        self.assertIn("not attributable", report["warning"])


# ------------------------------------------------------------ evaluation ---
class TestEvaluation(unittest.TestCase):
    def test_absent_answers_are_absent_not_zero(self):
        from thesis.conditions.base import ConditionResult

        results = [ConditionResult(query_id="q", condition="baseline")]
        report = evaluate(results, in_memory_query_set([{"query_id": "q", "query": "x"}]))
        self.assertIsNone(report["answers"])
        self.assertIn("absent rather than zero", report["answers_note"])

    def test_retrieval_report_counts(self):
        from thesis.conditions.base import ConditionResult

        results = [ConditionResult(
            query_id="q", condition="baseline",
            admitted=[AdmittedEvidence("c1", "d1", "pmc-fulltext", "t", 1)],
            rejected=[AdmittedEvidence("c2", "d2", "pubmed-abstract", "t", 2)])]
        report = retrieval_report(results)
        self.assertEqual(report["evidence_admitted"], 1)
        self.assertAlmostEqual(report["admission_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
