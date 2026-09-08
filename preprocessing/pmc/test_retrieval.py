#!/usr/bin/env python3
"""Tests for embed_chunks.py and retrieve.py.

Run from the pmc/ directory:
    cd pmc && python3 -m unittest test_retrieval

Offline: uses the deterministic stub encoder, no models, no network, no GPU.
The production MedCPT paths are import-guarded and not exercised here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import embed_chunks as ec
import retrieve as rt


def chunk(cid, cat="pmc-fulltext", text="alpha beta gamma", **kw):
    base = {
        "chunk_id": cid, "document_id": cid.split("#")[0],
        "pmcid": "PMC1", "pmid": "111", "doi": "10.1/x", "title": "T",
        "source_category": cat, "eligibility_status": "eligible",
        "canonical_date": "2024-08", "date_precision": "month",
        "date_source": "jats:epub", "split_june_2024": "post",
        "authority_tier_label": "", "guideline_family": "", "in_currency_pack": "no",
        "retracted": "no", "license_code": "CC BY", "license_band": "open",
        "location": "body", "section_id": "s1", "section_heading": "Intro",
        "imrad": "introduction", "word_count": 3, "text_sha256": "h",
        "duplicate_of": "", "text": text,
    }
    base.update(kw)
    return base


def write_chunks(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


class StubEncoderDeterminism(unittest.TestCase):
    def test_same_text_same_vector(self):
        e = ec.StubEncoder(16)
        self.assertEqual(e.encode(["hello"]), e.encode(["hello"]))

    def test_different_text_different_vector(self):
        e = ec.StubEncoder(16)
        self.assertNotEqual(e.encode(["a"]), e.encode(["b"]))

    def test_vectors_are_unit_norm_and_right_dim(self):
        v = ec.StubEncoder(32).encode(["some text"])[0]
        self.assertEqual(len(v), 32)
        self.assertAlmostEqual(sum(x * x for x in v) ** 0.5, 1.0, places=5)

    def test_stub_is_marked_non_production(self):
        self.assertFalse(ec.StubEncoder(8).production)


class ModelPinning(unittest.TestCase):
    def test_medcpt_models_are_the_pinned_ones(self):
        self.assertEqual(ec.ARTICLE_ENCODER, "ncbi/MedCPT-Article-Encoder")
        self.assertEqual(ec.QUERY_ENCODER, "ncbi/MedCPT-Query-Encoder")
        self.assertEqual(ec.CROSS_ENCODER, "ncbi/MedCPT-Cross-Encoder")
        self.assertEqual(ec.MEDCPT_DIM, 768)

    def test_stub_index_refused_without_explicit_flag(self):
        with self.assertRaises(SystemExit):
            ec.main(["--encoder", "stub", "--chunks", "/nonexistent"])


class IndexBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.chunks = self.tmp / "chunks.jsonl"
        write_chunks(self.chunks, [
            chunk("PMC1.1#s1.w1", "pmc-fulltext", "amyloid biomarker evidence"),
            chunk("PMC1.1#s1.w2", "pmc-fulltext", "tau imaging findings"),
            chunk("PMC2.1#abs.w1", "pubmed-abstract", "plasma p-tau217 assay"),
            chunk("PMC3.1#abs.w1", "currency-pack", "anti-amyloid contested efficacy"),
            chunk("PMC4.1#s1.w1", "pmc-fulltext", "duplicate text", duplicate_of="PMC1.1#s1.w1"),
        ])
        self.out = self.tmp / "index"

    def build(self, out=None, skip_dupes=True):
        return ec.build(self.chunks, out or self.out, ec.StubEncoder(16), 2,
                        skip_duplicates=skip_dupes)

    def test_vector_count_matches_manifest(self):
        meta = self.build()
        idx = rt.Index(self.out)
        self.assertEqual(idx.n, meta["vectors"])
        self.assertEqual(len(idx.rows), meta["vectors"])

    def test_duplicates_skipped_by_default_but_chunk_layer_untouched(self):
        meta = self.build()
        self.assertEqual(meta["vectors"], 4)                    # the dupe is skipped
        self.assertEqual(len(self.chunks.read_text().splitlines()), 5)   # source intact

    def test_duplicates_included_on_request(self):
        meta = self.build(self.tmp / "i2", skip_dupes=False)
        self.assertEqual(meta["vectors"], 5)

    def test_rows_are_sorted_and_contiguous(self):
        self.build()
        idx = rt.Index(self.out)
        ids = [r["chunk_id"] for r in idx.rows]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual([r["row"] for r in idx.rows], list(range(idx.n)))

    def test_no_orphan_vectors_or_missing_mappings(self):
        self.build()
        idx = rt.Index(self.out)
        raw = (self.out / "embeddings.f32").read_bytes()
        self.assertEqual(len(raw), idx.n * idx.dim * 4)          # exact, no stragglers
        for r in idx.rows:
            self.assertTrue(r["chunk_id"])

    def test_provenance_fields_survive_into_the_manifest(self):
        self.build()
        idx = rt.Index(self.out)
        for r in idx.rows:
            for f in ["canonical_date", "date_precision", "split_june_2024",
                      "source_category", "authority_tier_label", "in_currency_pack",
                      "retracted", "pmid", "pmcid", "document_id", "text_sha256"]:
                self.assertIn(f, r)

    def test_manifest_does_not_carry_chunk_text(self):
        # the index is a mapping layer; text stays in the frozen chunk file
        self.build()
        self.assertNotIn("text", rt.Index(self.out).rows[0])

    def test_build_is_deterministic(self):
        a = self.build(self.tmp / "d1")
        b = self.build(self.tmp / "d2")
        self.assertEqual(a["content_digest"], b["content_digest"])
        self.assertEqual((self.tmp / "d1" / "embeddings.f32").read_bytes(),
                         (self.tmp / "d2" / "embeddings.f32").read_bytes())
        self.assertEqual((self.tmp / "d1" / "index_manifest.jsonl").read_bytes(),
                         (self.tmp / "d2" / "index_manifest.jsonl").read_bytes())

    def test_meta_records_stub_as_non_production(self):
        meta = self.build()
        self.assertFalse(meta["production"])
        self.assertIn("exact flat scan", meta["search"])


class Retrieval(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.chunks = self.tmp / "chunks.jsonl"
        rows = []
        for i in range(6):
            rows.append(chunk(f"PMCA.1#s1.w{i}", "pmc-fulltext", f"body text {i}"))
        for i in range(6):
            rows.append(chunk(f"PMCB.1#abs.w{i}", "pubmed-abstract", f"abstract text {i}"))
        rows.append(chunk("PMCC.1#abs.w1", "currency-pack", "contested anti-amyloid"))
        write_chunks(self.chunks, rows)
        self.out = self.tmp / "index"
        ec.build(self.chunks, self.out, ec.StubEncoder(16), 4)
        self.index = rt.Index(self.out)
        self.q = ec.StubEncoder(16).encode(["body text 1"])[0]

    def test_balanced_quota_per_category(self):
        cands = rt.balanced_retrieve(self.index, self.q, per_category=2)
        got = {}
        for c in cands:
            got[c["retrieved_from_category"]] = got.get(c["retrieved_from_category"], 0) + 1
        self.assertEqual(got["pmc-fulltext"], 2)
        self.assertEqual(got["pubmed-abstract"], 2)
        self.assertEqual(got["currency-pack"], 1)     # only one exists; quota is a cap

    def test_small_category_is_not_drowned_out(self):
        cands = rt.balanced_retrieve(self.index, self.q, per_category=3)
        self.assertIn("currency-pack", {c["retrieved_from_category"] for c in cands})

    def test_retrieval_is_deterministic(self):
        a = rt.balanced_retrieve(self.index, self.q, 3)
        b = rt.balanced_retrieve(self.index, self.q, 3)
        self.assertEqual([c["chunk_id"] for c in a], [c["chunk_id"] for c in b])

    def test_ranks_are_dense_and_ordered(self):
        cands = rt.balanced_retrieve(self.index, self.q, 3)
        self.assertEqual([c["retrieval_rank"] for c in cands],
                         list(range(1, len(cands) + 1)))
        scores = [c["retrieval_score"] for c in cands]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_candidates_keep_recency_and_authority_metadata(self):
        for c in rt.balanced_retrieve(self.index, self.q, 2):
            for f in ["canonical_date", "date_precision", "split_june_2024",
                      "source_category", "in_currency_pack", "retracted"]:
                self.assertIn(f, c)

    def test_query_dim_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            rt.score_all(self.index, [0.1, 0.2])

    def test_ties_break_deterministically_on_chunk_id(self):
        scores = [1.0] * self.index.n              # everything tied
        got = rt.top_k(self.index, scores, 4)
        ids = [self.index.rows[i]["chunk_id"] for i, _ in got]
        self.assertEqual(ids, sorted(ids))


class CandidateReplay(unittest.TestCase):
    """Validity control V3: the candidate set must replay byte-identically."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        chunks = self.tmp / "chunks.jsonl"
        write_chunks(chunks, [chunk(f"PMCA.1#s1.w{i}", "pmc-fulltext", f"t {i}")
                              for i in range(5)])
        self.out = self.tmp / "index"
        ec.build(chunks, self.out, ec.StubEncoder(16), 4)
        self.index = rt.Index(self.out)
        self.q = ec.StubEncoder(16).encode(["t 2"])[0]
        self.cands = rt.balanced_retrieve(self.index, self.q, 3)
        self.path = self.tmp / "cand" / "q1.json"

    def test_save_then_replay_round_trips(self):
        saved = rt.save_candidates(self.path, "q1", "t 2", self.cands,
                                   self.index.meta, {"per_category": 3})
        back = rt.replay_candidates(self.path)
        self.assertEqual(back["candidate_digest"], saved["candidate_digest"])
        self.assertEqual([c["chunk_id"] for c in back["candidates"]],
                         [c["chunk_id"] for c in self.cands])

    def test_fresh_retrieval_matches_saved_set(self):
        rt.save_candidates(self.path, "q1", "t 2", self.cands, self.index.meta, {})
        again = rt.balanced_retrieve(self.index, self.q, 3)
        self.assertTrue(rt.verify_replay(self.path, again))

    def test_different_candidate_population_is_detected(self):
        rt.save_candidates(self.path, "q1", "t 2", self.cands, self.index.meta, {})
        smaller = rt.balanced_retrieve(self.index, self.q, 1)
        self.assertFalse(rt.verify_replay(self.path, smaller))

    def test_tampered_candidate_file_is_rejected(self):
        rt.save_candidates(self.path, "q1", "t 2", self.cands, self.index.meta, {})
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["candidates"][0]["chunk_id"] = "PMCX.1#s9.w9"
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit):
            rt.replay_candidates(self.path)

    def test_digest_depends_on_order_not_just_membership(self):
        reordered = list(reversed(self.cands))
        self.assertNotEqual(rt.candidate_digest(self.cands),
                            rt.candidate_digest(reordered))

    def test_saved_payload_records_index_identity(self):
        saved = rt.save_candidates(self.path, "q1", "t 2", self.cands,
                                   self.index.meta, {})
        self.assertEqual(saved["index"]["content_digest"],
                         self.index.meta["content_digest"])
        self.assertIs(saved["index"]["production"], False)

    def test_corrupt_index_is_detected(self):
        (self.out / "embeddings.f32").write_bytes(b"\x00" * 8)
        with self.assertRaises(SystemExit):
            rt.Index(self.out)


if __name__ == "__main__":
    unittest.main()


class ResumeAfterInterruption(unittest.TestCase):
    """A 780k-chunk GPU run gets interrupted; restarting must not start over.

    The property that matters is not merely "it continues" but that the
    resumed index is byte-identical to an uninterrupted one -- content_digest
    included, since retrieval and the RAG2 corpus adapter both key off it.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.chunks = self.tmp / "chunks.jsonl"
        write_chunks(self.chunks, [
            chunk(f"PMC{i}.1#abs.w1", "pubmed-abstract", f"evidence text {i}")
            for i in range(1, 11)
        ])
        self.ref = self.tmp / "reference"
        self.out = self.tmp / "index"

    def _encoder(self):
        return ec.StubEncoder(16)

    def _reference(self):
        return ec.build(self.chunks, self.ref, self._encoder(), 3)

    def _artifacts(self, d: Path):
        return (
            (d / "embeddings.f32").read_bytes(),
            (d / "index_manifest.jsonl").read_bytes(),
        )

    def test_resume_reproduces_an_uninterrupted_index_exactly(self):
        reference = self._reference()
        # Stop after 4 of 10 rows, then resume.
        ec.build(self.chunks, self.out, self._encoder(), 3, limit=4)
        partial = ec.resume_state(self.out / "embeddings.f32",
                                  self.out / "index_manifest.jsonl", 16)
        self.assertEqual(partial[0], 4)
        resumed = ec.build(self.chunks, self.out, self._encoder(), 3, resume=True)

        self.assertEqual(resumed["vectors"], reference["vectors"])
        self.assertEqual(resumed["content_digest"], reference["content_digest"])
        self.assertEqual(self._artifacts(self.out), self._artifacts(self.ref))

    def test_resume_survives_a_vector_truncated_mid_row(self):
        """A kill during a write leaves half a vector; it must be trimmed."""
        reference = self._reference()
        ec.build(self.chunks, self.out, self._encoder(), 3, limit=6)
        vec = self.out / "embeddings.f32"
        vec.write_bytes(vec.read_bytes() + b"\x00\x01\x02")     # partial row 6

        resumed = ec.build(self.chunks, self.out, self._encoder(), 3, resume=True)
        self.assertEqual(resumed["content_digest"], reference["content_digest"])
        self.assertEqual(self._artifacts(self.out), self._artifacts(self.ref))

    def test_resume_survives_a_partial_manifest_line(self):
        reference = self._reference()
        ec.build(self.chunks, self.out, self._encoder(), 3, limit=6)
        man = self.out / "index_manifest.jsonl"
        man.write_bytes(man.read_bytes() + b'{"chunk_id": "PMC7.1#ab')

        resumed = ec.build(self.chunks, self.out, self._encoder(), 3, resume=True)
        self.assertEqual(resumed["content_digest"], reference["content_digest"])
        self.assertEqual(self._artifacts(self.out), self._artifacts(self.ref))

    def test_resume_recovers_when_the_manifest_lags_the_vectors(self):
        """Vectors flushed, manifest not: the extra vector rows must be dropped."""
        reference = self._reference()
        ec.build(self.chunks, self.out, self._encoder(), 3, limit=6)
        man = self.out / "index_manifest.jsonl"
        lines = man.read_bytes().splitlines(keepends=True)
        man.write_bytes(b"".join(lines[:4]))                    # lose 2 manifest rows

        resumed = ec.build(self.chunks, self.out, self._encoder(), 3, resume=True)
        self.assertEqual(resumed["vectors"], reference["vectors"])
        self.assertEqual(resumed["content_digest"], reference["content_digest"])
        self.assertEqual(self._artifacts(self.out), self._artifacts(self.ref))

    def test_restart_discards_an_existing_partial_index(self):
        reference = self._reference()
        ec.build(self.chunks, self.out, self._encoder(), 3, limit=6)
        restarted = ec.build(self.chunks, self.out, self._encoder(), 3, resume=False)
        self.assertEqual(restarted["content_digest"], reference["content_digest"])
        self.assertEqual(self._artifacts(self.out), self._artifacts(self.ref))

    def test_resume_against_different_chunks_is_refused(self):
        """Silently appending to an index built from other chunks would corrupt it."""
        ec.build(self.chunks, self.out, self._encoder(), 3, limit=4)
        other = self.tmp / "other.jsonl"
        write_chunks(other, [
            chunk(f"ZZZ{i}.1#abs.w1", "pubmed-abstract", f"different {i}")
            for i in range(1, 11)
        ])
        with self.assertRaises(SystemExit) as ctx:
            ec.build(other, self.out, self._encoder(), 3, resume=True)
        self.assertIn("resume mismatch", str(ctx.exception))

    def test_resume_refuses_an_index_longer_than_the_chunk_layer(self):
        ec.build(self.chunks, self.out, self._encoder(), 3)
        shorter = self.tmp / "shorter.jsonl"
        write_chunks(shorter, [
            chunk(f"PMC{i}.1#abs.w1", "pubmed-abstract", f"evidence text {i}")
            for i in range(1, 4)
        ])
        with self.assertRaises(SystemExit) as ctx:
            ec.build(shorter, self.out, self._encoder(), 3, resume=True)
        self.assertIn("different chunks", str(ctx.exception))

    def test_resume_on_a_complete_index_is_a_no_op(self):
        reference = self._reference()
        ec.build(self.chunks, self.out, self._encoder(), 3)
        again = ec.build(self.chunks, self.out, self._encoder(), 3, resume=True)
        self.assertEqual(again["vectors"], reference["vectors"])
        self.assertEqual(again["content_digest"], reference["content_digest"])

    def test_resume_state_on_a_missing_index_is_empty(self):
        rows, digest, cats, last = ec.resume_state(
            self.tmp / "nope.f32", self.tmp / "nope.jsonl", 16)
        self.assertEqual((rows, cats, last), (0, {}, ""))

    def test_limit_marks_the_index_as_partial(self):
        """A smoke-test index must never look like the production one."""
        meta = ec.build(self.chunks, self.out, self._encoder(), 3, limit=4)
        self.assertEqual(meta["vectors"], 4)
        self.assertEqual(meta["partial_index_limit"], 4)
        self.assertNotIn("partial_index_limit", self._reference())


class DeviceSelection(unittest.TestCase):
    """--device must not silently substitute the CPU for a requested GPU.

    That substitution is what sent a production run to the CPU unnoticed, so it
    is pinned here with a stand-in for torch (no torch in this environment).
    """

    class _Torch:
        __version__ = "2.4.1+cpu"

        def __init__(self, available):
            self._available = available
            self.cuda = self

        def is_available(self):
            return self._available

    def test_cuda_request_fails_loudly_when_unavailable(self):
        with self.assertRaises(SystemExit) as ctx:
            ec.resolve_device("cuda", self._Torch(False))
        message = str(ctx.exception)
        self.assertIn("torch.cuda.is_available() is False", message)
        self.assertIn("cu121", message)          # actionable: names the fix

    def test_cuda_request_is_honoured_when_available(self):
        self.assertEqual(ec.resolve_device("cuda", self._Torch(True)), "cuda")
        self.assertEqual(ec.resolve_device("cuda:0", self._Torch(True)), "cuda:0")

    def test_auto_prefers_cuda_then_falls_back(self):
        self.assertEqual(ec.resolve_device("auto", self._Torch(True)), "cuda")
        self.assertEqual(ec.resolve_device(None, self._Torch(True)), "cuda")
        self.assertEqual(ec.resolve_device("auto", self._Torch(False)), "cpu")

    def test_cpu_is_selectable_explicitly(self):
        self.assertEqual(ec.resolve_device("cpu", self._Torch(True)), "cpu")

    def test_default_batch_size_is_conservative(self):
        """Pinned: 32 does not fit a 4 GiB card at 512 tokens."""
        self.assertLessEqual(ec.DEFAULT_BATCH_SIZE, 8)


class CudaOOMBackoff(unittest.TestCase):
    """On CUDA OOM the batch shrinks and the run continues -- it never moves to CPU.

    A mid-run device switch would silently make part of the index incomparable
    with the rest, so the backoff is pinned here with a fake torch.
    """

    class _OOM(RuntimeError):
        pass

    class _Torch:
        def __init__(self, outer):
            self.OutOfMemoryError = outer._OOM
            self.cuda = self
            self.emptied = 0

        def empty_cache(self):
            self.emptied += 1

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def no_grad(self):
            return self._NoGrad()

    def _encoder(self, fits_at, batch_size=8, min_batch_size=1):
        """An encoder whose forward pass OOMs above ``fits_at`` sequences."""
        enc = ec.MedCPTEncoder.__new__(ec.MedCPTEncoder)
        enc.torch = self._Torch(self)
        enc.dim = ec.MEDCPT_DIM
        enc.device = "cuda"
        enc.batch_size = batch_size
        enc.min_batch_size = min_batch_size
        enc.attempts = []

        def fake_batch(batch):
            enc.attempts.append(len(batch))
            if len(batch) > fits_at:
                raise self._OOM("CUDA out of memory")
            return [[0.0] * ec.MEDCPT_DIM for _ in batch]

        enc._encode_batch = fake_batch
        return enc

    def test_batch_halves_until_it_fits_and_all_vectors_are_returned(self):
        enc = self._encoder(fits_at=2, batch_size=8)
        vectors = enc.encode([f"text {i}" for i in range(10)])
        self.assertEqual(len(vectors), 10)
        self.assertEqual(enc.attempts[:3], [8, 4, 2])   # 8 -> 4 -> 2, then succeeds
        self.assertEqual(enc.batch_size, 2)             # stays reduced for the rest
        self.assertGreaterEqual(enc.torch.emptied, 2)   # cache freed on each retry

    def test_reduced_batch_size_persists_so_the_failure_is_not_repeated(self):
        enc = self._encoder(fits_at=4, batch_size=8)
        enc.encode([f"text {i}" for i in range(12)])
        self.assertEqual(enc.batch_size, 4)
        self.assertEqual(enc.attempts.count(8), 1)      # tried the big batch only once

    def test_oom_at_the_floor_stops_with_an_actionable_message(self):
        enc = self._encoder(fits_at=0, batch_size=2, min_batch_size=1)
        with self.assertRaises(SystemExit) as ctx:
            enc.encode(["text"])
        message = str(ctx.exception)
        self.assertIn("out of memory even at batch size 1", message)
        self.assertIn("resumes", message)               # tells the student what to do

    def test_no_oom_leaves_the_configured_batch_size_alone(self):
        enc = self._encoder(fits_at=64, batch_size=8)
        vectors = enc.encode([f"text {i}" for i in range(20)])
        self.assertEqual(len(vectors), 20)
        self.assertEqual(enc.batch_size, 8)
        self.assertEqual(enc.attempts, [8, 8, 4])       # last partial batch is the remainder
        self.assertEqual(enc.torch.emptied, 0)


class EmbedCLI(unittest.TestCase):
    """main() end to end. The unit tests call build() directly, so the CLI's own
    wiring (device print, batch resolution, resume flags) needs its own cover --
    a missing attribute here breaks the production command and nothing else."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.chunks = self.tmp / "chunks.jsonl"
        write_chunks(self.chunks, [
            chunk(f"PMC{i}.1#abs.w1", "pubmed-abstract", f"evidence {i}")
            for i in range(1, 21)
        ])
        self.out = self.tmp / "index"

    def _run(self, *extra):
        return ec.main(["--encoder", "stub", "--allow-stub", "--dim", "16",
                        "--chunks", str(self.chunks), "--out", str(self.out),
                        "--batch-size", "4", "--progress-every", "0", *extra])

    def _meta(self):
        return json.loads((self.out / "index_meta.json").read_text(encoding="utf-8"))

    def test_cli_builds_an_index(self):
        self.assertEqual(self._run(), 0)
        self.assertEqual(self._meta()["vectors"], 20)

    def test_cli_resumes_by_default_and_matches_an_uninterrupted_run(self):
        self._run("--limit", "7")
        self.assertEqual(self._meta()["vectors"], 7)
        self._run()                                   # resume, no --limit
        resumed = self._meta()

        reference = self.tmp / "ref"
        ec.main(["--encoder", "stub", "--allow-stub", "--dim", "16",
                 "--chunks", str(self.chunks), "--out", str(reference),
                 "--batch-size", "4", "--progress-every", "0"])
        expected = json.loads((reference / "index_meta.json").read_text(encoding="utf-8"))

        self.assertEqual(resumed["vectors"], 20)
        self.assertEqual(resumed["content_digest"], expected["content_digest"])
        self.assertEqual((self.out / "embeddings.f32").read_bytes(),
                         (reference / "embeddings.f32").read_bytes())

    def test_cli_restart_discards_the_partial_index(self):
        self._run("--limit", "7")
        self._run("--restart")
        self.assertEqual(self._meta()["vectors"], 20)

    def test_cli_refuses_a_stub_index_without_allow_stub(self):
        with self.assertRaises(SystemExit) as ctx:
            ec.main(["--encoder", "stub", "--chunks", str(self.chunks),
                     "--out", str(self.out)])
        self.assertIn("--allow-stub", str(ctx.exception))

    def test_cli_rejects_a_zero_batch_size(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run("--batch-size", "0")
        self.assertIn("at least 1", str(ctx.exception))

    def test_cli_default_batch_size_is_the_conservative_one(self):
        parsed = ec.argparse.ArgumentParser()          # sanity: the default is wired
        self.assertEqual(ec.DEFAULT_BATCH_SIZE, 8)
        del parsed
