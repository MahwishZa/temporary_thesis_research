#!/usr/bin/env python3
"""Tests for verify_index.py.

Run from the pmc/ directory:
    cd pmc && python3 -m unittest test_verify_index

A gate is only worth having if it fires. Every check is exercised twice: once on
a clean index, and once on an index with that specific corruption injected.

Offline: builds tiny 768-dim indexes by hand, no models, no network, no GPU.
"""

from __future__ import annotations

import io
import json
import math
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import verify_index as vi

DIM = vi.MEDCPT_DIM


def unit_vector(seed: int) -> list[float]:
    """A deterministic L2-normalised 768-dim vector."""
    raw = [math.sin(seed * 0.7 + i * 0.013) for i in range(DIM)]
    norm = math.sqrt(sum(v * v for v in raw))
    return [v / norm for v in raw]


class IndexFixture:
    """A clean, self-consistent index that individual tests then corrupt."""

    def __init__(self, root: Path, count: int = 6):
        self.root = root
        self.index = root / "index"
        self.index.mkdir(parents=True, exist_ok=True)
        self.chunks_path = root / "chunks.jsonl"
        self.count = count

        self.chunk_ids = sorted(f"PMC{i:03d}.1#abs.w1" for i in range(1, count + 1))
        self.vectors = [unit_vector(i) for i in range(count)]

        with self.chunks_path.open("w", encoding="utf-8", newline="\n") as fh:
            for cid in self.chunk_ids:
                fh.write(json.dumps({
                    "chunk_id": cid, "document_id": cid.split("#")[0],
                    "source_category": "pubmed-abstract", "canonical_date": "2024-08",
                    "date_precision": "month", "split_june_2024": "post",
                    "eligibility_status": "eligible", "retracted": "no",
                    "duplicate_of": "", "text": "evidence",
                }, sort_keys=True) + "\n")
        self.write()

    def write(self):
        import hashlib
        digest = hashlib.sha256()
        with (self.index / "embeddings.f32").open("wb") as vf, \
                (self.index / "index_manifest.jsonl").open(
                    "w", encoding="utf-8", newline="\n") as mf:
            for row, (cid, vec) in enumerate(zip(self.chunk_ids, self.vectors)):
                raw = struct.pack(f"<{DIM}f", *vec)
                vf.write(raw)
                digest.update(cid.encode("utf-8"))
                digest.update(raw)
                mf.write(json.dumps({
                    "row": row, "chunk_id": cid, "document_id": cid.split("#")[0],
                    "source_category": "pubmed-abstract", "canonical_date": "2024-08",
                    "date_precision": "month", "split_june_2024": "post",
                    "eligibility_status": "eligible", "retracted": "no",
                }, sort_keys=True) + "\n")
        (self.index / "index_meta.json").write_text(json.dumps({
            "encoder": "ncbi/MedCPT-Article-Encoder", "production": True,
            "dim": DIM, "vectors": len(self.chunk_ids),
            "content_digest": digest.hexdigest(), "device": "cuda",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    def run(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = vi.verify(self.index, self.chunks_path)
        return code, buffer.getvalue()

    def patch_meta(self, **changes):
        path = self.index / "index_meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(changes)
        path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")


class VerifyIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fixture = IndexFixture(self.tmp)

    def assertFails(self, output, label):
        self.assertIn(f"[FAIL] {label}", output)

    # -- the clean case ----------------------------------------------------
    def test_a_clean_index_passes_every_check(self):
        code, output = self.fixture.run()
        self.assertEqual(code, 0, output)
        self.assertNotIn("[FAIL]", output)
        self.assertIn("6 vectors x 768 dims", output)

    # -- each check, proven to fire ---------------------------------------
    def test_missing_artifact_is_caught(self):
        (self.fixture.index / "embeddings.f32").unlink()
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "embeddings.f32 exists")
        self.assertIn("run pmc/embed_chunks.py first", output.lower())

    def test_stub_index_is_caught(self):
        self.fixture.patch_meta(production=False, encoder="stub-sha256")
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "index is a production index")

    def test_partial_limit_index_is_caught(self):
        self.fixture.patch_meta(partial_index_limit=3)
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "index is not a --limit smoke test")

    def test_wrong_dimension_is_caught(self):
        self.fixture.patch_meta(dim=384)
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "vector dimension is MedCPT's 768")

    def test_trailing_partial_row_is_caught(self):
        path = self.fixture.index / "embeddings.f32"
        path.write_bytes(path.read_bytes() + b"\x00\x01\x02\x03\x04")
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "embeddings.f32 is a whole number of rows")

    def test_vector_manifest_count_mismatch_is_caught(self):
        path = self.fixture.index / "index_manifest.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "vector count == manifest rows")

    def test_misaligned_row_is_caught(self):
        """The failure that silently returns the wrong document."""
        path = self.fixture.index / "index_manifest.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        rows[2]["chunk_id"] = "PMC999.1#abs.w1"        # points at the wrong chunk
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "row alignment")
        self.assertIn("first at row 2", output)

    def test_row_number_not_matching_position_is_caught(self):
        path = self.fixture.index / "index_manifest.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        rows[3]["row"] = 99
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "every manifest row declares its own position")

    def test_nan_vector_is_caught(self):
        self.fixture.vectors[1] = [float("nan")] + [0.0] * (DIM - 1)
        self.fixture.write()
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "no NaN or Inf components")
        self.assertIn("[1]", output)

    def test_inf_vector_is_caught(self):
        self.fixture.vectors[0] = [float("inf")] + [0.0] * (DIM - 1)
        self.fixture.write()
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "no NaN or Inf components")

    def test_unnormalised_vector_is_caught(self):
        self.fixture.vectors[2] = [v * 3.0 for v in self.fixture.vectors[2]]
        self.fixture.write()
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "vectors are L2-normalised")

    def test_digest_mismatch_is_caught(self):
        """A vector edited after the fact must not pass as the recorded index."""
        self.fixture.patch_meta(content_digest="0" * 64)
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "content_digest recomputes to the recorded value")

    def test_silently_edited_vector_breaks_the_digest(self):
        path = self.fixture.index / "embeddings.f32"
        raw = bytearray(path.read_bytes())
        raw[0] ^= 0xFF                                  # flip a byte in row 0
        path.write_bytes(bytes(raw))
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "content_digest recomputes to the recorded value")

    def test_missing_provenance_field_is_caught(self):
        path = self.fixture.index / "index_manifest.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        for row in rows:
            del row["canonical_date"]                   # the recency study needs this
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "manifest carries the provenance the thesis needs")
        self.assertIn("canonical_date", output)

    def test_chunk_count_mismatch_is_caught(self):
        with self.fixture.chunks_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"chunk_id": "PMC900.1#abs.w1", "duplicate_of": "",
                                 "text": "extra"}, sort_keys=True) + "\n")
        code, output = self.fixture.run()
        self.assertEqual(code, 1)
        self.assertFails(output, "vector count == embeddable chunks in the chunk layer")

    def test_duplicate_chunks_are_excluded_from_the_expected_count(self):
        """embed_chunks skips duplicate_of chunks, so the verifier must too."""
        with self.fixture.chunks_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"chunk_id": "PMC901.1#abs.w1",
                                 "duplicate_of": "PMC001.1#abs.w1",
                                 "text": "dupe"}, sort_keys=True) + "\n")
        code, output = self.fixture.run()
        self.assertEqual(code, 0, output)

    def test_absent_chunk_layer_skips_alignment_without_failing(self):
        self.fixture.chunks_path.unlink()
        code, output = self.fixture.run()
        self.assertEqual(code, 0, output)
        self.assertIn("alignment not checked", output)

    def test_cli_returns_the_exit_code(self):
        self.assertEqual(
            vi.main(["--index", str(self.fixture.index),
                     "--chunks", str(self.fixture.chunks_path)]), 0)


if __name__ == "__main__":
    unittest.main()
