"""The adapter that lets the baseline retrieve over this repository's corpus.

``pmc/`` produces a chunk layer and a MedCPT index in its own format, not the
article/embedding layout of the authors' release. These tests pin the join
between the two files, the row-alignment invariant that keeps a retrieved vector
pointing at the right passage, and the provenance-carrying policy.
"""

import json
import os
import struct

import pytest

numpy = pytest.importorskip("numpy")

from rag2.config import CorpusConfig
from rag2.corpora.base import build_corpus
from rag2.corpora.thesis_chunks import (
    ThesisChunkCorpus,
    build_thesis_corpora,
    index_categories,
)

DIM = 4

# One row per vector, in embeddings.f32 order -- the alignment contract.
MANIFEST = [
    {"row": 0, "chunk_id": "PMC1#abs.w1", "document_id": "PMC1", "title": "T1",
     "source_category": "pubmed-abstract", "canonical_date": "2013-05-01",
     "date_precision": "day", "authority_tier_label": "", "in_currency_pack": "no",
     "retracted": "no", "duplicate_of": ""},
    {"row": 1, "chunk_id": "PMC2#abs.w1", "document_id": "PMC2", "title": "T2",
     "source_category": "pmc-fulltext", "canonical_date": "2021-02", "date_precision": "month",
     "authority_tier_label": "", "in_currency_pack": "no", "retracted": "no",
     "duplicate_of": ""},
    {"row": 2, "chunk_id": "PMC3#abs.w1", "document_id": "PMC3", "title": "T3",
     "source_category": "currency-pack", "canonical_date": "2025-08-14",
     "date_precision": "day", "authority_tier_label": "clinical-practice-guideline",
     "in_currency_pack": "yes", "retracted": "no", "duplicate_of": ""},
    {"row": 3, "chunk_id": "PMC4#abs.w1", "document_id": "PMC4", "title": "T4",
     "source_category": "pubmed-abstract", "canonical_date": "2007-01-01",
     "date_precision": "day", "authority_tier_label": "", "in_currency_pack": "no",
     "retracted": "no", "duplicate_of": "PMC1#abs.w1"},
]

CHUNKS = [
    {"chunk_id": r["chunk_id"], "document_id": r["document_id"],
     "text": f"body text for {r['chunk_id']}"}
    for r in MANIFEST
]


def _write_index(tmp_path, manifest=MANIFEST, chunks=CHUNKS, meta=None, dim=DIM):
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    with open(index_dir / "index_manifest.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in manifest:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with open(index_dir / "embeddings.f32", "wb") as fh:
        for i in range(len(manifest)):
            vector = [0.0] * dim
            vector[i % dim] = 1.0
            fh.write(struct.pack(f"<{dim}f", *vector))
    payload = {"dim": dim, "production": True, "encoder": "ncbi/MedCPT-Article-Encoder",
               "vectors": len(manifest), "content_digest": "deadbeef"}
    payload.update(meta or {})
    (index_dir / "index_meta.json").write_text(json.dumps(payload), encoding="utf-8")

    chunks_path = tmp_path / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, sort_keys=True) + "\n")
    return str(index_dir), str(chunks_path)


def _corpus(tmp_path, category="pubmed-abstract", **options):
    index_dir, chunks_path = _write_index(tmp_path)
    return ThesisChunkCorpus(
        CorpusConfig(
            name=category,
            loader="thesis_chunks",
            options={"index_dir": index_dir, "chunks_path": chunks_path,
                     "source_category": category, **options},
        )
    )


def test_loads_only_the_requested_source_category(tmp_path):
    """One corpus per category is what makes balanced retrieval possible here."""
    corpus = _corpus(tmp_path, "pubmed-abstract")
    # PMC4 is in the same category but flagged as an exact duplicate.
    assert len(corpus) == 1
    assert corpus.passage(0).passage_id == "PMC1#abs.w1"

    assert len(_corpus(tmp_path, "pmc-fulltext")) == 1
    assert len(_corpus(tmp_path, "currency-pack")) == 1


def test_duplicate_chunks_can_be_kept(tmp_path):
    corpus = _corpus(tmp_path, "pubmed-abstract", drop_duplicate_chunks=False)
    assert len(corpus) == 2


def test_text_is_joined_from_the_chunk_file(tmp_path):
    """The index manifest carries no text; it must come from chunks.jsonl."""
    corpus = _corpus(tmp_path, "pmc-fulltext")
    assert corpus.passage(0).text == "body text for PMC2#abs.w1"


def test_provenance_is_carried_as_metadata(tmp_path):
    corpus = _corpus(tmp_path, "currency-pack")
    evidence = corpus.passage(0)
    assert evidence.doc_id == "PMC3"
    assert evidence.source == "currency-pack"
    assert evidence.metadata["canonical_date"] == "2025-08-14"
    assert evidence.metadata["authority_tier_label"] == "clinical-practice-guideline"
    assert evidence.metadata["in_currency_pack"] == "yes"
    # Identity fields are promoted, not duplicated into metadata.
    assert "chunk_id" not in evidence.metadata
    assert "document_id" not in evidence.metadata


def test_embedding_shards_align_with_passages(tmp_path):
    """Row i of the yielded matrix must be the vector of passage i.

    This is the invariant that stops retrieval decoding to the wrong document.
    """
    corpus = _corpus(tmp_path, "pubmed-abstract", drop_duplicate_chunks=False)
    shards = list(corpus.embedding_shards())
    matrix = numpy.vstack([m for _, m in shards])
    assert matrix.shape == (2, DIM)
    assert matrix.dtype == numpy.float32
    # PMC1 is index row 0 -> one-hot at 0; PMC4 is index row 3 -> one-hot at 3.
    assert matrix[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert matrix[1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_shards_are_chunked_without_changing_content(tmp_path):
    small = _corpus(tmp_path, "pubmed-abstract", drop_duplicate_chunks=False, shard_rows=1)
    whole = _corpus(tmp_path, "pubmed-abstract", drop_duplicate_chunks=False)
    assert len(list(small.embedding_shards())) == 2
    assert numpy.array_equal(
        numpy.vstack([m for _, m in small.embedding_shards()]),
        numpy.vstack([m for _, m in whole.embedding_shards()]),
    )


def test_shard_offsets_are_corpus_local_and_contiguous(tmp_path):
    corpus = _corpus(tmp_path, "pubmed-abstract", drop_duplicate_chunks=False, shard_rows=1)
    assert [offset for offset, _ in corpus.embedding_shards()] == [0, 1]


def test_a_stub_index_is_refused(tmp_path):
    """A stub-encoder index must never be mistaken for a real one."""
    index_dir, chunks_path = _write_index(
        tmp_path, meta={"production": False, "encoder": "stub-sha256"}
    )
    options = {"index_dir": index_dir, "chunks_path": chunks_path,
               "source_category": "pubmed-abstract"}
    with pytest.raises(ValueError, match="production=false"):
        ThesisChunkCorpus(CorpusConfig(name="pubmed-abstract", options=dict(options)))
    # ...but an explicit wiring test may opt out.
    relaxed = dict(options, require_production_index=False)
    assert len(ThesisChunkCorpus(CorpusConfig(name="pubmed-abstract", options=relaxed))) == 1


def test_manifest_out_of_row_order_is_rejected(tmp_path):
    scrambled = [dict(MANIFEST[1], row=1), dict(MANIFEST[0], row=0)]
    index_dir, chunks_path = _write_index(tmp_path, manifest=scrambled)
    with pytest.raises(ValueError, match="row order"):
        ThesisChunkCorpus(
            CorpusConfig(name="x", options={"index_dir": index_dir,
                                            "chunks_path": chunks_path,
                                            "source_category": "pmc-fulltext"})
        )


def test_truncated_embedding_file_is_rejected(tmp_path):
    index_dir, chunks_path = _write_index(tmp_path)
    path = os.path.join(index_dir, "embeddings.f32")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 4)
    with pytest.raises(ValueError, match="out of step"):
        ThesisChunkCorpus(
            CorpusConfig(name="x", options={"index_dir": index_dir,
                                            "chunks_path": chunks_path,
                                            "source_category": "pubmed-abstract"})
        )


def test_missing_chunk_text_is_rejected(tmp_path):
    index_dir, chunks_path = _write_index(tmp_path, chunks=CHUNKS[1:])
    with pytest.raises(ValueError, match="no text"):
        ThesisChunkCorpus(
            CorpusConfig(name="x", options={"index_dir": index_dir,
                                            "chunks_path": chunks_path,
                                            "source_category": "pubmed-abstract"})
        )


def test_unknown_category_names_the_ones_that_exist(tmp_path):
    index_dir, chunks_path = _write_index(tmp_path)
    with pytest.raises(ValueError, match="categories present"):
        ThesisChunkCorpus(
            CorpusConfig(name="x", options={"index_dir": index_dir,
                                            "chunks_path": chunks_path,
                                            "source_category": "textbook"})
        )


def test_missing_files_say_which_stage_builds_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_chunks"):
        ThesisChunkCorpus(
            CorpusConfig(name="x", options={"index_dir": str(tmp_path / "nope"),
                                            "chunks_path": str(tmp_path / "nope.jsonl")})
        )


def test_index_categories_reports_row_counts(tmp_path):
    index_dir, _ = _write_index(tmp_path)
    assert index_categories(index_dir) == {
        "currency-pack": 1,
        "pmc-fulltext": 1,
        "pubmed-abstract": 2,
    }


def test_build_thesis_corpora_covers_every_category(tmp_path):
    index_dir, chunks_path = _write_index(tmp_path)
    corpora = build_thesis_corpora(index_dir, chunks_path)
    assert [c.name for c in corpora] == ["currency-pack", "pmc-fulltext", "pubmed-abstract"]
    assert all(len(c) >= 1 for c in corpora)


def test_registered_under_the_config_loader_key(tmp_path):
    index_dir, chunks_path = _write_index(tmp_path)
    corpus = build_corpus(
        CorpusConfig(name="pmc-fulltext", loader="thesis_chunks",
                     options={"index_dir": index_dir, "chunks_path": chunks_path})
    )
    assert isinstance(corpus, ThesisChunkCorpus)
    assert corpus.describe()["encoder"] == "ncbi/MedCPT-Article-Encoder"


def test_balanced_retrieval_runs_over_the_thesis_corpora(tmp_path):
    """End-to-end: the adapter satisfies the retrieval stage's contract."""
    from rag2.config import RetrievalConfig
    from rag2.retrieval.balanced import balanced_retrieve, corpus_distribution

    index_dir, chunks_path = _write_index(tmp_path)
    corpora = build_thesis_corpora(index_dir, chunks_path)
    query = numpy.zeros((1, DIM), dtype=numpy.float32)
    query[0, 0] = 1.0

    pooled = balanced_retrieve(
        corpora, query, RetrievalConfig(embedding_dim=DIM, candidates_per_corpus=1)
    )
    assert corpus_distribution(pooled[0]) == {
        "currency-pack": 1,
        "pmc-fulltext": 1,
        "pubmed-abstract": 1,
    }
    assert all(e.text for e in pooled[0])


def test_shipped_thesis_config_wires_the_three_corpora():
    """configs/thesis_corpus.yaml must stay in step with the corpus categories.

    A renamed category here is silent until retrieval returns nothing, so pin it.
    """
    from rag2.config import load_config

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(os.path.join(root, "configs", "thesis_corpus.yaml"))
    corpora = config.retrieval.corpora
    assert [c.name for c in corpora] == ["pubmed-abstract", "pmc-fulltext", "currency-pack"]
    assert all(c.loader == "thesis_chunks" for c in corpora)
    assert all(c.options["require_production_index"] is True for c in corpora)
    # Baseline fidelity: MedCPT, and the paper's reading of the two [D] switches.
    assert config.retrieval.query_encoder == "ncbi/MedCPT-Query-Encoder"
    assert config.retrieval.reranker == "ncbi/MedCPT-Cross-Encoder"
    assert config.retrieval.embedding_dim == 768
    assert config.retrieval.rerank_query == "initial"
    assert config.retrieval.shard_merge == "score"
