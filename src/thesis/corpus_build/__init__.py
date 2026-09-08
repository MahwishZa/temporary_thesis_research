#!/usr/bin/env python3
"""Construction of the frozen Alzheimer's evidence corpus, one stage per module.

The corpus is built once and then frozen; these modules are the record of how.
They are ordered, each stage consuming the previous stage's committed output:

    acquisition -> parsing -> qc -> metadata -> chunking -> embedding -> indexing

``acquisition``  PubMed E-utilities search; PMC open-access inventory; MD5-verified
                 JATS XML download.
``parsing``      JATS XML -> one structured JSON record per article.
``qc``           independent detectors and the integrity gates that must pass
                 before a downstream stage is allowed to consume its input.
``metadata``     the M1-M4 corpus policy overlays: canonical dates, eligibility,
                 the CPG layer, the currency pack.
``chunking``     deterministic 256-word / 32-word-overlap retrieval units that
                 never cross a section boundary.
``embedding``    MedCPT article-encoder vectors plus the aligned row manifest.
``indexing``     exact inner-product search, balanced retrieval, reranking, and
                 the fingerprinted candidate replay that validity control V3
                 depends on.

A stage never imports a later stage. Where an earlier stage's helper is needed
the import is explicit and one-directional (``embedding`` reuses ``chunking``'s
``compose_embed_text``, so that what is indexed and what is scored can never
drift apart).
"""
