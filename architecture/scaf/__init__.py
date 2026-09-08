"""SCAF -- the thesis's evidence-admission policy. **Not part of the RAG2 baseline.**

This package sits *above* ``rag2/``: it imports the baseline's interfaces and
never modifies them. The dependency runs one way only, which is what keeps the
baseline a clean comparison point:

    scaf/   thesis extension     imports rag2, is never imported by it
    rag2/   original RAG2        imports neither scaf nor pmc
    pmc/    corpus + index       imports neither

It lives outside ``rag2/rag2/`` deliberately. That tree is guarded by
``rag2/tests/test_metadata_isolation.py``, which fails the build if any baseline
module so much as names a publication-date or currency field in executable code.
SCAF's whole purpose is to read those fields, so it belongs on this side of the
line -- and the guard staying green is the evidence that the baseline it is
compared against is still untouched.

Importing this package registers the ``scaf`` filter with the baseline's filter
registry, so ``rag2.filtering.base.build_filter(FilterConfig(kind="scaf"))``
resolves once ``import scaf`` has run.
"""

from .policy import (  # noqa: F401
    DEFAULT_ADMIT_THRESHOLD,
    DEFAULT_AUTHORITY_TIERS,
    DEFAULT_CATEGORY_AUTHORITY,
    DEFAULT_HALF_LIFE_YEARS,
    DEFAULT_SUPERSEDED_DISCOUNT,
    DEFAULT_WEIGHTS,
    AuthorityScorer,
    CurrencyScorer,
    SCAFDecision,
    SCAFFilter,
    SupportScorer,
    tokenize,
)

__all__ = [
    "SCAFFilter", "SCAFDecision", "SupportScorer", "CurrencyScorer",
    "AuthorityScorer", "tokenize", "DEFAULT_WEIGHTS", "DEFAULT_AUTHORITY_TIERS",
    "DEFAULT_CATEGORY_AUTHORITY", "DEFAULT_HALF_LIFE_YEARS",
    "DEFAULT_SUPERSEDED_DISCOUNT", "DEFAULT_ADMIT_THRESHOLD",
]
