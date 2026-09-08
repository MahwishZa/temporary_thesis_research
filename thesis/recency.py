#!/usr/bin/env python3
"""The temporal-policy boundary.

This is the seam the thesis's recency work will attach to, and **it is a seam,
not a method.** No policy here scores, reweights, filters or reorders evidence by
publication date. The thesis has not yet fixed its recency algorithm, and
inventing one so the architecture looks complete would put an unjustified method
in the baseline's comparison set and misrepresent an interface as a contribution.

So what exists is:

* ``TemporalPolicy`` -- the interface a future policy implements;
* ``NullTemporalPolicy`` -- the baseline: dates are carried, never read;
* ``UnimplementedTemporalPolicy`` -- a named, configurable placeholder that
  **raises when applied**, so a run configured for a policy that does not exist
  fails loudly instead of silently producing baseline numbers under a recency
  label;
* a registry, so a real policy is added by registering it and naming it in
  config -- no restructuring.

The temporal fields the corpus already carries and a policy will read are
``canonical_date``, ``date_precision`` and ``split_june_2024`` (see
``thesis.provenance.CARRIED_TEMPORAL_FIELDS``). They travel through retrieval
untouched; ``thesis/tests/test_condition_isolation.py`` proves the baseline path
never reads them.

Design note on why the boundary is *here* and not inside the RAG2 filter: the
thesis measures what the original filter does with evidence of different ages.
Teaching that filter about dates would destroy the thing being measured. A
temporal policy therefore acts on the candidate set around the filter, and the
filter itself stays byte-identical to the reproduction.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


class TemporalPolicyError(RuntimeError):
    """Raised when a configured temporal policy cannot be applied."""


class TemporalPolicy(abc.ABC):
    """Decides how publication dates affect a candidate set.

    A policy receives the retrieved candidates *after* retrieval and *before* the
    condition's admission step, and returns a candidate list. It may reorder,
    annotate or drop; it must not fabricate evidence, and it must preserve each
    record's corpus provenance.
    """

    name: str = ""
    #: Whether this policy reads publication dates at all. The baseline asserts
    #: this is False for its own policy, which is how "the baseline is temporally
    #: blind" becomes a checkable property rather than a claim.
    reads_dates: bool = True

    @abc.abstractmethod
    def apply(
        self, candidates: Sequence[Mapping[str, Any]], context: Optional[Mapping[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Return the candidate set this policy admits, in admission order."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "reads_dates": self.reads_dates}


class NullTemporalPolicy(TemporalPolicy):
    """The baseline: evidence passes through untouched and dates are not read.

    This is not a degenerate case to be replaced later -- it is the control arm.
    Its behaviour must remain exactly "identity", because every temporal effect
    the thesis reports is measured against it.
    """

    name = "none"
    reads_dates = False

    def apply(
        self, candidates: Sequence[Mapping[str, Any]], context: Optional[Mapping[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        return [dict(candidate) for candidate in candidates]


class UnimplementedTemporalPolicy(TemporalPolicy):
    """A named policy the thesis intends but has not yet defined.

    Constructing it is fine -- config validation and wiring tests need that.
    Applying it raises. The failure mode this prevents is a run configured as a
    recency arm quietly producing baseline numbers, which would look like "the
    recency condition made no difference" rather than "the recency condition does
    not exist yet".
    """

    reads_dates = True

    def __init__(self, name: str, reason: str = "") -> None:
        self.name = name
        self.reason = reason or (
            "the thesis has not yet fixed its recency method; this is a declared "
            "interface, not an implementation"
        )

    def apply(
        self, candidates: Sequence[Mapping[str, Any]], context: Optional[Mapping[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        raise TemporalPolicyError(
            f"temporal policy {self.name!r} is not implemented: {self.reason}. "
            "Implement it as a TemporalPolicy and register it with "
            "thesis.recency.register_policy, or run with recency.policy='none' "
            "(the baseline control) -- but do not report a run under a policy name "
            "whose behaviour does not exist."
        )

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "implemented": False, "reason": self.reason}


_REGISTRY: Dict[str, Callable[[Mapping[str, Any]], TemporalPolicy]] = {}


def register_policy(key: str):
    """Register a temporal policy under a config-selectable name."""

    def decorator(factory: Callable[[Mapping[str, Any]], TemporalPolicy]):
        if key in _REGISTRY:
            raise ValueError(f"temporal policy {key!r} already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_policies() -> List[str]:
    return sorted(_REGISTRY)


@register_policy("none")
def _build_none(options: Mapping[str, Any]) -> TemporalPolicy:
    return NullTemporalPolicy()


#: Policy names the thesis proposal anticipates. They are registered so that a
#: config naming one validates and the run *record* shows what was requested --
#: and then refuses to run, rather than silently degrading to the baseline.
PLANNED_POLICIES = {
    "recency_weighted": "weight evidence utility by age; weighting function not yet fixed",
    "currency_three_state": "the three-state currency term of the SCAF proposal",
    "supersession": "demote evidence superseded by a newer study; supersession relation not yet defined",
}

for _planned, _reason in PLANNED_POLICIES.items():
    register_policy(_planned)(
        lambda options, _n=_planned, _r=_reason: UnimplementedTemporalPolicy(_n, _r)
    )


def build_policy(name: str, options: Optional[Mapping[str, Any]] = None) -> TemporalPolicy:
    """Instantiate the configured temporal policy."""
    if name not in _REGISTRY:
        raise TemporalPolicyError(
            f"unknown temporal policy {name!r}; registered: {available_policies()}. "
            "Add one with thesis.recency.register_policy."
        )
    return _REGISTRY[name](dict(options or {}))
