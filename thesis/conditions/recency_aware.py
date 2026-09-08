#!/usr/bin/env python3
"""Recency-aware condition: an inner arm plus a declared temporal policy.

Composition rather than a new pipeline. The recency arm is *some existing
condition* -- the baseline, or RAG2 -- with a non-null
:class:`~thesis.recency.TemporalPolicy` applied to the candidate set before
admission. That is what makes it a controlled comparison: the inner condition,
the corpus, the query set and the candidate population are all identical to the
arm it is compared against, and the temporal policy is the only difference.

    recency arm  ==  inner condition  +  temporal policy
    control arm  ==  inner condition  +  NullTemporalPolicy

Configured as::

    condition:
      name: recency
    recency:
      policy: recency_weighted        # must name a registered policy
      options: {inner: baseline}      # which condition the policy wraps

**No recency algorithm is implemented here.** The thesis has not fixed one, so
every planned policy name resolves to
:class:`~thesis.recency.UnimplementedTemporalPolicy`, which raises when applied.
Running this condition with ``policy: none`` is refused too -- that configuration
is just the control arm wearing a recency label, and reporting it as a recency
result would be a false claim about what was measured.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..queries import ResearchQuery
from ..recency import NullTemporalPolicy, TemporalPolicyError
from ..retrieval import RetrievalResult
from .base import ConditionResult, ExperimentCondition, build_condition, register_condition


class RecencyAwareCondition(ExperimentCondition):
    """Wraps an inner condition, supplying it a temporal policy."""

    name = "recency"

    def __init__(self, config, policy=None, inner: Optional[ExperimentCondition] = None) -> None:
        super().__init__(config, policy)
        if isinstance(self.policy, NullTemporalPolicy):
            raise TemporalPolicyError(
                "condition 'recency' was selected with recency.policy='none'. That is the "
                "control arm, not a recency condition: it would produce baseline numbers "
                "under a recency label. Name a real policy, or set condition.name='baseline'."
            )
        self.inner_name = str(config.recency.options.get("inner", "baseline"))
        if self.inner_name == self.name:
            raise ValueError("recency condition cannot wrap itself; set recency.options.inner")
        self._inner = inner

    def inner(self) -> ExperimentCondition:
        """The condition being wrapped, built with *this* arm's temporal policy."""
        if self._inner is None:
            from copy import deepcopy

            inner_config = deepcopy(self.config)
            inner_config.condition.name = self.inner_name
            # The inner condition receives the policy explicitly, so the wrapping
            # is visible at one point rather than smuggled through config.
            self._inner = build_condition(inner_config, policy=self.policy)
        return self._inner

    @property
    def generates(self) -> bool:  # type: ignore[override]
        return self.inner().generates

    def run(self, query: ResearchQuery, retrieved: RetrievalResult) -> ConditionResult:
        result = self.inner().run(query, retrieved)
        result.condition = f"{self.name}({self.inner_name})"
        result.temporal_policy = self.policy.name
        result.diagnostics = {
            **result.diagnostics,
            "inner_condition": self.inner_name,
            "temporal_policy": self.policy.describe(),
        }
        return result

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "inner_condition": self.inner_name}


@register_condition("recency")
def _build(config, policy=None) -> ExperimentCondition:
    return RecencyAwareCondition(config, policy)
