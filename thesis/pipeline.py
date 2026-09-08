#!/usr/bin/env python3
"""The orchestrator: research query in, structured provenance-bearing result out.

Composes the stages and nothing else. Each stage lives in its own module and is
independently testable; this file's whole job is to run them in order, in the
same order for every condition, and to record what happened.

    load config
      -> open corpus            identity verified against the configured digest
      -> load query set         one population, digest recorded
      -> build retrieval        MedCPT, exact, balanced, replayable
      -> build condition        the arm, carrying its temporal policy
      -> per query: retrieve -> condition.run -> collect
      -> evaluate               one protocol across arms
      -> write run record       corpus, models, config, git, environment

The run record is written **whether or not the run is reportable**, with
``reportable: false`` and the reasons attached. A wiring test that silently
looked like a research result would be worse than one that says what it is.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .conditions.base import ConditionResult, ExperimentCondition, build_condition
from .config import ThesisConfig
from .corpus import CorpusHandle
from .evaluation import evaluate
from .provenance import RunRecord, temporal_fields_carried
from .queries import QuerySet, load_query_set
from .retrieval import RetrievalService


@dataclass
class PipelineOutcome:
    """Everything one run produced."""

    config: ThesisConfig
    query_set: QuerySet
    results: List[ConditionResult] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    record: Optional[RunRecord] = None
    output_dir: str = ""

    @property
    def reportable(self) -> bool:
        return bool(self.record and self.record.is_reportable()[0])


def run_pipeline(
    config: ThesisConfig,
    corpus: Optional[CorpusHandle] = None,
    query_set: Optional[QuerySet] = None,
    retrieval: Optional[RetrievalService] = None,
    condition: Optional[ExperimentCondition] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    write: bool = True,
) -> PipelineOutcome:
    """Run one experimental condition over one query set.

    Every collaborator can be injected, which is how the offline smoke test
    exercises the real orchestration with deterministic stand-ins for the parts
    that need model weights.
    """
    _seed(config.seed)

    corpus = corpus if corpus is not None else CorpusHandle.open(config)
    query_set = query_set if query_set is not None else load_query_set(
        config.path(config.queries.path),
        name=config.queries.name,
        version=config.queries.version,
        limit=config.queries.limit,
    )
    retrieval = retrieval if retrieval is not None else RetrievalService(config)
    condition = condition if condition is not None else build_condition(config)

    output_dir = config.path(config.resolved_output_dir())
    candidates_dir = os.path.join(output_dir, "candidates")

    results: List[ConditionResult] = []
    all_candidates: List[Dict[str, Any]] = []
    for index, query in enumerate(query_set, start=1):
        retrieved = retrieval.retrieve(query)
        all_candidates.extend(retrieved.candidates)
        if write and config.output.write_candidates and not retrieved.replayed_from:
            retrieval.save(retrieved, candidates_dir)
        results.append(condition.run(query, retrieved))
        if progress:
            progress(index, len(query_set))

    report = evaluate(results, query_set, config.evaluation.metrics)
    report["condition"] = condition.describe()
    report["candidate_digest"] = _single_digest(results)

    record = RunRecord(
        run_name=config.name,
        condition=config.condition.name,
        temporal_policy=condition.policy.name,
        seed=config.seed,
        config_fingerprint=config.fingerprint(),
        retrieval_fingerprint=retrieval.fingerprint(),
        query_set=query_set.describe(),
        corpus=corpus.stamp(),
        models=retrieval.models,
        retrieval={
            "per_category": config.retrieval.per_category,
            "final_top_k": config.retrieval.final_top_k,
            "rerank": config.retrieval.rerank,
            "replayed": bool(config.retrieval.replay_from),
            # Whether a recency arm is even possible on this candidate set.
            "temporal_fields_carried": temporal_fields_carried(all_candidates),
        },
        config=config.to_dict(),
        notes=config.condition.notes,
    )

    outcome = PipelineOutcome(
        config=config, query_set=query_set, results=results,
        report=report, record=record, output_dir=output_dir,
    )
    if write:
        write_outputs(outcome)
    return outcome


def write_outputs(outcome: PipelineOutcome) -> str:
    """Write results, report and run record. Returns the output directory."""
    directory = outcome.output_dir
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "results.jsonl"), "w", encoding="utf-8", newline="\n") as h:
        for result in outcome.results:
            h.write(json.dumps(result.to_dict(), ensure_ascii=False, default=str) + "\n")

    reportable, reasons = outcome.record.is_reportable() if outcome.record else (False, ["no record"])
    report = {**outcome.report, "reportable": reportable, "not_reportable_because": reasons}
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8", newline="\n") as h:
        json.dump(report, h, indent=2, ensure_ascii=False, default=str)
        h.write("\n")

    if outcome.record:
        outcome.record.write(directory)
    return directory


def _single_digest(results: Sequence[ConditionResult]) -> str:
    """One digest identifying the evidence population, when there is exactly one."""
    from .provenance import stable_hash

    digests = [r.candidate_digest for r in results if r.candidate_digest]
    return stable_hash(digests) if digests else ""


def _seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
