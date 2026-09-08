"""Generator INTERFACE tests. These do NOT execute a real model.

    python3 -m pytest architecture/scaf/tests/test_generation.py

torch, transformers and the 8B checkpoint are absent from this container, so
nothing here loads a model. What is tested is the contract around it: that both
arms receive the *same* generator object, that the prompt handed to the model is
the baseline's own answer prompt over the admitted evidence, that decoding
settings propagate, that resolved identity reaches the manifest, and that every
missing dependency produces an actionable error instead of a downgrade.

A fake LLM is injected through ``build_generator(..., llm=...)``, which the
scientific path does not expose. Real-model execution is a Windows/GPU concern
and is explicitly out of scope here -- see docs/runbooks/windows_experiment_runbook.md.
"""

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_ROOT, "architecture"),
           os.path.join(_ROOT, "architecture", "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag2.config import FilterConfig
from rag2.schema import Evidence, Question

from scaf.compare import fairness_report, run_arm
from scaf.generation import (
    SCIENTIFIC_BACKENDS,
    ArmGenerator,
    GeneratorSpec,
    GeneratorUnavailable,
    build_generator,
)
from scaf.policy import SCAFFilter
from scaf.tests.test_frozen_and_compare import AlwaysFilter, make_set


class FakeLLM:
    """Records the prompts it is given and returns a deterministic answer.

    Stands in for HFCausalLM at the same interface (``generate`` + ``describe``).
    It proves the wiring; it proves nothing about model behaviour.
    """

    def __init__(self, revision="abc123"):
        self.prompts = []
        self.kwargs = []
        self.revision = revision

    def generate(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        self.kwargs.append(kwargs)
        return [f"ANSWER({len(p)})" for p in prompts]

    def describe(self):
        return {"name": "fake/model", "revision": self.revision, "class": "FakeLLM",
                "chat_template": True}


def spec(**kw):
    base = dict(backend="huggingface", model="meta-llama/Meta-Llama-3-8B-Instruct",
                temperature=0.0, max_new_tokens=64, seed=42)
    base.update(kw)
    return GeneratorSpec(**base)


def question(text="Which biomarker is used for early Alzheimer diagnosis?"):
    return Question(qid="q1", question=text, options={}, answer=None, metadata={})


def evidences(n=2):
    return [Evidence(text=f"evidence snippet {i}", source="pmc-fulltext",
                     doc_id=f"PMC{i}", passage_id=f"PMC{i}#abs.w1") for i in range(n)]


# ---------------------------------------------------------------- config
class TestGeneratorSpec:
    def test_valid_spec_passes(self):
        spec().validate()

    @pytest.mark.parametrize("field,value,message", [
        ("model", "", "no checkpoint"),
        ("backend", "", "backend is empty"),
        ("max_new_tokens", 0, "max_new_tokens"),
        ("top_p", 0.0, "top_p"),
        ("top_p", 1.5, "top_p"),
        ("max_input_tokens", -1, "max_input_tokens"),
    ])
    def test_invalid_configuration_is_rejected(self, field, value, message):
        with pytest.raises(GeneratorUnavailable, match=message):
            spec(**{field: value}).validate()

    def test_non_greedy_decoding_is_rejected(self):
        """The paper decodes at temperature 0; a sampled run is not reproducible."""
        with pytest.raises(GeneratorUnavailable, match="non-greedy"):
            spec(temperature=0.7).validate()

    def test_from_mapping_ignores_unknown_keys(self):
        built = GeneratorSpec.from_mapping(
            {"model": "x", "temperature": 0.0, "not_a_field": 1, "prompt_version": "v1"})
        assert built.model == "x" and built.prompt_version == "v1"

    def test_spec_maps_onto_the_baseline_config_objects(self):
        s = spec(max_new_tokens=128, max_input_tokens=2048, dtype="float16")
        llm_config = s.to_llm_config()
        assert llm_config.model == s.model
        assert llm_config.max_new_tokens == 128
        assert llm_config.max_input_tokens == 2048
        assert llm_config.dtype == "float16"
        assert s.to_generation_config().max_new_tokens == 128


# ------------------------------------------------------- missing dependencies
class TestActionableErrors:
    def test_missing_transformers_names_the_install_command(self):
        """torch/transformers are genuinely absent here, so this is the real path."""
        pytest.importorskip  # noqa: B018  (documented: we WANT them missing)
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("torch/transformers present; the missing-dependency path "
                        "cannot be exercised here")
        with pytest.raises(GeneratorUnavailable) as excinfo:
            build_generator(spec())
        message = str(excinfo.value)
        assert "pip install torch" in message
        assert "transformers" in message
        assert "will not fall back" in message

    def test_a_non_scientific_backend_is_refused(self):
        """'stub' must never produce a reported answer."""
        with pytest.raises(GeneratorUnavailable, match="may not drive a reported run"):
            build_generator(spec(backend="stub"))
        assert "stub" not in SCIENTIFIC_BACKENDS

    def test_invalid_config_is_caught_before_any_model_load(self):
        with pytest.raises(GeneratorUnavailable, match="max_new_tokens"):
            build_generator(spec(max_new_tokens=-5))


# ------------------------------------------------------------- the contract
class TestGeneratorContract:
    def _generator(self, **kw):
        llm = FakeLLM()
        return build_generator(spec(**kw), llm=llm), llm

    def test_injected_llm_bypasses_loading_and_still_uses_the_real_prompt(self):
        generator, llm = self._generator()
        answer = generator(question(), evidences(2))
        assert answer.startswith("ANSWER(")
        assert len(llm.prompts) == 1
        prompt = llm.prompts[0]
        # The baseline's answer prompt, with the evidence block.
        assert "multiple choice questions about medical knowledge" in prompt
        assert "Here are the retrieved documents:" in prompt
        assert "evidence snippet 0" in prompt and "evidence snippet 1" in prompt

    def test_no_evidence_uses_the_closed_book_prompt(self):
        generator, llm = self._generator()
        generator(question(), [])
        assert "Here are the retrieved documents:" not in llm.prompts[0]
        assert "Here is the question:" in llm.prompts[0]

    def test_only_admitted_evidence_reaches_the_prompt(self):
        generator, llm = self._generator()
        generator(question(), evidences(3)[:1])
        assert "evidence snippet 0" in llm.prompts[0]
        assert "evidence snippet 1" not in llm.prompts[0]

    def test_decoding_parameters_propagate_to_the_backend(self):
        generator, llm = self._generator(max_new_tokens=77, temperature=0.0)
        generator(question(), evidences(1))
        assert llm.kwargs[0]["max_new_tokens"] == 77
        assert llm.kwargs[0]["temperature"] == 0.0

    def test_render_prompt_matches_what_the_backend_receives(self):
        generator, llm = self._generator()
        q, ev = question(), evidences(2)
        rendered = generator.render_prompt(q, ev)
        generator(q, ev)
        assert rendered == llm.prompts[0]

    def test_describe_records_resolved_identity_for_the_manifest(self):
        generator, _ = self._generator(revision="")
        described = generator.describe()
        assert described["backend"] == "huggingface"
        assert described["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert described["resolved_revision"] == "abc123"   # from the backend, not config
        assert described["configured_revision"] is None
        assert described["greedy"] is True
        assert described["seed"] == 42
        assert described["prompt_fingerprint"]
        assert described["prompt_version"]

    def test_generation_is_deterministic_for_the_same_inputs(self):
        q, ev = question(), evidences(2)
        first, _ = self._generator()
        second, _ = self._generator()
        assert first(q, ev) == second(q, ev)


# -------------------------------------------------- both arms, one generator
class TestBothArmsShareOneGenerator:
    def test_the_same_generator_object_serves_both_arms(self):
        llm = FakeLLM()
        generator = build_generator(spec(), llm=llm)
        frozen = [make_set("q1", n=3), make_set("q2", n=3)]

        arm_a = run_arm("rag2", AlwaysFilter(True), frozen, generator)
        arm_b = run_arm("scaf", SCAFFilter(FilterConfig(kind="scaf")), frozen, generator)

        assert generator.calls == 4                     # 2 questions x 2 arms
        assert all(r.generated for r in arm_a + arm_b)
        assert all(r.answer for r in arm_a + arm_b)
        assert all(not r.generation_error for r in arm_a + arm_b)

    def test_arms_differ_only_in_which_evidence_reaches_the_prompt(self):
        llm = FakeLLM()
        generator = build_generator(spec(), llm=llm)
        frozen = [make_set("q1", n=4)]

        run_arm("all", AlwaysFilter(True), frozen, generator)
        prompt_all = llm.prompts[-1]
        run_arm("none", AlwaysFilter(False), frozen, generator)
        prompt_none = llm.prompts[-1]

        assert prompt_all != prompt_none
        assert "Here are the retrieved documents:" in prompt_all
        assert "Here are the retrieved documents:" not in prompt_none
        # The question is identical in both -- only the evidence block moved.
        assert "Which biomarker is used for early Alzheimer diagnosis?" in prompt_all
        assert "Which biomarker is used for early Alzheimer diagnosis?" in prompt_none

    def test_the_prompt_is_recorded_in_the_trace(self):
        generator = build_generator(spec(), llm=FakeLLM())
        result = run_arm("rag2", AlwaysFilter(True), [make_set("q1", n=2)], generator)[0]
        assert result.prompt and result.prompt_chars == len(result.prompt)
        assert result.to_dict()["generated"] is True

    def test_a_generation_failure_is_recorded_not_swallowed(self):
        class Exploding:
            def __call__(self, question, evidences):
                raise RuntimeError("CUDA out of memory")

            def describe(self):
                return {}

        results = run_arm("rag2", AlwaysFilter(True), [make_set("q1")], Exploding())
        assert results[0].generation_error.startswith("RuntimeError: CUDA out of memory")
        assert results[0].generated is False
        # The admission decisions survive: they are valid regardless.
        assert results[0].num_candidates == 4
        assert not results[0].error

    def test_fairness_sees_identical_generation_settings(self):
        llm = FakeLLM()
        generator = build_generator(spec(), llm=llm)
        frozen = [make_set("q1")]
        a = run_arm("rag2", AlwaysFilter(True), frozen, generator)
        b = run_arm("scaf", SCAFFilter(FilterConfig(kind="scaf")), frozen, generator)
        described = generator.describe()
        report = fairness_report(frozen, a, b, {
            "arm_a_generator": "huggingface", "arm_b_generator": "huggingface",
            "arm_a_generation": described, "arm_b_generation": described,
            "arm_a_index": "i", "arm_b_index": "i",
        })
        assert report["all_passed"], report["failed"]

    def test_summary_counts_generated_answers_and_failures(self):
        from scaf.compare import summarise

        generator = build_generator(spec(), llm=FakeLLM())
        results = run_arm("rag2", AlwaysFilter(True),
                          [make_set("q1"), make_set("q2")], generator)
        summary = summarise(results)
        assert summary["answers_generated"] == 2
        assert summary["generation_failures"] == 0
        assert summary["mean_prompt_chars"] > 0


class TestNoMockOnTheScientificPath:
    def test_build_generator_does_not_accept_an_llm_from_the_cli(self):
        """The injection point exists for tests only; the runner must not expose it.

        Read as text rather than imported: the runner is a CLI script under
        experiments/, not an importable module of this package.
        """
        path = os.path.join(_ROOT, "experiments", "scripts", "run_comparison.py")
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        assert "llm=" not in source, "the runner must never inject a fake LLM"

    @pytest.mark.parametrize("backend", ["stub", "mock", "fake", "echo", ""])
    def test_scientific_run_refuses_a_non_scientific_backend_early(self, backend, tmp_path):
        """A fake backend must be refused before torch or a checkpoint is loaded.

        build_generator() refuses it too, but only after the filter is built --
        on the GPU machine that means loading torch and a multi-GB checkpoint
        first. The runner checks it alongside the other cheap gates so 'stub'
        fails in the same second that 'none' does.
        """
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"model_type": "t5"}', encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"\x00" * 64)

        result = subprocess.run(
            [sys.executable,
             os.path.join(_ROOT, "experiments", "scripts", "run_comparison.py"),
             "--scientific", "--rag2-filter", "rag2_perplexity",
             "--rag2-checkpoint", str(checkpoint), "--generator", backend],
            capture_output=True, text=True, cwd=_ROOT)
        combined = result.stdout + result.stderr
        assert result.returncode != 0, f"{backend!r} was not refused:\n{combined}"
        assert "will not run with --generator" in combined or "requires a real generator" in combined
        # Refused before any model machinery was touched.
        assert "torch" not in combined.lower() or "will not run" in combined

    def test_huggingface_is_not_refused_by_the_backend_gate(self, tmp_path):
        """The gate must reject fakes without also blocking the real backend."""
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"model_type": "t5"}', encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"\x00" * 64)

        result = subprocess.run(
            [sys.executable,
             os.path.join(_ROOT, "experiments", "scripts", "run_comparison.py"),
             "--scientific", "--rag2-filter", "rag2_perplexity",
             "--rag2-checkpoint", str(checkpoint), "--generator", "huggingface"],
            capture_output=True, text=True, cwd=_ROOT)
        combined = result.stdout + result.stderr
        assert "will not run with --generator" not in combined, combined

    def test_arm_generator_requires_a_real_backend_object(self):
        generator = ArmGenerator(spec(), FakeLLM())
        assert generator.llm is not None


class TestTrainingPathDependencies:
    """The RAG2 filter-training path must declare everything it imports.

    04_train_filter.py shells out to the authors' classifier/run_classifier.py,
    so that script's module-level imports are real requirements of this
    repository even though no file here imports them. nltk was missing from
    requirements.txt and would have failed filter training on the GPU machine at
    the first import.
    """

    def _requirements(self):
        path = os.path.join(_ROOT, "architecture", "rag2", "requirements.txt")
        with open(path, encoding="utf-8") as fh:
            return [line.strip() for line in fh
                    if line.strip() and not line.strip().startswith("#")]

    def _declared(self):
        import re
        return {re.split(r"[=<>!\[; ]", line)[0].lower() for line in self._requirements()}

    @pytest.mark.parametrize("package", ["torch", "transformers", "datasets",
                                         "accelerate", "nltk", "numpy"])
    def test_training_script_imports_are_declared(self, package):
        assert package in self._declared(), (
            f"classifier/run_classifier.py imports {package} at module level but "
            "architecture/rag2/requirements.txt does not declare it; filter training would fail "
            "on a fresh environment")

    def test_the_authors_script_still_imports_what_we_claim(self):
        """If upstream changes, this guard must be updated rather than drift."""
        path = os.path.join(_ROOT, "architecture", "rag2", "classifier", "run_classifier.py")
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        for package in ("import torch", "import nltk", "import datasets",
                        "from accelerate import Accelerator"):
            assert package in source, f"expected {package!r} in run_classifier.py"


class TestCheckpointIdentity:
    """The manifest must identify WHICH trained filter produced a result."""

    def _checkpoint(self, tmp_path, config=True, weights=True):
        from scaf.compare import checkpoint_identity
        directory = tmp_path / "ckpt"
        directory.mkdir(exist_ok=True)
        if config:
            (directory / "config.json").write_text('{"model_type": "t5"}', encoding="utf-8")
        if weights:
            (directory / "model.safetensors").write_bytes(b"\x00" * 128)
        return checkpoint_identity(str(directory))

    def test_a_trained_checkpoint_is_identified(self, tmp_path):
        identity = self._checkpoint(tmp_path)
        assert identity["exists"] and identity["looks_trained"]
        assert identity["has_config_json"]
        assert identity["weight_files"] == ["model.safetensors"]
        assert len(identity["file_listing_digest"]) == 64
        assert identity["newest_mtime"]

    def test_weights_without_config_is_not_a_trained_checkpoint(self, tmp_path):
        assert self._checkpoint(tmp_path, config=False)["looks_trained"] is False

    def test_config_without_weights_is_not_a_trained_checkpoint(self, tmp_path):
        assert self._checkpoint(tmp_path, weights=False)["looks_trained"] is False

    def test_missing_checkpoint_is_reported_not_crashed(self):
        from scaf.compare import checkpoint_identity
        assert checkpoint_identity(None)["exists"] is False
        assert checkpoint_identity("")["reason"] == "no checkpoint configured"
        assert checkpoint_identity("/no/such/dir")["reason"] == "not a directory"

    def test_digest_changes_when_the_checkpoint_changes(self, tmp_path):
        from scaf.compare import checkpoint_identity
        first = self._checkpoint(tmp_path)
        (tmp_path / "ckpt" / "model.safetensors").write_bytes(b"\x01" * 999)
        second = checkpoint_identity(str(tmp_path / "ckpt"))
        assert first["file_listing_digest"] != second["file_listing_digest"], (
            "a re-trained checkpoint must be distinguishable in the manifest")
