"""Answer generation for the comparison. **A thin adapter, not a new generator.**

Everything here delegates to machinery the RAG2 baseline already has:

    rag2.llm.base.build_llm       backend registry ("huggingface" -> HFCausalLM)
    rag2.llm.hf.HFCausalLM        the transformers backbone, chat template,
                                  greedy decoding, batching, truncation
    rag2.generation.generate_answers   stage 4
    rag2.prompts.PromptSet.render_answer_prompt   the paper's answer prompt

Writing a second generator would break the thing the experiment is for: both
arms must be prompted identically, and the only prompt that is defensible is the
baseline's own. So this module builds **one** LLM, wraps it in **one** callable,
and hands the *same object* to both arms. Neither arm can diverge, because
neither owns a generator.

What the arms pass in is the list of Evidence their admission policy kept -- not
a pre-rendered context string. The evidence block is then rendered by
``PromptSet.render_answer_prompt``, exactly as ``rag2.generation`` does it, so
Arm A and Arm B differ in *which* snippets reach the prompt and in nothing else.

This module never fabricates an answer. If transformers is missing, the model is
missing, or the decoding configuration is invalid, it raises with a message that
names the fix. There is no fallback and no mock on the scientific path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_ROOT / "architecture"), str(_ROOT / "architecture" / "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag2.config import GenerationConfig, LLMConfig            # noqa: E402
from rag2.prompts import DEFAULT_PROMPTS, PromptSet            # noqa: E402
from rag2.schema import Evidence, Question                     # noqa: E402

#: Backends that may drive a reported run. "stub" is excluded on purpose: it is
#: the offline wiring backend and must never produce a reported answer.
SCIENTIFIC_BACKENDS = ("huggingface", "hf", "vllm", "openai")


class GeneratorUnavailable(RuntimeError):
    """The generator cannot be built here. The message must name the fix."""


@dataclass
class GeneratorSpec:
    """Everything that decides what an answer looks like. Goes in the manifest.

    Both arms are configured from one spec, so "same generator, same decoding"
    is a property of the object rather than a claim in a report.
    """

    backend: str = "huggingface"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    revision: str = ""
    dtype: str = "bfloat16"
    device: str = "auto"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    max_input_tokens: int = 0
    chat_template: bool = True
    batch_size: int = 8
    seed: int = 42
    prompt_version: str = ""

    def validate(self) -> None:
        """Reject a configuration that could not produce a defensible answer."""
        problems: List[str] = []
        if not self.backend:
            problems.append("generation.backend is empty")
        if not self.model:
            problems.append("generation.model is empty (no checkpoint to load)")
        if self.max_new_tokens < 1:
            problems.append(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature < 0:
            problems.append(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            problems.append(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_input_tokens < 0:
            problems.append(f"max_input_tokens must be >= 0, got {self.max_input_tokens}")
        if self.temperature > 0:
            problems.append(
                f"temperature={self.temperature} is non-greedy. The paper decodes at "
                "temperature 0 (appendix A.3), and a sampled run is not reproducible "
                "between arms. Set generation.temperature: 0.0, or override deliberately "
                "in the config and record why."
            )
        if problems:
            raise GeneratorUnavailable(
                "invalid generation configuration:\n  - " + "\n  - ".join(problems))

    def to_llm_config(self) -> LLMConfig:
        return LLMConfig(
            backend=self.backend, model=self.model, revision=self.revision,
            dtype=self.dtype, device=self.device, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, top_p=self.top_p,
            max_input_tokens=self.max_input_tokens, chat_template=self.chat_template,
            batch_size=self.batch_size,
        )

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(max_new_tokens=self.max_new_tokens,
                                temperature=self.temperature)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, Any]]) -> "GeneratorSpec":
        data = dict(payload or {})
        known = {k: data[k] for k in cls.__annotations__ if k in data}
        return cls(**known)


class ArmGenerator:
    """One LLM, shared by both arms. Callable as ``generator(question, evidences)``.

    Holds the resolved model identity and decoding parameters so the manifest can
    record what actually ran, not what was requested.
    """

    def __init__(self, spec: GeneratorSpec, llm: Any,
                 prompts: Optional[PromptSet] = None) -> None:
        self.spec = spec
        self.llm = llm
        self.prompts = prompts or DEFAULT_PROMPTS
        self.calls = 0

    def __call__(self, question: Question, evidences: Sequence[Evidence]) -> str:
        from rag2.generation import generate_answers

        self.calls += 1
        answers = generate_answers(
            self.llm, [question], [list(evidences)],
            config=self.spec.to_generation_config(),
            prompts=self.prompts,
            batch_size=1,
        )
        return answers[0] if answers else ""

    def render_prompt(self, question: Question, evidences: Sequence[Evidence]) -> str:
        """The exact string the model is given. Used by tests and traces."""
        return self.prompts.render_answer_prompt(question, list(evidences))

    def describe(self) -> Dict[str, Any]:
        """Resolved identity + decoding, for the run manifest."""
        described = self.llm.describe() if hasattr(self.llm, "describe") else {}
        return {
            "backend": self.spec.backend,
            "model": self.spec.model,
            "configured_revision": self.spec.revision or None,
            "resolved_revision": described.get("revision"),
            "llm_class": described.get("class"),
            "dtype": self.spec.dtype,
            "device": self.spec.device,
            "chat_template": described.get("chat_template", self.spec.chat_template),
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "max_new_tokens": self.spec.max_new_tokens,
            "max_input_tokens": self.spec.max_input_tokens,
            "greedy": self.spec.temperature == 0.0,
            "seed": self.spec.seed,
            "prompt_version": self.prompts.version,
            "prompt_fingerprint": self.prompts.fingerprint(),
        }


def _seed_everything(seed: int) -> None:
    """Seed what is present. Greedy decoding is deterministic anyway; this covers
    any stochastic path a future config might enable."""
    import random

    random.seed(seed)
    for module, call in (("numpy", "seed"), ("torch", "manual_seed")):
        try:
            imported = __import__(module)
            getattr(imported.random if module == "numpy" else imported, call)(seed)
            if module == "torch" and imported.cuda.is_available():   # pragma: no cover
                imported.cuda.manual_seed_all(seed)
        except Exception:
            pass


def build_generator(spec: GeneratorSpec, prompts: Optional[PromptSet] = None,
                    llm: Optional[Any] = None) -> ArmGenerator:
    """Build the one generator both arms share.

    ``llm`` is an injection point for tests: it bypasses model loading while
    still exercising the real prompt path. It is never used on the scientific
    path -- ``run_comparison.py`` does not expose it.

    Raises :class:`GeneratorUnavailable` with an actionable message when the
    backend, its dependencies, or the checkpoint are missing.
    """
    spec.validate()
    _seed_everything(spec.seed)

    if llm is not None:
        return ArmGenerator(spec, llm, prompts)

    if spec.backend not in SCIENTIFIC_BACKENDS:
        raise GeneratorUnavailable(
            f"generation.backend {spec.backend!r} may not drive a reported run.\n"
            f"  Allowed: {', '.join(SCIENTIFIC_BACKENDS)}.\n"
            "  'stub' is the offline wiring backend and must never produce a "
            "reported answer."
        )

    if spec.backend in ("huggingface", "hf"):
        _require_transformers(spec)

    from rag2.llm.base import build_llm

    try:
        built = build_llm(spec.to_llm_config())
    except GeneratorUnavailable:
        raise
    except OSError as exc:
        # transformers raises OSError for "repo not found" / "not a local folder"
        # / gated-repo-without-token. All three need the same three answers.
        raise GeneratorUnavailable(
            f"could not load the generator checkpoint {spec.model!r}"
            + (f" at revision {spec.revision!r}" if spec.revision else "") + ".\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  Check, in this order:\n"
            "   1. the name is right, or the path exists if it is a local folder;\n"
            "   2. the model is downloaded, or the machine is online;\n"
            "   3. for a gated repo (Llama-3 is gated) you have accepted its licence\n"
            "      and are authenticated: huggingface-cli login\n"
            "  This is an environment problem, not a code problem -- the comparison "
            "will not substitute another model."
        ) from exc
    except Exception as exc:
        raise GeneratorUnavailable(
            f"failed to build the {spec.backend!r} generator for {spec.model!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return ArmGenerator(spec, built, prompts)


def _require_transformers(spec: GeneratorSpec) -> None:
    """Turn two import errors into one message that says what to install."""
    missing: List[str] = []
    for module in ("torch", "transformers"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if not missing:
        return
    raise GeneratorUnavailable(
        f"the 'huggingface' generator needs {' and '.join(missing)}, "
        f"which {'is' if len(missing) == 1 else 'are'} not installed.\n"
        "  On the GPU machine:\n"
        "    pip install torch==2.4.1+cu121 --index-url "
        "https://download.pytorch.org/whl/cu121\n"
        "    pip install transformers accelerate sentencepiece\n"
        f"  Then re-run. The comparison will not fall back to another backend, and "
        f"{spec.model!r} will not be silently replaced."
    )
