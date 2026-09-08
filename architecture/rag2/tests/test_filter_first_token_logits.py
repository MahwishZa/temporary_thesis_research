"""The two code paths that read the filter's decision logits must agree.

``classifier/run_classifier.py`` (lines 696-712) decides a snippet by calling
``model.generate(..., output_scores=True)`` and keeping ``scores[0]`` -- the
logits at the **first** decoded position -- then softmaxing over just the
``[HELPFUL]`` / ``[NOT_HELPFUL]`` columns.

``RAG2PerplexityFilter._first_token_logits`` computes those same logits with a
single forward pass under ``decoder_input_ids=[[decoder_start_token_id]]``,
which is far cheaper than a generate() call per snippet. The two are equivalent
because the first decoded position is a function of the encoder output and the
decoder start token alone, and nothing else has been decoded yet.

These tests pin the part of that equivalence this repository owns: that both
branches read the same position, slice the same label columns, and produce the
same keep/drop decision. The remaining step -- that HuggingFace's
``generate().scores[0]`` is itself that forward pass -- is a property of
``transformers``, not of this code; ``filter.options.use_generate: true`` exists
so the literal release path can be run if it is ever in doubt.

Requires torch (tensor semantics), not transformers: the model is a stand-in
that exposes both interfaces over the same logits.
"""

import pytest

torch = pytest.importorskip("torch")

from rag2.config import FilterConfig
from rag2.filtering.rag2_filter import RAG2PerplexityFilter, helpful_probability

HELPFUL_ID, NOT_HELPFUL_ID = 7, 9
VOCAB = 12
DECODER_START = 0


class _FakeSeq2Seq:
    """Exposes ``__call__`` and ``generate`` over one fixed logit tensor.

    ``generate`` returns the same first-position logits the forward pass does,
    which is exactly the release's assumption; the test then checks that both
    branches of ``_first_token_logits`` extract them identically.
    """

    class _Config:
        decoder_start_token_id = DECODER_START
        pad_token_id = DECODER_START

    def __init__(self, first_token_logits):
        self.config = self._Config()
        self._first = first_token_logits
        self.seen_decoder_input_ids = None

    def __call__(self, input_ids=None, attention_mask=None, decoder_input_ids=None):
        self.seen_decoder_input_ids = decoder_input_ids
        batch = input_ids.shape[0]
        # A three-position decoder output; only position 0 may be read.
        logits = torch.zeros((batch, 3, VOCAB))
        logits[:, 0, :] = self._first
        logits[:, 1:, :] = -999.0  # reading any later position would be a bug
        return type("Out", (), {"logits": logits})()

    def generate(self, input_ids=None, attention_mask=None, max_length=None,
                 return_dict_in_generate=False, output_scores=False):
        return type("Gen", (), {"scores": [self._first]})()


def _filter(model, use_generate):
    """Build the filter without loading Flan-T5 (``__init__`` needs weights)."""
    obj = RAG2PerplexityFilter.__new__(RAG2PerplexityFilter)
    obj.config = FilterConfig(kind="rag2_perplexity", checkpoint="fake")
    obj.model = model
    obj.helpful_id = HELPFUL_ID
    obj.not_helpful_id = NOT_HELPFUL_ID
    obj.use_generate = use_generate
    return obj


def _encoded(batch=2):
    return {
        "input_ids": torch.ones((batch, 5), dtype=torch.long),
        "attention_mask": torch.ones((batch, 5), dtype=torch.long),
    }


def _logits(pairs):
    """A (batch, vocab) tensor with the two label columns set as given."""
    out = torch.full((len(pairs), VOCAB), -5.0)
    for row, (helpful, not_helpful) in enumerate(pairs):
        out[row, HELPFUL_ID] = helpful
        out[row, NOT_HELPFUL_ID] = not_helpful
    return out


def test_forward_pass_and_generate_return_the_same_logits():
    first = _logits([(3.0, 1.0), (-1.0, 2.5)])
    encoded = _encoded()
    direct = _filter(_FakeSeq2Seq(first), use_generate=False)._first_token_logits(encoded)
    viagen = _filter(_FakeSeq2Seq(first), use_generate=True)._first_token_logits(_encoded())
    assert torch.equal(direct, viagen)
    assert torch.equal(direct, first)


def test_forward_pass_reads_position_zero_only():
    """Later decoder positions are poisoned; reading one would show up here."""
    first = _logits([(3.0, 1.0)])
    logits = _filter(_FakeSeq2Seq(first), use_generate=False)._first_token_logits(_encoded(1))
    assert logits.min().item() > -100.0


def test_decoder_is_started_with_the_configured_start_token():
    model = _FakeSeq2Seq(_logits([(1.0, 0.0), (0.0, 1.0)]))
    _filter(model, use_generate=False)._first_token_logits(_encoded())
    assert model.seen_decoder_input_ids.shape == (2, 1)
    assert torch.all(model.seen_decoder_input_ids == DECODER_START)


def test_pad_token_is_used_when_no_decoder_start_token_is_set():
    model = _FakeSeq2Seq(_logits([(1.0, 0.0)]))
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = 3
    _filter(model, use_generate=False)._first_token_logits(_encoded(1))
    assert torch.all(model.seen_decoder_input_ids == 3)


def test_both_paths_give_the_same_helpful_probability():
    """The decision, not just the tensor: this is what keeps or drops a snippet."""
    pairs = [(3.0, 1.0), (-1.0, 2.5), (0.0, 0.0)]
    first = _logits(pairs)
    for use_generate in (False, True):
        logits = _filter(_FakeSeq2Seq(first), use_generate)._first_token_logits(_encoded(3))
        stacked = torch.stack(
            [logits[:, HELPFUL_ID], logits[:, NOT_HELPFUL_ID]], dim=0
        )
        probs = torch.nn.functional.softmax(stacked, dim=0)[0].tolist()
        expected = [helpful_probability(h, n) for h, n in pairs]
        assert probs == pytest.approx(expected, abs=1e-6)
