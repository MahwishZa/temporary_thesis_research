# Evidence-quality annotation -- instructions

You are judging ONE thing: does this passage help answer this question?

You are NOT judging whether the passage is true, whether it is recent, whether
it is from a good journal, or whether the system should have admitted it. Judge
only usefulness as evidence for THIS question.

## The scale

  0  Not relevant -- does not help answer the question.
  1  Partially relevant -- weak, indirect, or background support. It touches the
     topic, or supports part of the question, but does not answer it.
  2  Clearly relevant -- useful supporting evidence. A reader answering the
     question would want to see this passage.

Put the number in the `human_label` field. Leave `human_notes` empty unless
something is worth recording.

## Hard cases -- decide them this way

* **Medically related but off-question.** A passage about Alzheimer's that does
  not bear on what was asked is **0**, not 1. "About the same disease" is not
  relevance.
* **Useful background that does not answer the question.** Definitions,
  epidemiology, general mechanism -- **1**.
* **Contradicts other evidence, or contradicts what you believe.** Judge
  relevance only. A passage that directly addresses the question but reports a
  contrary finding is still **2**. Disagreement is not irrelevance.
* **Partly relevant, partly off-topic.** Judge the part that bears on the
  question. If that part is useful, **2**; if it is thin, **1**.
* **Truncated mid-sentence.** Judge what is there. Do not guess the rest.
* **You genuinely cannot tell.** Choose **1** and say why in `human_notes`.
  Do not leave it blank -- a blank means "not yet done" and blocks the analysis.

## Two rules that protect the result

1. **Do not look up the machine scores.** They are deliberately kept in a
   separate key file. If you judge with the score in view, the labels partly
   measure your agreement with the score, and the whole exercise collapses.
2. **Do not skip rows you find hard.** Skipped hard rows bias the sample toward
   easy ones. Label them 1 with a note.

Roughly 1-2 minutes per row. You can stop and resume; run `check` to see
progress.
