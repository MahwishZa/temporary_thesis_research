# Reading the passages — a step-by-step guide

> **The first attempt has to be done again.** The interface used to show a
> computer suggestion beside every passage, and the audit found that all 120
> answers matched that suggestion exactly. That means the answers recorded
> agreement with the computer rather than an independent reading, so they cannot
> be used. Nothing was your fault — the interface was built wrong.
>
> The suggestion has been removed. This second pass covers **exactly the same
> 120 passages**, with no suggestion of any kind. Your reading is now the only
> signal, which is the whole point.
>
> The first set of answers is kept, unchanged, in `pilot_anchored/`. It is
> labelled invalid for reporting and is not deleted.

This guide is for the person doing the reading. You do not need to know anything
about the code, and you do not need to understand the words "annotation",
"SCAF" or "RAG2" to do this well.

## What you are doing, in one sentence

You will read 120 short medical passages, one at a time, and for each one press
a button saying whether it **helps answer the question shown above it**.

That is the whole job. The computer has already picked which passages to show
you. Your answers are the part the computer cannot produce, and the reason the
result means anything.

---

## What to judge, and what to ignore

Judge **only** this: does this passage help answer this question?

Ignore all of these, even though they may feel relevant:

| Do not judge | Why |
| --- | --- |
| Whether the passage is *true* | You are not fact-checking the paper. |
| Whether it is *recent* | The date is deliberately not shown to you. |
| Whether the journal is *good* | Also deliberately not shown. |
| Whether the computer *should have picked it* | That is the thing being measured. |

## The three buttons

**0 — Not relevant.** It does not help answer the question.

**1 — Partially relevant.** It touches the topic, or answers part of the
question, but does not really answer it. Background, definitions and general
facts about the disease go here.

**2 — Clearly relevant.** Someone answering the question would want to see this.

### The hard cases, decided in advance

So that you decide the same way on day one and day three:

- **About Alzheimer's, but not about what was asked** → **0**. "Same disease" is
  not relevance.
- **Useful background that does not answer the question** → **1**.
- **It contradicts other passages, or contradicts what you believe** → judge
  relevance only. If it addresses the question, it is still **2**. Disagreeing
  is not the same as being irrelevant.
- **Partly on topic, partly not** → judge the part that bears on the question.
  Useful → **2**; thin → **1**.
- **It stops in the middle of a sentence** → judge what is there. Do not guess
  the rest.
- **You genuinely cannot tell** → press **1** and type why in the notes box.
  Do not skip it. Skipping the hard ones quietly biases the result toward the
  easy ones, which is worse than an imperfect answer.

---

## There is no computer suggestion any more

The corrected pass shows you the question, the passage, and the three buttons.
Nothing else. No suggestion is displayed, and none is even calculated — so there
is nothing to agree or disagree with.

If you find yourself unsure, that is normal and it is information. Press **1**
and write a short note. Do not go looking for the old answers.

---

## Running it

Open a terminal (on Windows: PowerShell) in the repository folder.

```powershell
python experiments\scripts\annotate.py --no-suggestions --sheet experiments\results\rag2_vs_scaf_alzheimer\evidence_quality\annotation_sheet_v2.jsonl
```

That is one long line. `--no-suggestions` is what removes the computer's
opinion; `annotation_sheet_v2.jsonl` is the corrected sheet covering the same
120 passages.

A web page opens on your own computer. Nothing is uploaded and nothing leaves
the machine — the passages, the questions and your answers all stay in this
folder.

If the page does not open by itself, the terminal prints an address that looks
like `http://127.0.0.1:8765/annotate`. Copy it into your browser.

**Stopping and restarting.** Every button press is saved immediately. Close the
window whenever you like, press `Ctrl+C` in the terminal, and run the same
command again later — it starts at the first passage you have not answered.
Roughly 1–2 minutes per passage.

**Changing an answer.** Answer it again. The page tells you what you chose last
time and replaces it when you press a new button.

### Options

Add these to the command above if you need them:

```powershell
--port 9000        # if something else is already using port 8765
--no-browser       # do not open a browser window automatically
```

### Checking how far you have got

```powershell
python experiments\scripts\evidence_quality.py check --sheet experiments\results\rag2_vs_scaf_alzheimer\evidence_quality\annotation_sheet_v2.jsonl
```

Prints how many are done, how many are left, and whether anything is invalid.

---

## Two rules that protect the result

1. **Do not open `annotation_key.jsonl`.** It holds the computer's scores for
   these exact passages. If you read it, your answers partly measure your
   agreement with those scores instead of your own reading, and the experiment
   is spoiled with no way to detect it afterwards. The program refuses to open
   that file for the same reason, and it is excluded from version control so it
   cannot arrive by accident.

2. **Do not skip rows you find hard.** Press 1 and leave a note.

3. **Do not open the old answers.** They are in `pilot_anchored/`, and they are
   the computer's suggestions in all but name. Reading them would put you right
   back where the first attempt ended.

When the last passage is done, the page says so. Tell your supervisor — they run
the analysis.
