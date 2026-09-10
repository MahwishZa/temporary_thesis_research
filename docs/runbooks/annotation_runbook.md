# Reading the passages — a step-by-step guide

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

## The computer's suggestion

Each passage comes with a suggestion, and a one-line reason for it.

**The suggestion counts how many words from the question appear in the passage.
That is all it does.** It has no medical knowledge whatsoever. It cannot tell
that "memory decline" and "cognitive impairment" mean nearly the same thing, so
it will be confidently wrong every time a passage answers the question in
different words.

Those rows — where the suggestion is wrong and you can see that it is wrong —
are the ones where your reading is worth something. Please read the passage
before you look at the suggestion, and press the button *you* believe is right.

Both answers are stored: yours as `human_label`, the computer's as
`ai_suggested_label`. They are never mixed. If it later turns out that the two
agree on almost every row, that is treated as a warning that the suggestion was
being followed rather than as a confirmation that it was right.

---

## Running it

Open a terminal (on Windows: PowerShell) in the repository folder.

```powershell
python experiments\scripts\annotate.py
```

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

```powershell
python experiments\scripts\annotate.py --no-suggestions   # hide the suggestion
python experiments\scripts\annotate.py --port 9000        # if 8765 is in use
python experiments\scripts\annotate.py --no-browser       # do not auto-open
python experiments\scripts\annotate.py --sheet <path>     # a different file
```

`--no-suggestions` shows the same passages with the suggestion hidden. Doing a
handful of rows that way — first, before you have seen any suggestions — gives a
comparison point for how much the suggestions moved your answers.

### Checking how far you have got

```powershell
python experiments\scripts\evidence_quality.py check
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

When the last passage is done, the page says so. Tell your supervisor — they run
the analysis.
