# Linguistic Decisions

This note explains preprocessing and label choices in plain English for defense
in graduate-admissions interviews.

## 1. Task definition

We classify product reviews as **positive**, **negative**, or **neutral**.

**Neutral policy:** Curated Neutral rows use a simple product-review guide
(mixed, average, or factual with no clear praise/complaint). Optional reference
CSVs may still drop Neutral when preparing *binary-only* external test files.
On older binary model artifacts, low calibrated confidence can surface as
`neutral` so sparse inputs (including unseen Urdu-script tokens) are not forced
into positive/negative.

## 2. What we clean

| Step | Rationale |
|------|-----------|
| Lowercasing | Reduces spelling sparsity for Latin-script text |
| URL / email / @mention removal | Social and listing noise, not polarity |
| Keep hashtag *words* | Product tags can carry sentiment-bearing tokens |
| Keep Arabic / Urdu letters | Nastaliq reviews must not be wiped empty by Latin-only filters |
| Strip most punctuation / emoji | TF-IDF tokens stay readable; punctuation rarely helps bag-of-ngrams |
| Whitespace normalization | Stable tokenization |

## 3. What we deliberately keep

| Cue | Examples | Why |
|-----|----------|-----|
| English negation | not, no | Flips polarity (*good* vs *not good*) |
| Roman Urdu negation | nahi, nahin, mat, na | Central to the research question |
| Urdu-script negation | نہیں and related cues in source text | Kept when present in Nastaliq reviews |
| Contrast | but, lekin, magar, لیکن, مگر | Second clause often carries the reviewer's verdict |
| Intensifiers | bohat, very | Strengthens polarity |

**Contrast with Reference 1:** their stopword list removes `nahi`, `nahin`, and
`lekin`. That would hide exactly the phenomena we want to study. We do **not**
apply a large stopword purge.

## 4. Roman Urdu spelling

We do **not** force a full phonetic spelling normalizer. Aggressive mapping can
erase distinctions and invent false equivalence.

We **do** apply a **closed-set** negation spelling map for a handful of known
informal variants of *nahi*:

`nahin`, `naheen`, `nahe`, `nae`, `nai`, `naie`, `nhi`, `nhin` → `nahi`

This targets a documented probe failure (`nahin`) without rewriting the rest of
the lexicon. Other informal spellings (e.g. `qlty`, `zabrdast`) remain as-is so
error analysis can still surface sparsity issues.

## 4b. Contrast emphasis (light feature)

Contrast markers (`but`, `lekin`, `magar`, `however`, `though`) are preserved.
In addition, `clean_text` appends a `__contrast__` marker and duplicates the
**post-contrast clause** so bag-of-ngrams TF-IDF can weight the verdict side
more heavily (e.g. *acha hai lekin quality kharab*). This is a transparent,
deterministic feature — not a neural attention mechanism.

## 5. Code-switching

Mixed English–Roman Urdu reviews are kept as single strings. We do not translate
Urdu segments into English before classification. Translation would change the
object of study.

## 6. Same pipeline for train and inference

`clean_text` is wired as the TF-IDF `preprocessor` inside the saved sklearn
Pipeline. Training and the Streamlit app therefore apply **identical** cleaning.

## 7. Ethical / interpretive limits

Model output is a **label**, not a claim about a reviewer's true feelings.
Ambiguous or mixed reviews may reasonably receive either label; error analysis
documents such cases instead of forcing artificial accuracy.
