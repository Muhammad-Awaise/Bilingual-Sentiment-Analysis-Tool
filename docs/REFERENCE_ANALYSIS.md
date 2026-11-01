# Reference Repository Analysis

**Inspection date:** 2026-10-27  
**Workspace:** Bilingual-Sentiment-Analysis-Tool  
**Method:** Shallow clone into `references/` for read-only inspection. No reference scripts were executed. No reference files were edited.

---

## 1. Repository inventory

| # | Remote URL | Local path | Cloned? |
|---|------------|------------|---------|
| 1 | https://github.com/ahsaanjameel595/roman_urdu_sentiment_analysis | `references/roman_urdu_sentiment_analysis/` | Yes |
| 2 | https://github.com/Gardezi-Enterprises/AIML-Bilingual-Sentiment-Analysis-of-Daraz-Reviews | `references/AIML-Bilingual-Sentiment-Analysis-of-Daraz-Reviews/` | Yes |

---

## 2. Reference 1 — Roman Urdu Sentiment Analysis

### Purpose
Streamlit app that classifies **Roman Urdu** text as **positive** or **negative** using TF-IDF features and classical ML (Naive Bayes, Logistic Regression, LinearSVC).

### Entry points
| File | Role |
|------|------|
| `app.py` | Streamlit UI; loads joblib models + TF-IDF |
| `train.py` | Load CSV → preprocess → split → train → save models/plots |
| `preprocess.py` | Text cleaning (lowercase, strip URLs/mentions, remove stopwords) |
| `urdu-sentiment-analysis.zip` | Bundled copy of data, pretrained pickles, plots |

### Dependencies (`requirements.txt`)
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib` (version floors only).

### Dataset (as present in the zip)
| Property | Value |
|----------|-------|
| File | `RomanUrduSentiment.csv` (~1.4 MB) |
| Stated source | Kaggle — [owaisraza009/roman-urdu-dataset](https://www.kaggle.com/datasets/owaisraza009/roman-urdu-dataset) |
| Approx. rows | 11,000 |
| Labels observed | `pos` (5699), `neg` (5300), 1 empty |
| Language | Roman Urdu (Latin-script Urdu), social / media style |
| Schema | No header; columns treated as `[sentiment, text]` |

### Preprocessing (important for linguistics)
Their stopword list **removes** `nahi`, `nahin`, `na`, `mat`, and contrast marker `lekin`. That erases negation and contrast cues central to this project's research question. **We do not reuse that stopword policy.**

### Models / artifacts
Pretrained `*.pkl` files exist inside the zip. Treated as **untrusted third-party artifacts** — not used by our app.

### License / provenance notes
- README claims MIT and links to `LICENSE`, but **no LICENSE file** was present in the clone.
- GitHub API returned `license: null` for this repository.
- **Dataset license is separate** from the code license and was **not verified** for redistribution via this repo.
- **Decision:** Do not copy their application code. Do not commit their CSV or pickles into our public tree. Architectural ideas only (TF-IDF + Logistic Regression + Streamlit).

### Useful ideas (not copied code)
- TF-IDF with unigrams + bigrams
- Compare Logistic Regression vs LinearSVC
- Separate train script from Streamlit app
- Persist vectorizer + model with joblib

### Problems / risks
- Missing enforceable open-source license file in clone
- Unclear dataset redistribution rights
- Stopword policy destroys negation/contrast
- README metrics are **their** results — never claim as ours

---

## 3. Reference 2 — AIML Bilingual Daraz Reviews

### Purpose
Sentiment classification for **code-mixed Urdu/English product reviews**, served by a minimal Streamlit app with a pickled SVC + TF-IDF vectorizer.

### Entry points
| File | Role |
|------|------|
| `main.py` | Streamlit: load pickles, predict positive/negative |
| `Jupyter file.ipynb` | Merging, balancing, training (not executed) |
| `svcModel(ok).pickle`, `vectorizer.pickle` | Serialized inference artifacts |
| `Urdustopwords.txt` | Custom Urdu stopword list |
| `Daraz FYP Dataset.csv` | Reviews with Sentiment + Date |
| `daraz-code-mixed-product-reviews.csv` | Code-mixed reviews |

### Dependencies
No `requirements.txt` in the repo root. README mentions Python, scikit-learn, pandas, Streamlit, Matplotlib. `how_to_run_web_app.txt` only says to install Streamlit.

### Datasets observed

**A. `Daraz FYP Dataset.csv` (~959 KB)**  
- Header: `Review`, `Sentiment`, `Date`  
- Rows: 8,000  
- Labels: positive 4443, negative 2457, **neutral 1100**  
- Content: bilingual / Roman Urdu product-style reviews  

**B. `daraz-code-mixed-product-reviews.csv` (~1.5 MB)**  
- Header: `Sentiments`, `Reviews`  
- Rows: 16,990  
- Labels: positive 10167, negative 4362, **neutral 2461**  
- Content: English, Roman Urdu, and code-mixed product reviews  

### Preprocessing / training
Notebook-based (not executed). README describes undersampling, bilingual stopwords, TF-IDF, SVC. Pickle loading in `main.py` applies **no** cleaning before `vectorizer.transform`.

### License / provenance notes
- **No LICENSE file** in the repository.
- GitHub API returned `license: null`.
- Dataset provenance (scrape vs. public release, terms of Daraz.pk, redistribution rights) is **not documented** in the README.
- Pickled models are untrusted binary blobs — **not used**.

### Useful ideas
- Product-review domain matches our research question better than general social posts
- Code-mixing is first-class in the data
- Neutral exists → three-class is possible **if** we legally use this data and keep label definitions consistent

### Problems / risks
- No clear code or dataset license
- Large CSVs should not be republished without rights
- Label casing differs across files (`Positive` vs `positive`)
- Neutral class present — must not silently mix with binary-only sources without documenting exclusions

---

## 4. Cross-cutting decisions for our project

| Topic | Decision |
|-------|----------|
| Copy apps? | **No** — original Streamlit app and `src/` package |
| Use their pickles? | **No** |
| Use their stopword lists as-is? | **No** — preserve negation/contrast |
| Commit their CSVs to GitHub? | **No** — unclear redistribution rights |
| Local experimental use? | Optional: `scripts/prepare_dataset.py` may read reference CSVs **locally** into `data/processed/` when the user opts in; large outputs stay gitignored |
| Primary shippable dataset | Original curated bilingual product-review CSV under `data/raw/` with clear authorship and binary labels |
| Classification task | **Binary** (positive / negative) for the deliverable; Neutral rows from Daraz are excluded if that optional path is used |
| Reported metrics | Only metrics we compute ourselves in `reports/` |

---

## 5. Unavailable or blocked items

| Item | Status |
|------|--------|
| Clear MIT LICENSE file in Ref 1 clone | Missing despite README claim |
| Any LICENSE in Ref 2 | Missing |
| Verified Kaggle dataset redistribution terms via this workspace | Not independently verified; treat as non-redistributable unless user confirms |
| Execution of reference train/notebook scripts | Skipped (untrusted) |

**Conclusion:** References are useful for architecture and for understanding bilingual product-review challenges. They are **not** a safe source of redistributable data or pretrained models. The project proceeds with an original curated dataset, optional local preparation from references, and a clean independent codebase.
