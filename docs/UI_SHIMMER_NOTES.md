# UI + Shimmer Notes (from sibling project)

**Source of truth to copy from:**  
`C:\Users\GrowMore Devs\Desktop\Projects\Urdu-English-Code-Switching Analyzer`

This is the closest sibling: also Streamlit + Python NLP research UI. Do **not** copy React/Android siblings for this app.

---

## 1. How their UI is structured

| Piece | Sibling path | Role |
|-------|--------------|------|
| Entry / home | `streamlit_app.py` | `set_page_config`, `init_app()`, CSS, dashboard |
| Shared UI helpers | `streamlit_lib/ui.py` | CSS, shimmer, headers, metrics |
| Bootstrap | `streamlit_lib/bootstrap.py` | one-time app init |
| Cached data | `streamlit_lib/cache.py` | `@st.cache_*` loaders |
| Formatters | `streamlit_lib/formatters.py` | compact metric strings |
| Pages | `pages/*.py` | Streamlit multipage sidebar (1_Analyzer, 2_Datasets, …) |

**Pattern on every page:**

```python
st.set_page_config(...)
init_app()
inject_base_css()
page_header("Title", "Short subtitle")

# optional shimmer while loading
skel = st.empty()
with skel.container():
    shimmer_block(2)          # or shimmer_metrics(4)
with st.spinner("Loading …"):
    data = cached_something()
skel.empty()                  # remove skeleton, show real UI

# then real widgets / metrics / tables
```

Home dashboard (`streamlit_app.py`) uses the same idea with `placeholder = st.empty()`.

---

## 2. How shimmer works (exact mechanism)

Not a Streamlit built-in. It is **custom HTML + CSS** injected once via `st.markdown(..., unsafe_allow_html=True)`.

### CSS (in `inject_base_css`)

- `@keyframes cs-shimmer` moves a light gradient left → right
- `.cs-shimmer` = animated grey gradient bars
- `.cs-shimmer-card` = white card shell with border/radius
- Line variants: `.sm` / `.md` / `.lg` widths + heights

### Helpers

| Helper | Use when |
|--------|----------|
| `shimmer_metrics(n)` | Placeholder for a row of `st.metric` cards |
| `shimmer_block(rows)` | Placeholder for content panels / tables |
| `with_loading(label)` | Thin wrapper → `st.spinner(label)` |

### Critical UX detail

1. Draw skeleton into `st.empty()` container  
2. Load data under `st.spinner(...)`  
3. Call `placeholder.empty()` / `skel.empty()`  
4. Render real content  

Skeleton is **temporary layout**, not left on screen after load.

---

## 3. What to port into Bilingual-Sentiment-Analysis-Tool

Current app is single-file `app.py` with radio/nav pages. Keep that for now; port the **helper module + loading pattern**, not necessarily full multipage migration on day one.

### Step A — Add a small UI module

Create e.g. `src/ui.py` (or `streamlit_lib/ui.py`) with:

- `inject_base_css()`
- `page_header(title, subtitle="")`
- `metric_row(items)`
- `shimmer_metrics(n=4)`
- `shimmer_block(rows=2)`
- `with_loading(label="Loading…")`

Copy CSS from sibling `streamlit_lib/ui.py` (lines ~87–148). Optionally rename class prefix `cs-` → `bsa-` so styles don’t clash if both apps run in one browser profile.

### Step B — Call CSS once at app start

In `app.py` `main()` / top of script (after `set_page_config`):

```python
from src.ui import inject_base_css, page_header, shimmer_block, shimmer_metrics

inject_base_css()
```

### Step C — Wrap slow paths with shimmer

**Highest value places in this project:**

1. **Evaluation page** — loading `evaluation_metrics.json`, confusion matrix, ablation/Daraz reports  
2. **Batch upload** — while `predict_dataframe` / `build_results_zip` runs (replace bare spinner)  
3. **Single review** — optional short shimmer only if model load is cold  
4. **Sidebar model status** — if artifact/metrics load is noticeable  

Example for Evaluation:

```python
skel = st.empty()
with skel.container():
    shimmer_metrics(4)
    shimmer_block(2)
with st.spinner("Loading evaluation report…"):
    metrics = _load_metrics()
skel.empty()
# then existing _render_metrics_block(...)
```

Example for Batch run:

```python
skel = st.empty()
with skel.container():
    shimmer_block(2)
with st.spinner("Running batch prediction…"):
    result = predict_dataframe(...)
    _, meta = build_results_zip(...)
skel.empty()
```

### Step D — Optional structure polish (later)

Sibling uses Streamlit **pages/** multipage. Optional later migration:

- `app.py` → home / overview  
- `pages/1_Single_Review.py`  
- `pages/2_Batch_Upload.py`  
- `pages/3_Evaluation.py`  
- `pages/4_Research_Notes.py`  

Not required to get shimmer working.

---

## 4. Design tokens they use (keep consistent)

From sibling CSS / chips:

- Page max width ~ `1200px`, top padding ~ `1.4rem`
- Shimmer greys: `#eef2f7` / `#f8fafc`
- Card border: `#e2e8f0`, radius ~ `12–14px`
- Metric tabular nums via `font-variant-numeric: tabular-nums`
- Tight headings: `letter-spacing: -0.02em`

For **this** bilingual product app, keep research-honest tone (no “AI understands emotions” claims). Shimmer is polish only — do not use it to hide failed loads; still show `st.error` / warnings.

---

## 5. Do’s and don’ts

**Do**
- Inject CSS once per page run
- Always `empty()` the skeleton after data is ready
- Pair shimmer with `st.spinner` for accessibility / clarity
- Keep helpers in one module (don’t paste CSS into every page)
- Preserve existing batch/export fixes when touching Batch UI

**Don’t**
- Leave shimmer HTML visible after content loads
- Block the UI with shimmer when work is synchronous and fast (<100ms)
- Port React/Android skeleton libraries — overkill for Streamlit
- Invent metrics behind a shimmer — still load real `reports/*.json`

---

## 6. Minimal copy checklist for implementer

1. Read sibling `streamlit_lib/ui.py` (`inject_base_css`, `shimmer_*`)  
2. Read sibling `streamlit_app.py` lines 36–43 (placeholder → load → empty)  
3. Read sibling `pages/2_Datasets.py` lines 27–32 (same pattern)  
4. Add `src/ui.py` to this repo  
5. Wire Evaluation + Batch first  
6. Manually verify: skeleton appears briefly, then real metrics/table, no leftover grey cards  

---

## 7. Reference snippets (sibling)

**Placeholder + shimmer + clear** (`streamlit_app.py`):

```python
placeholder = st.empty()
with placeholder.container():
    shimmer_metrics(4)
    shimmer_block(2)

with st.spinner("Loading dashboard…"):
    data = cached_dashboard_overview()
placeholder.empty()
```

**Core CSS idea** (`streamlit_lib/ui.py`):

```css
@keyframes cs-shimmer {
  0% { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
.cs-shimmer {
  background: linear-gradient(90deg, #eef2f7 25%, #f8fafc 37%, #eef2f7 63%);
  background-size: 800px 100%;
  animation: cs-shimmer 1.15s ease-in-out infinite;
  border-radius: 12px;
}
```

---

*Notes only — no UI code was ported into the bilingual app in this step. Implement when ready (Agent mode).*
