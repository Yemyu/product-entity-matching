# Bilingual static research site

**English** | [简体中文](README.zh-CN.md)

Open [English](en/index.html) or [简体中文](zh-CN/index.html). Each language has
four complete HTML pages: overview, data and method, results, and reproduction.
The language link goes to the corresponding page. With JavaScript enabled it
also retains the current anchor, AP/F1 selection, teaching scenario and open model/configuration panel. Text,
figures and tables can be read without JavaScript; both metric plots and all
three teaching scenarios then remain visible. JavaScript adds scenario and
metric selection, section navigation and command copying. On the data and method page,
the Model card and Final configuration buttons expand readable panels below the buttons;
click again to close or press Escape while focus is in the panel. Without JavaScript,
both panels remain visible and the buttons link to their headings.


The overview starts with the matching task and synthetic field examples, then
explains the scoring procedure, saved results and code entry points. The method
page introduces the data source and project roles before normalization and the
model; token budgets and training settings can be expanded for detail. Technical
document links open rendered repository documents and are labeled **GitHub**.
Result limitations are readable within the Results page; CSV and JSON links are
explicit downloads of machine-readable files.

Create the project environment in the [README quickstart](../README.md#quickstart), then run from the repository root:

macOS / Linux:

```bash
.venv/bin/python scripts/build_showcase.py
.venv/bin/python scripts/build_showcase.py --check
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

Windows / PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/build_showcase.py
.\.venv\Scripts\python.exe scripts/build_showcase.py --check
.\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1
```

The first command generates eight HTML files and `assets/fixed-results.csv`.
The check command renders them in memory and compares the bytes, without
writing files. The site reads `results/fixed-results.json`,
`reproducibility/FINAL_MODEL.json`, and `reproducibility/FEATURES.json` locally
at build time. Three invented record pairs are passed through the public
normalization and feature functions to produce the teaching tables. The browser
switches between those precomputed tables; it does not calculate a model score. No IDF
is fitted, no neural model is run, and no network request is made. Open
`http://127.0.0.1:8000/showcase/en/index.html` for local browser inspection.

Edit `content/en.json` and `content/zh-CN.json` for wording, `templates/page.html`
for the shared page shell, or `assets/site.css` for local styling; rebuild after
changing generated content. Numeric values come from the public JSON sources.
The standard-library builder preserves the feature order and unrounded CSV
values. It does not alter the source facts.

## GitHub Pages

The [Pages workflow](../.github/workflows/pages.yml) runs the public test suite,
verifies the generated pages and publishes on pushes to `main`. It uses Python's
standard library and does not install or run model dependencies. The repository
must select **GitHub Actions** as its Pages publishing source. The root address
opens the Chinese overview; the language link switches to the matching English page.
`.venv/bin/python scripts/build_pages.py` (or `.\.venv\Scripts\python.exe scripts/build_pages.py`
on Windows) creates the same deployment tree in `dist/pages`
from the verified public inventory, including the files linked from the pages.

See [NOTICE.md](NOTICE.md) for Academic/Nerfies attribution and the separate
CC BY-SA 4.0 website and MIT model-code scopes. `UPSTREAM.json` records the
verified pinned template source. Bulma v0.9.1 is local and has its own MIT notice.

The site presents saved aggregate results and synthetic field examples. Model prediction runs through the CLI and requires local product data, snapshots and trained weights. The Notebook walkthrough is linked at the top of the repository README.
