# Bilingual static research site

**English** | [简体中文](README.zh-CN.md)

Open [English](en/index.html) or [简体中文](zh-CN/index.html). Each language has
four complete HTML pages: overview, data and method, results, and reproduction.
The language link goes to the corresponding page. With JavaScript enabled it
also retains the current anchor, AP/F1 selection and teaching scenario. Text,
figures and tables can be read without JavaScript; both metric plots and all
three teaching scenarios then remain visible. JavaScript adds scenario and
metric selection, section navigation and command copying.

Create the project environment in the [README quickstart](../README.md#quickstart), then run from the repository root:

```bash
python scripts/build_showcase.py
python scripts/build_showcase.py --check
python -m http.server 8000 --bind 127.0.0.1
```

The first command generates eight HTML files and `assets/fixed-results.csv`.
The check command renders them in memory and compares the bytes, without
writing files. The site reads `results/fixed-results.json`,
`reproducibility/FINAL_MODEL.json`, and `reproducibility/FEATURES.json` locally
at build time. Three invented record pairs are passed through the public
normalization and feature functions to produce the teaching tables. No IDF
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
`python scripts/build_pages.py` creates the same deployment tree in `dist/pages`
from the verified public inventory, including the files linked from the pages.

See [NOTICE.md](NOTICE.md) for Academic/Nerfies attribution and the separate
CC BY-SA 4.0 website and MIT model-code scopes. `UPSTREAM.json` records the
verified pinned template source. Bulma v0.9.1 is local and has its own MIT notice.

The site presents saved aggregate results and synthetic field examples. Model prediction runs through the CLI and requires local product data, snapshots and trained weights. The Notebook walkthrough is linked at the top of the repository README.
