"""Explicit public inventory; reject unexpected files rather than ship them."""
from pathlib import Path
TOP={'README.md','README.zh-CN.md','LICENSE','.gitignore','pyproject.toml','requirements-train.lock.txt','index.html'}
SCRIPTS={'matching.py','public_smoke.py','run_notebook.py','release_files.py','build_public_manifest.py','verify_public_release.py','run_tests.py','build_showcase.py','build_pages.py'}
TESTS={'test_public_release.py','test_final_cli.py','test_final_features.py','test_final_metrics.py','test_final_references.py','test_final_roles.py','test_bilingual_notebook.py','test_showcase.py','test_source_reconstruction.py'}
DOCS={'CODE_MAP.md','CORE_USAGE.md','DATA_CARD.md','LIMITATIONS.md','MODEL_CARD.md','PROJECT_CARD.md','REPRODUCIBILITY.md'}
NOTEBOOKS={'product-matching-walkthrough.ipynb','product-matching-walkthrough.zh-CN.ipynb','walkthrough_text.json'}
CORE={'__init__.py','snapshot.py','data.py','serialization.py','normalization.py','errors.py','lexical.py','cli.py','model.py','runtime.py','features.py','roles.py','references.py','semantic.py','metrics.py','source.py'}
SHOWCASE={
    'showcase/LICENSE','showcase/NOTICE.md','showcase/README.md','showcase/README.zh-CN.md',
    'showcase/UPSTREAM.json','showcase/templates/page.html',
    'showcase/content/en.json','showcase/content/zh-CN.json',
    'showcase/assets/academic.css','showcase/assets/site.css','showcase/assets/site.js','showcase/assets/favicon.svg',
    'showcase/assets/fixed-results.csv','showcase/assets/vendor/bulma.min.css','showcase/assets/vendor/BULMA-LICENSE',
    'showcase/en/index.html','showcase/en/method.html','showcase/en/results.html','showcase/en/reproduce.html',
    'showcase/zh-CN/index.html','showcase/zh-CN/method.html','showcase/zh-CN/results.html','showcase/zh-CN/reproduce.html',
}
METADATA={'FINAL_MODEL.json','EXPERIMENT_LOCK.json','ROBERTA_SNAPSHOT.json','FEATURES.json','PUBLIC_MANIFEST.json','ROLE_RECONSTRUCTION.json'}
RUNTIME={'.git','.venv','.conda','__pycache__','.pytest_cache','build','dist'}
def allowed(name):
    if name=='.github/workflows/pages.yml':return True
    parts=Path(name).parts
    if parts and parts[0]=='showcase':return name in SHOWCASE
    if len(parts)==1:return name in TOP
    if len(parts)==2:
        directory,file=parts
        return ((directory=='scripts' and file in SCRIPTS) or (directory=='tests' and file in TESTS)
            or (directory=='docs' and file in DOCS) or (directory=='results' and file=='fixed-results.json')
            or (directory=='notebooks' and file in NOTEBOOKS)
            or (directory=='reproducibility' and file in METADATA))
    return len(parts)==3 and ((parts[:2]==('src','product_matching') and parts[2] in CORE)
        or (parts[:2]==('docs','zh-CN') and parts[2] in DOCS))
def release_files(root):
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if any(part in RUNTIME or part.endswith('.egg-info') for part in rel.parts):continue
        if path.name=='.DS_Store':continue
        if path.is_symlink():raise ValueError('Release symlink: '+str(rel))
        if path.is_file():
            if not allowed(rel.as_posix()):raise ValueError('Unexpected release file: '+str(rel))
            yield path
