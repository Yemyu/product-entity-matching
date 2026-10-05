"""Check published displays against their source facts and offline build."""
import csv
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.items = []
        self.stack = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        item = {'tag': tag, 'attrs': dict(attrs), 'text': '', 'parent': self.stack[-1] if self.stack else None}
        self.items.append(item)
        if tag not in {'meta', 'link', 'br', 'hr', 'img', 'input', 'source', 'wbr'}:
            self.stack.append(item)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]['tag'] == tag:
                del self.stack[index:]
                break

    def handle_data(self, value):
        for item in self.stack:
            item['text'] += value

    def select(self, key, value=None):
        return [item for item in self.items if key in item['attrs']
                and (value is None or item['attrs'][key] == value)]


class ShowcaseTests(unittest.TestCase):
    def setUp(self):
        self.facts = json.loads((ROOT / 'results/fixed-results.json').read_text())

    def page(self, language, name):
        return Elements((ROOT / f'showcase/{language}/{name}.html').read_text(encoding='utf-8'))

    def test_reader_entry_points_explain_model_and_keep_notebooks_in_readme(self):
        for language in ('en', 'zh-CN'):
            for name in ('index', 'method', 'results', 'reproduce'):
                page = self.page(language, name)
                links = [item['attrs']['href'] for item in page.select('href')]
                self.assertFalse(any('.ipynb' in link or '/notebooks/' in link for link in links))
                definition, = page.select('class', 'model-definition')
                self.assertIn('C+', definition['text'])
                self.assertIn('RoBERTa', definition['text'])
                self.assertIn('42', definition['text'])
            reproduce = self.page(language, 'reproduce')
            self.assertEqual(len(reproduce.select('id', 'quickstart')), 1)
            for identity in ('setup-unix', 'setup-windows'):
                command, = reproduce.select('id', identity)
                self.assertIn('public_smoke.py', command['text'])
                self.assertIn('http.server', command['text'])
            readme = (ROOT / ('README.md' if language == 'en' else 'README.zh-CN.md')).read_text()
            self.assertIn('[Notebook](notebooks/', readme.split('##', 1)[0])

    def test_build_is_current_without_writing_and_has_no_model_dependency(self):
        paths = sorted((ROOT / 'showcase').rglob('*'))
        before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                  for p in paths if p.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'scripts/build_showcase.py'), '--check'],
                                    cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status": "passed"', result.stdout)
        self.assertEqual(before, {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                                  for p in paths if p.is_file()})

    def test_documents_open_on_github_and_machine_readable_files_are_downloads(self):
        github = 'https://github.com/Yemyu/product-entity-matching/blob/main/'
        for language in ('en', 'zh-CN'):
            for name in ('index', 'method', 'results', 'reproduce'):
                page = self.page(language, name)
                for link in (item for item in page.select('href') if item['tag'] == 'a'):
                    href = link['attrs']['href']
                    if href.endswith('.md'):
                        self.assertTrue(href.startswith(github), href)
                        self.assertIn('GitHub', link['text'])
                        if '/docs/' in href:
                            directory = 'docs/zh-CN/' if language == 'zh-CN' else 'docs/'
                            self.assertTrue(href.startswith(github + directory), href)
                            if language == 'en':
                                self.assertNotIn('/docs/zh-CN/', href)
                    if href.endswith(('.json', '.csv')):
                        self.assertIn('download', link['attrs'], href)
                        self.assertTrue('JSON' in link['text'] or 'CSV' in link['text'], link['text'])
            results = self.page(language, 'results')
            buttons = [item for item in results.select('href', '#limitations')
                       if 'button' in item['attrs'].get('class', '').split()]
            self.assertEqual(len(buttons), 1)

    def test_public_commands_use_the_explicit_project_interpreter(self):
        for language in ('en', 'zh-CN'):
            page = self.page(language, 'reproduce')
            for item in (item for item in page.items if item['tag'] == 'pre'):
                lines = item['text'].splitlines()
                self.assertFalse(any(line.startswith('python ') for line in lines))
                self.assertFalse(any('activate' in line for line in lines))
            for platform, interpreter in [('unix', '.venv/bin/python'),
                                          ('windows', r'.\.venv\Scripts\python.exe')]:
                command, = page.select('id', 'build-command-' + platform)
                self.assertTrue(all(line.startswith(interpreter + ' ') for line in command['text'].splitlines()))

    def test_inline_resources_are_accessible_readable_and_use_frozen_configuration(self):
        model = json.loads((ROOT / 'reproducibility/FINAL_MODEL.json').read_text())
        expected = [model['model_revision'], ', '.join(map(str, model['seeds'])), str(model['selected_epoch']),
                    f"{model['micro_batch']} / {model['effective_batch']}",
                    f"{model['encoder_lr']} / {model['head_lr']}",
                    f"{model['scheduler_epochs']} / {model['warmup_fraction']}",
                    f"{model['weight_decay']} / {model['gradient_clip']}", str(model['dropout'])]
        for language in ('en', 'zh-CN'):
            page = self.page(language, 'method')
            panels = {item['attrs']['data-resource-panel']: item for item in page.select('data-resource-panel')}
            self.assertEqual(set(panels), {'model', 'config'})
            for button in page.select('data-resource-toggle'):
                panel = panels[button['attrs']['data-resource-toggle']]
                self.assertEqual(button['attrs']['role'], 'button')
                self.assertEqual(button['attrs']['aria-controls'], panel['attrs']['id'])
                self.assertEqual(button['attrs']['href'], '#' + panel['attrs']['id'])
                self.assertNotIn('hidden', panel['attrs'])  # static fallback if JS fails
                self.assertEqual(len(page.select('id', panel['attrs']['aria-labelledby'])), 1)
            config = panels['config']
            cells = []
            for item in page.select('class', 'config-value'):
                ancestor = item['parent']
                while ancestor is not None and ancestor is not config:
                    ancestor = ancestor['parent']
                if ancestor is config:
                    cells.append(item['text'])
            self.assertEqual(len(cells), 10)
            self.assertEqual(cells[:8], expected)
            self.assertIn(str(self.facts['metrics']['C+']['threshold']), config['text'])
            for checkpoint in model['checkpoints']:
                self.assertIn(checkpoint['sha256'], config['text'])
                self.assertIn(f"{checkpoint['bytes']:,} bytes", config['text'])
            for name in ('index', 'results', 'reproduce'):
                self.assertFalse(self.page(language, name).select('data-resource-panel'))

    def test_both_languages_show_original_metrics_in_plots_and_table(self):
        for language in ('en', 'zh-CN'):
            page = self.page(language, 'results')
            for figure in page.select('data-metric-plot'):
                self.assertEqual(figure['attrs']['data-axis-min'], '0')
                self.assertEqual(figure['attrs']['data-axis-max'], '1')
                metric = figure['attrs']['data-metric-plot']
                rows = [item for item in page.select('data-method') if item['parent'] is figure]
                # Rows live directly in the figure; exact values and displayed rounding are checked.
                self.assertEqual({row['attrs']['data-method'] for row in rows}, set(self.facts['metrics']))
                for row in rows:
                    expected = self.facts['metrics'][row['attrs']['data-method']][metric]
                    self.assertEqual(float(row['attrs']['data-value']), expected)
                    self.assertIn(f'{expected:.4f}', row['text'])
            cells = [item for item in page.select('data-metric') if item['tag'] == 'td']
            self.assertEqual(len(cells), 30)
            for cell in cells:
                method = cell['parent']['attrs']['data-method']
                metric = cell['attrs']['data-metric']
                expected = self.facts['metrics'][method][metric]
                self.assertEqual(float(cell['attrs']['data-value']), expected)
                self.assertEqual(float(cell['text'].strip()), float(f'{expected:.2f}' if metric == 'p_at_100' else f'{expected:.6f}'))
            self.assertEqual({item['attrs']['data-confusion'] for item in page.select('data-confusion')}, {'tn','fp','fn','tp'})
            for cell in page.select('data-confusion'):
                value = self.facts['metrics']['C+']['confusion'][cell['attrs']['data-confusion']]
                self.assertIn(str(value), cell['text'])

    def test_role_counts_feature_order_and_homepage_values_match_sources(self):
        features = json.loads((ROOT / 'reproducibility/FEATURES.json').read_text())['names']
        for language in ('en','zh-CN'):
            method = self.page(language, 'method')
            for role, value in self.facts['roles'].items():
                item, = method.select('data-source', 'roles.' + role)
                self.assertEqual(int(item['text'].strip().replace(',', '')), value)
            entries = method.select('data-feature-index')
            self.assertEqual([int(e['attrs']['data-feature-index']) for e in entries], list(range(42)))
            for item, name in zip(entries, features):
                self.assertIn(name, item['text'])
            overview = self.page(language, 'index')
            for metric in ('ap','f1','p_at_100'):
                item, = overview.select('data-source', 'metrics.C+.' + metric)
                self.assertEqual(float(item['attrs']['data-value']), self.facts['metrics']['C+'][metric])

    def test_csv_uses_full_values_and_integer_counts(self):
        with (ROOT / 'showcase/assets/fixed-results.csv').open(newline='', encoding='utf-8') as source:
            rows = list(csv.DictReader(source))
        self.assertEqual({row['method'] for row in rows}, set(self.facts['metrics']))
        for row in rows:
            metric = self.facts['metrics'][row['method']]
            for key in ('ap','f1','p_at_100','precision','recall','threshold'):
                self.assertEqual(float(row[key]), metric[key])
            for key in ('tp','fp','fn','tn'):
                self.assertEqual(int(row[key]), metric['confusion'][key])

    def test_public_inventory_rejects_unlisted_site_and_model_files(self):
        spec = importlib.util.spec_from_file_location('site_inventory', ROOT / 'scripts/release_files.py')
        inventory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inventory)
        for name in ('showcase/content/evaluation.csv', 'showcase/assets/private.json',
                     'showcase/en/unknown.html', 'src/product_matching/legacy_model.py'):
            with self.subTest(name=name):
                self.assertFalse(inventory.allowed(name))

    def test_teaching_cases_distinguish_spelling_digits_and_missing_evidence(self):
        expected = {
            'spelling': {'title_token_jaccard':.5, 'brand_equal':1., 'brand_missing_any':0.,
                         'price_comparable':1., 'code_exact_overlap':1.,
                         'code_same_letters_different_digits':0., 'native_model_compact_exact':1.},
            'digits': {'title_token_jaccard':.5, 'brand_equal':1., 'brand_missing_any':0.,
                       'price_comparable':1., 'code_exact_overlap':0.,
                       'code_same_letters_different_digits':1., 'native_model_compact_exact':0.},
            'missing': {'title_token_jaccard':.5, 'brand_equal':0., 'brand_missing_any':1.,
                        'price_comparable':0., 'price_similarity':0., 'code_exact_overlap':1.,
                        'code_same_letters_different_digits':0., 'native_model_compact_exact':1.},
        }
        for language in ('en','zh-CN'):
            page = self.page(language,'index')
            panels = page.select('data-example-panel')
            self.assertEqual({p['attrs']['data-example-panel'] for p in panels}, set(expected))
            for panel in panels:
                self.assertNotIn('hidden', panel['attrs'])  # readable if scripts fail
                actual = {}
                for row in page.select('data-demo-feature'):
                    ancestor = row['parent']
                    while ancestor is not None and ancestor is not panel:
                        ancestor = ancestor['parent']
                    if ancestor is panel:
                        actual[row['attrs']['data-demo-feature']] = float(row['attrs']['data-demo-value'])
                self.assertEqual(len(actual),8)
                for key, value in expected[panel['attrs']['data-example-panel']].items():
                    self.assertEqual(actual[key],value)

    def test_gain_chart_is_absolute_percentage_points_against_named_controls(self):
        for language in ('en','zh-CN'):
            page = self.page(language,'results')
            deltas = page.select('data-delta-reference')
            self.assertEqual(len(deltas),4)
            for row in deltas:
                reference, metric = row['attrs']['data-delta-reference'], row['attrs']['data-delta-metric']
                expected = (self.facts['metrics']['C+'][metric] - self.facts['metrics'][reference][metric])*100
                self.assertEqual(float(row['attrs']['data-value']),expected)
                self.assertIn(f'+{expected:.4f}',row['text'])


if __name__ == '__main__':
    unittest.main()
