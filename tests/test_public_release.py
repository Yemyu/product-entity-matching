"""Public workflows and fail-closed distribution boundaries."""
import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PublicReleaseTests(unittest.TestCase):
    def run_script(self,name):
        p=subprocess.run([sys.executable,str(ROOT/'scripts'/name)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('"status": "passed"',p.stdout)
    def test_standard_library_smoke(self):self.run_script('public_smoke.py')
    def test_notebook_saved_outputs_match_execution(self):self.run_script('run_notebook.py')
    def test_release_arithmetic_and_complete_manifest(self):self.run_script('verify_public_release.py')
    def test_pages_artifact_preserves_verified_files_without_private_git_history(self):
        manifest=json.loads((ROOT/'reproducibility/PUBLIC_MANIFEST.json').read_text())
        expected={entry['path']:entry['sha256'] for entry in manifest['files']}
        expected['reproducibility/PUBLIC_MANIFEST.json']=hashlib.sha256((ROOT/'reproducibility/PUBLIC_MANIFEST.json').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'pages'
            result=subprocess.run([sys.executable,'-I','-S',str(ROOT/'scripts/build_pages.py'),'--output',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            actual={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
            self.assertEqual(actual,expected)
            self.assertFalse((output/'.git').exists())
            self.assertIn('url=showcase/zh-CN/index.html',(output/'index.html').read_text())
            self.assertTrue((output/'showcase/zh-CN/index.html').is_file())
    def test_inventory_rejects_unexpected_private_file_and_symlink(self):
        spec=importlib.util.spec_from_file_location('release_files_test',ROOT/'scripts/release_files.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'README.md').write_text('synthetic')
            (root/'weights.safetensors').write_bytes(b'synthetic')
            with self.assertRaises(ValueError):list(module.release_files(root))
            (root/'weights.safetensors').unlink();(root/'LICENSE').symlink_to(root/'README.md')
            with self.assertRaises(ValueError):list(module.release_files(root))
if __name__=='__main__':unittest.main()
