"""Copy the verified public inventory into a standalone GitHub Pages artifact."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_files import release_files

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory: ' + str(output))
    if ROOT == output or output in ROOT.parents:
        raise ValueError('Output cannot replace the source checkout')
    # Fail before creating the artifact if the inventory, links or generated pages differ.
    for script, args in (('verify_public_release.py', []), ('build_showcase.py', ['--check'])):
        subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'scripts' / script), *args], check=True)
    files = list(release_files(ROOT))
    for path in files:
        destination = output / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    return len(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'pages')
    args = parser.parse_args()
    print(json.dumps({'status': 'passed', 'copied_files': build(args.output), 'model_run': False}))
