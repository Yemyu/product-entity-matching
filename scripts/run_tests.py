"""Run the public suite from a source checkout without installing packages."""
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({'status':'passed' if result.wasSuccessful() else 'failed',
                      'tests':result.testsRun,'skipped':len(result.skipped)}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
