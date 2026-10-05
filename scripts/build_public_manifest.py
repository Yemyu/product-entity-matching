"""Regenerate the exact allowed file set after deliberate public edits."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from release_files import release_files
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reproducibility/PUBLIC_MANIFEST.json'
def main():
    entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
              'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in release_files(ROOT) if p!=OUTPUT]
    OUTPUT.write_text(json.dumps({'schema':'product-matching-public-manifest-v2','self_excluded':True,
                                 'file_count':len(entries),'files':entries},indent=2)+'\n')
    print(json.dumps({'status':'written','file_count':len(entries)}))
if __name__=='__main__':main()
