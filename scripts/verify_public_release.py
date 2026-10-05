"""Verify aggregate arithmetic, complete file inventory and public boundaries."""
import hashlib
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from release_files import DOCS, release_files
ROOT=Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/name).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-14)
class PageReferences(HTMLParser):
    """Read local resources and anchors without executing page scripts."""
    def __init__(self):
        super().__init__();self.references=[];self.ids=set();self.language=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids,'Duplicate HTML id'
            self.ids.add(attrs['id'])
        if tag=='html':self.language=attrs.get('lang')
        for name in ('href','src'):
            if attrs.get(name):self.references.append(attrs[name])

def local_target(path,target):
    url=urlsplit(target)
    if url.scheme or url.netloc:return None
    assert not url.path.startswith('/'),'Local links must work under a subdirectory'
    resolved=(path.parent/unquote(url.path)).resolve() if url.path else path.resolve()
    assert resolved.is_relative_to(ROOT.resolve()),'Link escapes public checkout'
    assert resolved.exists(),'Missing local link: '+str(path.relative_to(ROOT))+' -> '+target
    return resolved,url.fragment

def main():
    result=load('results/fixed-results.json');model=load('reproducibility/FINAL_MODEL.json')
    lock=load('reproducibility/EXPERIMENT_LOCK.json');features=load('reproducibility/FEATURES.json')
    assert result['roles']==lock['roles']==model['roles']=={'fit':3738,'dev':199,'calibration':209,'evaluation':555}
    assert result['model']==lock['model']
    assert model['model_revision']==result['model']['revision'] and model['selected_epoch']==4
    assert result['model']['seeds']==model['seeds']==[42,43,44]
    assert result['model']['aggregation']==model['aggregation']=='equal_probability_mean'
    assert len(features['names'])==len(set(features['names']))==42
    assert lock['fresh_sample_validation'] is model['independent_new_sample_validation'] is False
    assert result['reproduction']['fresh_training_reproduced'] is False
    assert result['reproduction']['fixed_artifact_replay_cli_commands']==20
    assert result['reproduction']['independently_checked_result_groups']==23
    assert set(result['metrics'])=={'B22','L30','S33','L42','C+','C0'}
    for metric in result['metrics'].values():
        c=metric['confusion'];tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'))
        assert all(type(v) is int and v>=0 for v in (tp,fp,fn,tn))
        assert tp+fp+fn+tn==metric['count']==555 and tp+fn==metric['positive_count']==190
        close(metric['precision'],tp/(tp+fp));close(metric['recall'],tp/(tp+fn))
        close(metric['f1'],2*tp/(2*tp+fp+fn))
        assert all(math.isfinite(metric[k]) and 0<=metric[k]<=1 for k in ('ap','f1','p_at_100','threshold'))
    close(result['metrics']['C+']['ap'],0.9659975315302057)
    close(result['metrics']['C+']['f1'],0.9016393442622951)
    close(result['metrics']['C+']['threshold'],lock['threshold'])
    assert lock['idf_role']=='fit' and lock['threshold_role']=='calibration'
    manifest=load('reproducibility/PUBLIC_MANIFEST.json')
    assert manifest['schema']=='product-matching-public-manifest-v2' and manifest['self_excluded'] is True
    entries=manifest['files'];names=[e['path'] for e in entries]
    assert len(names)==len(set(names))==manifest['file_count']
    actual={p.relative_to(ROOT).as_posix():p for p in release_files(ROOT)
            if p!=ROOT/'reproducibility/PUBLIC_MANIFEST.json'}
    assert set(names)==set(actual),'Manifest does not cover exact file set'
    for entry in entries:
        path=actual[entry['path']]
        assert path.stat().st_size==entry['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    forbidden=('/Use'+'rs/','D:'+ '/', 'C:/'+'Users','192.'+'168.',
               'BEGIN OPENSSH '+'PRIVATE KEY','authorized'+'_keys')
    links=0;html_links=0;html_pages={}
    for path in actual.values():
        if path.suffix=='.html' and 'templates' not in path.relative_to(ROOT).parts:
            page=PageReferences();page.feed(path.read_text(encoding='utf-8'))
            html_pages[path.resolve()]=page
    for path in actual.values():
        text=path.read_text(encoding='utf-8')
        assert not any(v in text for v in forbidden),'Private value in '+str(path.relative_to(ROOT))
        if path.suffix in ('.md','.ipynb'):
            if path.suffix=='.ipynb':
                text='\n'.join(''.join(c['source']) for c in json.loads(text)['cells'] if c['cell_type']=='markdown')
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                if target.startswith('#'):continue
                if local_target(path,target):links+=1
        if path.resolve() in html_pages:
            for target in html_pages[path.resolve()].references:
                resolved=local_target(path,target)
                if resolved:
                    destination,fragment=resolved
                    if fragment and destination.suffix=='.html':
                        assert fragment in html_pages[destination].ids,'Missing HTML anchor: '+target
                    html_links+=1
    assert not (ROOT/'src/product_entity_matching').exists()
    assert (ROOT/'README.zh-CN.md').is_file()
    for filename in DOCS:
        assert (ROOT/'docs'/filename).is_file() and (ROOT/'docs/zh-CN'/filename).is_file()
    expected_pages={ROOT/'showcase'/lang/page for lang in ('en','zh-CN')
                    for page in ('index.html','method.html','results.html','reproduce.html')}
    expected_pages.add(ROOT/'index.html')
    assert set(html_pages)=={p.resolve() for p in expected_pages},'Incomplete bilingual page set'
    for path,page in html_pages.items():
        assert page.language==('zh-CN' if path==ROOT/'index.html' else path.parent.name),'Page language and route differ'
    for directory in ('data','models','reports','archives','handoff'):
        assert not (ROOT/directory).exists(),'Private directory in checkout'
    print(json.dumps({'status':'passed','manifest_files':len(entries),'local_links':links,
                      'html_resource_links':html_links,'html_files':len(html_pages),'model_run':False}))
if __name__=='__main__':main()
