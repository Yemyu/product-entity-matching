"""Synthetic-only feature and seed aggregation check; no model fitting."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from product_matching.data import business_record,native_record
from product_matching.features import FEATURE_NAMES_42,fit_title_evidence,prepare_rows
from product_matching.metrics import mean_scores

def synthetic_pair():
    return {'pair_id':'synthetic:acme',
        'left':business_record({'title':'Acme ZX-100 16GB','brand':'Acme','price':'19.99','priceCurrency':'USD'}),
        'right':business_record({'title':'ACME ZX100 16 GB','brand':'ACME','price':'20.00','priceCurrency':'USD'}),
        'left_native':native_record({'modelno':'ZX-100','category':'electronics'}),
        'right_native':native_record({'modelno':'ZX100','category':'electronics'})}
def main():
    row=synthetic_pair();idf=fit_title_evidence([{**row,'label':1}])
    prepared=prepare_rows([row],role='dev',fit_idf=idf)
    values=dict(zip(FEATURE_NAMES_42,prepared[0]['x42']))
    assert len(values)==42 and values['native_model_compact_exact']==1
    assert values['brand_equal']==1 and values['price_comparable']==1 and 'label' not in prepared[0]
    seeds={s:[{'pair_id':'synthetic:acme','score':v}] for s,v in zip((42,43,44),(.9,.3,.3))}
    assert mean_scores(['synthetic:acme'],seeds)==[{'pair_id':'synthetic:acme','score':.5}]
    assert not {'torch','transformers','sklearn','sentence_transformers'} & set(sys.modules)
    print(json.dumps({'status':'passed','synthetic':True,'feature_count':42,'model_run':False,'new_fit_calls':0}))
if __name__=='__main__':main()
