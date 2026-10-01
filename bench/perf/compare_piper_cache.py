#!/usr/bin/env python3
"""Compare baseline and request-scoped Piper espeak cache in isolation."""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np

def load(path, tag, bundle):
    spec=importlib.util.spec_from_file_location(tag, path); mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
    mod.MODEL_DIR=str(Path(bundle).parent); mod.PRELOAD_LANGS=[Path(bundle).name]; mod.DEFAULT_LANG=Path(bundle).name
    return mod

def one(path, tag, bundle, cases):
    import onnxruntime as ort
    ort.set_seed(123); np.random.seed(123)
    p=load(path, tag, bundle); b=p.PiperRKNNBackend(); b.preload(); m=b._models[Path(bundle).name]
    out={'source':path,'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'cases':{}}
    try:
      for c in cases:
        cache={}; seg=list(p._frontend_segments(c['text'],m,cache)); ids=[p._frontend_tokens(x,m,cache)[1] for x in seg]
        np.random.seed(123); ort.set_seed(123); wav,meta=b.synthesize(c['text']);
        out['cases'][c['id']]={'segments':seg,'segment_token_counts':[len(x) for x in ids],'token_ids_sha256':[hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest() for x in ids],'wav_sha256':hashlib.sha256(wav).hexdigest(),'duration':meta['duration'],'samples':int(meta['duration']*m.sample_rate)}
    finally:b.cleanup()
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',required=True); ap.add_argument('--patched',required=True); ap.add_argument('--bundle',required=True); ap.add_argument('--cases',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    rows=json.loads(Path(a.cases).read_text())['prompts']; want={'en_short_01','en_domain_01','en_long_03','en_unpunct_01'}; cases=[x for x in rows if x['id'] in want]
    result={'baseline':one(a.baseline,'piper_baseline',a.bundle,cases),'patched':one(a.patched,'piper_patched',a.bundle,cases)}
    for cid in want:
      x=result['baseline']['cases'][cid]; y=result['patched']['cases'][cid]
      x['segment_equal']=x['segments']==y['segments']; x['token_counts_equal']=x['segment_token_counts']==y['segment_token_counts']; x['token_ids_equal']=x['token_ids_sha256']==y['token_ids_sha256']; x['wav_hash_equal']=x['wav_sha256']==y['wav_sha256']
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'out':a.out,'cases':sorted(want)}))
if __name__=='__main__':main()
