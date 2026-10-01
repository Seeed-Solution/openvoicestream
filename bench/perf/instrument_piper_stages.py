#!/usr/bin/env python3
"""Isolated Piper stage timing; never edits the installed runtime."""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
import importlib.util
from pathlib import Path

class Timed:
    def __init__(self, obj, name, stats): self._obj=obj; self._name=name; self._stats=stats
    def __getattr__(self, k): return getattr(self._obj, k)
    def inference(self, *a, **kw):
        t=time.perf_counter(); out=self._obj.inference(*a, **kw)
        self._stats[self._name].append((time.perf_counter()-t)*1000)
        return out
    def run(self, *a, **kw):
        t=time.perf_counter(); out=self._obj.run(*a, **kw)
        self._stats[self._name].append((time.perf_counter()-t)*1000)
        return out

def stats(values):
    if not values: return {"count":0,"total_ms":0.0,"p50_ms":0.0,"p90_ms":0.0,"values_ms":[]}
    x=sorted(values); p=lambda f: x[min(len(x)-1,int((len(x)-1)*f))]
    return {"count":len(x),"total_ms":sum(x),"p50_ms":statistics.median(x),"p90_ms":p(.9),"values_ms":x}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bundle',required=True); ap.add_argument('--cases',required=True)
    ap.add_argument('--case-ids',required=True); ap.add_argument('--warmup',type=int,default=2); ap.add_argument('--reps',type=int,default=3); ap.add_argument('--out',required=True)
    a=ap.parse_args(); os.environ['PIPER_MODEL_DIR']=str(Path(a.bundle).parent); os.environ['PIPER_LANGUAGES']=Path(a.bundle).name; os.environ['PIPER_DEFAULT_LANG']=Path(a.bundle).name; os.environ['PIPER_ENABLE_FRONTEND_NPU']='1'; os.environ['PIPER_SEQ_LEN']='256'
    import numpy as np
    source = os.environ.get('PIPER_STAGE_SOURCE')
    if source:
        spec = importlib.util.spec_from_file_location('piper_stage_source', source)
        p = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(p)
    else:
        import rkvoice_stream.backends.tts.piper as p
    p.MODEL_DIR=str(Path(a.bundle).parent); p.PRELOAD_LANGS=[Path(a.bundle).name]; p.DEFAULT_LANG=Path(a.bundle).name
    cases=json.loads(Path(a.cases).read_text())['prompts']; wanted=set(a.case_ids.split(',')); cases=[x for x in cases if x['id'] in wanted]
    b=p.PiperRKNNBackend(); b.preload(); m=b._models[Path(a.bundle).name]
    allrows=[]
    try:
      for c in cases:
        text=str(c['text']); row={'id':c['id'],'text':text,'seq_len':m.seq_len,'stats':{},'runs':[]}
        for _ in range(a.warmup): b.synthesize(text)
        for mode in ('sync','stream'):
          vals=[]
          for rep in range(a.reps):
            s={"prefix_npu":[],"remainder_ort":[],"decoder_npu":[],"phonemize":[],"phonemize_texts":[]}
            old1,old2,old3=m._frontend_rknn,m._remainder,m._rknn
            m._frontend_rknn=Timed(old1,'prefix_npu',s); m._remainder=Timed(old2,'remainder_ort',s); m._rknn=Timed(old3,'decoder_npu',s)
            orig=p.text_to_phonemes
            def timed_ph(text, voice, espeak_cache=None):
              t=time.perf_counter()
              try:return orig(text, voice, espeak_cache=espeak_cache)
              finally:
                s['phonemize'].append((time.perf_counter()-t)*1000)
                s['phonemize_texts'].append(str(text))
            p.text_to_phonemes=timed_ph; t=time.perf_counter()
            if mode=='sync':
              wav,meta=b.synthesize(text)
              output={'samples':int(meta.get('duration',0)*m.sample_rate),'segments':None}
            else:
              chunks=[]; metas=[]
              for chunk,meta in b.synthesize_stream(text): chunks.append(chunk); metas.append(meta)
              output={'samples':sum(len(x) for x in chunks),'segments':len(chunks)}
            total=(time.perf_counter()-t)*1000
            p.text_to_phonemes=orig; m._frontend_rknn,m._remainder,m._rknn=old1,old2,old3
            stage_out={k:stats(v) for k,v in s.items() if k != 'phonemize_texts'}
            stage_out['phonemize_text_calls']=len(s['phonemize_texts'])
            stage_out['phonemize_unique_texts']=len(set(s['phonemize_texts']))
            stage_out['phonemize_repeated_calls']=len(s['phonemize_texts'])-len(set(s['phonemize_texts']))
            vals.append({'mode':mode,'total_ms':total,'output':output,'stages':stage_out})
          row['runs'].extend(vals)
        allrows.append(row)
      Path(a.out).write_text(json.dumps({'bundle':a.bundle,'cases':allrows},indent=2)+'\n')
      print(json.dumps({'out':a.out,'seq_len':m.seq_len,'rows':len(allrows)}))
    finally:b.cleanup()
if __name__=='__main__':main()
