#!/usr/bin/env python3
"""Isolated evidence for Piper espeak subprocess/fallback call duplication."""
import argparse, hashlib, json, os, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bundle',required=True); ap.add_argument('--cases',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); bpath=Path(a.bundle); os.environ.update(PIPER_MODEL_DIR=str(bpath.parent),PIPER_LANGUAGES=bpath.name,PIPER_DEFAULT_LANG=bpath.name,PIPER_ENABLE_FRONTEND_NPU='1')
    import rkvoice_stream.backends.tts.piper as p
    p.MODEL_DIR=str(bpath.parent); p.PRELOAD_LANGS=[bpath.name]; p.DEFAULT_LANG=bpath.name
    cases=json.loads(Path(a.cases).read_text())['prompts']; wanted={'en_short_01','en_long_03'}; cases=[x for x in cases if x['id'] in wanted]
    b=p.PiperRKNNBackend(); b.preload(); rows=[]
    try:
      for forced in (False,True):
        if forced: p._HAS_PIPER_PHONEMIZE=False
        calls=[]; sub=[]; orig_run=p._run_espeak; orig_sub=p._phonemize_subprocess
        def run(text,voice):
          t=time.perf_counter(); out=orig_run(text,voice); calls.append({'sha256':hashlib.sha256(text.encode()).hexdigest(),'chars':len(text),'voice':voice,'ms':(time.perf_counter()-t)*1000}); return out
        def subproc(text,voice):
          t=time.perf_counter(); out=orig_sub(text,voice); sub.append({'sha256':hashlib.sha256(text.encode()).hexdigest(),'chars':len(text),'voice':voice,'ms':(time.perf_counter()-t)*1000}); return out
        p._run_espeak=run; p._phonemize_subprocess=subproc
        for c in cases:
          t=time.perf_counter(); wav,meta=b.synthesize(c['text']); rows.append({'case':c['id'],'forced_subprocess':forced,'native_available':bool(getattr(p,'_HAS_PIPER_PHONEMIZE',False)),'total_ms':(time.perf_counter()-t)*1000,'run_espeak_calls':len(calls),'subprocess_calls':len(sub),'run_espeak':list(calls),'subprocess':list(sub)})
          calls.clear(); sub.clear()
        p._run_espeak=orig_run; p._phonemize_subprocess=orig_sub
    finally: b.cleanup()
    Path(a.out).write_text(json.dumps({'rows':rows},indent=2)+'\n'); print(json.dumps({'out':a.out,'rows':len(rows)}))
if __name__=='__main__': main()
