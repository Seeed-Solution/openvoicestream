#!/usr/bin/env python3
"""Temporary matched text benchmark; never a production phase result."""
import argparse, concurrent.futures, hashlib, importlib, json, os, re, sys, threading, time, types
from pathlib import Path

SAMPLING={'temperature':0.0,'top_p':1.0,'top_k':1,'max_tokens':32,'enable_thinking':False,'skip_special_tokens':True}

class RuntimeSourceMismatch(RuntimeError):
 pass

def _runtime_source(module, expected_sha256):
 path=getattr(module,'__file__',None)
 result={'status':'UNBOUND' if expected_sha256 is None else 'ERROR','expected_sha256':expected_sha256,'actual_path':path}
 if expected_sha256 is None:
  if isinstance(path,str) and os.path.isabs(path) and not os.path.islink(path) and os.path.isfile(path):
   h=hashlib.sha256(Path(path).read_bytes()).hexdigest(); result['actual_sha256']=h
  return result
 if not isinstance(expected_sha256,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',expected_sha256):
  raise RuntimeSourceMismatch('runtime_engine_sha256 pin must be a 64-character hexadecimal SHA256')
 if not isinstance(path,str) or not os.path.isabs(path) or os.path.islink(path) or not os.path.isfile(path):
  raise RuntimeSourceMismatch(f'runtime engine module path is not a regular non-symlink file: {path!r}')
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 actual=h.hexdigest(); result.update(actual_sha256=actual,status='UNBOUND' if expected_sha256 is None else ('PASS' if actual==expected_sha256 else 'ERROR'))
 if expected_sha256 is None:
  return result
 if actual != expected_sha256:
  raise RuntimeSourceMismatch(f'runtime engine source SHA mismatch: expected {expected_sha256}, actual {actual}, path {path}')
 return result

def one(llm,label,prompt,params):
 start=time.monotonic(); first=None; ids=[]; text=[]; reason=None; deltas=[]; prompt_tokens=None
 for d in llm.generate_stream([{'role':'user','content':prompt}],params):
  now=time.monotonic()
  if first is None and (getattr(d,'text','') or getattr(d,'token_ids',None)): first=now
  di=list(getattr(d,'token_ids',None) or []); ids.extend(di); text.append(getattr(d,'text','') or '')
  prompt_tokens=getattr(d,'prompt_tokens',prompt_tokens); reason=getattr(d,'finish_reason',None) or reason
  deltas.append({'t_mono':now,'text':getattr(d,'text',''),'token_ids':di,'finished':bool(getattr(d,'finished',False)),'finish_reason':getattr(d,'finish_reason',None),'prompt_tokens':getattr(d,'prompt_tokens',None)})
 end=time.monotonic(); ttft=None if first is None else first-start; tpot=None if len(ids)<=1 or ttft is None else max(0.0,(end-start)-ttft)/(len(ids)-1)
 return {'label':label,'prompt':prompt,'request_start_mono':start,'request_end_mono':end,'elapsed_s':end-start,'ttft_s':ttft,'tpot_s':tpot,'token_ids':ids,'text':''.join(text),'prompt_tokens':prompt_tokens,'finish_reason':reason,'finished':bool(deltas and deltas[-1]['finished']),'qualified_32':len(ids)==32 and reason=='length' and bool(deltas and deltas[-1]['finished']),'deltas':deltas}

def round_run(llm,width,prompts,params,index):
 barrier=threading.Barrier(width) if width>1 else None
 def task(i):
  if barrier: barrier.wait(timeout=10)
  return one(llm,f'r{index}-{i}',prompts[i],params)
 if width==1: rows=[task(0),task(1)]
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool: rows=list(pool.map(task,range(width)))
 wall=max(x['request_end_mono'] for x in rows)-min(x['request_start_mono'] for x in rows); tokens=sum(len(x['token_ids']) for x in rows)
 return {'round':index,'width':width,'rows':rows,'round_wall_s':wall,'tokens':tokens,'throughput_tokens_s':tokens/wall,'qualified_32':all(x['qualified_32'] for x in rows)}

def main(path):
 p=json.loads(Path(path).read_text()); width=p.get('width'); engine_batch_size=p.get('engine_batch_size',width); media_configured='load_media_engines' in p; load_media_engines=p.get('load_media_engines',True)
 if (isinstance(width,bool) or not isinstance(width,int) or width < 1
     or isinstance(engine_batch_size,bool) or not isinstance(engine_batch_size,int)
     or engine_batch_size < 1 or width > engine_batch_size
     or not isinstance(load_media_engines,bool)):
  raise ValueError('width and engine_batch_size must be positive integers with width <= engine_batch_size; load_media_engines must be bool')
 out={'schema':'matched-text-benchmark.v2','status':'STARTED','width':width,'workload_width':width,'engine_batch_size':engine_batch_size,'runtime_capacity':engine_batch_size,'load_media_engines':load_media_engines,'load_media_engines_configured':media_configured,'load_media_engines_source':'explicit' if media_configured else 'inferred-legacy-default','rounds':[],'warmup':[],'source_pins':p.get('source_pins',{}),'bundle':p.get('bundle'),'sampling':SAMPLING,'close_complete':False}
 llm=None; error=None
 try:
  from experimental.server.runtime import engine_build
  engine_module=importlib.import_module('experimental.server.runtime.engine')
  expected_runtime_sha=p.get('source_pins',{}).get('runtime_engine_sha256')
  try:
   out['actual_runtime_source']=_runtime_source(engine_module,expected_runtime_sha)
  except RuntimeSourceMismatch as exc:
   details=getattr(exc,'args',[str(exc)])[0]
   actual_path=getattr(engine_module,'__file__',None)
   actual_sha=None
   if isinstance(actual_path,str) and os.path.isfile(actual_path) and not os.path.islink(actual_path):
    h=hashlib.sha256(Path(actual_path).read_bytes()).hexdigest(); actual_sha=h
   out['actual_runtime_source']={'status':'ERROR','expected_sha256':expected_runtime_sha,'actual_sha256':actual_sha,'actual_path':actual_path}
   raise
  from experimental.server.runtime.engine import SamplingParams,load_model
  o=engine_build.BuildOptions(max_input_len=p['max_input_len'],max_batch_size=engine_batch_size,max_kv_cache_capacity=p['max_kv_cache_capacity']); bundle=engine_build.bundle_cache_path(p['model'],p['cache'],o)
  if bundle!=p['bundle'] or not engine_build._is_ready(p['model'],bundle,o): raise RuntimeError('bundle admission failed')
  guard=types.ModuleType('experimental.builder.cli'); guard.main=lambda argv: (_ for _ in ()).throw(RuntimeError('builder invocation rejected')); sys.modules['experimental.builder.cli']=guard
  load_kwargs={'model':p['model'],'cache_dir':p['cache'],'max_input_len':p['max_input_len'],'max_batch_size':engine_batch_size,'max_kv_cache_capacity':p['max_kv_cache_capacity'],'enable_in_flight_batching':p['enable_in_flight_batching'],'context_cache_config':p.get('context_cache_config',{})}
  if media_configured: load_kwargs['load_media_engines']=load_media_engines
  llm=load_model(**load_kwargs)
  runtime_capacity=getattr(llm,'max_batch_size',None)
  if runtime_capacity != engine_batch_size: raise RuntimeError('loaded runtime capacity does not match engine_batch_size')
  out['runtime_capacity']=runtime_capacity
  params=SamplingParams(**SAMPLING); prompts=p['prompts']; out['warmup']=[one(llm,'warmup-numeric',prompts[0],params),one(llm,'warmup-color',prompts[1],params)]
  measured=[prompts[0],prompts[1]] if p['width']==1 else [prompts[i%2] for i in range(p['width'])]
  for i in range(p['measured_rounds']): out['rounds'].append(round_run(llm,p['width'],measured,params,i))
  if not all(x['qualified_32'] for x in out['rounds']): raise RuntimeError('measured output failed exact 32-token qualification')
  out['status']='DONE'
 except BaseException as exc:
  error={'type':type(exc).__name__,'message':str(exc)}
  if isinstance(exc,RuntimeSourceMismatch):
   error.update(out.get('actual_runtime_source',{}))
  out['status']='ERROR'; out['error']=error
 finally:
  if llm is not None:
   try: llm.close(); out['close_complete']=True
   except BaseException as exc:
    out['close_complete']=False; out['close_error']={'type':type(exc).__name__,'message':str(exc)}
    if out.get('status') != 'ERROR': out['status']='ERROR'; out['error']={'type':type(exc).__name__,'message':'close failed: '+str(exc)}
  out['finished_ns']=time.time_ns(); Path(p['output']).write_text(json.dumps(out,separators=(',',':')))
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--profile',required=True); a=ap.parse_args(); r=main(a.profile); print(json.dumps({'status':r['status'],'width':r['width'],'rounds':len(r['rounds']),'close_complete':r['close_complete'],'error':r.get('error')})); raise SystemExit(0 if r['status']=='DONE' else 1)
