import hashlib,importlib.util,json,sys,tempfile,threading,time,types
from pathlib import Path
E=Path(__file__).parent; spec=importlib.util.spec_from_file_location('bench',E/'edgellm_llm_matched_perf.py'); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
class D:
 def __init__(self,t='',ids=(),fin=False,reason=None): self.text=t; self.token_ids=list(ids); self.finished=fin; self.finish_reason=reason; self.prompt_tokens=4
class L:
 def __init__(self,short=False,closefail=False,fail_after=None,**kw): self.kw=kw; self.short=short; self.closefail=closefail; self.fail_after=fail_after; self.calls=0; self.prompts=[]; self.params=[]; self.active=0; self.peak=0; self.closed=False; self.lock=threading.Lock()
 def generate_stream(self,m,params):
  with self.lock:self.active+=1; self.peak=max(self.peak,self.active); self.prompts.append(m[0]['content']); self.params.append(params.kw)
  try:
   time.sleep(.001); self.calls+=1
   if self.fail_after is not None and self.calls>self.fail_after: raise RuntimeError('request failure')
   n=4 if self.short else 32; yield D('x',list(range(n))); yield D('',(),True,'stop' if self.short else 'length')
  finally:
   with self.lock:self.active-=1
 def close(self):
  self.closed=True
  if self.closefail: raise RuntimeError('close failure')
 @property
 def max_batch_size(self): return self.kw['max_batch_size']
class S:
 def __init__(self,**kw): self.kw=kw
class Fake:
 def __init__(self,short=False,closefail=False,fail_after=None):
  self.short=short; self.closefail=closefail; self.fail_after=fail_after; self.instances=[]; self.native_calls=[]; self.builder_called=False; self.reject_media_kw=False; self.load_calls=0
  self.engine_file=Path(tempfile.mkstemp(prefix='fake-runtime-engine-',suffix='.py')[1]); self.engine_file.write_text('# fake runtime source\n'); self.engine_sha=hashlib.sha256(self.engine_file.read_bytes()).hexdigest()
 def install(self):
  eb=types.ModuleType('experimental.server.runtime.engine_build'); eng=types.ModuleType('experimental.server.runtime.engine'); rt=types.ModuleType('experimental.server.runtime'); server=types.ModuleType('experimental.server'); ex=types.ModuleType('experimental')
  class O:
   def __init__(self,**kw): self.__dict__.update(kw)
  eb.BuildOptions=O; eb.bundle_cache_path=lambda model,cache,opt:'/bundle-'+str(opt.max_batch_size); eb._is_ready=lambda model,bundle,opt:True; eng.__file__=str(self.engine_file)
  def load_model(**kw):
   self.load_calls+=1
   if self.reject_media_kw and 'load_media_engines' in kw: raise TypeError('legacy load_model does not accept load_media_engines')
   self.native_calls.append({'multimodalEngineDir':'/bundle-'+str(kw['max_batch_size']) if kw.get('load_media_engines',True) else '', 'max_input_len':kw['max_input_len'], 'max_kv_cache_capacity':kw['max_kv_cache_capacity'], 'max_batch_size':kw['max_batch_size']})
   x=L(short=self.short,closefail=self.closefail,fail_after=self.fail_after,**kw); self.instances.append(x); return x
  eng.SamplingParams=S; eng.load_model=load_model
  sys.modules.update({'experimental':ex,'experimental.server':server,'experimental.server.runtime':rt,'experimental.server.runtime.engine_build':eb,'experimental.server.runtime.engine':eng})
NUM='Here are 32 bounded tokens related to numbers, ranging from basic arithmetic to advanced mathematical concepts. Continue with exactly 32 tokens and finish at the cap.'
COLOR='Here are 32 bounded tokens related to colors, ranging from basic hues to specific shades and color properties. Continue with exactly 32 tokens and finish at the cap.'
def profile(root,width,rounds,ifb,short=False,closefail=False,engine_batch_size=None,load_media=True):
 f=Fake(short,closefail); q=root/f'w{width}-{rounds}.json'; out=root/f'w{width}-{rounds}.out.json'; d={'model':'/model','cache':'/cache','bundle':'/bundle-'+str(engine_batch_size if engine_batch_size is not None else width),'output':str(out),'width':width,'max_input_len':8192,'max_kv_cache_capacity':8192,'enable_in_flight_batching':ifb,'context_cache_config':{},'prompts':['numeric prompt','color prompt'],'measured_rounds':rounds,'source_pins':{'runtime_engine_sha256':f.engine_sha}}; 
 if engine_batch_size is not None: d['engine_batch_size']=engine_batch_size
 if load_media is not None: d['load_media_engines']=load_media
 q.write_text(json.dumps(d)); return q,f,out
def test_matched_perf_cpu():
 root=Path(tempfile.mkdtemp(prefix='matched-text-cpu-'))
 for width,rounds,ifb in ((1,20,True),(2,20,True),(4,10,True)):
  q,f,out=profile(root,width,rounds,ifb,engine_batch_size=4,load_media=False); doc=json.loads(q.read_text()); doc.update(engine_batch_size=4,bundle='/bundle-4',prompts=[NUM,COLOR] if width < 4 else [NUM,COLOR,NUM,COLOR], perf_texts=[NUM,COLOR] if width < 4 else [NUM,COLOR,NUM,COLOR], phase=f'matched.text.w{width}.engine4', warmup_rounds=2, schema=1, sampling={'temperature':0.0,'top_p':1.0,'top_k':1,'max_tokens':32,'enable_thinking':False,'skip_special_tokens':True}); q.write_text(json.dumps(doc)); f.install(); r=b.main(q); inst=f.instances[-1]; measured=[NUM,COLOR] if width < 4 else [NUM,COLOR,NUM,COLOR]; expected=[NUM,COLOR]+measured*rounds; assert r['status']=='DONE' and r['close_complete'] and r['actual_runtime_source']['status']=='PASS' and r['actual_runtime_source']['actual_sha256']==f.engine_sha and r['load_media_engines'] is False and r['load_media_engines_configured'] is True and r['load_media_engines_source']=='explicit' and r['engine_batch_size']==4 and r['workload_width']==width and r['runtime_capacity']==4 and len(r['rounds'])==rounds and all(x['qualified_32'] for x in r['rounds']) and inst.peak>=width and inst.prompts[:2]==[NUM,COLOR] and sorted(inst.prompts[2:])==sorted(expected[2:]) and all(x == b.SAMPLING for x in inst.params) and inst.kw['model']=='/model' and inst.kw['cache_dir']=='/cache' and inst.kw['max_input_len']==8192 and inst.kw['max_kv_cache_capacity']==8192 and inst.kw['max_batch_size']==4 and inst.kw['enable_in_flight_batching']==ifb and f.native_calls[-1]['multimodalEngineDir']=='' and f.native_calls[-1]['max_input_len']==8192 and f.native_calls[-1]['max_kv_cache_capacity']==8192 and f.native_calls[-1]['max_batch_size']==4 and not f.builder_called and out.exists(); print(json.dumps({'status':'PASS','width':width,'engine_batch_size':4,'runtime_capacity':4,'rounds':rounds,'peak':inst.peak,'ifb':ifb,'load_media_engines':False,'native_multimodalEngineDir':'','runtime_source':'verified','profile_fields':'verified','prompt_sequence':'verified','sampling':'verified'}))
 q,f,out=profile(root,1,1,False,load_media=None); f.reject_media_kw=True; doc=json.loads(q.read_text()); doc.pop('source_pins'); doc.update(prompts=[NUM,COLOR],perf_texts=[NUM,COLOR],warmup_rounds=2,schema=1,sampling={'temperature':0.0,'top_p':1.0,'top_k':1,'max_tokens':32,'enable_thinking':False,'skip_special_tokens':True}); q.write_text(json.dumps(doc)); f.install(); r=b.main(q); inst=f.instances[-1]; assert r['status']=='DONE' and r['actual_runtime_source']['status']=='UNBOUND' and r['actual_runtime_source']['actual_sha256']==f.engine_sha and r['load_media_engines'] is True and r['load_media_engines_configured'] is False and r['load_media_engines_source']=='inferred-legacy-default' and r['engine_batch_size']==1 and r['workload_width']==1 and r['runtime_capacity']==1 and inst.kw['max_batch_size']==1 and inst.kw['enable_in_flight_batching'] is False and f.native_calls[-1]['multimodalEngineDir']=='/bundle-1'; print(json.dumps({'status':'PASS','case':'default-width-engine-compatibility','legacy_unknown_kw':'not_sent','runtime_source':'unbound'}))
 q,f,out=profile(root,1,1,False,load_media=None); doc=json.loads(q.read_text()); doc['source_pins']['runtime_engine_sha256']='f'*64; q.write_text(json.dumps(doc)); f.install(); r=b.main(q); assert r['status']=='ERROR' and r['error']['type']=='RuntimeSourceMismatch' and r['actual_runtime_source']['expected_sha256']=='f'*64 and r['actual_runtime_source']['actual_sha256']==f.engine_sha and f.load_calls==0 and not f.instances and out.exists(); print(json.dumps({'status':'PASS','case':'runtime-source-mismatch-before-load','load_calls':0,'expected_sha256':'f'*64,'actual_sha256':f.engine_sha}))
 q,f,out=profile(root,1,1,False); f.install(); sys.modules['experimental.server.runtime.engine'].__file__=None; r=b.main(q); assert r['status']=='ERROR' and r['error']['type']=='RuntimeSourceMismatch' and r['actual_runtime_source']['actual_path'] is None and f.load_calls==0; print(json.dumps({'status':'PASS','case':'runtime-source-missing-file-before-load','load_calls':0}))
 q,f,out=profile(root,1,1,False); f.install(); link=root/'runtime-engine-link.py'; link.symlink_to(f.engine_file); sys.modules['experimental.server.runtime.engine'].__file__=str(link); r=b.main(q); assert r['status']=='ERROR' and r['error']['type']=='RuntimeSourceMismatch' and r['actual_runtime_source']['actual_path']==str(link) and f.load_calls==0; print(json.dumps({'status':'PASS','case':'runtime-source-leaf-symlink-before-load','load_calls':0}))
 q,f,out=profile(root,1,1,False,load_media=None); doc=json.loads(q.read_text()); doc.pop('source_pins'); q.write_text(json.dumps(doc)); f.install(); sys.modules['experimental.server.runtime.engine'].__file__=None; f.reject_media_kw=True; r=b.main(q); assert r['status']=='DONE' and r['actual_runtime_source']['status']=='UNBOUND' and r['actual_runtime_source']['actual_path'] is None and f.load_calls==1; print(json.dumps({'status':'PASS','case':'legacy-unbound-missing-file-continues','load_calls':1}))
 q,f,out=profile(root,1,1,False); doc=json.loads(q.read_text()); doc['source_pins']['runtime_engine_sha256']='g'*64; q.write_text(json.dumps(doc)); f.install(); r=b.main(q); assert r['status']=='ERROR' and r['error']['type']=='RuntimeSourceMismatch' and '64-character hexadecimal' in r['error']['message'] and f.load_calls==0; print(json.dumps({'status':'PASS','case':'runtime-source-nonhex-pin-before-load','load_calls':0}))
 q,f,out=profile(root,2,1,True,short=True); f.install(); r=b.main(q); assert r['status']=='ERROR' and r['close_complete'] and '32-token' in r['error']['message']; print(json.dumps({'status':'PASS','case':'short-output-error'}))
 q,f,out=profile(root,1,1,False,closefail=True); f.install(); r=b.main(q); assert r['status']=='ERROR' and not r['close_complete'] and 'close_error' in r and r['error']['message'].startswith('close failed'); print(json.dumps({'status':'PASS','case':'success-close-error-rejected'}))
 q,f,out=profile(root,2,2,True,closefail=True); f.install(); r=b.main(q); assert r['status']=='ERROR' and r['error']['message'].startswith('close failed') and 'close_error' in r; print(json.dumps({'status':'PASS','case':'success-and-close-errors-preserved'}))
 q,f,out=profile(root,2,3,True,closefail=True); f.fail_after=6; f.install(); r=b.main(q); assert r['status']=='ERROR' and r['error']['message']=='request failure' and 'close_error' in r and len(r['rounds'])==2; print(json.dumps({'status':'PASS','case':'run-and-close-errors-preserved'}))
 q,f,out=profile(root,2,3,True); f.fail_after=6; f.install(); r=b.main(q); assert r['status']=='ERROR' and len(r['rounds'])==2 and r['close_complete']; print(json.dumps({'status':'PASS','case':'partial-rounds-retained'}))
 for bad_width,bad_engine in ((5,4),(1,0)):
  q,f,out=profile(root,bad_width,1,True,engine_batch_size=bad_engine); f.install()
  try: b.main(q)
  except ValueError as exc: assert 'width and engine_batch_size' in str(exc)
  else: raise AssertionError('invalid width/engine_batch_size accepted')
  print(json.dumps({'status':'PASS','case':'invalid-width-or-engine-batch-rejected','width':bad_width,'engine_batch_size':bad_engine}))
 q,f,out=profile(root,1,1,ifb=True,load_media='false'); f.install()
 try: b.main(q)
 except ValueError as exc: assert 'load_media_engines must be bool' in str(exc)
 else: raise AssertionError('non-bool load_media_engines accepted')
 print(json.dumps({'status':'PASS','case':'non-bool-load-media-engines-rejected'}))
