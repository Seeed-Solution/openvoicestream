import websocket,json,wave,pathlib,time,datetime,hashlib
def workers():
 out={}
 for p in pathlib.Path("/proc").iterdir():
  if p.name.isdigit():
   try:
    e=(p/"exe").resolve()
    if e.name.startswith("qwen3_asr_worker"):out[p.name]={"exe":str(e),"sha":hashlib.sha256((p/"exe").read_bytes()).hexdigest()}
   except OSError:pass
 return out
def connect(mode):
 u="ws://127.0.0.1:18621/"+("v2v/stream" if mode=="v2v" else "asr/stream?language=English&vad=none")
 w=websocket.create_connection(u,timeout=10)
 if mode=="v2v":w.send(json.dumps({"type":"config","asr_language":"English","sample_rate":16000,"vad":"none","multi_utterance":False}))
 return w
with wave.open("/work/seeed/agent/tests/e2e/fixtures/wav/cmd_en_go_home.wav","rb") as f:pcm=f.readframes(f.getnframes())
before=workers();print(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"worker_before":before}),flush=True)
for mode in ["v2v","asr"]:
 w=connect(mode);w.send_binary(pcm[:3200]);w.close()
 t=time.monotonic();time.sleep(.1)
 w=connect(mode)
 print(json.dumps({"mode":mode,"reconnect_after_close_ms":(time.monotonic()-t)*1000}),flush=True)
 for i in range(0,len(pcm),3200):w.send_binary(pcm[i:i+3200])
 if mode=="v2v":w.send(json.dumps({"type":"asr_eos"}))
 else:w.send_binary(b"")
 events=[]
 while True:
  raw=w.recv()
  if not raw:break
  e=json.loads(raw);events.append(e)
  if e.get("type") in ["asr_final","final"]:break
 w.close()
 print(json.dumps({"mode":mode,"events":events}),flush=True)
 assert any(e.get("text")=="Go home." and e.get("type") in ["asr_final","final"] for e in events)
after=workers();print(json.dumps({"worker_after":after,"unchanged":before==after}),flush=True)
assert before and before==after
