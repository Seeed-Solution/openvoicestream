import requests, websocket, json, wave, pathlib, datetime, hashlib
base="http://127.0.0.1:18621"
root=pathlib.Path("/work/seeed/agent/tests/e2e/fixtures/wav")
print(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"container":"spark-asrfix-canary","endpoint":base}),flush=True)
for file,lang in [("weather.wav","Chinese"),("weather.wav",None),("cmd_en_go_home.wav","English"),("cmd_en_go_home.wav",None)]:
 p=root/file
 with p.open("rb") as f:r=requests.post(base+"/asr",params={"language":lang} if lang else {},files={"file":(file,f,"audio/wav")},timeout=30)
 print(json.dumps({"http":file,"language":lang,"sha":hashlib.sha256(p.read_bytes()).hexdigest(),"status":r.status_code,"response":r.json()},ensure_ascii=False),flush=True)
 r.raise_for_status()
for file,lang in [("weather.wav","Chinese"),("cmd_en_go_home.wav","English")]:
 with wave.open(str(root/file),"rb") as f:pcm=f.readframes(f.getnframes())
 ws=websocket.create_connection("ws://127.0.0.1:18621/v2v/stream",timeout=15)
 ws.send(json.dumps({"type":"config","asr_language":lang,"sample_rate":16000,"vad":"none","multi_utterance":False}))
 for i in range(0,len(pcm),3200):ws.send_binary(pcm[i:i+3200])
 ws.send(json.dumps({"type":"asr_eos"}))
 events=[]
 while True:
  raw=ws.recv()
  if not raw:break
  e=json.loads(raw);events.append(e)
  if e.get("type")=="asr_final":break
 ws.close()
 print(json.dumps({"v2v":file,"language":lang,"events":events},ensure_ascii=False),flush=True)
 assert any(e.get("type")=="asr_final" and e.get("text") for e in events)
