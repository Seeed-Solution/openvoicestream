#!/usr/bin/env python3
"""Decode full-CPU and captured DP256 prefix paths through the same Piper decoder."""
import argparse, hashlib, json, os, wave
from pathlib import Path
import numpy as np
import onnxruntime as ort

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def wav_write(path, audio, rate):
    a=np.asarray(audio,dtype=np.float32)
    finite=bool(np.isfinite(a).all())
    if not finite:
        raise ValueError(f"non-finite audio cannot be written: {path}")
    peak=float(np.max(np.abs(a))) if a.size else 0.0
    pcm=np.clip(a,-1,1); pcm=(pcm*32767).astype(np.int16)
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    return {'path':str(path),'sha256':sha(path),'samples':int(a.size),'finite':finite,
            'clipped_samples':int(np.count_nonzero(np.abs(a)>1.0)),'peak':peak}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--capture',required=True)
    ap.add_argument('--tokens',required=True)
    ap.add_argument('--source',required=True)
    ap.add_argument('--bundle',required=True)
    ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--text-encoder-sha256',required=True)
    ap.add_argument('--remainder-sha256',required=True)
    args=ap.parse_args()
    os.environ['PIPER_ENABLE_FRONTEND_NPU']='1'
    import sys; sys.path.insert(0,'/opt/venv/lib/python3.11/site-packages')
    from rkvoice_stream.backends.tts.piper import _LangModel, _trim_silence
    bundle=Path(args.bundle)
    artifacts={name:sha(bundle/name) for name in ('manifest.json','text_encoder.rknn','remainder.onnx','flow_decoder.rknn')}
    expected={'manifest.json':args.manifest_sha256,'text_encoder.rknn':args.text_encoder_sha256,'remainder.onnx':args.remainder_sha256}
    for name,want in expected.items():
        if artifacts[name] != want:
            raise RuntimeError(f"artifact SHA mismatch for {name}: got {artifacts[name]}, expected {want}")
    model=_LangModel('en_US',bundle)
    try:
        model.load()
        if model.seq_len != 256: raise RuntimeError(f'unexpected seq_len {model.seq_len}')
        names=[x['name'] if isinstance(x,dict) else x for x in model._frontend_manifest['prefix']['outputs']]
        if len(names) != 6 or '/dp/Mul_output_0' not in names:
            raise RuntimeError(f"refusing non-DP256 capture: prefix outputs={names!r}")
        rows={x['id']:x for x in json.load(open(args.tokens))['rows']}; ids=rows[args.case]['tokens']; n=len(ids)
        tokens=np.zeros((1,model.seq_len),np.int64); tokens[0,:n]=ids
        lengths=np.array([n],np.int64); scales=np.array([0.,1.,0.],np.float32)
        x_mask=np.zeros((1,1,model.seq_len),np.float32)
        x_mask[0,0,:n]=1.0
        base={'input':tokens,'input_lengths':lengths,'scales':scales,'x_mask':x_mask}
        source=ort.InferenceSession(args.source,providers=['CPUExecutionProvider'])
        sin={x.name for x in source.get_inputs()}; full={k:v for k,v in base.items() if k in sin}
        if 'sid' in sin: full['sid']=np.array([0],np.int64)
        zfull,yfull=source.run(None,full)[:2]
        with np.load(args.capture,allow_pickle=False) as cap:
            missing=[f'{args.case}|{name}' for name in names if f'{args.case}|{name}' not in cap.files]
            if missing: raise RuntimeError(f"capture missing required outputs: {missing}")
            captured={name:cap[f'{args.case}|{name}'] for name in names}
            capture_keys=list(cap.files)
            case_keys=[key for key in capture_keys if key.startswith(f'{args.case}|')]
            if len(case_keys) != 6:
                raise RuntimeError(f"capture must contain exactly 6 outputs for {args.case}, got {case_keys}")
        feeds=dict(base); feeds.update(captured)
        for inp in model._remainder.get_inputs():
            if inp.name in feeds: continue
            if inp.name=='sid': feeds[inp.name]=np.array([0],np.int64)
            elif inp.name=='audio_length': feeds[inp.name]=np.zeros(tuple(int(x) for x in inp.shape),np.float32)
            elif inp.name=='cumulative_durations': feeds[inp.name]=np.zeros(tuple(int(x) for x in inp.shape),np.float32)
        allowed={x.name for x in model._remainder.get_inputs()}; feeds={k:v for k,v in feeds.items() if k in allowed}
        zf,yf=model._remainder.run(None,feeds)[:2]
        af=model._decode_mel(np.asarray(zfull),np.asarray(yfull),int(zfull.shape[2])); bf=model._decode_mel(np.asarray(zf),np.asarray(yf),int(zf.shape[2]))
        tf=_trim_silence(af); tb=_trim_silence(bf); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
        result={'case':args.case,'tokens':n,'token_ids':ids,'source_sha256':sha(args.source),'artifacts':artifacts,'capture_sha256':sha(args.capture),'capture_keys':capture_keys,'prefix_outputs':names,'runtime_piper_sha256':sha('/opt/venv/lib/python3.11/site-packages/rkvoice_stream/backends/tts/piper.py'),'raw_mel_frames':{'full':int(zfull.shape[2]),'fused':int(zf.shape[2])},'raw':{},'trimmed':{},'wav':{}}
        for label,a,t in [('fullcpu',af,tf),('dp256',bf,tb)]:
            finite=bool(np.isfinite(a).all())
            if not finite: raise ValueError(f'non-finite {label} waveform')
            result['raw'][label]={'samples':int(a.size),'finite':finite,'peak':float(np.max(np.abs(a))) if a.size else 0.0,'total_trimmed_samples':int(a.size-t.size)}
            result['trimmed'][label]={'samples':int(t.size),'finite':bool(np.isfinite(t).all()),'peak':float(np.max(np.abs(t))) if t.size else 0.0}
            result['wav'][label]=wav_write(out/f'{args.case}.{label}.wav',t,22050)
        (out/f'{args.case}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        model.release()
if __name__=='__main__': main()
