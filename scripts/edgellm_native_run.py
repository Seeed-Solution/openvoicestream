#!/usr/bin/env python3
"""Bounded CPU producer: frozen inputs, c0dd guardian, raw evidence only."""
from __future__ import annotations
import hashlib,json,math,os,pwd,subprocess,sys,time,importlib.util,re,signal,selectors
from pathlib import Path

class Bound(Exception): pass

def _guardian_proc_identity(pid, argv, guardian_path):
    spec=importlib.util.spec_from_file_location('accepted_guardian', guardian_path)
    if spec is None or spec.loader is None: raise RuntimeError('guardian identity helper unavailable')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.proc_identity(pid, argv)

def remaining(deadline, reserve=0.0):
    r=deadline-time.monotonic()-reserve
    if r <= 0: raise Bound("deadline/reserve exhausted")
    return r

def no_symlink_under(p, root, allow_missing=True):
    p=Path(p); root=Path(root)
    if not p.is_absolute() or not root.is_absolute(): raise ValueError("absolute path required")
    try: rel=p.relative_to(root)
    except ValueError: raise ValueError("path outside HOME")
    if any(part in {"..", "."} for part in rel.parts): raise ValueError("lexical path traversal")
    cur=root
    if cur.is_symlink(): raise ValueError("HOME symlink")
    for part in rel.parts:
        cur=cur/part
        if cur.is_symlink(): raise ValueError("symlink path component")
        if cur.exists() and not (cur.is_file() or cur.is_dir()): raise ValueError("invalid path component")
        if not cur.exists() and not allow_missing: raise ValueError("missing path component")
    return p

def stat_id(p):
    s=os.stat(p,follow_symlinks=False)
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def no_symlink_ancestors(p):
    p = Path(p)
    current = Path(p.anchor)
    for part in p.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f'path contains symlink ancestor: {current}')
    return p

def stream_hash(p, deadline, require_regular=True):
    p=Path(p)
    if p.is_symlink() or not p.is_file(): raise ValueError(f"regular file required: {p}")
    before=stat_id(p); h=hashlib.sha256(); n=0
    with p.open('rb') as f:
        while True:
            remaining(deadline)
            b=f.read(1<<20)
            if not b: break
            h.update(b); n+=len(b)
    after=stat_id(p)
    if before != after: raise RuntimeError(f"file mutated during hash: {p}")
    if n != before[2]: raise RuntimeError(f"file size changed during hash: {p}")
    return {'path':str(p),'size':n,'sha256':h.hexdigest(),'regular':True,'stat_before':before,'stat_after':after}

def snapshot(cmd, deadline):
    q=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=remaining(deadline))
    if q.returncode!=0: raise RuntimeError(f"snapshot rc={q.returncode}: {q.stderr[:200]}")
    rows=json.loads(q.stdout)
    if not isinstance(rows,list): raise ValueError('snapshot must be JSON array')
    req={'Id','ImageID','Name','Running','Pid','RestartCount'}; out=[]
    for x in rows:
        if not isinstance(x,dict) or set(x)!=req: raise ValueError('snapshot fields must be exact')
        if not isinstance(x['Id'],str) or not x['Id'] or not isinstance(x['ImageID'],str) or not isinstance(x['Name'],str) or not isinstance(x['Running'],bool) or type(x['Pid']) is not int or type(x['RestartCount']) is not int: raise ValueError('snapshot field types invalid')
        out.append(x)
    return sorted(out,key=lambda x:(x['Id'],x['Name']))

def check_cfg(c):
    if 'client_result_sha256' in c: raise ValueError('client_result_sha256 is not an accepted input pin')
    whole=float(c.get('whole_s')); reserve=float(c.get('reserve_s'))
    if not math.isfinite(whole) or not math.isfinite(reserve) or whole<=0 or reserve<0 or reserve>=whole: raise ValueError('invalid whole/reserve budget')
    if type(c.get('max_output_bytes')) is not int or c['max_output_bytes']<=0: raise ValueError('invalid max output')
    if type(c.get('expected_uid')) is not int or c['expected_uid']<0: raise ValueError('invalid expected uid')
    return whole,reserve

def _preflight_pin(value, label):
    if not isinstance(value, dict) or set(value) != {'path','sha256','size'} or not isinstance(value.get('path'), str) or not value['path'].startswith('/'):
        raise ValueError(f'{label} origin pin is invalid')
    if not isinstance(value.get('sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', value['sha256']) is None:
        raise ValueError(f'{label} origin sha256 is invalid')
    if type(value.get('size')) is not int or value['size'] < 1:
        raise ValueError(f'{label} origin size is invalid')
    return value

def _preflight_paths(c, env, preflight):
    if not isinstance(preflight, dict) or set(preflight) != {'module','write_paths','min_free_bytes'}:
        raise ValueError('launch_preflight must be an object')
    module = preflight.get('module')
    if (not isinstance(module, dict) or set(module) != {'name','origin'} or
            not isinstance(module.get('name'), str) or not module['name'] or not isinstance(module.get('origin'), dict)):
        raise ValueError('launch_preflight module name/origin is required')
    origin = _preflight_pin(module['origin'], 'launch_preflight module')
    no_symlink_ancestors(origin['path'])
    paths = preflight.get('write_paths', [])
    if not isinstance(paths, list) or not paths or any(not isinstance(x, str) or not x.startswith('/') for x in paths):
        raise ValueError('launch_preflight write_paths must be a nonempty absolute string list')
    floors = preflight.get('min_free_bytes', {})
    if not isinstance(floors, dict) or any(not isinstance(k, str) or not k.startswith('/') or type(v) is not int or v < 1 for k, v in floors.items()):
        raise ValueError('launch_preflight min_free_bytes must map absolute paths to positive integers')
    out_parent = str(Path(c['outdir']).parent)
    home = Path(c['home']).resolve(strict=True)
    approved = [Path(out_parent).resolve()]
    for raw in paths:
        path = no_symlink_ancestors(raw)
        resolved = path.resolve()
        try:
            resolved.relative_to(home)
        except ValueError:
            raise ValueError(f'launch_preflight write path outside HOME: {path}')
        approved.append(resolved)
    for key in ('TMPDIR', 'CUDA_CACHE_PATH'):
        value = env.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.startswith('/'):
                raise ValueError(f'launch_preflight {key} must be an absolute path')
            if str(Path(value).resolve()) not in {str(x) for x in approved}:
                raise ValueError(f'launch_preflight {key} requires an explicit approved write path')
    ordered = list(dict.fromkeys([out_parent] + paths))
    return {'module': {'name': module['name'], 'origin': origin},
            'write_paths': ordered, 'min_free_bytes': dict(floors)}

def _bounded_probe(cmd, cwd, env, deadline, max_bytes):
    out_read, out_write = os.pipe(); err_read, err_write = os.pipe()
    for fd in (out_read, err_read): os.set_blocking(fd, False)
    proc = None; term_sent = False; handoff = False; timed_out = False; output_limited = False
    stdout = bytearray(); stderr = bytearray(); stdout_total = 0; stderr_total = 0
    sel = selectors.DefaultSelector()
    sel.register(out_read, selectors.EVENT_READ, 'stdout'); sel.register(err_read, selectors.EVENT_READ, 'stderr')

    def drain():
        nonlocal output_limited, stdout_total, stderr_total
        for key, _ in sel.select(0):
            fd, target = key.fd, stdout if key.data == 'stdout' else stderr
            while True:
                try: chunk = os.read(fd, 65536)
                except BlockingIOError: break
                except OSError: chunk = b''
                if not chunk:
                    try: sel.unregister(fd)
                    except KeyError: pass
                    break
                if key.data == 'stdout': stdout_total += len(chunk)
                else: stderr_total += len(chunk)
                room = max_bytes - len(stdout) - len(stderr)
                if len(chunk) > room:
                    target.extend(chunk[:max(0, room)])
                    output_limited = True
                    break
                target.extend(chunk)
                if len(stdout) + len(stderr) > max_bytes:
                    output_limited = True
                    break
            if output_limited: break

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out_write, stderr=err_write,
                                cwd=cwd, env=env, start_new_session=True)
        os.close(out_write); out_write = None; os.close(err_write); err_write = None
        while proc.poll() is None and not output_limited:
            drain()
            if output_limited: break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            sel.select(min(0.02, max(0.001, deadline - time.monotonic())))
            drain()
        if proc.poll() is not None and not output_limited:
            drain()
        if (timed_out or output_limited) and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM); term_sent = True
            except ProcessLookupError:
                pass
            reap_deadline = min(deadline + 0.25, time.monotonic() + 0.25)
            while proc.poll() is None and time.monotonic() < reap_deadline:
                drain(); time.sleep(0.01)
        rc = proc.poll()
        if rc is None:
            handoff = True
        else:
            try:
                os.killpg(proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                pass
            else:
                handoff = True
        drain()
        return {'returncode': rc, 'stdout': bytes(stdout).decode('utf-8', 'replace'),
                'stderr': bytes(stderr).decode('utf-8', 'replace'), 'term_issued': term_sent,
                'survivor_handoff': handoff, 'timed_out': timed_out,
                'output_limited': output_limited,
                'stdout_bytes': stdout_total, 'stderr_bytes': stderr_total,
                'stdout_evidence_bytes': len(stdout), 'stderr_evidence_bytes': len(stderr)}
    finally:
        try: sel.close()
        finally:
            for fd in (out_read, err_read, out_write, err_write):
                if fd is not None:
                    try: os.close(fd)
                    except OSError: pass

def _launch_preflight(c, env, runtime_path, deadline):
    spec = _preflight_paths(c, env, c['launch_preflight'])
    origin = Path(spec['module']['origin']['path'])
    origin_rec = stream_hash(origin, deadline)
    if origin_rec['sha256'] != spec['module']['origin']['sha256'] or origin_rec['size'] != spec['module']['origin']['size']:
        raise RuntimeError('launch_preflight module origin pin mismatch')
    write_paths = []
    for raw in spec['write_paths']:
        path = no_symlink_ancestors(raw)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ValueError(f'launch_preflight write path must be an existing regular directory: {path}')
        st = path.stat()
        try:
            statvfs = os.statvfs(path)
            free = statvfs.f_bavail * statvfs.f_frsize
        except OSError as exc:
            raise RuntimeError(f'launch_preflight statvfs failed for {path}: {exc}') from exc
        floor = spec['min_free_bytes'].get(str(path), 1)
        row = {'path': str(path), 'uid': st.st_uid, 'free_bytes': free, 'min_free_bytes': floor}
        write_paths.append(row)
        if st.st_uid != c['expected_uid']:
            raise PermissionError(f'launch_preflight write path uid mismatch: {path}')
        if free < floor:
            raise OSError(f'launch_preflight free space below floor: {path}')
    probe_code = r'''import importlib.util, json, os, pathlib, sys, tempfile
paths=json.loads(sys.argv[1]); expected_uid=int(sys.argv[2]); module_name=sys.argv[3]; origin=str(pathlib.Path(sys.argv[4]).resolve())
if os.getuid() != expected_uid: raise RuntimeError(f"uid mismatch: {os.getuid()} != {expected_uid}")
for raw in paths:
    p=pathlib.Path(raw)
    fd, name=tempfile.mkstemp(prefix='.slv-launch-preflight-', dir=str(p))
    try:
        os.write(fd, b'preflight'); os.close(fd); os.unlink(name)
    finally:
        try: os.close(fd)
        except OSError: pass
        try: os.unlink(name)
        except FileNotFoundError: pass
spec=importlib.util.find_spec(module_name)
actual=None if spec is None else spec.origin
if not actual or str(pathlib.Path(actual).resolve()) != origin: raise RuntimeError(f"module origin mismatch: {actual!r} != {origin!r}")
print(json.dumps({'uid': os.getuid(), 'module': module_name, 'origin': str(pathlib.Path(actual).resolve()), 'write_paths': paths}, sort_keys=True))
'''
    cmd = [str(runtime_path), '-c', probe_code, json.dumps([x['path'] for x in write_paths]), str(c['expected_uid']), spec['module']['name'], str(origin)]
    try:
        q = _bounded_probe(cmd, str(c['cwd']), env, deadline=min(deadline, time.monotonic() + remaining(deadline, 0.05)), max_bytes=c['max_output_bytes'])
    except Exception as exc:
        raise RuntimeError(f'launch_preflight probe failed: {type(exc).__name__}: {exc}') from exc
    protocol = None
    if q['returncode'] == 0 and not q['survivor_handoff'] and not q['timed_out'] and not q['output_limited']:
        lines = q['stdout'].splitlines()
        if len(lines) == 1:
            try:
                protocol = json.loads(lines[0])
            except json.JSONDecodeError:
                protocol = None
    expected_probe = {'uid': c['expected_uid'], 'module': spec['module']['name'],
                      'origin': str(origin.resolve()), 'write_paths': [x['path'] for x in write_paths]}
    protocol_valid = (isinstance(protocol, dict) and set(protocol) == set(expected_probe) and
                      type(protocol.get('uid')) is int and protocol['uid'] == expected_probe['uid'] and
                      isinstance(protocol.get('module'), str) and protocol['module'] == expected_probe['module'] and
                      isinstance(protocol.get('origin'), str) and protocol['origin'] == expected_probe['origin'] and
                      isinstance(protocol.get('write_paths'), list) and
                      all(isinstance(x, str) for x in protocol['write_paths']) and
                      protocol['write_paths'] == expected_probe['write_paths'])
    evidence = {'status': 'PASS' if q['returncode'] == 0 and protocol_valid else 'FAILED', 'command_argv': cmd,
                'returncode': q['returncode'], 'stdout': q['stdout'], 'stderr': q['stderr'],
                'stdout_bytes': q['stdout_bytes'], 'stderr_bytes': q['stderr_bytes'],
                'stdout_evidence_bytes': q['stdout_evidence_bytes'], 'stderr_evidence_bytes': q['stderr_evidence_bytes'],
                'term_issued': q['term_issued'], 'survivor_handoff': q['survivor_handoff'],
                'timed_out': q['timed_out'], 'output_limited': q['output_limited'],
                'write_paths': write_paths, 'module_origin': origin_rec}
    if evidence['status'] != 'PASS':
        error = RuntimeError(f"launch_preflight probe failed rc={q['returncode']}: {q['stderr'][:200]}")
        error.evidence = evidence
        raise error
    evidence['probe'] = protocol
    return evidence

def run(cfgpath):
    start=time.monotonic(); deadline=None; out=None; out_created=False
    gp=None; after_captured=False
    result={'status':'UNPROVEN','phase':'llm.b1','start_mono':start,'foreign_before':None,'foreign_after':None,'guardian_rc':None,'guardian_result':None}
    try:
        c=json.loads(Path(cfgpath).read_text()); whole,reserve=check_cfg(c); deadline=start+whole
        uid=os.getuid(); actual_home=Path(os.environ.get('HOME','')).resolve(strict=True)
        if uid != c['expected_uid'] or pwd.getpwuid(uid).pw_dir != str(actual_home): raise ValueError('UID/HOME identity mismatch')
        if Path(c['home']).resolve(strict=True) != actual_home: raise ValueError('config HOME differs from actual HOME')
        out=Path(c['outdir']); no_symlink_under(out,actual_home); no_symlink_under(Path(c['cwd']),actual_home,allow_missing=False)
        if os.path.lexists(out): raise ValueError('fresh output directory required')
        remaining(deadline,reserve)
        artifacts=[]
        for x in c['artifacts']:
            e=stream_hash(x['path'],deadline); artifacts.append(e)
            if e['sha256']!=x['sha256'] or e['size']!=x['size']: raise RuntimeError(f"pin mismatch {x['role']}")
        runtime_path=Path(c['runtime']['python']).resolve(strict=True)
        runtime=stream_hash(runtime_path,deadline)
        guardian=stream_hash(c['guardian']['path'],deadline)
        if guardian['sha256'] != c['guardian']['sha256']: raise RuntimeError('guardian pin mismatch')
        if not isinstance(c['runtime'].get('sha256'),str) or runtime['sha256'] != c['runtime']['sha256']: raise RuntimeError('runtime pin mismatch')
        before=snapshot(c['snapshot_cmd'],deadline); result['foreign_before']=before
        child_out=Path(c['client_result_path']); no_symlink_under(child_out,out.parent); no_symlink_under(child_out,out)
        if os.path.lexists(child_out): raise RuntimeError('client result exists')
        env=dict(os.environ); env.update({str(k):str(v) for k,v in c.get('env',{}).items()})
        if 'HOME' in env and env['HOME'] != os.environ.get('HOME'): raise ValueError('config may not override HOME')
        if 'launch_preflight' in c:
            try:
                result['launch_preflight'] = _launch_preflight(c, env, runtime_path, deadline)
            except Exception as exc:
                if hasattr(exc, 'evidence'): result['launch_preflight'] = exc.evidence
                else: result['launch_preflight'] = {'status': 'FAILED', 'reason': f'{type(exc).__name__}: {exc}'}
                raise
        out.mkdir(); out_created=True
        guarddir=out/'guardian'; logs=out/'transport'; logs.mkdir()
        remaining(deadline,reserve)
        gargv=[c['runtime']['python'],c['guardian']['path'],'--outdir',str(guarddir),'--timeout',str(remaining(deadline,reserve)),'--reserve','0.05','--max-output',str(c['max_output_bytes']),'--min-mem-available-bytes',str(c['min_mem_available_bytes']),'--min-shm-free-bytes',str(c['min_shm_free_bytes'])]+list(c.get('guardian_args',[]))+['--']+list(c['client_argv'])
        result['actual_guardian_argv']=gargv
        with (logs/'guardian.stdout').open('x') as so, (logs/'guardian.stderr').open('x') as se:
            gp=subprocess.Popen(gargv,stdin=subprocess.DEVNULL,stdout=so,stderr=se,cwd=str(c['cwd']),env=env)
            result['guardian_pid']=gp.pid
            result['guardian_identity'] = _guardian_proc_identity(gp.pid, gargv, c['guardian']['path'])
            try: grc=gp.wait(timeout=remaining(deadline,reserve))
            except subprocess.TimeoutExpired:
                gr=result.get('guardian_result')
                rp=guarddir/'result.json'
                if rp.is_file() and not rp.is_symlink():
                    try: result['guardian_result']=json.loads(rp.read_text())
                    except Exception: pass
                result['guardian_handoff']={'pid':gp.pid,'running':gp.poll() is None,'identity':result.get('guardian_identity'),'argv':gargv,'guardpath':str(guarddir),'reason':'parent deadline','child_identity':(result.get('guardian_result') or {}).get('identity') if isinstance(result.get('guardian_result'),dict) else None}
                raise Bound('guardian exceeded parent deadline')
        result['guardian_rc']=grc; result['actual_guardian_argv']=gargv
        rp=guarddir/'result.json'
        if rp.is_file() and not rp.is_symlink(): result['guardian_result']=json.loads(rp.read_text())
        if not child_out.is_file() or child_out.is_symlink(): raise RuntimeError('client result missing or symlink')
        child_rec=stream_hash(child_out,deadline); result['child_result']=child_rec
        result['raw_guardian_stdout']=stream_hash(logs/'guardian.stdout',deadline); result['raw_guardian_stderr']=stream_hash(logs/'guardian.stderr',deadline)
        artifacts_after=[stream_hash(x['path'],deadline) for x in c['artifacts']]
        result['artifacts_before']=artifacts; result['artifacts_after']=artifacts_after
        if any(a['sha256']!=b['sha256'] or a['size']!=b['size'] or a['stat_before']!=b['stat_before'] for a,b in zip(artifacts,artifacts_after)):
            raise RuntimeError('artifact changed during producer run')
        after=snapshot(c['snapshot_cmd'],deadline); result['foreign_after']=after; after_captured=True
        gr=result['guardian_result'] or {}; ident=gr.get('identity') if isinstance(gr,dict) else None
        expected_exe=str(runtime_path)
        good=(gr.get('outcome') in {'EXITED_TERM_EXITED','EXITED'} and type(gr.get('child_rc')) is int and gr.get('child_rc')==0 and type(gr.get('term_count')) is int and gr.get('term_count')==0 and gr.get('survivor_handoff') is False and isinstance(ident,dict) and type(ident.get('pid')) is int and ident['pid']>0 and type(ident.get('start_ticks')) is int and ident['start_ticks']>0 and ident.get('argv')==list(c['client_argv']) and str(Path(ident.get('exe','')).resolve())==expected_exe and ident.get('exe_sha256')==runtime['sha256'])
        for k,n in [('stdout','child.stdout'),('stderr','child.stderr')]:
            x=gr.get(k) if isinstance(gr,dict) else None; gpth=(guarddir/n).resolve()
            if not isinstance(x,dict) or Path(str(x.get('path',''))).resolve()!=gpth or x.get('bytes')!=stat_id(gpth)[2] or x.get('sha256')!=stream_hash(gpth,deadline)['sha256']: good=False
        if good and before==after: result['status']='RAW_OBSERVED'; result['close_complete']=True
        else: result['reason']='guardian/client/identity/log/foreign evidence failed validation'
    except Exception as e:
        if isinstance(c if 'c' in locals() else None, dict) and 'launch_preflight' in c and 'launch_preflight' not in result:
            result['launch_preflight'] = {'status': 'FAILED', 'reason': f'{type(e).__name__}: {e}'}
        result['reason']=f'{type(e).__name__}: {e}'
    finally:
        result['end_mono']=time.monotonic(); result['elapsed_s']=result['end_mono']-start
        # Preserve the operational after-snapshot even when client-result validation
        # fails.  Only capture after the owned guardian has actually reaped; an
        # alive guardian/child must remain an explicit unresolved handoff.
        if result.get('foreign_before') is not None and not after_captured:
            if gp is not None and gp.poll() is not None:
                try:
                    if deadline is None or deadline <= time.monotonic():
                        raise Bound('after snapshot deadline exhausted')
                    result['foreign_after']=snapshot(c['snapshot_cmd'],deadline)
                    after_captured=True
                    result['foreign_after_observed']='guardian_reaped_fallback'
                except Exception as e:
                    result['foreign_after_error']={'status':'UNPROVEN','reason':f'{type(e).__name__}: {e}'}
            else:
                result['foreign_after_error']={'status':'UNPROVEN','reason':'owned guardian not reaped'}
        if result.get('foreign_before') is not None and result.get('foreign_after') is not None:
            result['foreign_after_match']=result['foreign_before']==result['foreign_after']
        if out is not None and out_created and out.exists():
            payload=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
            fd=os.open(out/'producer-result.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            try: os.write(fd,payload)
            finally: os.close(fd)
    return result

if __name__=='__main__':
    try: r=run(sys.argv[1]); print(json.dumps(r,sort_keys=True)); raise SystemExit(0 if r.get('status')=='RAW_OBSERVED' else 1)
    except Exception as e: print(json.dumps({'status':'UNPROVEN','reason':f'{type(e).__name__}: {e}'})); raise SystemExit(1)
