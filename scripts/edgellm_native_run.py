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
    admission = c.get('startup_admission')
    if admission is not None:
        allowed = {'min_mem_available_bytes', 'min_shm_free_bytes', 'min_root_physical_free_bytes', 'filesystem_free_floors_bytes'}
        if not isinstance(admission, dict) or set(admission) - allowed:
            raise ValueError('invalid startup_admission schema')
        for key in allowed - {'filesystem_free_floors_bytes'}:
            if key in admission and (type(admission[key]) is not int or admission[key] < 1):
                raise ValueError(f'invalid startup admission floor: {key}')
        floors = admission.get('filesystem_free_floors_bytes', {})
        if not isinstance(floors, dict) or any(not isinstance(k, str) or not k.startswith('/') or type(v) is not int or v < 1 for k, v in floors.items()):
            raise ValueError('invalid startup admission filesystem floors')
    build = c.get('build_output_contract')
    if build is not None:
        if 'client_result_path' in c:
            raise ValueError('build_output_contract is mutually exclusive with client_result_path')
        required_build = {'component','engine_path','config_path','engine_dir','config_constraints','argv','argv_pin','max_batch_size','code_len'}
        if not isinstance(build, dict) or set(build) != required_build:
            raise ValueError('invalid build_output_contract schema')
        if build.get('component') != 'code2wav':
            raise ValueError('only code2wav build component is supported')
        for key in ('engine_path', 'config_path'):
            value = build[key]
            if not isinstance(value, str) or not value.startswith('/') or any(part in {'', '.', '..'} for part in value.split('/')[1:]):
                raise ValueError(f'invalid build output path: {key}')
        if build['engine_path'] == build['config_path']:
            raise ValueError('build output paths must be distinct')
        if (not isinstance(build['argv'], list) or not build['argv'] or
                any(type(x) is not str or not x for x in build['argv']) or
                build['argv'] != c.get('client_argv')):
            raise ValueError('build recipe argv does not match client_argv')
        if type(build['engine_dir']) is not str or not build['engine_dir'].startswith('/'):
            raise ValueError('build engine_dir must be an absolute path')
        if type(build['max_batch_size']) is not int or build['max_batch_size'] <= 0:
            raise ValueError('build max_batch_size must be a positive integer')
        code_len = build['code_len']
        if (not isinstance(code_len, dict) or set(code_len) != {'min_code_len','opt_code_len','max_code_len'} or
                any(type(code_len[k]) is not int or code_len[k] <= 0 for k in code_len) or
                not (code_len['min_code_len'] <= code_len['opt_code_len'] <= code_len['max_code_len'])):
            raise ValueError('build code_len must be ordered positive integers')
        pin = build['argv_pin']
        if (not isinstance(pin, dict) or set(pin) != {'sha256','size'} or
                type(pin.get('sha256')) is not str or re.fullmatch(r'[0-9a-f]{64}', pin['sha256']) is None or
                type(pin.get('size')) is not int or pin['size'] <= 0):
            raise ValueError('build argv_pin is invalid')
        protected = {'--components': 'code2wav', '--engine-dir': build['engine_dir'],
                     '--max-batch-size': str(build['max_batch_size'])}
        for key in ('min_code_len','opt_code_len','max_code_len'):
            protected[f'--{key.replace("_", "-")}'] = str(code_len[key])
        seen = {}
        argv = build['argv']
        for i, token in enumerate(argv):
            if not isinstance(token, str) or not token.startswith('--'):
                continue
            if token == '--':
                continue
            name, equal, inline = token.partition('=')
            if name not in protected:
                if any(option.startswith(name) for option in protected if name != option):
                    raise ValueError(f'build argv uses abbreviated protected option: {name}')
                continue
            if name in seen:
                raise ValueError(f'build argv repeats protected option: {name}')
            if equal:
                value = inline
            elif i + 1 < len(argv) and isinstance(argv[i + 1], str) and not argv[i + 1].startswith('--'):
                value = argv[i + 1]
            else:
                raise ValueError(f'build argv missing value for protected option: {name}')
            if value != protected[name]:
                raise ValueError(f'build argv value mismatch for protected option: {name}')
            seen[name] = value
        if set(seen) != set(protected):
            raise ValueError('build argv lacks code2wav component, engine-dir, batch, or code lengths')
        if c.get('guardian_args', []):
            raise ValueError('build mode does not allow guardian_args')
        constraints = build['config_constraints']
        if (not isinstance(constraints, dict) or set(constraints) != {'model_type','code2wav_config','builder_config'} or
                constraints.get('model_type') != 'qwen3_tts_code2wav' or
                not isinstance(constraints.get('code2wav_config'), dict) or not constraints['code2wav_config'] or
                not isinstance(constraints.get('builder_config'), dict) or
                set(constraints['builder_config']) != {'min_code_len','opt_code_len','max_code_len'} or
                any(type(constraints['builder_config'][key]) is not int or constraints['builder_config'][key] <= 0
                    for key in ('min_code_len','opt_code_len','max_code_len')) or
                not (constraints['builder_config']['min_code_len'] <= constraints['builder_config']['opt_code_len'] <=
                     constraints['builder_config']['max_code_len']) or
                constraints['builder_config'] != code_len):
            raise ValueError('invalid code2wav config constraints')
    elif 'client_result_path' not in c:
        raise ValueError('client_result_path is required outside build mode')
    return whole,reserve

def _build_path(path, out):
    value = Path(path)
    no_symlink_ancestors(value)
    try:
        value.relative_to(out)
    except ValueError:
        raise ValueError(f'build output path outside outdir: {value}')
    return value

def _build_output_contract(c, out):
    build = c.get('build_output_contract')
    if build is None:
        return None
    engine = _build_path(build['engine_path'], out)
    config = _build_path(build['config_path'], out)
    if build['engine_dir'] != str(out):
        raise ValueError('build engine_dir must equal fresh output directory')
    if (engine.name != 'code2wav.engine' or config.name != 'config.json' or
            config.parent != out / 'code2wav' or engine.parent != out / 'code2wav'):
        raise ValueError('code2wav output paths must be code2wav/code2wav.engine and code2wav/config.json')
    if engine == config:
        raise ValueError('build output paths must be distinct')
    for label, path in (('engine', engine), ('config', config)):
        if os.path.lexists(path):
            raise RuntimeError(f'build {label} output exists')
    argv_bytes = json.dumps(build['argv'], separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    if build['argv_pin']['size'] != len(argv_bytes) or build['argv_pin']['sha256'] != hashlib.sha256(argv_bytes).hexdigest():
        raise ValueError('build frozen argv pin mismatch')
    return {'engine_path': str(engine), 'config_path': str(config),
            'component': 'code2wav', 'engine_dir': str(out), 'argv': list(build['argv']),
            'argv_pin': dict(build['argv_pin']), 'max_batch_size': build['max_batch_size'],
            'code_len': dict(build['code_len']),
            'config_constraints': dict(build['config_constraints'])}

def _build_artifact(path, deadline, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'build {label} output is missing or symlink')
    record = stream_hash(path, deadline)
    if record['size'] < 1:
        raise RuntimeError(f'build {label} output is empty')
    return record

def _guarded_json_read(path, deadline, max_size=1 << 20):
    path = Path(path); no_symlink_ancestors(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try: fd = os.open(path, flags)
    except OSError as exc: raise RuntimeError(f'build config open failed: {exc}') from exc
    try:
        before = os.fstat(fd)
        if not __import__('stat').S_ISREG(before.st_mode): raise RuntimeError('build config must be regular')
        if before.st_size > max_size: raise RuntimeError('build config exceeds max size')
        data = bytearray()
        while True:
            remaining(deadline)
            chunk = os.read(fd, min(65536, max_size + 1 - len(data)))
            if not chunk: break
            data.extend(chunk)
            if len(data) > max_size: raise RuntimeError('build config exceeds max size')
        after = os.fstat(fd)
        path_after = stat_id(path)
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or path_after != (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns):
            raise RuntimeError('build config identity changed during read')
    finally: os.close(fd)
    try: document = json.loads(bytes(data))
    except json.JSONDecodeError as exc: raise RuntimeError(f'build config JSON invalid: {exc}') from exc
    return document, {'path':str(path),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'regular':True,
                     'stat_before':(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns),
                     'stat_after':(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)}

def _json_exact(actual, expected):
    if type(actual) is not type(expected): return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_json_exact(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list): return len(actual) == len(expected) and all(_json_exact(a,e) for a,e in zip(actual,expected))
    return actual == expected

def _validate_build_config(path, constraints, deadline):
    document, record = _guarded_json_read(path, deadline)
    if not isinstance(document, dict):
        raise RuntimeError('build config JSON must be an object')
    if set(document) != set(constraints):
        raise RuntimeError('build config top-level fields mismatch')
    for key, expected in constraints.items():
        if not _json_exact(document[key], expected):
            raise RuntimeError(f'build config constraint mismatch: {key}')
    builder = document['builder_config']
    if any(type(builder[k]) is not int or builder[k] <= 0 for k in builder) or not (builder['min_code_len'] <= builder['opt_code_len'] <= builder['max_code_len']):
        raise RuntimeError('build config code lengths invalid')
    return document, record

def _startup_admission(c, guardian_path):
    spec = c.get('startup_admission')
    if spec is None:
        return None
    module = importlib.util.spec_from_file_location('startup_admission_guardian', guardian_path)
    if module is None or module.loader is None:
        raise RuntimeError('startup admission guardian helper unavailable')
    guardian = importlib.util.module_from_spec(module)
    module.loader.exec_module(guardian)
    floors = dict(spec.get('filesystem_free_floors_bytes', {}))
    snapshot = guardian.resource_snapshot(floors)
    failure = guardian.resource_guard_failure(
        snapshot,
        spec.get('min_mem_available_bytes', 1),
        spec.get('min_shm_free_bytes', 1),
        spec.get('min_root_physical_free_bytes', 1),
        floors,
    )
    evidence = {'status': 'FAILED' if failure else 'PASS', 'snapshot': snapshot, 'floors': spec}
    if failure:
        evidence['failure'] = failure
        error = RuntimeError('startup admission resource floor failed')
        error.evidence = evidence
        raise error
    return evidence

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

def _check_sdk_module_argv(c, build_contract):
    """Reject the known SDK direct-file invocation before any producer launch."""
    if build_contract is None or 'launch_preflight' not in c:
        return
    name = c['launch_preflight'].get('module', {}).get('name') if isinstance(c.get('launch_preflight'), dict) else None
    if name != 'experimental.builder.cli':
        return
    argv = c.get('client_argv', [])
    expected_prefix = [str(c['runtime']['python']), '-m', 'experimental.builder.cli']
    if argv[:3] != expected_prefix:
        raise ValueError('SDK build client_argv must use python -m experimental.builder.cli')

def run(cfgpath, preflight_only=False):
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
        build_contract = _build_output_contract(c, out)
        if build_contract is not None:
            result['build_output_contract'] = build_contract
        _check_sdk_module_argv(c, build_contract)
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
        result['artifacts'] = artifacts
        result['runtime'] = runtime
        result['guardian'] = guardian
        before=snapshot(c['snapshot_cmd'],deadline); result['foreign_before']=before
        child_out = None
        if build_contract is None:
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
        if preflight_only:
            result['startup_admission'] = {'status': 'NOT_EVALUATED', 'reason': 'preflight-only mode does not inspect or launch GPU admission'}
            after = snapshot(c['snapshot_cmd'], deadline)
            result['foreign_after'] = after
            result['foreign_after_match'] = before == after
            if not result['foreign_after_match']:
                raise RuntimeError('foreign snapshot changed during preflight')
            artifacts_after = [stream_hash(x['path'], deadline) for x in c['artifacts']]
            runtime_after = stream_hash(runtime_path, deadline)
            guardian_after = stream_hash(c['guardian']['path'], deadline)
            result['artifacts_after'] = artifacts_after
            result['runtime_after'] = runtime_after
            result['guardian_after'] = guardian_after
            if (any(a['sha256'] != b['sha256'] or a['size'] != b['size'] or a['stat_before'] != b['stat_before']
                    for a, b in zip(artifacts, artifacts_after)) or
                    runtime != runtime_after or guardian != guardian_after):
                raise RuntimeError('input pin drift during preflight')
            result['status'] = 'INPUTS_PREFLIGHT_VERIFIED'
            result['close_complete'] = True
            return result
        if 'startup_admission' in c:
            try:
                result['startup_admission'] = _startup_admission(c, c['guardian']['path'])
            except Exception as exc:
                if hasattr(exc, 'evidence'): result['startup_admission'] = exc.evidence
                else: result['startup_admission'] = {'status': 'FAILED', 'reason': f'{type(exc).__name__}: {exc}'}
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
        if build_contract is None:
            if not child_out.is_file() or child_out.is_symlink(): raise RuntimeError('client result missing or symlink')
            child_rec=stream_hash(child_out,deadline); result['child_result']=child_rec
        else:
            result['engine_artifact'] = _build_artifact(build_contract['engine_path'], deadline, 'engine')
            result['build_config'], result['config_artifact'] = _validate_build_config(build_contract['config_path'], build_contract['config_constraints'], deadline)
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
        if build_contract is not None:
            result['recipe'] = {'argv': list(c['client_argv']), 'argv_pin': dict(build_contract['argv_pin']), 'input_pins': artifacts}
            result['input_pins'] = {'argv': dict(build_contract['argv_pin']), 'artifacts': artifacts}
            result['profile_validation'] = 'UNPROVEN'
            result['production_qualification'] = 'UNPROVEN'
            good = good and before == after
            if good:
                result['build_status'] = 'OUTPUT_VERIFIED'
        if good and before==after:
            result['status']='BUILD_OUTPUT_VERIFIED' if build_contract is not None else 'RAW_OBSERVED'; result['close_complete']=True
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
    try:
        if len(sys.argv) == 3 and sys.argv[1] == '--preflight-only':
            r = run(sys.argv[2], preflight_only=True)
        elif len(sys.argv) == 2:
            r = run(sys.argv[1])
        else:
            raise ValueError('usage: edgellm_native_run.py [--preflight-only] CONFIG')
        print(json.dumps(r,sort_keys=True))
        raise SystemExit(0 if r.get('status') in {'RAW_OBSERVED','BUILD_OUTPUT_VERIFIED','INPUTS_PREFLIGHT_VERIFIED'} else 1)
    except Exception as e: print(json.dumps({'status':'UNPROVEN','reason':f'{type(e).__name__}: {e}'})); raise SystemExit(1)
