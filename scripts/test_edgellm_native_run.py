import hashlib,json,os,pwd,signal,subprocess,sys,tempfile,shutil,time
import importlib.util
from pathlib import Path
ROOT=Path(__file__).with_name('edgellm_native_run.py').parent; PROD=Path(__file__).with_name('edgellm_native_run.py'); GUARD=Path(__file__).with_name('edgellm_native_guardian.py')
assert '/tmp/slv-v011-llm-guardian-root-recovery-20261004' not in PROD.read_text() and '/tmp/slv-v011-llm-guardian-root-recovery-20261004' not in GUARD.read_text()
def proc_identity(pid):
 try:
  stat_text=Path('/proc',str(pid),'stat').read_text()
  tail=stat_text.rsplit(')',1)[1].split()
  state=tail[0]; start_ticks=int(tail[19])
  exe=os.readlink('/proc/'+str(pid)+'/exe')
  return {'pid':pid,'state':state,'start_ticks':start_ticks,'exe':exe}
 except (FileNotFoundError,ProcessLookupError,ValueError,OSError):
  return None
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,s): Path(p).write_text(s); return Path(p)
def base(tag, mode='ok', timeout=.8):
 d=Path(tempfile.mkdtemp(prefix='producer-'+tag+'-',dir=os.environ['HOME'])); art=d/'client.py'; out=d/'out'; result=out/'client-result.json'; snap=d/'snapshot.py'; staged_guard=d/'guardian-staged.py'; shutil.copyfile(GUARD,staged_guard)
 write(art,'''import json,sys,time
mode=sys.argv[1]; out=sys.argv[2]
print("O"*1500000,flush=True); print("E"*1500000,file=sys.stderr,flush=True)
if mode == "sleep": time.sleep(2)
if mode in ("bad", "bad_snapshot_error", "bad_foreign_change"): raise SystemExit(7)
if mode == "mutate": open(__file__,"ab").write(b"mutation")
json.dump({"status":"DONE","closed":True,"source_contract":{"bundle_matches":True,"is_ready":True},"text":"4"},open(out,"w"))
''')
 write(snap,'''#!/usr/bin/env python3
import json,sys
mode=sys.argv[1]; counter=sys.argv[2]
try: n=int(open(counter).read())+1
except FileNotFoundError: n=1
open(counter,"w").write(str(n))
if mode == "bad_snapshot_error" and n > 1: raise SystemExit(9)
ident="foreign-after" if mode in ("foreign_change", "bad_foreign_change") and n > 1 else "foreign-id"
print(json.dumps([{"Id":ident,"ImageID":"sha256:foreign","Name":"/foreign","Running":True,"Pid":123,"RestartCount":0}]))
'''); snap.chmod(0o755)
 cfg={'whole_s':timeout,'reserve_s':.1,'expected_uid':os.getuid(),'max_output_bytes':4_000_000,'min_mem_available_bytes':1,'min_shm_free_bytes':1,'outdir':str(out),'home':os.environ['HOME'],'cwd':str(d),'runtime':{'python':sys.executable,'sha256':sha(sys.executable)},'guardian':{'path':str(staged_guard),'sha256':sha(staged_guard)},'artifacts':[{'role':'client','path':str(art),'sha256':sha(art),'size':art.stat().st_size}], 'snapshot_cmd':[str(snap),mode,str(d/'snapshot-count')], 'client_argv':[sys.executable,str(art),mode,str(result)], 'client_result_path':str(result),'env':{}}
 c=d/'config.json'; c.write_text(json.dumps(cfg)); return d,c,cfg,out

def run(tag,mode='ok',timeout=.8,mut=None):
 d,c,cfg,out=base(tag,mode,timeout)
 if mut: mut(cfg); c.write_text(json.dumps(cfg))
 p=subprocess.run([sys.executable,str(PROD),str(c)],capture_output=True,text=True); return p,d,c,out
rows=[]
p,d,c,o=run('ok'); rows.append(('success',p.returncode,p.stdout[-200:]))
ok=json.loads(p.stdout); assert p.returncode==0 and ok['status']=='RAW_OBSERVED' and ok['guardian_rc']==0
p,d,c,o=run('bad','bad'); rows.append(('nonzero',p.returncode,p.stdout[-200:]))
bad_result=json.loads(p.stdout)
assert p.returncode!=0 and bad_result['status']!='RAW_OBSERVED' and bad_result['guardian_rc']!=0
assert bad_result['foreign_after_observed']=='guardian_reaped_fallback' and bad_result['foreign_after_match'] is True
assert 'client result missing' in bad_result['reason']
p,d,c,o=run('bad_snapshot_error','bad_snapshot_error'); rows.append(('after_snapshot_error',p.returncode,p.stdout[-200:]))
snapshot_error_result=json.loads(p.stdout)
assert p.returncode!=0 and snapshot_error_result['foreign_after'] is None
assert snapshot_error_result['foreign_after_error']['status']=='UNPROVEN'
assert 'client result missing' in snapshot_error_result['reason']
p,d,c,o=run('bad_foreign_change','bad_foreign_change'); rows.append(('missing_client_foreign_mismatch',p.returncode,p.stdout[-200:]))
missing_mismatch_result=json.loads(p.stdout)
assert p.returncode!=0 and missing_mismatch_result['foreign_after_match'] is False
assert 'client result missing' in missing_mismatch_result['reason']
p,d,c,o=run('foreign_change','foreign_change'); rows.append(('foreign_after_mismatch',p.returncode,p.stdout[-200:]))
mismatch_result=json.loads(p.stdout)
assert p.returncode!=0 and mismatch_result['foreign_after_match'] is False
assert mismatch_result['reason']=='guardian/client/identity/log/foreign evidence failed validation'
p,d,c,o=run('sleep','sleep',.5); rows.append(('timeout',p.returncode,p.stdout[-200:]))
timeout_result=json.loads(p.stdout); assert p.returncode!=0 and isinstance(timeout_result.get('guardian_handoff',{}).get('identity'),dict) and timeout_result['guardian_handoff']['identity']['pid']>0
p,d,c,o=run('stale',mut=lambda x:x.update(client_result_sha256='0'*64)); rows.append(('stale_client_hash',p.returncode,p.stdout[-200:]))
assert p.returncode!=0 and 'client_result_sha256' in p.stdout
p,d,c,o=run('pin',mut=lambda x:x['artifacts'][0].update(sha256='f'*64)); rows.append(('wrong_artifact_pin',p.returncode,p.stdout[-200:]))
p,d,c,o=run('mutate','mutate'); rows.append(('artifact_mutation',p.returncode,p.stdout[-200:]))
p,d,c,o=run('escape',mut=lambda x:x.update(client_result_path=str(Path(x['home'])/'outside-result.json'))); rows.append(('path_escape',p.returncode,p.stdout[-200:]))
p,d,c,o=run('dotdot',mut=lambda x:x.update(client_result_path=str(Path(x['outdir']).parent/'..'/'escape-result.json'))); rows.append(('dotdot_escape',p.returncode,p.stdout[-200:]))
p,d,c,o=run('wrong_uid',mut=lambda x:x.update(expected_uid=x['expected_uid']+1)); rows.append(('wrong_uid',p.returncode,p.stdout[-200:]))
p,d,c,o=run('wrong_home',mut=lambda x:x.update(home=str(Path(x['home'])/'other-home'))); rows.append(('wrong_home',p.returncode,p.stdout[-200:]))
p,d,c,o=run('identity_missing','sleep',.8,mut=lambda x:x.update(guardian_args=['--identity-test','missing'])); rows.append(('identity_missing',p.returncode,p.stdout[-200:]))
p,d,c,o=run('existing'); (o).mkdir(parents=True,exist_ok=True); sentinel=o/'producer-result.json'; sentinel.write_text('SENTINEL\n'); before=sha(sentinel); p2=subprocess.run([sys.executable,str(PROD),str(c)],capture_output=True,text=True); rows.append(('existing_output',p2.returncode,p2.stdout[-200:])); assert p2.returncode!=0 and sha(sentinel)==before and sentinel.read_text()=='SENTINEL\n'
print(json.dumps(rows,indent=2))

# Optional launch admission: prove the configured runtime/env/module and
# writable filesystems before native guardian Popen.  These tests call run()
# directly so a forbidden guardian launch can be observed without a process.
spec = importlib.util.spec_from_file_location('native_run_under_test', PROD)
native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)

def preflight_cfg(tag):
    d,c,cfg,out = base('preflight-' + tag)
    module_name = 'experimental_entry_' + tag.replace('-', '_')
    module_root = d / 'module-src'; module_root.mkdir()
    module = module_root / (module_name + '.py')
    module.write_text('VALUE = 1\n')
    cfg['env'] = {'PYTHONPATH': str(module_root)}
    cfg['launch_preflight'] = {
        'module': {'name': module_name, 'origin': {'path': str(module), 'sha256': sha(module), 'size': module.stat().st_size}},
        'write_paths': [str(d)], 'min_free_bytes': {str(d): 1},
    }
    c.write_text(json.dumps(cfg))
    return d,c,cfg,out

def assert_no_guardian_launch(cfg_path, *, low_space=False):
    calls = []
    original_popen = native.subprocess.Popen
    original_statvfs = native.os.statvfs
    guardian_path = json.loads(Path(cfg_path).read_text())['guardian']['path']
    def forbidden(*args, **kwargs):
        argv = list(args[0]) if args else []
        if guardian_path in argv:
            calls.append(argv)
            raise AssertionError('guardian Popen must not be reached after failed launch preflight')
        return original_popen(*args, **kwargs)
    class Low:
        f_bavail = 0
        f_frsize = 4096
    try:
        native.subprocess.Popen = forbidden
        if low_space: native.os.statvfs = lambda _path: Low()
        result = native.run(str(cfg_path))
    finally:
        native.subprocess.Popen = original_popen
        native.os.statvfs = original_statvfs
    assert not calls, calls
    assert result['status'] == 'UNPROVEN'
    assert result['launch_preflight']['status'] == 'FAILED'
    return result

d,c,cfg,out = preflight_cfg('success')
success = subprocess.run([sys.executable, str(PROD), str(c)], capture_output=True, text=True)
success_result = json.loads(success.stdout)
assert success.returncode == 0 and success_result['status'] == 'RAW_OBSERVED'
assert success_result['launch_preflight']['status'] == 'PASS'

d,c,cfg,out = preflight_cfg('missing-module')
cfg['launch_preflight']['module']['name'] = 'experimental_missing'
c.write_text(json.dumps(cfg))
missing_module = assert_no_guardian_launch(c)
assert 'launch_preflight probe failed rc=' in missing_module['reason']

d,c,cfg,out = preflight_cfg('wrong-env')
wrong = d / 'wrong-import-root'; wrong.mkdir()
cfg['env']['PYTHONPATH'] = str(wrong)
c.write_text(json.dumps(cfg))
wrong_env = assert_no_guardian_launch(c)
assert 'launch_preflight probe failed rc=' in wrong_env['reason']

d,c,cfg,out = preflight_cfg('low-space')
low_space = assert_no_guardian_launch(c, low_space=True)
assert 'free space below floor' in low_space['reason']

def package_cfg(tag, init_body):
    d,c,cfg,out = preflight_cfg(tag)
    root = d / 'module-src'; package = root / ('pkg_' + tag.replace('-', '_')); package.mkdir()
    name = package.name + '.entry'
    (package / '__init__.py').write_text(init_body)
    module = package / 'entry.py'; module.write_text('VALUE = 2\n')
    cfg['launch_preflight']['module'] = {'name': name, 'origin': {'path': str(module), 'sha256': sha(module), 'size': module.stat().st_size}}
    cfg['env']['PYTHONPATH'] = str(root)
    c.write_text(json.dumps(cfg))
    return d,c,cfg

# A plain module body is not executed: its print(None) and large output do
# not pollute the one-object control protocol.
d,c,cfg,_ = preflight_cfg('module-output')
(d / 'module-src' / cfg['launch_preflight']['module']['name'].__str__()).write_text('print(None)\nprint("X" * 1000000)\n')
module_output = subprocess.run([sys.executable, str(PROD), str(c)], capture_output=True, text=True)
assert module_output.returncode == 0 and json.loads(module_output.stdout)['launch_preflight']['status'] == 'PASS'

_,c,_, = package_cfg('huge-output', 'print("X" * 5000000)\n')
huge_output = assert_no_guardian_launch(c)
assert huge_output['launch_preflight']['output_limited'] is True

_,c,_ = package_cfg('spawn-timeout', 'import subprocess,sys,time; subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"]); time.sleep(10)\n')
spawn_timeout = assert_no_guardian_launch(c)
assert spawn_timeout['launch_preflight']['timed_out'] is True and spawn_timeout['launch_preflight']['survivor_handoff'] is False

d,c,cfg,_ = preflight_cfg('wrong-origin-parent')
real_parent = d / 'module-src'; link_parent = d / 'module-link'; link_parent.symlink_to(real_parent, target_is_directory=True)
cfg['launch_preflight']['module']['origin']['path'] = str(link_parent / Path(cfg['launch_preflight']['module']['origin']['path']).name)
c.write_text(json.dumps(cfg))
wrong_origin_parent = assert_no_guardian_launch(c)
assert 'symlink ancestor' in wrong_origin_parent['reason']

d,c,cfg,_ = preflight_cfg('invalid-schema')
cfg['launch_preflight']['unexpected'] = True; c.write_text(json.dumps(cfg))
invalid_schema = assert_no_guardian_launch(c)
assert 'launch_preflight must be an object' in invalid_schema['reason']

d,c,cfg,_ = preflight_cfg('write-failure')
bad_path = d / 'write-file'; bad_path.write_text('not a directory')
cfg['launch_preflight']['write_paths'] = [str(bad_path)]; c.write_text(json.dumps(cfg))
write_failure = assert_no_guardian_launch(c)
assert 'existing regular directory' in write_failure['reason']

d,c,cfg,_ = preflight_cfg('uid-boundary')
cfg['expected_uid'] += 1; c.write_text(json.dumps(cfg))
uid_boundary = assert_no_guardian_launch(c)
assert 'UID/HOME identity mismatch' in uid_boundary['reason']

print(json.dumps({'launch_preflight': {
    'success': {'exitcode': success.returncode, 'status': success_result['launch_preflight']['status']},
    'missing_module': {'status': missing_module['launch_preflight']['status'], 'guardian_launched': False},
    'wrong_env': {'status': wrong_env['launch_preflight']['status'], 'guardian_launched': False},
    'low_space': {'status': low_space['launch_preflight']['status'], 'guardian_launched': False},
    'huge_output': {'status': huge_output['launch_preflight']['status'], 'output_limited': huge_output['launch_preflight']['output_limited']},
    'spawn_timeout': {'status': spawn_timeout['launch_preflight']['status'], 'survivor_handoff': spawn_timeout['launch_preflight']['survivor_handoff']},
    'wrong_origin_parent': {'status': wrong_origin_parent['launch_preflight']['status'], 'guardian_launched': False},
    'invalid_schema': {'status': invalid_schema['launch_preflight']['status'], 'guardian_launched': False},
    'write_failure': {'status': write_failure['launch_preflight']['status'], 'guardian_launched': False},
    'uid_boundary': {'status': uid_boundary['launch_preflight']['status'], 'guardian_launched': False},
}}, sort_keys=True))

# Persist the r6 direct boundary checks.  These variants exercise the actual
# _launch_preflight validation and run() admission boundary; only the probe
# transport is replaced so each malformed protocol is deterministic.
def protocol_case(tag, mutate, *, expect_pass=False):
    d,c,cfg,out = preflight_cfg('protocol-' + tag)
    expected = {
        'uid': cfg['expected_uid'],
        'module': cfg['launch_preflight']['module']['name'],
        'origin': str(Path(cfg['launch_preflight']['module']['origin']['path']).resolve()),
        'write_paths': [str(Path(d).resolve())],
    }
    protocol_text = mutate(dict(expected))
    if not isinstance(protocol_text, str):
        protocol_text = json.dumps(protocol_text)
    # _preflight_paths prepends the output parent, then the configured path.
    def fake_bounded(cmd, cwd, env, deadline, max_bytes):
        return {'returncode': 0, 'stdout': protocol_text + '\n', 'stderr': '',
                'term_issued': False, 'survivor_handoff': False, 'timed_out': False,
                'output_limited': False, 'stdout_bytes': len(protocol_text.encode()) + 1,
                'stderr_bytes': 0, 'stdout_evidence_bytes': len(protocol_text.encode()) + 1,
                'stderr_evidence_bytes': 0}
    calls = []
    original_probe = native._bounded_probe
    original_popen = native.subprocess.Popen
    guardian_path = cfg['guardian']['path']
    def forbidden(*args, **kwargs):
        argv = list(args[0]) if args else []
        if guardian_path in argv:
            calls.append(argv)
            raise AssertionError('guardian Popen must not be reached after protocol rejection')
        return original_popen(*args, **kwargs)
    try:
        native._bounded_probe = fake_bounded
        if not expect_pass:
            native.subprocess.Popen = forbidden
        env = dict(os.environ); env.update({str(k):str(v) for k,v in cfg.get('env',{}).items()})
        if expect_pass:
            preflight = native._launch_preflight(cfg, env, Path(cfg['runtime']['python']), time.monotonic() + 3)
            result = {'status': 'RAW_OBSERVED', 'launch_preflight': preflight}
        else:
            result = native.run(str(c))
    finally:
        native._bounded_probe = original_probe
        native.subprocess.Popen = original_popen
    if expect_pass:
        assert not calls, calls
        assert result['status'] == 'RAW_OBSERVED'
        assert result['launch_preflight']['status'] == 'PASS'
    else:
        assert not calls, calls
        assert result['status'] == 'UNPROVEN'
        assert result['launch_preflight']['status'] == 'FAILED'
    return result

protocol_rows = {}
# Every object case starts with the complete expected protocol and changes only
# the field under test.  The malformed JSON case is sent as literal text.
protocol_control = protocol_case('control', lambda expected: json.dumps(expected), expect_pass=True)
protocol_rows['control'] = protocol_control
protocol_rows['array'] = protocol_case('array', lambda expected: '[]')
protocol_rows['malformed'] = protocol_case('malformed', lambda expected: '{"uid":')
protocol_rows['extra'] = protocol_case('extra', lambda expected: dict(expected, extra=True))
protocol_rows['booluid'] = protocol_case('booluid', lambda expected: dict(expected, uid=True))
protocol_rows['wrongpaths'] = protocol_case('wrongpaths', lambda expected: dict(expected, write_paths=['/tmp/not-approved']))

# The real bounded probe must cap retained evidence and fail closed on output
# overflow, while TERM-reaping the owned leader.
probe_deadline = time.monotonic() + 3
huge_probe = native._bounded_probe(
    [sys.executable, '-c', 'import sys; sys.stdout.write("X" * (20 * 1024 * 1024)); sys.stdout.flush()'],
    str(Path.cwd()), dict(os.environ), probe_deadline, 1024)
assert huge_probe['output_limited'] is True and huge_probe['term_issued'] is True
assert huge_probe['stdout_evidence_bytes'] + huge_probe['stderr_evidence_bytes'] <= 1024
assert huge_probe['survivor_handoff'] is False and huge_probe['returncode'] is not None

# A leader can exit while a forked descendant keeps the capture pipe open.
# _bounded_probe must report the unresolved handoff; the test then terminates
# that owned process group with TERM and waits for the descendant to disappear.
holder_pid = Path(tempfile.mkdtemp(prefix='probe-holder-', dir=os.environ['HOME'])) / 'pid'
term_marker = holder_pid.with_name('term-marker')
holder_code = ('import os,signal,time; child=os.fork(); '
               'open(' + repr(str(holder_pid)) + ', "w").write(str(os.getpid())) if child == 0 else os._exit(0); '
               'signal.signal(signal.SIGTERM, lambda *_: (open(' + repr(str(term_marker)) + ', "w").write("term"), os._exit(0))); '
               'time.sleep(30)')
fork_probe = native._bounded_probe(
    [sys.executable, '-c', holder_code], str(Path.cwd()), dict(os.environ),
    time.monotonic() + 1, 1024)
assert holder_pid.exists()
descendant_pid = int(holder_pid.read_text())
try:
    os.killpg(os.getpgid(descendant_pid), signal.SIGTERM)
except ProcessLookupError:
    raise AssertionError('owned descendant vanished before TERM cleanup')
for _ in range(500):
    if term_marker.exists():
        try:
            with open('/proc/' + str(descendant_pid) + '/stat') as f:
                proc_state = f.read().split()[2]
        except FileNotFoundError:
            proc_state = 'gone'
        if proc_state == 'gone':
            break
        if proc_state == 'Z':
            raise AssertionError('owned descendant became zombie without reaping proof')
    time.sleep(.01)
else:
    raise AssertionError('owned descendant survived TERM cleanup')
assert term_marker.read_text() == 'term'
assert fork_probe['survivor_handoff'] is True

# Build mode uses only fresh engine/config outputs as its semantic result.  It
# deliberately has no client_result_path, so the legacy client contract cannot
# accidentally make a builder pass.
def build_case(tag, *, mode='ok', mutate=None):
    d,c,cfg,out = base('build-' + tag)
    builder = d / 'builder.py'
    builder.write_text('''import json,sys,time,os
engine,config,mode=sys.argv[1:4]
os.makedirs(os.path.dirname(engine), exist_ok=True)
valid={"model_type":"qwen3_tts_code2wav","code2wav_config":{"sample_rate":24000},"builder_config":{"min_code_len":1,"opt_code_len":2,"max_code_len":3}}
if mode == "term": open(engine,"wb").write(b"engine"); json.dump(valid, open(config,"w")); time.sleep(3)
elif mode == "missing": json.dump(valid, open(config,"w"))
elif mode == "empty": open(engine,"wb").close(); json.dump(valid, open(config,"w"))
elif mode == "symlink": open(engine,"wb").write(b"engine"); open(config,"w").write("{}"); os.unlink(config); os.symlink(engine, config)
else:
    open(engine,"wb").write(b"engine")
    json.dump(valid, open(config,"w"))
''')
    engine = out / 'code2wav' / 'code2wav.engine'; config = out / 'code2wav' / 'config.json'
    cfg.pop('client_result_path', None)
    cfg['client_argv'] = [sys.executable, str(builder), str(engine), str(config), mode, '--components', 'code2wav', '--engine-dir', str(out), '--max-batch-size', '1', '--min-code-len', '1', '--opt-code-len', '2', '--max-code-len', '3']
    argv_pin_bytes = json.dumps(cfg['client_argv'], separators=(',', ':'), ensure_ascii=False).encode()
    cfg['build_output_contract'] = {'component':'code2wav', 'engine_path': str(engine), 'config_path': str(config), 'engine_dir': str(out), 'argv': list(cfg['client_argv']),
                                    'argv_pin': {'sha256':hashlib.sha256(argv_pin_bytes).hexdigest(),'size':len(argv_pin_bytes)}, 'max_batch_size':1,
                                    'code_len': {'min_code_len':1,'opt_code_len':2,'max_code_len':3},
                                    'config_constraints': {'model_type': 'qwen3_tts_code2wav', 'code2wav_config': {'sample_rate': 24000}, 'builder_config': {'min_code_len': 1, 'opt_code_len': 2, 'max_code_len': 3}}}
    if mutate:
        mutate(cfg, engine, config)
    c.write_text(json.dumps(cfg))
    p = subprocess.run([sys.executable, str(PROD), str(c)], capture_output=True, text=True)
    return p, d, cfg, engine, config

def repin_build_argv(cfg):
    argv = cfg['client_argv']
    encoded = json.dumps(argv, separators=(',', ':'), ensure_ascii=False).encode()
    cfg['build_output_contract']['argv'] = list(argv)
    cfg['build_output_contract']['argv_pin'] = {'sha256': hashlib.sha256(encoded).hexdigest(), 'size': len(encoded)}

def rejected_build_case(tag, mutate, expected_reason=None):
    rejected, rejected_dir, _, _, _ = build_case(tag, mutate=mutate)
    value = json.loads(rejected.stdout)
    assert rejected.returncode != 0 and value['status'] == 'UNPROVEN'
    if expected_reason is not None:
        assert value['reason'] == expected_reason
    assert not (rejected_dir / 'out').exists()
    assert not (rejected_dir / 'out' / 'guardian').exists()
    return value

build_success, build_dir, build_cfg, build_engine, build_config = build_case('success')
build_value = json.loads(build_success.stdout)
assert build_success.returncode == 0 and build_value['status'] == 'BUILD_OUTPUT_VERIFIED'
assert build_value['build_status'] == 'OUTPUT_VERIFIED' and build_value['engine_artifact']['size'] > 0
assert build_value['config_artifact']['size'] > 0 and build_value['build_config']['model_type'] == 'qwen3_tts_code2wav'
assert build_value['recipe']['argv'] == build_cfg['client_argv']

def read_receipt(d):
    receipt = d / 'out' / 'producer-result.json'
    assert receipt.is_file() and not receipt.is_symlink() and receipt.stat().st_size > 0
    value = json.loads(receipt.read_text())
    assert isinstance(value, dict)
    return receipt, value

success_receipt, success_value = read_receipt(build_dir)
assert success_value['status'] == build_value['status'] == 'BUILD_OUTPUT_VERIFIED'
assert success_value['build_status'] == build_value['build_status'] == 'OUTPUT_VERIFIED'
assert success_value['input_pins'] == build_value['input_pins']
assert success_value['recipe']['argv_pin'] == build_value['recipe']['argv_pin']

# CPU-only input preflight validates every pinned input and the launch probe,
# then captures a stable foreign snapshot without creating out or launching
# the guardian/client.  The client below is an intentional failure sentinel.
pre_d, pre_c, pre_cfg, pre_out = preflight_cfg('mode-valid')
pre_cfg['client_argv'] = [sys.executable, '-c', 'raise SystemExit(91)']
pre_c.write_text(json.dumps(pre_cfg))
pre_run = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(pre_c)], capture_output=True, text=True)
pre_value = json.loads(pre_run.stdout)
assert pre_run.returncode == 0 and pre_value['status'] == 'INPUTS_PREFLIGHT_VERIFIED'
assert pre_value['startup_admission']['status'] == 'NOT_EVALUATED'
assert pre_value['foreign_after_match'] is True
assert pre_value['runtime']['sha256'] == pre_cfg['runtime']['sha256']
assert pre_value['guardian']['sha256'] == pre_cfg['guardian']['sha256']
assert not pre_out.exists()

missing_d, missing_c, missing_cfg, missing_out = preflight_cfg('mode-missing-artifact')
missing_cfg['artifacts'][0]['path'] = str(missing_d / 'missing-client.py')
missing_c.write_text(json.dumps(missing_cfg))
missing_run = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(missing_c)], capture_output=True, text=True)
missing_value = json.loads(missing_run.stdout)
assert missing_run.returncode != 0 and missing_value['status'] == 'UNPROVEN'
assert 'regular file required' in missing_value['reason'] and not missing_out.exists()

wrong_home_d, wrong_home_c, wrong_home_cfg, wrong_home_out = preflight_cfg('mode-wrong-home')
wrong_home_cfg['home'] = str(wrong_home_d / 'incorrect-home')
Path(wrong_home_cfg['home']).mkdir()
wrong_home_c.write_text(json.dumps(wrong_home_cfg))
wrong_home_run = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(wrong_home_c)], capture_output=True, text=True)
wrong_home_value = json.loads(wrong_home_run.stdout)
assert wrong_home_run.returncode != 0 and 'config HOME differs from actual HOME' in wrong_home_value['reason']
assert not wrong_home_out.exists()

no_home_env = dict(os.environ); no_home_env.pop('HOME', None)
no_home_run = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(pre_c)], capture_output=True, text=True, env=no_home_env)
no_home_value = json.loads(no_home_run.stdout)
assert no_home_run.returncode != 0 and 'UID/HOME identity mismatch' in no_home_value['reason']

symlink_d, symlink_c, symlink_cfg, symlink_out = preflight_cfg('mode-runtime-symlink')
runtime_link = symlink_d / 'python-runtime-link'
runtime_link.symlink_to(Path(sys.executable).resolve())
symlink_cfg['runtime']['python'] = str(runtime_link)
symlink_c.write_text(json.dumps(symlink_cfg))
symlink_run = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(symlink_c)], capture_output=True, text=True)
symlink_value = json.loads(symlink_run.stdout)
assert symlink_run.returncode == 0 and symlink_value['status'] == 'INPUTS_PREFLIGHT_VERIFIED'
assert not symlink_out.exists()

drift_d, drift_c, drift_cfg, drift_out = preflight_cfg('mode-input-drift')
drift_artifact = Path(drift_cfg['artifacts'][0]['path'])
drift_snapshot = Path(drift_cfg['snapshot_cmd'][0])
drift_snapshot.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
marker = pathlib.Path(sys.argv[2] + ".drift-marker")
if not marker.exists():
    pathlib.Path(''' + repr(str(drift_artifact)) + ''').open("ab").write(b"drift")
    marker.write_text("1")
print(json.dumps([{"Id":"foreign-id","ImageID":"sha256:foreign","Name":"/foreign","Running":True,"Pid":123,"RestartCount":0}]))
''')
drift_c.write_text(json.dumps(drift_cfg))
drift_run = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(drift_c)], capture_output=True, text=True)
drift_value = json.loads(drift_run.stdout)
assert drift_run.returncode != 0 and 'input pin drift during preflight' in drift_value['reason']
assert not drift_out.exists()

late_d, late_c, late_cfg, late_out = preflight_cfg('mode-second-snapshot-drift')
late_artifact = Path(late_cfg['artifacts'][0]['path'])
late_snapshot = Path(late_cfg['snapshot_cmd'][0])
late_snapshot.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
counter = pathlib.Path(sys.argv[2])
try: n = int(counter.read_text()) + 1
except FileNotFoundError: n = 1
counter.write_text(str(n))
if n >= 2:
    pathlib.Path(''' + repr(str(late_artifact)) + ''').open("ab").write(b"late-drift")
print(json.dumps([{"Id":"foreign-id","ImageID":"sha256:foreign","Name":"/foreign","Running":True,"Pid":123,"RestartCount":0}]))
''')
late_c.write_text(json.dumps(late_cfg))
late_run = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(late_c)], capture_output=True, text=True)
late_value = json.loads(late_run.stdout)
assert late_run.returncode != 0 and 'input pin drift during preflight' in late_value['reason']
assert late_value['foreign_after_match'] is True and not late_out.exists()

# The SDK build contract accepts the module form only when its launch
# preflight identifies the SDK CLI.  The direct-file form is rejected before
# any launcher; the module form passes the CPU preflight without running body.
def sdk_mode_case(tag, module_form):
    _, d, cfg, _, _ = build_case('sdk-' + tag)
    c = d / 'config.json'
    out = d / 'out'
    shutil.rmtree(out)
    module_root = d / 'module-src' / 'experimental' / 'builder'
    module_root.mkdir(parents=True)
    (module_root / '__init__.py').write_text('')
    (module_root.parent / '__init__.py').write_text('')
    module = module_root / 'cli.py'
    module.write_text('raise SystemExit(99)\n')
    cfg['env'] = {'PYTHONPATH': str(d / 'module-src')}
    cfg['launch_preflight'] = {'module': {'name': 'experimental.builder.cli', 'origin': {'path': str(module), 'sha256': sha(module), 'size': module.stat().st_size}}, 'write_paths': [str(d)], 'min_free_bytes': {str(d): 1}}
    if module_form:
        cfg['client_argv'] = [sys.executable, '-m', 'experimental.builder.cli'] + cfg['client_argv'][2:]
    repin_build_argv(cfg)
    c.write_text(json.dumps(cfg))
    p = subprocess.run([sys.executable, str(PROD), '--preflight-only', str(c)], capture_output=True, text=True)
    return p, d

sdk_direct, sdk_direct_d = sdk_mode_case('direct', False)
sdk_direct_value = json.loads(sdk_direct.stdout)
assert sdk_direct.returncode != 0 and 'must use python -m experimental.builder.cli' in sdk_direct_value['reason']
assert not (sdk_direct_d / 'out').exists()
sdk_module, sdk_module_d = sdk_mode_case('module', True)
sdk_module_value = json.loads(sdk_module.stdout)
assert sdk_module.returncode == 0 and sdk_module_value['status'] == 'INPUTS_PREFLIGHT_VERIFIED'
assert not (sdk_module_d / 'out').exists()
assert success_value['foreign_after_match'] is True
assert success_value['profile_validation'] == success_value['production_qualification'] == 'UNPROVEN'
for key in ('engine_artifact', 'config_artifact'):
    assert success_value[key]['size'] > 0 and success_value[key]['regular'] is True

def legal_flag_order(cfg, engine, config):
    out = cfg['outdir']
    cfg['client_argv'] = [cfg['client_argv'][0], cfg['client_argv'][1], engine.__str__(), config.__str__(), 'ok',
                          '--engine-dir=' + out, '--plugin', 'dense', '--components', 'code2wav',
                          '--max-batch-size=1', '--max-code-len', '3', '--min-code-len=1', '--opt-code-len', '2']
    repin_build_argv(cfg)
legal_order, legal_order_dir, _, _, _ = build_case('legal-flag-order', mutate=legal_flag_order)
assert legal_order.returncode == 0 and json.loads(legal_order.stdout)['status'] == 'BUILD_OUTPUT_VERIFIED'

def duplicate_components(cfg, engine, config):
    cfg['client_argv'] += ['--', '--components', 'code2wav']; repin_build_argv(cfg)
duplicate_components_value = rejected_build_case('duplicate-components', duplicate_components)
assert 'repeats protected option: --components' in duplicate_components_value['reason']

def duplicate_engine_dir(cfg, engine, config):
    cfg['client_argv'] += ['--engine-dir', cfg['outdir']]; repin_build_argv(cfg)
duplicate_engine_dir_value = rejected_build_case('duplicate-engine-dir', duplicate_engine_dir)
assert 'repeats protected option: --engine-dir' in duplicate_engine_dir_value['reason']

def duplicate_code_len(cfg, engine, config):
    cfg['client_argv'] += ['--opt-code-len=2']; repin_build_argv(cfg)
duplicate_code_len_value = rejected_build_case('duplicate-code-len', duplicate_code_len)
assert 'repeats protected option: --opt-code-len' in duplicate_code_len_value['reason']

def abbreviated_component(cfg, engine, config):
    cfg['client_argv'][cfg['client_argv'].index('--components')] = '--comp'; repin_build_argv(cfg)
abbreviated_component_value = rejected_build_case('abbreviated-component', abbreviated_component)
assert 'abbreviated protected option: --comp' in abbreviated_component_value['reason']

def wrong_engine_dir(cfg, engine, config):
    cfg['client_argv'][cfg['client_argv'].index('--engine-dir') + 1] = str(Path(cfg['outdir']) / 'other'); repin_build_argv(cfg)
wrong_engine_dir_value = rejected_build_case('wrong-engine-dir', wrong_engine_dir)
assert 'value mismatch for protected option: --engine-dir' in wrong_engine_dir_value['reason']

def nested_output_parent(cfg, engine, config):
    nested = str(Path(cfg['outdir']) / 'A' / 'code2wav')
    cfg['build_output_contract']['engine_path'] = nested + '/code2wav.engine'
    cfg['build_output_contract']['config_path'] = nested + '/config.json'
nested_output_parent_value = rejected_build_case('nested-output-parent', nested_output_parent)
assert 'code2wav output paths must be code2wav/code2wav.engine and code2wav/config.json' in nested_output_parent_value['reason']

def guardian_override(cfg, engine, config):
    cfg['guardian_args'] = ['--max-output', '1']
gd_override_value = rejected_build_case('build-guardian-override', guardian_override)
assert 'build mode does not allow guardian_args' in gd_override_value['reason']

for mode in ('missing', 'empty', 'symlink', 'term'):
    failed, failed_dir, _, _, _ = build_case(mode, mode=mode)
    assert failed.returncode != 0, mode
    failed_value = json.loads(failed.stdout)
    assert failed_value['status'] == 'UNPROVEN', mode
    expected_reason = {'missing':'RuntimeError: build engine output is missing or symlink',
                       'empty':'RuntimeError: build engine output is empty',
                       'symlink':f"ValueError: path contains symlink ancestor: {failed_dir / 'out' / 'code2wav' / 'config.json'}",
                       'term':'Bound: guardian exceeded parent deadline'}[mode]
    assert failed_value['reason'] == expected_reason, (mode, failed_value['reason'])
    _, failed_receipt = read_receipt(failed_dir)
    assert failed_receipt['reason'] == expected_reason
    assert failed_receipt.get('status') != 'BUILD_OUTPUT_VERIFIED'
    assert failed_receipt.get('build_status') != 'OUTPUT_VERIFIED'
    assert failed_receipt.get('foreign_after_match') is not False
    if mode == 'term':
        handoff = failed_receipt.get('guardian_handoff')
        assert isinstance(handoff, dict) and handoff.get('pid', 0) > 0
        ident = handoff.get('identity')
        assert isinstance(ident, dict) and ident.get('start_ticks', 0) > 0
        assert ident.get('exe') and ident.get('argv') == failed_value['actual_guardian_argv']
        pid = int(handoff['pid'])
        before_term = proc_identity(pid)
        assert before_term is not None and before_term['state'] != 'Z'
        assert before_term['start_ticks'] == ident['start_ticks']
        assert os.path.realpath(before_term['exe']) == os.path.realpath(ident['exe'])
        os.kill(pid, signal.SIGTERM)
        cleanup_deadline = time.monotonic() + 2
        while time.monotonic() < cleanup_deadline:
            current = proc_identity(pid)
            if current is None or current['start_ticks'] != ident['start_ticks']:
                break
            time.sleep(.01)
        else:
            final = proc_identity(pid)
            raise AssertionError('owned guardian handoff survived safe TERM' if final and final['state'] != 'Z'
                                 else 'owned guardian handoff remained zombie after safe TERM deadline')
        final = proc_identity(pid)
        assert final is None or final['start_ticks'] != ident['start_ticks']

preexisting, d, preexisting_cfg_doc, engine, _ = build_case('preexisting')
# The enclosing output namespace is itself fresh; any pre-existing child is
# therefore rejected by the same early fresh-output gate.
(d / 'out').mkdir(exist_ok=True)
old_receipt = d / 'out' / 'producer-result.json'
if old_receipt.exists(): old_receipt.unlink()
preexisting_cfg = d / 'config.json'
preexisting_cfg.write_text(json.dumps(preexisting_cfg_doc))
preexisting = subprocess.run([sys.executable, str(PROD), str(preexisting_cfg)], capture_output=True, text=True)
assert preexisting.returncode != 0 and 'fresh output directory required' in preexisting.stdout
assert not old_receipt.exists()

def mismatch(cfg, engine, config):
    cfg['build_output_contract']['config_constraints']['model_type'] = 'wrong-model'
mismatched, mismatched_dir, _, _, _ = build_case('config-mismatch', mutate=mismatch)
assert mismatched.returncode != 0 and 'invalid code2wav config constraints' in mismatched.stdout
assert not (mismatched_dir / 'out').exists()

def builder_constraint_scalar(key, value):
    def mutate(cfg, engine, config):
        cfg['build_output_contract']['config_constraints']['builder_config'][key] = value
    return mutate

for key, value in (
    ('min_code_len', True), ('min_code_len', 1.0),
    ('opt_code_len', True), ('opt_code_len', 2.0),
    ('max_code_len', True), ('max_code_len', 3.0),
):
    rejected_build_case('builder-constraint-' + key + '-' + type(value).__name__, builder_constraint_scalar(key, value),
                        'ValueError: invalid code2wav config constraints')

def drift(cfg, engine, config):
    cfg['client_argv'][-1] = 'drift'
drifted, drifted_dir, _, _, _ = build_case('argv-drift', mutate=drift)
assert drifted.returncode != 0
assert not (drifted_dir / 'out').exists()

def foreign_after_change(cfg, engine, config):
    cfg['snapshot_cmd'][1] = 'foreign_change'
foreign_failed, foreign_dir, _, _, _ = build_case('foreign-after', mutate=foreign_after_change)
foreign_value = json.loads(foreign_failed.stdout)
assert foreign_failed.returncode != 0 and foreign_value['status'] == 'UNPROVEN'
foreign_reason = 'guardian/client/identity/log/foreign evidence failed validation'
assert foreign_value['reason'] == foreign_reason
_, foreign_receipt = read_receipt(foreign_dir)
assert foreign_receipt['foreign_after_match'] is False
assert foreign_receipt['reason'] == foreign_reason
assert foreign_receipt.get('build_status') != 'OUTPUT_VERIFIED'

def admission_floor(cfg, engine, config):
    cfg['startup_admission'] = {'min_mem_available_bytes': 1 << 60,
                                'min_shm_free_bytes': 1,
                                'min_root_physical_free_bytes': 1}
admission_failed, admission_dir, _, _, _ = build_case('startup-admission', mutate=admission_floor)
admission_value = json.loads(admission_failed.stdout)
assert admission_failed.returncode != 0 and admission_value['status'] == 'UNPROVEN'
assert admission_value['startup_admission']['status'] == 'FAILED'
assert not (admission_dir / 'out').exists()

print(json.dumps({'input_preflight_mode': {
    'valid': {'status': pre_value['status'], 'foreign_after_match': pre_value['foreign_after_match'], 'out_created': pre_out.exists()},
    'missing_artifact': {'returncode': missing_run.returncode, 'reason': missing_value['reason'], 'out_created': missing_out.exists()},
    'wrong_home': {'returncode': wrong_home_run.returncode, 'reason': wrong_home_value['reason'], 'out_created': wrong_home_out.exists()},
    'no_home_env': {'returncode': no_home_run.returncode, 'reason': no_home_value['reason']},
    'runtime_symlink': {'status': symlink_value['status'], 'out_created': symlink_out.exists()},
    'input_drift': {'returncode': drift_run.returncode, 'reason': drift_value['reason'], 'out_created': drift_out.exists()},
    'second_snapshot_drift': {'returncode': late_run.returncode, 'reason': late_value['reason'], 'foreign_after_match': late_value['foreign_after_match'], 'out_created': late_out.exists()},
    'sdk_direct': {'returncode': sdk_direct.returncode, 'reason': sdk_direct_value['reason'], 'out_created': (sdk_direct_d / 'out').exists()},
    'sdk_module': {'status': sdk_module_value['status'], 'out_created': (sdk_module_d / 'out').exists()},
}, 'r6_persistent_boundaries': {
    'protocol_rejected': {k: {'status': v['launch_preflight']['status'], 'guardian_launched': False}
                          for k,v in protocol_rows.items()},
    'huge_output': {'output_limited': huge_probe['output_limited'],
                    'stdout_evidence_bytes': huge_probe['stdout_evidence_bytes'],
                    'term_issued': huge_probe['term_issued'],
                    'survivor_handoff': huge_probe['survivor_handoff']},
    'fork_descendant_pipe': {'survivor_handoff': fork_probe['survivor_handoff']},
}}, sort_keys=True))
