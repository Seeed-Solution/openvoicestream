import hashlib,json,os,pwd,signal,subprocess,sys,tempfile,shutil,time
import importlib.util
from pathlib import Path
ROOT=Path(__file__).with_name('edgellm_native_run.py').parent; PROD=Path(__file__).with_name('edgellm_native_run.py'); GUARD=Path(__file__).with_name('edgellm_native_guardian.py')
assert '/tmp/slv-v011-llm-guardian-root-recovery-20261004' not in PROD.read_text() and '/tmp/slv-v011-llm-guardian-root-recovery-20261004' not in GUARD.read_text()
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

print(json.dumps({'r6_persistent_boundaries': {
    'protocol_rejected': {k: {'status': v['launch_preflight']['status'], 'guardian_launched': False}
                          for k,v in protocol_rows.items()},
    'huge_output': {'output_limited': huge_probe['output_limited'],
                    'stdout_evidence_bytes': huge_probe['stdout_evidence_bytes'],
                    'term_issued': huge_probe['term_issued'],
                    'survivor_handoff': huge_probe['survivor_handoff']},
    'fork_descendant_pipe': {'survivor_handoff': fork_probe['survivor_handoff']},
}}, sort_keys=True))
