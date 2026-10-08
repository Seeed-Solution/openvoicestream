#!/usr/bin/env python3
"""CPU-only contract check for the optional Edge-LLM media runtime mode.

The supplied runtime source is imported with a local fake native/build boundary;
no CUDA, model weights, engine build, or forward pass is performed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, tempfile, types
from enum import Enum
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def emit(path: Path | None, doc: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, sort_keys=True, indent=2) + '\n')
    print(json.dumps(doc, sort_keys=True))


def fail(output: Path | None, kind: str, message: str, source=None):
    doc = {'status': 'ERROR', 'error_type': kind, 'message': message}
    if source is not None:
        doc['source'] = source
    emit(output, doc)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime-source', required=True)
    ap.add_argument('--expected-sha256')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    source = Path(args.runtime_source)
    if not source.is_file() or source.is_symlink():
        return fail(args.output, 'source_not_regular_file', str(source))
    actual_sha = sha256(source)
    source_info = {'path': str(source), 'sha256': actual_sha, 'size': source.stat().st_size}
    if args.expected_sha256 is not None and args.expected_sha256 != actual_sha:
        return fail(args.output, 'source_sha256_mismatch', f'expected {args.expected_sha256}, got {actual_sha}', source_info)
    root = Path(tempfile.mkdtemp(prefix='edgellm-media-check-'))
    model, bundle = root / 'model', root / 'bundle'
    model.mkdir(); bundle.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    (bundle / 'config.json').write_text(json.dumps({'builder_config': {'max_input_len': 8192, 'max_batch_size': 4, 'max_kv_cache_capacity': 8192}, 'engine_role': 'llm'}))

    pkg = types.ModuleType('candpkg'); pkg.__path__ = []
    runtimepkg = types.ModuleType('candpkg.runtime'); runtimepkg.__path__ = []
    config = types.ModuleType('candpkg.config')
    class ContextCacheConfig:
        def __init__(self, enabled=False): self.enabled = enabled
        @classmethod
        def parse(cls, value): return cls(False)
    config.DEFAULT_MAX_QUEUED_REQUESTS = 4; config.ContextCacheConfig = ContextCacheConfig
    parsing = types.ModuleType('candpkg.parsing'); parsing.__path__ = []
    tool = types.ModuleType('candpkg.parsing.tool_calling')
    class ToolConfig: pass
    tool.ToolConfig = ToolConfig
    tool.parse_assistant_output = lambda *a, **k: None
    tool.validate_tool_request = lambda *a, **k: types.SimpleNamespace(tools=[], tool_choice='none', forced_name=None)
    layout = types.ModuleType('candpkg.runtime.engine_layout')
    class EngineType(Enum): LLM = 'llm'; SPEC_DECODE = 'spec_decode'; UNKNOWN = 'unknown'
    class BundleLayout:
        def __init__(self, root_dir, visual_dir):
            self.root = root_dir; self.engine_type = EngineType.LLM; self.visual_dir = visual_dir
            self.audio_dir = self.talker_dir = self.code_predictor_dir = self.code2wav_dir = None; self.audio_model_type = ''
        @property
        def media_dir(self): return self.root if self.visual_dir else ''
        @property
        def has_speech(self): return False
    layout.BundleLayout = BundleLayout; layout.EngineType = EngineType
    layout.inspect_bundle = lambda p: BundleLayout(str(bundle), str(bundle / 'visual'))
    sys.modules.update({'candpkg': pkg, 'candpkg.config': config, 'candpkg.parsing': parsing,
                        'candpkg.parsing.tool_calling': tool, 'candpkg.runtime': runtimepkg,
                        'candpkg.runtime.engine_layout': layout})
    build = types.ModuleType('candpkg.runtime.engine_build')
    class BuildOptions:
        def __init__(self, **kw): self.__dict__.update(kw); self.spec_type = 'none'; self.draft_model_dir = ''; self.max_verify_tree_size = None; self.max_draft_tree_size = None
        @property
        def builder_spec_type(self): return self.spec_type
    class Prepared:
        bundle_dir = str(bundle); model_dir = str(model); draft_model_dir = ''; built = False
    prepare_calls = []
    build.BuildOptions = BuildOptions
    build.prepare_model = lambda model_dir, cache_dir, options, **kw: (prepare_calls.append((model_dir, cache_dir, options, kw)) or Prepared())
    build.cache_root = lambda value: str(Path(value or root).resolve())
    build.resolve_model_dir = lambda model_dir, cache_dir: str(model)
    sys.modules['candpkg.runtime.engine_build'] = build

    spec = importlib.util.spec_from_file_location('candpkg.runtime.engine', str(source))
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    native_calls = []; native_engines = []
    class NativeEngine:
        def __init__(self): self.shutdown_called = False
        def shutdown(self, *a): self.shutdown_called = True
    class Native:
        class ShutdownMode:
            DRAIN = 0
        def RequestEngine(self, engine_dir, media_dir, lora, model_dir, cc, **kw):
            native_calls.append({'engine_dir': engine_dir, 'media_dir': media_dir, 'model_dir': model_dir, 'max_batch_size': kw['max_batch_size']})
            obj = NativeEngine(); native_engines.append(obj); return obj
        def capture_decoding_cuda_graph(self): return True
    native = Native()
    try:
        spec.loader.exec_module(mod)
        mod._import_runtime = lambda: native
        mod._native_context_cache_config = lambda rt, cc: cc
        mod.ifb_unsupported_reason = lambda layout, model_type=None: None
        mod._resolve_spec_decode_runtime_options = lambda *a, **k: types.SimpleNamespace(top_k=1, step=1, verify_size=1, dflash_block_size=0)
        true_obj = mod.load_model(model=str(model), cache_dir=str(root / 'cache'), max_input_len=8192, max_batch_size=4, max_kv_cache_capacity=8192, enable_in_flight_batching=True, load_media_engines=True)
        false_obj = mod.load_model(model=str(model), cache_dir=str(root / 'cache'), max_input_len=8192, max_batch_size=4, max_kv_cache_capacity=8192, enable_in_flight_batching=True, load_media_engines=False)
        assert [x['media_dir'] for x in native_calls] == [str(bundle), '']
        assert all(x['max_batch_size'] == 4 for x in native_calls)
        assert len(prepare_calls) == 2 and all(x[2].max_input_len == 8192 and x[2].max_batch_size == 4 and x[2].max_kv_cache_capacity == 8192 for x in prepare_calls)
        image_calls = []
        mod._load_image_buffers = lambda *a: image_calls.append('image') or []
        mod._load_audio_buffers = lambda *a: image_calls.append('audio') or []
        mod._convert_messages_to_cpp = lambda *a: []
        false_obj._rt = native; false_obj._video_model_family = lambda: 'qwen'; false_obj._video_frame_limits = lambda: {}
        for media in ({'type': 'image_url', 'image_url': {'url': 'x'}}, {'type': 'input_audio', 'input_audio': {'data': 'x'}}, {'type': 'audio_url', 'audio_url': {'url': 'x'}}):
            try: false_obj._prepare_messages_for_runtime([{'role': 'user', 'content': [media]}])
            except ValueError as exc: assert 'load_media_engines=False' in str(exc)
            else: raise AssertionError('media input was accepted with load_media_engines=False')
        true_obj._rt = native; true_obj._video_model_family = lambda: 'qwen'; true_obj._video_frame_limits = lambda: {}
        true_obj._prepare_messages_for_runtime([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}])
        assert image_calls == ['image']
        false_obj._prepare_messages_for_runtime([{'role': 'user', 'content': 'plain text'}])
        assert image_calls == ['image', 'image']
        try: mod.load_model(model=str(model), load_media_engines='false')
        except TypeError: pass
        else: raise AssertionError('non-bool load_media_engines accepted')
        (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_tts'}))
        try: mod.load_model(model=str(model), load_media_engines=False)
        except ValueError as exc: assert 'only supported by LLM' in str(exc)
        else: raise AssertionError('TTS false accepted')
        for obj in (true_obj, false_obj):
            for name in ('close', 'shutdown'):
                fn = getattr(obj, name, None)
                if callable(fn): fn(); break
        result = {'status': 'PASS', 'source': source_info, 'native_args': native_calls,
                  'prepare_model_calls': len(prepare_calls), 'shape_contract': {'max_batch_size': 4, 'max_input_len': 8192, 'max_kv_cache_capacity': 8192},
                  'media_rejections': 3, 'true_media_loader_calls': 1, 'false_text_loader_calls': 1,
                  'negative_cases': ['nonbool_load_media_engines', 'tts_false', 'image_url_false', 'input_audio_false', 'audio_url_false']}
        emit(args.output, result); return 0
    except Exception as exc:
        return fail(args.output, 'cpu_contract_failed', f'{type(exc).__name__}: {exc}', source_info)

if __name__ == '__main__':
    raise SystemExit(main())
