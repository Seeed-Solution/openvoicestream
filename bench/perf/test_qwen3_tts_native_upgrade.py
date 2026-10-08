import hashlib
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import qwen3_tts_native_smoke as smoke
import edgellm_upgrade as upgrade


class NativeUpgradeCpuTests(unittest.TestCase):
    def _native_child_args(self, root: Path, *, speaker: str, language: str) -> list[str]:
        talker = root / "talker"
        talker.mkdir()
        (talker / "config.json").write_text(json.dumps({
            "speaker_id": {"serena": 3066, "vivian": 3065},
            "codec_language_id": {"english": 2050, "chinese": 2055},
        }))
        for name in ("code-predictor", "code2wav", "checkpoint"):
            (root / name).mkdir()
        extension = root / "extension.so"; extension.write_bytes(b"extension")
        plugin = root / "plugin.so"; plugin.write_bytes(b"plugin")
        return ["--child", "--talker", str(talker), "--code-predictor", str(root / "code-predictor"),
                "--code2wav", str(root / "code2wav"), "--checkpoint", str(root / "checkpoint"),
                "--extension", str(extension), "--plugin", str(plugin),
                "--expected-extension-sha256", hashlib.sha256(extension.read_bytes()).hexdigest(),
                "--expected-plugin-sha256", hashlib.sha256(plugin.read_bytes()).hexdigest(),
                "--output-wav", str(root / "out.wav"), "--speaker", speaker, "--language", language]

    def test_named_inputs_are_checked_before_native_import_and_runtime(self):
        for field, value in (("speaker", "Serena"), ("language", "English")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                args = self._native_child_args(root, speaker="serena" if field == "language" else value,
                                               language="english" if field == "speaker" else value)
                with mock.patch.object(smoke.ctypes, "CDLL") as cdll, mock.patch.object(smoke, "load_extension") as load:
                    with self.assertRaises(ValueError) as caught:
                        smoke.child(smoke.parser().parse_args(args))
                self.assertIn(field, str(caught.exception))
                cdll.assert_not_called()
                load.assert_not_called()

    def test_named_inputs_from_config_allow_native_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._native_child_args(root, speaker="serena", language="english")
            with mock.patch.object(smoke.ctypes, "CDLL"), mock.patch.object(smoke, "load_extension", return_value=smoke._FakeNative) as load:
                self.assertEqual(smoke.child(smoke.parser().parse_args(args)), 0)
            load.assert_called_once()

    def test_default_plan_remains_compatible_without_upgrade_env(self):
        ns = smoke.parser().parse_args([])
        self.assertIsNone(smoke.validate_upgrade_mode(ns))
        self.assertEqual(smoke.config(ns)["request_count"], 1)

    def test_upgrade_mode_rejects_missing_metadata_before_native_import(self):
        with tempfile.TemporaryDirectory() as td:
            ns = smoke.parser().parse_args(["--upgrade-result-json", str(Path(td) / "result.json")])
            with self.assertRaises(ValueError):
                smoke.validate_upgrade_mode(ns)

    def test_fake_observation_can_never_pass(self):
        observation = {"mode": "cpu_fake", "status": "ok", "finished": True,
                       "cancelled": False, "request_frame_count": 1, "pcm_frames": 1,
                       "bytes": 2, "wav": "/tmp/fake.wav"}
        self.assertFalse(smoke._native_observation_pass(observation))

    def test_child_upgrade_flag_rejected_even_with_forged_internal_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "result.json"
            wav = root / "audio.wav"
            env = os.environ.copy()
            env.update({"SLV_NATIVE_INTERNAL_CHILD": "1", "SLV_UPGRADE_ROW_ID": "tts.customvoice.b1",
                        "SLV_UPGRADE_PHASE": "tts.customvoice.b1", "SLV_UPGRADE_VARIANT": "customvoice",
                        "SLV_UPGRADE_CLOSURE_SHA256": "a" * 64})
            proc = __import__("subprocess").run(
                [sys.executable, str(Path(smoke.__file__)), "--child", "--upgrade-result-json", str(result),
                 "--output-wav", str(wav), "--speaker", "Serena", "--language", "English"],
                env=env, capture_output=True, text=True, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(result.exists())

    def test_valid_native_observation_writes_schema_seam_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav_path = root / "audio.wav"
            with wave.open(str(wav_path), "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000); stream.writeframes(b"\x01\x00" * 8)
            out = root / "raw.stdout"; err = root / "raw.stderr"; out.write_bytes(b"native\n"); err.write_bytes(b"")
            result = root / "upgrade.json"
            ns = smoke.parser().parse_args(["--upgrade-result-json", str(result), "--output-wav", str(wav_path), "--speaker", "Serena", "--language", "English"])
            env = {"SLV_UPGRADE_ROW_ID": "tts.customvoice.b1", "SLV_UPGRADE_PHASE": "tts.customvoice.b1", "SLV_UPGRADE_VARIANT": "customvoice", "SLV_UPGRADE_CLOSURE_SHA256": "a" * 64}
            old = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                observation = {"mode": "native_api_smoke", "status": "ok", "finished": True, "cancelled": False,
                               "request_frame_count": 8, "codec_frames": 8, "pcm_frames": 8, "bytes": 16,
                               "pcm_sha256": hashlib.sha256(b"\x01\x00" * 8).hexdigest(), "worker_reaped": True,
                               "wav": str(wav_path),
                               "text": "Hello.", "send_mono": 1.0, "first_chunk_mono": 1.1, "terminal_mono": 1.2,
                               "ttfa_ms": 100.0, "generation_ms": 200.0, "rtf": 600.0}
                smoke._write_upgrade_result(ns, observation, out, err)
            finally:
                for key, value in old.items():
                    if value is None: os.environ.pop(key, None)
                    else: os.environ[key] = value

            doc = json.loads(result.read_text())
            self.assertEqual(doc["functional_status"], "PASS")
            self.assertEqual({doc[k] for k in ("row_id", "phase", "variant", "closure_sha256")},
                             {"tts.customvoice.b1", "customvoice", "a" * 64})
            self.assertEqual(doc["wav"]["sha256"], hashlib.sha256(wav_path.read_bytes()).hexdigest())
            # Bind the generated client envelope into the same owner/artifact
            # proof seam used by validate_tts_result.
            def pin(path, remote):
                data = path.read_bytes()
                return {"local_path": path.name, "remote_path": remote,
                        "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            client_path = root / "client.json"; client_path.write_text(json.dumps(doc, sort_keys=True))
            guardian = {"child_rc": 0, "survivor_handoff": False, "term_count": 0}
            guardian_path = root / "guardian.json"; guardian_path.write_text(json.dumps(guardian, sort_keys=True))
            guardian_pin = pin(guardian_path, "/remote/guardian.json")
            producer = {"status": "RAW_OBSERVED", "close_complete": True, "guardian_rc": 0,
                        "child_result": {"path": "/remote/client.json"}, "guardian_result": guardian,
                        "foreign_before": [], "foreign_after": []}
            producer_path = root / "producer.json"; producer_path.write_text(json.dumps(producer, sort_keys=True))
            producer_pin = pin(producer_path, "/remote/producer.json")
            client_pin = pin(client_path, "/remote/client.json")
            producer["child_result"] = {"path": "/remote/client.json", "sha256": client_pin["sha256"], "size": client_pin["size"]}
            producer_path.write_text(json.dumps(producer, sort_keys=True)); producer_pin = pin(producer_path, "/remote/producer.json")
            value = {**doc, "client_result": doc, "producer_result": producer, "guardian_result": guardian,
                     "raw_proofs": {"client": client_pin, "producer": producer_pin, "guardian": guardian_pin},
                     "artifact_map": {"raw_stdout": {**pin(out, str(out)), "local_path": out.name},
                                      "raw_stderr": {**pin(err, str(err)), "local_path": err.name},
                                      "wav": {**pin(wav_path, str(wav_path)), "local_path": wav_path.name}}}
            closure = {"closure_sha256": "a" * 64, "validation": {"config": {"tts": {"phases": {
                "tts.customvoice.b1": {"voice_input": {"kind": "named_speaker", "speaker": "Serena", "language": "English"}}}}}},
                       "tts_frozen": {"tts.customvoice.b1": {"voice_input": {"kind": "named_speaker", "speaker": "Serena", "language": "English"}}}}
            self.assertTrue(upgrade.validate_tts_result(root, "tts.customvoice.b1", closure, value))
            os.environ.update(env)
            try:
                with self.assertRaises(FileExistsError):
                    smoke._write_upgrade_result(ns, observation, out, err)
            finally:
                for key, value in old.items():
                    if value is None: os.environ.pop(key, None)
                    else: os.environ[key] = value
    def test_bound_artifact_mismatch_never_writes_pass_envelope(self):
        cases = ("request_frame_count", "pcm_sha256", "worker_reaped", "timing", "zero", "format", "wav")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                root = Path(td); wav_path = root / "audio.wav"; pcm = b"\x01\x00" * 8
                with wave.open(str(wav_path), "wb") as stream:
                    stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000); stream.writeframes(pcm)
                out, err, result = root / "stdout", root / "stderr", root / "upgrade.json"
                out.write_bytes(b"native\n"); err.write_bytes(b"")
                ns = smoke.parser().parse_args(["--upgrade-result-json", str(result), "--output-wav", str(wav_path), "--speaker", "Serena", "--language", "English"])
                observation = {"mode": "native_api_smoke", "status": "ok", "finished": True, "cancelled": False,
                               "request_frame_count": 8, "codec_frames": 8, "pcm_frames": 8, "bytes": len(pcm),
                               "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "worker_reaped": True, "wav": str(wav_path),
                               "text": "Hello.", "send_mono": 1.0, "first_chunk_mono": 1.1, "terminal_mono": 1.2,
                               "ttfa_ms": 100.0, "generation_ms": 200.0, "rtf": 600.0}
                if case == "request_frame_count": observation[case] = 7
                elif case == "pcm_sha256": observation[case] = "0" * 64
                elif case == "worker_reaped": observation[case] = False
                elif case == "timing": observation["rtf"] = 0.1
                elif case == "wav": observation["wav"] = str(root / "other.wav")
                elif case == "zero":
                    zero = b"\0" * len(pcm)
                    with wave.open(str(wav_path), "wb") as stream:
                        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000); stream.writeframes(zero)
                    observation["pcm_sha256"] = hashlib.sha256(zero).hexdigest()
                elif case == "format":
                    with wave.open(str(wav_path), "wb") as stream:
                        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(24000); stream.writeframes(pcm)
                env = {"SLV_UPGRADE_ROW_ID": "tts.customvoice.b1", "SLV_UPGRADE_PHASE": "tts.customvoice.b1",
                       "SLV_UPGRADE_VARIANT": "customvoice", "SLV_UPGRADE_CLOSURE_SHA256": "a" * 64}
                old = {key: os.environ.get(key) for key in env}; os.environ.update(env)
                try:
                    with self.assertRaises((RuntimeError, OSError, wave.Error)):
                        smoke._write_upgrade_result(ns, observation, out, err)
                finally:
                    for key, value in old.items():
                        if value is None: os.environ.pop(key, None)
                        else: os.environ[key] = value
                self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()
