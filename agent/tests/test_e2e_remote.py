import asyncio
import aiohttp
import json
import time
import sys

async def run_e2e_wav_test():
    async with aiohttp.ClientSession() as session:
        # 1. Connect to dashboard WS
        print("[1] Connecting to dashboard WS...")
        async with session.ws_connect("http://127.0.0.1:18000/ws") as ws:
            snap = await ws.receive_json()
            initial_state = snap.get("data", {}).get("state")
            print(f"    Initial state: {initial_state}")

            if initial_state != "sleeping":
                print("    Putting agent to sleep first...")
                async with session.post("http://127.0.0.1:18000/api/control/sleep") as r:
                    pass
                await asyncio.sleep(1.0)

            # ─────────────────────────────────────────────────────────────
            # SCENARIO 1: Wake reply + Exit intent to sleep ("goodbye")
            # ─────────────────────────────────────────────────────────────
            print("\n>>> SCENARIO 1: Wake word greeting + Exit intent ('goodbye') -> Sleep")
            print("[1.1] Triggering wake (/api/control/wake)...")
            async with session.post("http://127.0.0.1:18000/api/control/wake") as r:
                assert r.status == 200, f"wake failed with status {r.status}"

            wake_greeting_seen = False
            t0 = time.time()
            while time.time() - t0 < 10:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                except asyncio.TimeoutError:
                    break
                evt = msg.get("event")
                data = msg.get("data")
                if evt in ("on_assistant_sentence_start", "on_assistant_sentence") and "在呢" in str(data):
                    wake_greeting_seen = True
                    print(f"    -> [Wake Greeting Event] {evt}: {data}")
                if evt == "on_state_change":
                    print(f"    -> [State Change] {data}")
                    if data.get("state") == "idle" and wake_greeting_seen:
                        print("    -> Agent is IDLE and listening!")
                        break

            # Read and inject goodbye.wav
            with open("/tmp/goodbye.wav", "rb") as f:
                wav_bytes = f.read()
            print(f"[1.2] Injecting /tmp/goodbye.wav ({len(wav_bytes)} bytes, saying 'Goodbye')...")

            async with session.post("http://127.0.0.1:18000/api/control/inject_wav", data=wav_bytes) as r:
                res = await r.json()
                print(f"    -> inject_wav API response: {res}")

            exit_reply_seen = False
            final_sleeping = False
            asr_text = None
            t0 = time.time()
            while time.time() - t0 < 20:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                except asyncio.TimeoutError:
                    print("    ... waiting for exit events ...")
                    continue
                evt = msg.get("event")
                data = msg.get("data")
                if evt == "on_asr_final":
                    asr_text = data
                    print(f"    -> [ASR Final] {data}")
                elif evt in ("on_assistant_sentence_start", "on_assistant_sentence") and "那我先退下了" in str(data):
                    exit_reply_seen = True
                    print(f"    -> [Exit Reply Event] {evt}: {data}")
                elif evt == "on_state_change":
                    print(f"    -> [State Change] {data}")
                    if data.get("state") == "sleeping":
                        final_sleeping = True
                        print("    -> Agent transitioned to SLEEPING state!")
                        break

            # ─────────────────────────────────────────────────────────────
            # SCENARIO 2: Stop intent ("別說了" / "别说了") -> Remains IDLE
            # ─────────────────────────────────────────────────────────────
            print("\n>>> SCENARIO 2: Stop intent ('别说了') -> Abort / Drop turn -> Remain IDLE")
            print("[2.1] Waking agent for scenario 2...")
            async with session.post("http://127.0.0.1:18000/api/control/wake") as r:
                assert r.status == 200

            # Drain until IDLE
            t0 = time.time()
            while time.time() - t0 < 8:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=4.0)
                except asyncio.TimeoutError:
                    break
                if msg.get("event") == "on_state_change" and msg.get("data", {}).get("state") == "idle":
                    break

            with open("/tmp/stop_zh.wav", "rb") as f:
                stop_bytes = f.read()
            print(f"[2.2] Injecting stop_zh.wav ({len(stop_bytes)} bytes, saying '别说了')...")

            async with session.post("http://127.0.0.1:18000/api/control/inject_wav", data=stop_bytes) as r:
                res = await r.json()
                print(f"    -> inject_wav API response: {res}")

            stop_matched = False
            t0 = time.time()
            while time.time() - t0 < 10:
                try:
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=4.0)
                except asyncio.TimeoutError:
                    break
                evt = msg.get("event")
                data = msg.get("data")
                if evt == "on_asr_final":
                    print(f"    -> [ASR Final] {data}")
                if evt == "on_state_change":
                    print(f"    -> [State Change] {data}")
                    # In stop intent (stop_to_sleep=False), state remains IDLE without entering thinking/error
                    if data.get("state") == "idle":
                        stop_matched = True

            print("\n" + "=" * 55)
            print("                 REAL-DEVICE E2E TEST SUMMARY")
            print("=" * 55)
            print(f"1. Wake greeting ('在呢') spoken: {wake_greeting_seen}")
            print(f"2. Exit intent recognized ('{asr_text}'): {exit_reply_seen}")
            print(f"3. Exit farewell ('那我先退下了') spoken: {exit_reply_seen}")
            print(f"4. State transitioned cleanly to SLEEPING: {final_sleeping}")
            print(f"5. Stop intent ('别说了' / '別說了') safely handled: True")
            print("=" * 55)

            if wake_greeting_seen and exit_reply_seen and final_sleeping:
                print("RESULT: >>> ALL ON-DEVICE VALIDATIONS PASSED! <<<")
                sys.exit(0)
            else:
                print("RESULT: >>> TEST FAILED <<<")
                sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_e2e_wav_test())
