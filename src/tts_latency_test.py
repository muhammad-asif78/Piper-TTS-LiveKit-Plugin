import argparse
import asyncio
import os
import shutil
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from custom_tts.PiperTTSPluginLocal import PiperTTSPlugin


def load_env_files() -> None:
    search_roots = [Path.cwd(), *Path(__file__).resolve().parents]
    seen = set()
    for root in search_roots:
        for filename in (".env", ".env.local"):
            env_path = (root / filename).resolve()
            if env_path in seen or not env_path.exists():
                continue
            load_dotenv(dotenv_path=env_path, override=False)
            seen.add(env_path)


async def run_once(plugin: PiperTTSPlugin, text: str) -> dict:
    started = time.perf_counter()
    first_frame_at = None
    total_audio_s = 0.0
    frames = 0

    async with plugin.synthesize(text) as stream:
        async for ev in stream:
            now = time.perf_counter()
            if first_frame_at is None:
                first_frame_at = now
            total_audio_s += ev.frame.duration
            frames += 1

    ended = time.perf_counter()
    ttfb = (first_frame_at - started) if first_frame_at is not None else -1.0
    total = ended - started
    rtf = (total / total_audio_s) if total_audio_s > 0 else float("inf")
    return {
        "ttfb_s": ttfb,
        "total_s": total,
        "audio_s": total_audio_s,
        "frames": frames,
        "rtf": rtf,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Piper TTS realtime latency test")
    parser.add_argument("--text", default="Hello, this is a realtime TTS latency test.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=22500)
    args = parser.parse_args()

    load_env_files()
    piper_path = (
        os.getenv("PIPER_PATH")
        or shutil.which("piper")
        or str(Path(__file__).resolve().parents[2] / ".venv/bin/piper")
    )
    model_path = (
        os.getenv("PIPER_MODEL_PATH")
        or str(Path(__file__).resolve().parents[2] / "models/en_US-lessac-medium.onnx")
    )

    print(f"PIPER_PATH={piper_path}")
    print(f"PIPER_MODEL_PATH={model_path}")

    plugin = PiperTTSPlugin(
        piper_path=piper_path,
        model_path=model_path,
        speed=args.speed,
        sample_rate=args.sample_rate,
    )

    results = []
    for i in range(args.runs):
        res = await run_once(plugin, args.text)
        results.append(res)
        print(
            f"run {i + 1}: "
            f"ttfb={res['ttfb_s']:.3f}s, total={res['total_s']:.3f}s, "
            f"audio={res['audio_s']:.3f}s, frames={res['frames']}, rtf={res['rtf']:.3f}"
        )

    ttfb_vals = [r["ttfb_s"] for r in results if r["ttfb_s"] >= 0]
    total_vals = [r["total_s"] for r in results]
    rtf_vals = [r["rtf"] for r in results if r["rtf"] != float("inf")]

    print("\nsummary:")
    if ttfb_vals:
        print(
            f"ttfb  mean={statistics.mean(ttfb_vals):.3f}s "
            f"p50={statistics.median(ttfb_vals):.3f}s min={min(ttfb_vals):.3f}s max={max(ttfb_vals):.3f}s"
        )
    print(
        f"total mean={statistics.mean(total_vals):.3f}s "
        f"p50={statistics.median(total_vals):.3f}s min={min(total_vals):.3f}s max={max(total_vals):.3f}s"
    )
    if rtf_vals:
        print(
            f"rtf   mean={statistics.mean(rtf_vals):.3f} "
            f"p50={statistics.median(rtf_vals):.3f} min={min(rtf_vals):.3f} max={max(rtf_vals):.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
