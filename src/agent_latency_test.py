import logging
import os
import shutil
import inspect
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    tts,
)
from livekit.plugins import deepgram, noise_cancellation, openai, silero

from custom_tts.PiperTTSPluginLocal import PiperTTSPlugin

logger = logging.getLogger("voice-agent-latency")


def _load_env_files() -> None:
    search_roots = [Path.cwd(), *Path(__file__).resolve().parents]
    seen = set()
    for root in search_roots:
        for filename in (".env", ".env.local"):
            env_path = (root / filename).resolve()
            if env_path in seen or not env_path.exists():
                continue
            load_dotenv(dotenv_path=env_path, override=False)
            seen.add(env_path)


_load_env_files()


class LatencyTracker:
    def __init__(self, room_name: str) -> None:
        self.room_name = room_name
        self.parts: dict[str, dict[str, float]] = {}

    def on_metrics_collected(self, event) -> None:
        metric = getattr(event, "metrics", event)

        stt_ms = self._stt_latency_ms(metric)
        llm_ms = self._llm_latency_ms(metric)
        tts_ms = self._tts_latency_ms(metric)

        if stt_ms is not None:
            logger.info("[%s] STT latency: %.2f ms", self.room_name, stt_ms)
        elif llm_ms is not None:
            logger.info("[%s] LLM latency: %.2f ms", self.room_name, llm_ms)
        elif tts_ms is not None:
            logger.info("[%s] TTS latency: %.2f ms", self.room_name, tts_ms)

        key = getattr(metric, "speech_id", None)
        if not key:
            return

        part_map = self.parts.setdefault(key, {})
        if stt_ms is not None:
            part_map["stt"] = stt_ms
        if llm_ms is not None:
            part_map["llm"] = llm_ms
        if tts_ms is not None:
            part_map["tts"] = tts_ms

        if all(k in part_map for k in ("stt", "llm", "tts")):
            total_ms = part_map["stt"] + part_map["llm"] + part_map["tts"]
            logger.info(
                "[%s] Total latency for speech_id %s: STT=%.2fms + LLM=%.2fms + TTS=%.2fms = %.2fms",
                self.room_name,
                key,
                part_map["stt"],
                part_map["llm"],
                part_map["tts"],
                total_ms,
            )
            self.parts.pop(key, None)

    @staticmethod
    def _stt_latency_ms(metric) -> float | None:
        if hasattr(metric, "end_of_utterance_delay"):
            return float(metric.end_of_utterance_delay) * 1000.0
        return None

    @staticmethod
    def _llm_latency_ms(metric) -> float | None:
        if hasattr(metric, "ttft"):
            return float(metric.ttft) * 1000.0
        return None

    @staticmethod
    def _tts_latency_ms(metric) -> float | None:
        if hasattr(metric, "ttfb"):
            return float(metric.ttfb) * 1000.0
        return None


class Assistant(Agent):
    def __init__(self) -> None:
        piper_bin = (
            os.getenv("PIPER_PATH")
            or shutil.which("piper")
            or str(Path(__file__).resolve().parents[2] / ".venv/bin/piper")
        )
        piper_model = (
            os.getenv("PIPER_MODEL_PATH")
            or str(Path(__file__).resolve().parents[2] / "models/en_US-lessac-medium.onnx")
        )

        base_tts = PiperTTSPlugin(piper_bin, piper_model, 1, 22500)
        super().__init__(
            instructions="You are a helpful voice assistant. Keep replies concise.",
            stt=deepgram.STT(
                model="nova-3",
                language="en-US",
                endpointing_ms=10,
                no_delay=True,
                interim_results=True,
            ),
            llm=openai.LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            tts=tts.StreamAdapter(
                tts=base_tts,
                sentence_tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=8),
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Say hello and ask how you can help.", allow_interruptions=True
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    logger.info("connecting to room %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info("starting voice assistant for participant %s", participant.identity)

    usage_collector = metrics.UsageCollector()
    tracker = LatencyTracker(ctx.room.name)

    session_kwargs = {
        "vad": ctx.proc.userdata["vad"],
        "min_endpointing_delay": 0.15,
        "max_endpointing_delay": 1.2,
        "min_interruption_duration": 0.2,
        "false_interruption_timeout": 1.0,
        "preemptive_generation": True,
    }
    supported = set(inspect.signature(AgentSession.__init__).parameters)
    session = AgentSession(**{k: v for k, v in session_kwargs.items() if k in supported})

    @session.on("metrics_collected")
    def _on_metrics_collected(event) -> None:
        metric = getattr(event, "metrics", event)
        metrics.log_metrics(metric)
        usage_collector.collect(metric)
        tracker.on_metrics_collected(event)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
