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
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
    tts,
    tokenize,
)
from livekit.plugins import (
    openai,
    deepgram,
    silero,
)

from livekit.plugins import noise_cancellation
from custom_tts.PiperTTSPluginLocal import PiperTTSPlugin

logger = logging.getLogger("voice-agent")


def _load_env_files() -> None:
    # Load env files from CWD and this file's parent chain so running from src/
    # can still pick up the repo-level .env.
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

class Assistant(Agent):
    def __init__(self) -> None:
        piper_bin = (
            os.getenv("PIPER_PATH")
            or shutil.which("piper")
            or str(Path(__file__).resolve().parents[2] / ".venv/bin/piper")
        )
        piper_model = (
            os.getenv("PIPER_MODEL_PATH")
            or str(Path(__file__).resolve().parents[2] / "models/es_ES-carlfm-x_low.onnx")
        )

        base_tts = PiperTTSPlugin(piper_bin, piper_model, 1, 22500)
        super().__init__(
            instructions="sample assistant to test piper",
            stt=deepgram.STT(
                model="nova-3",
                language="en-US",
                endpointing_ms=10,
                no_delay=True,
                interim_results=True,
            ),
            llm=openai.LLM(
                model="gpt-4o-mini",  
                temperature=0.7
            ),
            tts=tts.StreamAdapter(
                tts=base_tts,
                sentence_tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=8),
            ),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Hey, how can I help you today?", allow_interruptions=True
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    def on_metrics_collected(event):
        # LiveKit 1.3 emits MetricsCollectedEvent with `metrics` payload.
        agent_metrics = getattr(event, "metrics", event)
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

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

    session.on("metrics_collected", on_metrics_collected)

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
