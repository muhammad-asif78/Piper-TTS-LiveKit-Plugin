# LiveKit + Piper TTS Plugin

Low-latency local Text-to-Speech integration for LiveKit Agents using Piper ONNX voices.

This repository contains:
- A LiveKit voice agent using Deepgram STT + OpenAI LLM + Piper TTS
- Custom Piper TTS plugins
- Latency test scripts for realtime benchmarking
- Docker support for easy deployment

## Features
- Local Piper TTS via subprocess (`PiperTTSPluginLocal.py`)
- Optional Piper Python API plugin (`PiperTTSPlugin.py`)
- Streaming-compatible playback in LiveKit via `tts.StreamAdapter`
- English model support out of the box (`en_US-lessac-medium`)
- End-to-end latency logging (STT / LLM / TTS / total)
- Dockerfile + `.dockerignore` for containerized runs

## Project Structure
```text
LiveKit-PiperTTS-Plugin/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── agent.py                   # Main agent
│   ├── agent_latency_test.py      # Agent with latency breakdown logging
│   ├── tts_latency_test.py        # Standalone Piper TTS latency test
│   └── custom_tts/
│       ├── PiperTTSPluginLocal.py # Piper CLI subprocess backend
│       └── PiperTTSPlugin.py      # Piper Python API backend
└── README.md
```

## Requirements
- Python 3.12+
- LiveKit credentials
- API keys:
  - `LIVEKIT_URL`
  - `LIVEKIT_API_KEY`
  - `LIVEKIT_API_SECRET`
  - `OPENAI_API_KEY`
  - `DEEPGRAM_API_KEY`
- Piper binary and model files (`.onnx` + `.onnx.json`)

## Quick Start (Local)

### 1. Install dependencies
```bash
cd piper_tts/LiveKit-PiperTTS-Plugin
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download English Piper model
```bash
mkdir -p ../models
cd ../models
curl -fL -o en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -fL -o en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### 3. Configure environment
In your project `.env` (repo root), set:
```env
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENAI_API_KEY=...
DEEPGRAM_API_KEY=...

PIPER_PATH=/absolute/path/to/piper
PIPER_MODEL_PATH=/absolute/path/to/en_US-lessac-medium.onnx
```

### 4. Run agent
```bash
cd piper_tts/LiveKit-PiperTTS-Plugin/src
python3 agent.py console
```

## Latency Testing

### 1) Full agent latency (STT + LLM + TTS)
```bash
cd piper_tts/LiveKit-PiperTTS-Plugin/src
python3 agent_latency_test.py console
```

This logs:
- STT latency (`end_of_utterance_delay`)
- LLM latency (`ttft`)
- TTS latency (`ttfb`)
- Total latency per `speech_id` when all parts are available

### 2) Standalone TTS latency only
```bash
cd piper_tts/LiveKit-PiperTTS-Plugin/src
python3 tts_latency_test.py --runs 3 --text "Hello, this is a realtime latency test."
```

This prints:
- `ttfb` (time to first audio frame)
- total synthesis time
- produced audio duration
- real-time factor (RTF)

## Docker

### Build
```bash
cd piper_tts/LiveKit-PiperTTS-Plugin
docker build -t livekit-piper-tts .
```

### Run
```bash
docker run --rm -it \
  --env-file ../../.env \
  -v /absolute/path/to/piper_tts/models:/models \
  livekit-piper-tts
```

Default container settings:
- `PIPER_PATH=/usr/local/bin/piper`
- `PIPER_MODEL_PATH=/models/en_US-lessac-medium.onnx`

## Plugin Usage

### Local subprocess backend
```python
from custom_tts.PiperTTSPluginLocal import PiperTTSPlugin

tts = PiperTTSPlugin(
    piper_path="/path/to/piper",
    model_path="/path/to/model.onnx",
    speed=1.0,
    sample_rate=22500,
)
```

### Piper Python API backend
```python
from custom_tts.PiperTTSPlugin import PiperTTSPlugin

tts = PiperTTSPlugin(
    model="/path/to/model.onnx",
    speed=1.0,
    volume=1.0,
    noise_scale=0.667,
    noise_w=0.8,
    use_cuda=False,
)
```

## Troubleshooting
- `ValueError: ws_url is required`
  - Set `LIVEKIT_URL` in `.env`.
- `Deepgram API key is required`
  - Set `DEEPGRAM_API_KEY`.
- `OpenAI auth errors`
  - Verify `OPENAI_API_KEY` and account quota.
- `Piper model not found`
  - Check `PIPER_MODEL_PATH` and ensure `.onnx` exists.
- No audio output in container
  - Confirm model mount path and `PIPER_MODEL_PATH=/models/...`.
- `Unable to find image 'livekit-piper-tts:latest'`
  - Build first: `docker build -t livekit-piper-tts .`

## Notes
- Keep API keys out of git.
- Rotate keys immediately if they were exposed.
- For lower latency, tune:
  - Deepgram `endpointing_ms`
  - AgentSession endpointing settings
  - Piper model size (small/medium/large tradeoff)
