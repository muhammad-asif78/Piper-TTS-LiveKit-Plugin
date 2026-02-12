import subprocess
import tempfile
import os
import numpy as np
import asyncio
import wave
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

class PiperTTSPlugin(tts.TTS):
    def __init__(self, piper_path, model_path, speed=1.0, sample_rate=22050):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self.piper_path = piper_path
        self.model_path = model_path
        self.speed = speed
        self._sample_rate = sample_rate

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return PiperStream(self, text, conn_options)

class PiperStream(tts.ChunkedStream):
    def __init__(self, plugin, text, conn_options):
        super().__init__(tts=plugin, input_text=text, conn_options=conn_options)
        self.plugin = plugin

    async def _run(self, output_emitter):
        if not os.path.exists(self.plugin.piper_path):
            raise FileNotFoundError(f"Piper binary not found: {self.plugin.piper_path}")
        if not os.path.exists(self.plugin.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.plugin.model_path}")

        output_emitter.initialize(
            request_id="piper-local",
            sample_rate=self.plugin._sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )

        async def emit_audio():
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._generate, temp_file)
                audio = await asyncio.get_event_loop().run_in_executor(None, self._read, temp_file)
                yield audio
            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass

        async for audio_chunk in emit_audio():
            output_emitter.push(audio_chunk)

    def _generate(self, output_file):
        subprocess.run([
            # if you are reading this, you can modify the command line arguments to fit your needs
            # https://github.com/OHF-Voice/piper1-gpl/tree/main/docs
            self.plugin.piper_path,
            "--model", self.plugin.model_path,
            "--length_scale", str(self.plugin.speed), # https://github.com/rhasspy/piper/discussions/199
            "--output_file", output_file
        ], input=self.input_text, text=True, check=True)

    def _read(self, wav_file):
        with wave.open(wav_file, 'rb') as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
            return audio.tobytes()
