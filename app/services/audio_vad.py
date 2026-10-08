import asyncio
import os
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from silero_vad import get_speech_timestamps, load_silero_vad
from app.session import Session

# VAD Configuration
FS = 16000
VAD_SPEECH_THRESHOLD = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.90"))
SILENCE_LIMIT = 0.8
MAX_SECONDS = 20

# Minimum sustained speech duration (seconds) to confirm interruption
# Increased to reduce false barge-ins on line noise / far-end echo.
MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_INTERRUPTION_SECS", "0.70"))

# Maximum chunks kept in the interruption buffer (~2s of audio at 16kHz/512-sample chunks)
MAX_INTERRUPTION_BUFFER_CHUNKS = 64

# Dedicated thread pool for VAD inference — 2 threads avoids contention with
# the default ThreadPoolExecutor used by other asyncio I/O tasks.
_vad_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vad")


class VADService:
    def __init__(self):
        self.model = None
        self._model_failed = False

    def _ensure_model(self):
        """Lazy-load model to keep app startup fast and resilient on constrained hosts."""
        if self.model is not None:
            return self.model
        if self._model_failed:
            return None
        try:
            self.model = load_silero_vad()
            print("[VAD] Silero model loaded")
        except Exception as e:
            # Do not crash the app if VAD model cannot be loaded at runtime.
            self._model_failed = True
            self.model = None
            print(f"[VAD] Model load failed; interruption detection disabled: {e}")
        return self.model

    async def detect_interruption(self, session: Session, audio_tensor: torch.Tensor) -> bool:
        """
        Frame-based interruption detection.
        User must speak above threshold for MIN_SPEECH_DURATION seconds
        to trigger an interruption. Ignores the first ECHO_GRACE_SECS after
        SPEAKING starts to prevent self-interruption from echo.
        """
        if not await session.is_speaking():
            session.interruption_buffer = []
            if hasattr(session, 'interruption_active_frames'):
                session.interruption_active_frames = 0
                session.interruption_silence_frames = 0
            return False

        # Init tracking variables if they don't exist
        if not hasattr(session, 'interruption_active_frames'):
            session.interruption_active_frames = 0
            session.interruption_silence_frames = 0

        # Echo grace period: ignore VAD during the first 1.2s of SPEAKING
        # This prevents the assistant's own TTS audio bleeding into the mic
        # from triggering a false interruption.
        elapsed_since_speaking = time.time() - session.speaking_since
        if (not getattr(session, "allow_barge_in_immediate", False)) and elapsed_since_speaking < session.ECHO_GRACE_SECS:
            return False

        # Accumulate the buffer that we are testing for interruption
        # so we don't lose the first word of the user's speech!
        session.interruption_buffer.append(audio_tensor.numpy())
        # Cap to avoid unbounded growth during long TTS turns where the user stays silent
        if len(session.interruption_buffer) > MAX_INTERRUPTION_BUFFER_CHUNKS:
            session.interruption_buffer.pop(0)

        model = self._ensure_model()
        if model is None:
            return False

        with torch.no_grad():
            loop = asyncio.get_event_loop()
            prob = await loop.run_in_executor(
                _vad_executor, lambda: float(model(audio_tensor, FS).item())
            )

        if prob >= VAD_SPEECH_THRESHOLD:
            session.interruption_active_frames += 1
            session.interruption_silence_frames = 0 # reset silence streak
            
            # 512 samples @ 16kHz = 0.032s per frame
            current_speech_duration = session.interruption_active_frames * 0.032
            
            if current_speech_duration >= MIN_SPEECH_DURATION:
                print(f"[Session {session.session_id}] [INTERRUPTION DETECTED] "
                      f"prob={prob:.3f} sustained={current_speech_duration:.2f}s")
                await session.trigger_interruption()
                session.interruption_active_frames = 0
                return True
        else:
            if session.interruption_active_frames > 0:
                session.interruption_silence_frames += 1
                # If we get > 150ms of silence (about 5 frames), reset the interruption progress
                if session.interruption_silence_frames > 5:
                    session.interruption_active_frames = 0
                    session.interruption_silence_frames = 0
                    session.interruption_buffer = []
            else:
                # If no speech has started, keep a small sliding window of silence audio
                # (e.g. last 10 frames = 320ms) so when they do speak, Deepgram 
                # gets the full context immediately.
                if len(session.interruption_buffer) > 10:
                    session.interruption_buffer.pop(0)

        return False

    def get_speech_timestamps_sync(self, audio_tensor):
        model = self._ensure_model()
        if model is None:
            return []
        return get_speech_timestamps(audio_tensor, model, sampling_rate=FS)


vad_service = VADService()

