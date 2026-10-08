"""
Streaming Speech-to-Text via Deepgram Live WebSocket.

Keeps one WebSocket open for the whole call session and reuses it across turns.
If the socket drops unexpectedly, it reconnects on the next audio send.
"""

import asyncio
import json
import os
from urllib.parse import urlencode

import numpy as np
import websockets
from dotenv import load_dotenv
from websockets import State
import time

load_dotenv()

FS = 16000  # sample rate

# Tunables (defaults match the previous hard-coded values).
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2").strip() or "nova-2"
DEEPGRAM_ENDPOINTING_MS = os.getenv("DEEPGRAM_ENDPOINTING_MS", "600").strip() or "600"
DEEPGRAM_UTTERANCE_END_MS = os.getenv("DEEPGRAM_UTTERANCE_END_MS", "1500").strip() or "1500"


class DeepgramSTTStream:
    """
    Manages a persistent Deepgram live WebSocket connection for one session.

    Lifecycle:
        start()                - open or reuse WS connection
        ensure_connection()    - reconnect only when dropped
        send_audio(bytes)      - forward raw PCM-16 audio in real-time
        get_final_transcript() - read accumulated is_final segments
        reset_transcript()     - clear transcript between turns
        close()                - terminate at session end
    """

    def __init__(self):
        self._api_key: str = os.getenv("STT_DEEPGRAM", "")
        self._ws = None
        self._final_parts: list[str] = []
        self._latest_interim: str = ""
        self._lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._connected = False
        self._was_connected = False
        self._closing = False
        self._is_final_event = asyncio.Event()
        self._last_result_ts: float = 0.0
        self._utterance_end_flag: bool = False
        self._speech_final_flag: bool = False

    def _build_ws_config(self) -> tuple[str, dict[str, str]]:
        # Keep this conservative and broadly compatible. Some aggressive
        # endpointing/vad combinations can cause HTTP 400 rejects.
        params = urlencode({
            "encoding": "linear16",
            "sample_rate": FS,
            "channels": 1,
            "model": DEEPGRAM_MODEL,
            "language": "en-US",
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
            "endpointing": DEEPGRAM_ENDPOINTING_MS,
            "utterance_end_ms": DEEPGRAM_UTTERANCE_END_MS,
        })
        url = f"wss://api.deepgram.com/v1/listen?{params}"
        headers = {"Authorization": f"Token {self._api_key}"}
        return url, headers

    async def _cleanup_connection(self, send_close: bool) -> None:
        self._connected = False

        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None

        if self._ws:
            try:
                if send_close and self._ws.state == State.OPEN:
                    await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except (Exception, asyncio.CancelledError):
                pass
            finally:
                self._ws = None

    async def start(self) -> bool:
        """Open Deepgram live-transcription WebSocket (or reuse if already open)."""
        async with self._connect_lock:
            if self._closing:
                return False
            if not self._api_key:
                print("[DEEPGRAM] Missing API key. Set STT_DEEPGRAM or DEEPGRAM_API_KEY.")
                return False

            if self._ws and self._ws.state == State.OPEN and self._connected:
                return True

            # Clean stale handles before reconnecting.
            await self._cleanup_connection(send_close=False)

            url, headers = self._build_ws_config()
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    self._ws = await websockets.connect(
                        url,
                        additional_headers=headers,
                        ping_interval=20,
                        ping_timeout=10,
                        open_timeout=20,
                    )
                    self._closing = False
                    self._connected = True
                    self._was_connected = True
                    self._receive_task = asyncio.create_task(self._receive_loop())
                    self._keepalive_task = asyncio.create_task(self._keepalive_loop())
                    print("[DEEPGRAM] Connected to live transcription API")
                    return True
                except Exception as e:
                    self._connected = False
                    print(f"[DEEPGRAM] Failed to connect (Attempt {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(float(attempt))

            print(f"[DEEPGRAM] Exhausted all {max_retries} attempts. STT will be offline.")
            return False

    async def ensure_connection(self) -> bool:
        """Ensure connection is alive; reconnect only when dropped unexpectedly."""
        if self._closing:
            return False

        if self._ws and self._ws.state == State.OPEN and self._connected:
            return True

        print("[DEEPGRAM] Connection lost. Attempting to reconnect..." if self._was_connected
              else "[DEEPGRAM] Opening live transcription connection...")
        return await self.start()

    async def close(self) -> None:
        """Close connection only when session ends."""
        self._closing = True
        await self._cleanup_connection(send_close=True)
        print("[DEEPGRAM] Connection closed and cleaned up")

    async def _receive_loop(self) -> None:
        """Read Deepgram JSON frames and accumulate final transcript segments."""
        try:
            async for msg in self._ws:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "Results":
                    alt = data.get("channel", {}).get("alternatives", [{}])[0]
                    transcript = alt.get("transcript", "")
                    is_final = data.get("is_final", False)

                    if transcript:
                        self._last_result_ts = time.time()
                        if is_final:
                            async with self._lock:
                                self._final_parts.append(transcript)
                                self._latest_interim = ""
                                # speech_final = Deepgram's endpointer saw the
                                # speaker stop, so the turn can end right away.
                                if data.get("speech_final"):
                                    self._speech_final_flag = True
                            self._is_final_event.set()
                            print(f'[DEEPGRAM] Final segment: "{transcript}"')
                        else:
                            async with self._lock:
                                self._latest_interim = transcript
                            print(f'[DEEPGRAM] Interim: "{transcript}"')

                elif msg_type == "UtteranceEnd":
                    self._utterance_end_flag = True
                    self._last_result_ts = time.time()
                    print("[DEEPGRAM] UtteranceEnd received")

                elif msg_type == "Metadata":
                    req_id = data.get("request_id", "N/A")
                    print(f"[DEEPGRAM] Metadata: request_id={req_id}")

        except websockets.exceptions.ConnectionClosedOK:
            print("[DEEPGRAM] Connection closed normally")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[DEEPGRAM] Connection closed unexpectedly: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[DEEPGRAM] Receive loop error: {e}")
        finally:
            self._connected = False

    async def _keepalive_loop(self) -> None:
        """Send KeepAlive frames so Deepgram does not drop idle WS during TTS."""
        try:
            while self._connected and not self._closing:
                await asyncio.sleep(8)
                if self._ws and self._ws.state == State.OPEN:
                    try:
                        await self._ws.send(json.dumps({"type": "KeepAlive"}))
                    except Exception:
                        self._connected = False
                        break
        except asyncio.CancelledError:
            pass

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Forward raw PCM-16 LE audio bytes to Deepgram."""
        if not await self.ensure_connection():
            return

        if self._connected and self._ws and self._ws.state == State.OPEN:
            try:
                await self._ws.send(pcm_bytes)
                return
            except Exception as e:
                print(f"[DEEPGRAM] Send error: {e}")
                self._connected = False

        # One retry after reconnect for transient drop during send.
        if await self.ensure_connection() and self._ws and self._ws.state == State.OPEN:
            try:
                await self._ws.send(pcm_bytes)
            except Exception as e:
                print(f"[DEEPGRAM] Send retry failed: {e}")
                self._connected = False

    async def send_audio_float32(self, float_array: np.ndarray) -> None:
        """Convert float32 [-1, 1] numpy array back to PCM-16 bytes and send."""
        pcm = (float_array * 32768.0).clip(-32768, 32767).astype(np.int16)
        await self.send_audio(pcm.tobytes())

    async def get_final_transcript(self) -> str:
        """Return accumulated final transcript and interim fallback if needed.

        When finals already exist, returns immediately (0ms wait).
        When only interims are available, waits up to 10ms for a final
        to arrive — just enough to catch in-flight finals without
        adding perceptible latency.
        """
        # If we already have final segments, use them immediately.
        # Only do a brief wait when we have zero finals, to give Deepgram
        # one last chance to promote the current interim.
        if not self._final_parts and not self._is_final_event.is_set():
            try:
                await asyncio.wait_for(self._is_final_event.wait(), timeout=0.01)
            except asyncio.TimeoutError:
                pass

        async with self._lock:
            parts = list(self._final_parts)
            if self._latest_interim:
                parts.append(self._latest_interim)
                if not self._final_parts:
                    print(f'[DEEPGRAM] Using interim as primary: "{self._latest_interim}"')
            return " ".join(parts).strip()

    async def reset_transcript(self) -> None:
        """Clear transcript buffers for a fresh utterance without restarting WebSocket."""
        async with self._lock:
            self._final_parts.clear()
            self._latest_interim = ""
            self._is_final_event.clear()
            self._utterance_end_flag = False
            self._speech_final_flag = False

    def seconds_since_last_result(self) -> float:
        if self._last_result_ts <= 0:
            return 1e9
        return time.time() - self._last_result_ts

    def has_any_transcript(self) -> bool:
        return bool(self._final_parts) or bool(self._latest_interim)

    def has_any_final(self) -> bool:
        return bool(self._final_parts)

    def speech_final_received(self) -> bool:
        """True when a final segment arrived flagged speech_final (endpoint detected)."""
        return self._speech_final_flag

    def utterance_end_received(self) -> bool:
        """True when Deepgram has explicitly signalled end of utterance."""
        return self._utterance_end_flag
