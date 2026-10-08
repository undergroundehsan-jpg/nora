"""End-to-end turn loop over /ws/voice with STT, TTS, LLM, VAD and logging faked.

Exercises the real main.py control flow: endpointing → filters → classification →
policy → fixed line / LLM pipeline → call close. No network, audio devices or disk logs.
"""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.main as main
from app.session import AssistantState

CHUNK = b"\x10\x00" * 512  # 512 int16 samples = one VAD frame


class FakeSTT:
    """Hands out one scripted utterance per turn as soon as audio arrives."""

    script: list = []

    def __init__(self):
        self._final_parts = []
        self._latest_interim = ""
        self._speech_final = False
        self._last = 0.0

    async def ensure_connection(self):
        return True

    async def send_audio(self, pcm_bytes):
        if not self._final_parts and FakeSTT.script:
            self._final_parts = [FakeSTT.script.pop(0)]
            self._speech_final = True
            self._last = time.time()

    async def send_audio_float32(self, arr):
        pass

    async def get_final_transcript(self):
        return " ".join(self._final_parts)

    async def reset_transcript(self):
        self._final_parts = []
        self._speech_final = False

    def has_any_final(self):
        return bool(self._final_parts)

    def has_any_transcript(self):
        return bool(self._final_parts)

    def speech_final_received(self):
        return self._speech_final

    def utterance_end_received(self):
        return False

    def seconds_since_last_result(self):
        return time.time() - self._last if self._last else 1e9

    async def close(self):
        pass


class FakeVAD:
    def __call__(self, tensor, sr):
        class _P:
            @staticmethod
            def item():
                return 0.9
        return _P()


@pytest.fixture
def harness(monkeypatch):
    spoken: list[str] = []
    events: list[dict] = []
    llm_calls: list[dict] = []
    ENDED.clear()

    async def fake_speak_text(session, text, q, **kw):
        spoken.append(text)

    async def fake_play_cached(session, name, q, fallback_text="", expect_text=""):
        said = expect_text or fallback_text
        if said:
            spoken.append(said)

    async def fake_process_tts_queue(session, q, ws_q, **kw):
        while True:
            item = await q.get()
            if item is None:
                break
        if await session.get_state() == AssistantState.SPEAKING:
            await session.set_state(AssistantState.LISTENING)

    async def fake_llm(user_input, lang, session, tts_queue, **kw):
        llm_calls.append({"text": user_input, "intent": kw.get("primary_intent"), "stage": session.call_stage})
        reply = "Just a quick reminder about your account. Has it already been taken care of?"
        await tts_queue.put(reply)
        await tts_queue.put(None)
        session.chat_history.append({"role": "user", "content": user_input})
        session.chat_history.append({"role": "assistant", "content": reply})
        return reply

    async def noop(*a, **kw):
        pass

    async def fake_log_event(session, record):
        events.append(record)

    async def fake_log_session_summary(session):
        ENDED.append(session.session_id)

    class _NoGroq:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(main, "DeepgramSTTStream", FakeSTT)
    monkeypatch.setattr(main, "speak_text", fake_speak_text)
    monkeypatch.setattr(main, "play_cached_audio", fake_play_cached)
    monkeypatch.setattr(main, "process_tts_queue", fake_process_tts_queue)
    monkeypatch.setattr(main, "get_ai_response_stream", fake_llm)
    monkeypatch.setattr(main, "warm_up_llm_connection", noop)
    monkeypatch.setattr(main, "warm_up_tts_connection", noop)
    monkeypatch.setattr(main, "log_turn", noop)
    monkeypatch.setattr(main, "log_turn_latency", noop)
    monkeypatch.setattr(main, "log_session_summary", fake_log_session_summary)
    monkeypatch.setattr(main, "log_event", fake_log_event)
    monkeypatch.setattr(main, "END_CALL_GRACE_SECS", 0.0)
    monkeypatch.setattr(main.vad_service, "model", FakeVAD())
    import app.services.intent as intent_mod
    monkeypatch.setattr(intent_mod, "_client", _NoGroq())
    monkeypatch.setattr(main, "session_store", _KeepStore())
    # Deterministic greeting: variant 0 asks "is this <name>?" (see greeting_variant fixture use).
    monkeypatch.setattr(main.random, "randrange", lambda n: 0)
    # Don't wait 30 s for session expiry when the socket closes.
    monkeypatch.setattr(main.asyncio, "sleep", _fast_sleep)
    return spoken, events, llm_calls


ENDED: list[str] = []  # session ids whose server-side loop has exited


class _KeepStore(dict):
    """Session store that ignores expiry so tests can inspect the ended session."""

    def pop(self, key, default=None):
        return self.get(key, default)


_real_sleep = asyncio.sleep


async def _fast_sleep(delay, *a, **kw):
    await _real_sleep(min(delay, 0.01), *a, **kw)


def _wait_for_turn(session_id, turns_before, ws, timeout=5.0):
    """Wait until the turn has been processed and its background speech finished."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if session_id in ENDED:
            return
        session = main.session_store.get(session_id)
        if session and session.turn_count > turns_before:
            task = session.llm_task
            if task is None or task.done():
                return
        try:
            ws.send_bytes(CHUNK)  # keep the receive loop fed while we wait
        except Exception:
            return
        time.sleep(0.02)


def run_call(utterances, session_id):
    """Speak each utterance (12 VAD frames of audio).

    Returns (session, ended_by_server): whether the server ended the call
    before the client hung up.
    """
    FakeSTT.script = list(utterances)
    client = TestClient(main.app)
    ended_by_server = False
    with client.websocket_connect(f"/ws/voice?session_id={session_id}") as ws:
        try:
            for _ in utterances:
                session = main.session_store.get(session_id)
                turns_before = session.turn_count if session else 0
                for _ in range(12):
                    ws.send_bytes(CHUNK)
                _wait_for_turn(session_id, turns_before, ws)
            for _ in range(8):
                ws.send_bytes(CHUNK)
                time.sleep(0.05)
        except WebSocketDisconnect:
            pass
        time.sleep(0.2)
        ended_by_server = session_id in ENDED
    return main.session_store.get(session_id), ended_by_server


def test_confirm_then_already_paid(harness):
    spoken, events, llm_calls = harness
    session, closed = run_call(["yes speaking", "I already paid that"], "sim-paid")
    assert llm_calls and llm_calls[0]["stage"] == "REMIND", llm_calls
    assert session.identity_verified and session.reminder_delivered
    assert session.final_disposition == "ALREADY_PAID"
    assert any("note it as paid" in s for s in spoken)
    assert closed


def test_do_not_call_ends_immediately_and_records(harness):
    spoken, events, llm_calls = harness
    session, closed = run_call(["please stop calling me"], "sim-dnc")
    assert not llm_calls
    assert session.final_disposition == "DO_NOT_CALL"
    assert {"type": "suppression", "reason": "DO_NOT_CALL"} in events
    assert closed


def test_short_control_phrase_is_not_dropped_and_hold_resumes(harness):
    spoken, events, llm_calls = harness
    session, closed = run_call(["one sec", "okay I'm back what is this about"], "sim-hold")
    assert "Sure, take your time." in spoken
    assert session.hold_active is False
    assert llm_calls and llm_calls[-1]["intent"] == "GATEKEEPER"
    assert not closed


def test_busy_then_callback_time(harness):
    spoken, events, llm_calls = harness
    session, closed = run_call(["I'm too busy right now", "tomorrow at 3"], "sim-busy")
    assert session.final_disposition == "CALLBACK_SCHEDULED"
    assert session.callback_time == "tomorrow at 3"
    assert events and events[-1]["type"] == "callback_request"
    assert closed


def test_voicemail_leaves_message(harness):
    spoken, events, llm_calls = harness
    session, closed = run_call(["please leave a message after the tone"], "sim-vm")
    assert session.final_disposition == "VOICEMAIL_LEFT"
    assert spoken and "Please call us back" in spoken[-1]
    assert closed


def test_repeated_llm_failure_ends_with_goodbye(harness, monkeypatch):
    """Real LLM wrapper with a failing Groq client: retry line once, then a goodbye."""
    spoken, events, llm_calls = harness
    import app.services.llm as llm_mod

    class _Boom:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    raise RuntimeError("groq down")

    queued: list[str] = []

    async def recording_tts_queue(session, q, ws_q, **kw):
        while True:
            item = await q.get()
            if item is None:
                break
            queued.append(item)

    monkeypatch.setattr(llm_mod, "client", _Boom())
    monkeypatch.setattr(main, "get_ai_response_stream", llm_mod.get_ai_response_stream)
    monkeypatch.setattr(main, "process_tts_queue", recording_tts_queue)

    session, ended = run_call(["yes speaking", "what is this about"], "sim-llm-down")
    assert "Sorry, I missed that. Could you say it one more time?" in queued
    assert session.final_disposition == "TECH_TROUBLE"
    assert spoken and "technical trouble" in spoken[-1]
    assert ended


def test_goodbye_falls_back_to_cached_audio_when_tts_is_silent(harness, monkeypatch):
    spoken, events, llm_calls = harness
    played: list[str] = []

    async def recording_play_cached(session, name, q, fallback_text="", expect_text=""):
        played.append(name)

    monkeypatch.setattr(main, "play_cached_audio", recording_play_cached)
    session, ended = run_call(["please stop calling me"], "sim-silent-tts")
    assert session.final_disposition == "DO_NOT_CALL"
    assert "decline_bye" in played  # live goodbye produced no audio → cached goodbye
    assert ended


def test_every_greeting_asks_who_is_speaking():
    """A "yes" may only confirm identity when the greeting actually asked."""
    from app.utils.language import build_greeting_variants
    for _name, text in build_greeting_variants("Dr. Sarah Mitchell", "Mitchell Family Clinic"):
        lowered = text.lower()
        assert "is this dr. sarah mitchell?" in lowered or "am i speaking with dr. sarah mitchell?" in lowered
        assert len(text.split()) <= 20, f"greeting too long: {text}"


class FakeSTTStuckUtteranceEnd(FakeSTT):
    """Deepgram's UtteranceEnd flag is sticky until the transcript is reset.

    Reproduces the live-call defect where it kept ending empty turns (and grabbed
    a half-finished interim as a turn).
    """

    def utterance_end_received(self):
        return True


def test_utterance_end_without_words_never_starts_a_turn(harness, monkeypatch):
    spoken, events, llm_calls = harness
    monkeypatch.setattr(main, "DeepgramSTTStream", FakeSTTStuckUtteranceEnd)
    FakeSTT.script = []
    client = TestClient(main.app)
    with client.websocket_connect("/ws/voice?session_id=sim-stuck-ue") as ws:
        for _ in range(40):
            ws.send_bytes(CHUNK)
        time.sleep(0.4)
    session = main.session_store.get("sim-stuck-ue")
    assert session.turn_count == 0, "empty utterance-end should not start a turn"
    assert not llm_calls


def test_call_uses_the_lead_passed_on_the_socket(harness):
    """A real customer's details drive the greeting, the prompt and the logs."""
    FakeSTT.script = []
    client = TestClient(main.app)
    with client.websocket_connect("/ws/voice?session_id=sim-lead&lead_id=L-1001") as ws:
        ws.send_bytes(CHUNK)
        time.sleep(0.3)
    session = main.session_store.get("sim-lead")
    assert session.lead_id == "L-1001"
    assert session.lead_context["lead_name"] == "Ehsan Khan"
    assert session.lead_context["amount_due"] == "$85.00"
    greeting = [m["content"] for m in session.chat_history if m["role"] == "assistant"][0]
    assert "Ehsan Khan" in greeting and "Mitchell Family Clinic" in greeting


def test_unknown_lead_id_refuses_the_call(harness):
    FakeSTT.script = []
    client = TestClient(main.app)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/voice?session_id=sim-bad-lead&lead_id=NOPE") as ws:
            ws.send_bytes(CHUNK)
            time.sleep(0.2)
            ws.receive_bytes()
    assert main.session_store.get("sim-bad-lead") is None
