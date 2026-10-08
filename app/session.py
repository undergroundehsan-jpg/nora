import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional

# Sample lead data for testing. When the lead scraper is attached,
# this will be replaced by real data passed per-session.
SAMPLE_LEAD_CONTEXT = {
    "lead_name": "Dr. Sarah Mitchell",
    "practice_name": "Mitchell Family Clinic",
    "city": "Dallas, TX",
    "specialty": "Family Medicine",
    "designation": "Practice Owner",
    "email": "sarah.mitchell@mitchellfamily.com",  # may be empty when scraper has no email
    # REMINDER = payment/service reminder flow. "VOB" re-enables the legacy
    # benefits-verification answer overrides in main.py.
    "call_type": "REMINDER",
    # Dummy patient/provider identifiers for VOB testing
    "patient_name": "Ehsan Khan",
    "patient_dob": "01/15/1985",
    "member_id": "UHC-4729158",
    "provider_npi": "1234567890",
    "tax_id": "12-3456789",
}

class AssistantState(Enum):
    LISTENING   = "LISTENING"
    THINKING    = "THINKING"
    SPEAKING    = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"

class Session:
    """
    Per-connection state. Owns state machine + interrupt flags so that
    concurrent connections cannot interfere with each other.
    """
    def __init__(self, websocket, session_id: str, timeout_secs: int = 420):
        self.websocket = websocket
        self.session_id = session_id
        
        # Conversation data
        self.chat_history: List[Dict[str, str]] = []
        self.last_user_input: str = None
        self.last_response: str = None
        self.last_intent: Optional[str] = None
        self.current_mode: str = "INTRO"
        self.refusal_count: int = 0
        self.soft_refusal_count: int = 0
        self.hard_refusal_count: int = 0
        self.meeting_ask_count: int = 0
        self.ai_ask_count: int = 0         # times user asked "are you AI?"
        self.recent_openers: List[str] = []  # recent response opening words for variety
        self.last_objection: Optional[str] = None
        # Local outcome flags for downstream analytics / logging
        self.agreed_to_baa: bool = False
        self.agreed_to_meeting: bool = False
        self.email_captured: bool = False
        self.email_address: Optional[str] = None
        self.email_pending_confirmation: bool = False  # True after email captured, waiting for user OK
        self.lead_context: Optional[dict] = None  # resolved by app/services/leads.py
        self.lead_id: Optional[str] = None        # stable key for logs and the dashboard
        self.mood_trajectory: List[str] = []  # tracks lead's emotional arc across turns
        
        self.question_count: int = 0
        self.turn_count: int = 0          # user input + AI response = 1 turn
        # How many times the agent has tried to get/confirm the caller's name
        # without success. After 2 attempts, the prompt escalates to spelling.
        self.name_clarify_attempts: int = 0
        self.MAX_TURNS: int = 25          # allows objection handling + meeting ask before forced cutoff
        # Set True once the core pitch (free billing review offer) has been delivered.
        # Drives Discovery→Pitch→Close stage transitions in the system prompt.
        self.pitch_delivered: bool = False
        # Captured callback window from gatekeeper/busy leads (e.g. "Thursday after 4 PM")
        self.callback_time: Optional[str] = None
        self.last_transcript_normalized: str = ""
        self.last_transcript_time: float = 0.0
        self.start_time: float = time.time()
        self.last_active_time: float = time.time()
        self.timeout_secs = timeout_secs
        self.close_after_speaking: bool = False
        
        self.expected_speech_end_time: float = 0.0
        self.interruption_active_frames: int = 0
        self.interruption_silence_frames: int = 0
        self.interruption_buffer = []
        self.has_speech_in_buffer = False

        # Streaming STT (Deepgram WebSocket) — set in websocket_endpoint
        self.stt_stream = None

        # State machine
        self._state = AssistantState.LISTENING
        self._state_lock = asyncio.Lock()

        # Session-scoped interrupt flags
        self.interrupt_tts = asyncio.Event()
        self.interrupt_llm = asyncio.Event()
        
        # Generation counter for TTS audio. Incremented on each interruption.
        # Frontend uses this to discard stale in-flight chunks.
        self.tts_generation: int = 0
        
        # Echo grace period: ignore VAD for 1.2s after SPEAKING starts
        # (Accounts for WebSocket network ping + browser WebAudio buffering + acoustic echo)
        self.speaking_since: float = 0.0
        self.ECHO_GRACE_SECS = 0.8
        # If true, allow immediate barge-in (skip echo grace) for the next SPEAKING segment.
        # Useful for greetings where the user may say "hello?" or "who is this?" instantly.
        self.allow_barge_in_immediate: bool = False
        
        # Track active LLM pipeline task so we can cleanly cancel it
        self.llm_task: asyncio.Task = None

        # ── Call-flow state (driven by app/services/policy.py) ──
        # Slots describe what has actually happened on the call; the prompt
        # stage is derived from them instead of from turn_count.
        self.identity_verified: bool = False
        self.reminder_delivered: bool = False
        self.outcome: Optional[str] = None            # PAID, RESCHEDULED, CALLBACK, DISPUTE, ...
        self.final_disposition: Optional[str] = None  # set once when the call ends via policy
        self.preferences: Dict[str, str] = {}         # e.g. {"channel": "text", "avoid_time": "at work"}
        self.pending_callback_reason: Optional[str] = None  # waiting for a callback time
        self.callback_asks: int = 0
        self.escalation_reason: Optional[str] = None
        self.soft_objection_count: int = 0
        self.repeat_request_count: int = 0
        self.abuse_count: int = 0
        self.ai_disclosed: bool = False
        self.hold_active: bool = False
        self.hold_started_at: float = 0.0
        self.hold_checkin_sent: bool = False
        # Per-session TTS speed override (None = env default). Higher = slower speech (Rime timeScaleFactor).
        self.tts_time_scale: Optional[float] = None
        # Consecutive LLM generation failures (reset on a successful reply).
        self.llm_failures: int = 0
        # Audio frames actually sent to the client (lets goodbye paths detect silent TTS failures).
        self.audio_chunks_sent: int = 0

        # Per-turn latency marks (seconds since epoch), reset each turn.
        self.turn_metrics: Dict[str, float] = {}

    async def get_state(self) -> AssistantState:
        async with self._state_lock:
            return self._state

    async def add_speaking_duration(self, duration: float):
        now = time.time()
        if self.expected_speech_end_time < now:
            self.expected_speech_end_time = now + duration
        else:
            self.expected_speech_end_time += duration

    async def set_state(self, new_state: AssistantState):
        async with self._state_lock:
            old = self._state
            self._state = new_state
            # Track when we enter SPEAKING for echo grace period
            if new_state == AssistantState.SPEAKING and old != AssistantState.SPEAKING:
                self.speaking_since = time.time()
                
            # Keep silence timeouts perfectly synced no matter what component resets the state
            if new_state == AssistantState.LISTENING and old != AssistantState.LISTENING:
                self.last_active_time = time.time()
        print(f"[Session {self.session_id}] STATE: {old.value} -> {new_state.value}")

    async def is_speaking(self) -> bool:
        async with self._state_lock:
            return self._state == AssistantState.SPEAKING

    def clear_interrupts(self):
        self.interrupt_tts.clear()
        self.interrupt_llm.clear()
        self.interruption_active_frames = 0
        self.interruption_silence_frames = 0
        self.interruption_buffer = []
        # Prevent stale playback deadlines from delaying new turns/wrap-ups.
        self.expected_speech_end_time = time.time()

    def record_intent(self, intent_name: str):
        self.last_intent = intent_name

        # Track repeated AI detection asks
        if intent_name == "ASK_IF_AI":
            self.ai_ask_count += 1

        objection_intents = {
            "NOT_INTERESTED",
            "ALREADY_HAVE_BILLER",
            "HAPPY_WITH_PROVIDER",
            "TOO_BUSY",
            "ALREADY_AUDITED",
            "NO_BUDGET",
            "BAD_EXPERIENCE",
            "CANNOT_PROCEED",
        }
        if intent_name in objection_intents:
            self.last_objection = intent_name

        if intent_name in {"GATEKEEPER", "NOT_DECISION_MAKER", "IDENTITY_DENIAL", "THIRD_PARTY"}:
            self.current_mode = "GATEKEEPER"
        elif intent_name in objection_intents or intent_name in {"DISPUTE", "HARDSHIP"}:
            self.current_mode = "OBJECTION"
        elif intent_name in {
            "ASK_AUDIT_DEFINITION", "ASK_AUDIT_PROCESS", "ASK_COMPLIANCE_BAA",
            "ASK_MORE_INFO", "ASK_SOURCE", "ASK_IF_AI",
            "TRUST_CONCERN", "PRIVACY_CONCERN",
        }:
            self.current_mode = "INFO"
        elif intent_name in {
            "INTERESTED", "CALL_BACK_LATER", "EMAIL_REQUEST", "TRANSFER_TO_HUMAN",
            "RESCHEDULE", "PAYMENT_PLAN", "PAY_NOW", "ALREADY_PAID",
        }:
            self.current_mode = "SCHEDULING"
        elif intent_name in {
            "DO_NOT_CALL", "WRONG_NUMBER", "LANGUAGE_BARRIER",
            "ATTORNEY", "BANKRUPTCY", "DECEASED", "DISTRESS",
        }:
            self.current_mode = "EXIT"

        # High-level outcome flags used for local JSONL logging and later DB sync.
        # These are deliberately simple heuristics, not a full CRM.
        if intent_name == "INTERESTED":
            # User is open to the offer / next steps.
            self.agreed_to_meeting = True
        elif intent_name == "EMAIL_REQUEST":
            # User wants details sent; assume consent to receive BAA + outline.
            self.agreed_to_baa = True

    def record_refusal(self, intent_name: str = "") -> int:
        hard_intents = {"WRONG_NUMBER", "DO_NOT_CALL", "PRIVACY_CONCERN", "LANGUAGE_BARRIER"}
        
        if intent_name in hard_intents:
            self.hard_refusal_count += 1
        else:
            self.soft_refusal_count += 1
            
        self.refusal_count += 1
        return self.refusal_count

    def should_end_after_refusal(self) -> bool:
        return self.refusal_count >= 3

    def note_meeting_attempt(self):
        self.meeting_ask_count += 1

    async def trigger_interruption(self):
        """Called by VAD when user speaks over TTS. Immediately stops TTS and goes to LISTENING."""
        self.interrupt_tts.set()
        self.interrupt_llm.set()
        self.tts_generation += 1  # Invalidate all in-flight audio chunks
        # Go directly to LISTENING so user speech is captured immediately
        await self.set_state(AssistantState.LISTENING)

    def timed_out(self) -> bool:
        return time.time() - self.start_time > self.timeout_secs

    def turns_exceeded(self) -> bool:
        return self.turn_count >= self.MAX_TURNS

    @property
    def call_stage(self) -> str:
        """Stage derived from call-state slots (INTRO → VERIFY → REMIND → RESOLVE → CLOSE)."""
        if self.outcome:
            return "CLOSE"
        if self.reminder_delivered:
            return "RESOLVE"
        if self.identity_verified:
            return "REMIND"
        if self.turn_count <= 1:
            return "INTRO"
        return "VERIFY"

    def mark_turn(self, name: str) -> None:
        """Record a latency mark for the current turn (first write wins)."""
        if name not in self.turn_metrics:
            self.turn_metrics[name] = time.time()

    def turn_latency_ms(self) -> Dict[str, int]:
        """Return each mark as milliseconds since end of user speech."""
        start = self.turn_metrics.get("speech_end")
        if not start:
            return {}
        return {
            name: int((ts - start) * 1000)
            for name, ts in self.turn_metrics.items()
            if name != "speech_end"
        }

    def get_lead_email(self) -> Optional[str]:
        """Return email from lead scraper data, if available."""
        if self.lead_context and self.lead_context.get("email"):
            return self.lead_context["email"]
        return None

    def record_mood(self, intent_name: str):
        """Map the current intent to a mood label and append to mood_trajectory."""
        _INTENT_TO_MOOD = {
            "INTERESTED": "positive",
            "ASK_MORE_INFO": "warming",
            "ASK_AUDIT_DEFINITION": "warming",
            "ASK_AUDIT_PROCESS": "warming",
            "ASK_COMPLIANCE_BAA": "warming",
            "EMAIL_REQUEST": "positive",
            "CALL_BACK_LATER": "neutral",
            "NOT_INTERESTED": "cold",
            "ALREADY_HAVE_BILLER": "skeptical",
            "HAPPY_WITH_PROVIDER": "skeptical",
            "TOO_BUSY": "cold",
            "ALREADY_AUDITED": "skeptical",
            "NO_BUDGET": "skeptical",
            "BAD_EXPERIENCE": "frustrated",
            "TRUST_CONCERN": "skeptical",
            "PRIVACY_CONCERN": "skeptical",
            "CANNOT_PROCEED": "cold",
            "DO_NOT_CALL": "cold",
            "WRONG_NUMBER": "neutral",
            "ALREADY_PAID": "positive",
            "RESCHEDULE": "neutral",
            "DISPUTE": "frustrated",
            "HARDSHIP": "frustrated",
            "ABUSE": "frustrated",
            "DISTRESS": "frustrated",
        }
        mood = _INTENT_TO_MOOD.get(intent_name, "neutral")
        self.mood_trajectory.append(mood)
        # Keep only last 8 moods to avoid unbounded growth
        if len(self.mood_trajectory) > 8:
            self.mood_trajectory = self.mood_trajectory[-8:]
