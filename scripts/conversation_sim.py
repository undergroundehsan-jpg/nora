"""Play scripted conversations through the real agent and print the transcript.

Uses the live classifier, policy engine and language model, with no audio. It is
the quickest way to judge how NORA actually sounds before placing a call.

    python scripts/conversation_sim.py                 # every scenario
    python scripts/conversation_sim.py already_paid    # one scenario
    python scripts/conversation_sim.py --lead L-1004   # as a given customer
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import policy  # noqa: E402
from app.services.intent import classify_intent  # noqa: E402
from app.services.leads import load_lead  # noqa: E402
from app.services.llm import get_ai_response_stream  # noqa: E402
from app.session import SAMPLE_LEAD_CONTEXT, Session  # noqa: E402
from app.utils.language import build_greeting_variants  # noqa: E402

SCENARIOS: dict[str, list[str]] = {
    "already_paid": ["yes speaking", "what is this about?", "I already paid it last week"],
    "asks_questions": [
        "who is this?",
        "yes that's me",
        "what's it regarding?",
        "how much is it exactly?",
        "and when is it due?",
    ],
    "reschedule": ["speaking", "I can't talk right now", "call me tomorrow afternoon"],
    "dispute": ["yes", "I don't think I owe anything", "no, I want someone to check it"],
    "hardship": ["yes it's me", "I can't afford it this month", "friday after 3 would work"],
    "third_party": ["this is his wife", "he'll be back this evening"],
    "wants_human": ["yes speaking", "can I talk to a real person?", "tomorrow morning"],
    "opt_out": ["yes", "stop calling me please"],
    "abuse": ["yes", "this is bullshit", "fuck you"],
    "confused": ["hello?", "what?", "sorry, say that again"],
}


async def run_scenario(name: str, utterances: list[str], lead_id: str = "") -> None:
    lead = load_lead(lead_id) if lead_id else copy.deepcopy(SAMPLE_LEAD_CONTEXT)
    session = Session(websocket=None, session_id=f"sim-{name}")
    session.lead_context = lead
    session.lead_id = lead.get("lead_id")

    greeting = build_greeting_variants(lead.get("lead_name", ""), lead.get("practice_name", ""))[0][1]
    session.chat_history.append({"role": "assistant", "content": greeting})

    print(f"\n{'=' * 78}\n{name.upper()}   customer: {lead.get('lead_name')} "
          f"({lead.get('campaign', 'n/a')})\n{'=' * 78}")
    print(f"NORA     : {greeting}")

    for utterance in utterances:
        session.turn_count += 1
        print(f"CUSTOMER : {utterance}")

        intent = await classify_intent(utterance, turn_count=session.turn_count)
        session.record_intent(intent.primary_intent.value)
        session.record_mood(intent.primary_intent.value)
        action = policy.decide(session, utterance, intent.primary_intent.value)

        if action.kind != policy.LLM:
            label = "ENDS" if action.kind == policy.END else action.kind
            print(f"NORA     : {action.text}")
            print(f"           [{intent.primary_intent.value} → {label}"
                  f"{', ' + action.disposition if action.disposition else ''}]")
            if action.kind == policy.END:
                return
            session.chat_history += [
                {"role": "user", "content": utterance},
                {"role": "assistant", "content": action.text},
            ]
            session.last_response = action.text
            continue

        queue: asyncio.Queue = asyncio.Queue()
        stage = session.call_stage
        reply = await get_ai_response_stream(
            utterance, "en", session, queue,
            max_tokens=intent.max_tokens, max_words=intent.max_words,
            tier=intent.tier.value, primary_intent=intent.primary_intent.value,
            current_mode=session.current_mode, pitch_delivered=session.reminder_delivered,
        )
        policy.note_assistant_turn(session, stage, reply)
        session.last_response = reply
        words = len(reply.split())
        sentences = sum(reply.count(end) for end in ".!?")
        print(f"NORA     : {reply}")
        print(f"           [{intent.primary_intent.value} → LLM, stage {stage}, "
              f"{words} words, {sentences} sentence(s)]")

        if session.close_after_speaking:
            print("           [call closes after this line]")
            return


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run scripted conversations through NORA.")
    parser.add_argument("scenario", nargs="?", help="One scenario name (default: all)")
    parser.add_argument("--lead", default="", help="Lead id to use as the customer")
    args = parser.parse_args()

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            print(f"Unknown scenario '{name}'. Available: {', '.join(SCENARIOS)}")
            continue
        await run_scenario(name, SCENARIOS[name], args.lead)


if __name__ == "__main__":
    asyncio.run(main())
