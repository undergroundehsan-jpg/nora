"""LLM generation failures never leave the caller in silence."""
import asyncio

import pytest

import app.services.llm as llm
from app.services import policy
from app.session import Session


class _Boom:
    class chat:
        class completions:
            @staticmethod
            async def create(**kw):
                raise RuntimeError("groq down")


async def _drain(q):
    items = []
    while True:
        item = await q.get()
        if item is None:
            return items
        items.append(item)


def _run(session, intent="UNCLEAR"):
    async def go():
        q = asyncio.Queue()
        result = await llm.get_ai_response_stream("hello", "en", session, q, primary_intent=intent)
        return result, await _drain(q)
    return asyncio.run(go())


@pytest.fixture(autouse=True)
def failing_client(monkeypatch):
    monkeypatch.setattr(llm, "client", _Boom())


def test_first_failure_asks_caller_to_repeat():
    s = Session(None, "t")
    result, spoken = _run(s)
    assert result == ""
    assert spoken == [policy.template("LLM_RETRY")]
    assert s.llm_failures == 1


def test_second_failure_stays_quiet_for_main_to_end_call():
    s = Session(None, "t")
    _run(s)
    _result, spoken = _run(s)
    assert spoken == []
    assert s.llm_failures == 2


def test_wrap_up_failure_still_says_goodbye():
    s = Session(None, "t")
    _result, spoken = _run(s, intent="WRAP_UP_TIMEOUT")
    assert spoken and "have a great day" in spoken[0].lower()
    assert s.close_after_speaking is True
