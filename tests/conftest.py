"""Safety net for the whole suite: no test may place a real phone call.

A test that exercises dialling patches `place_twilio_outbound_call` itself; this
fixture makes sure a test that forgets still cannot reach Twilio.
"""
import pytest


@pytest.fixture(autouse=True)
def never_dial_for_real(monkeypatch):
    async def refuse(payload):
        return {"error": "Outbound calls are disabled during tests", "payload": payload}, 503

    monkeypatch.setattr("app.services.twilio_bridge.place_twilio_outbound_call", refuse,
                        raising=False)
