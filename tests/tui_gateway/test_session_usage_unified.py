"""session.usage must surface Anthropic unified (subscription) usage.

Invokes the REAL registered JSON-RPC handler out of ``tui_gateway.server._methods``
rather than re-implementing its body, so the test fails if the handler changes.
"""

from types import SimpleNamespace
from unittest.mock import patch

LIVE_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.27",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.42",
    "anthropic-ratelimit-unified-overage-utilization": "0.0",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
    "anthropic-ratelimit-unified-status": "allowed",
}


def _agent_with(headers):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._unified_usage_state = None
    agent._rate_limit_state = None
    agent.provider = "anthropic"
    if headers is not None:
        agent._capture_unified_usage(SimpleNamespace(headers=headers))
    return agent


def _call_session_usage(agent):
    """Dispatch the real `session.usage` handler against a stub session."""
    from tui_gateway import server

    session = {"agent": agent}
    handler = server._methods["session.usage"]

    with patch.object(server, "_sess_nowait", return_value=(session, None)), \
         patch.object(server, "_session_usage_snapshot",
                      return_value={"calls": 1, "input": 10, "output": 5, "total": 15}):
        response = handler(1, {"session_id": "s1"})

    assert "result" in response, response
    return response["result"]


class TestSessionUsageUnifiedProjection:
    def test_includes_unified_when_headers_seen(self):
        u = _call_session_usage(_agent_with(LIVE_HEADERS))["unified"]
        assert u["five_hour_percent"] == 27.0
        assert u["seven_day_percent"] == 42.0
        assert u["overage_percent"] == 0.0
        assert u["on_overage"] is False
        assert u["representative_claim"] == "five_hour"
        assert u["status"] == "allowed"

    def test_omits_unified_key_when_never_seen(self):
        # The client keys off presence to decide whether to render the item at
        # all — an all-zero block would show a misleading "0% / 0%".
        assert "unified" not in _call_session_usage(_agent_with(None))

    def test_omits_unified_for_non_anthropic_provider(self):
        agent = _agent_with({"x-ratelimit-limit-requests": "100"})
        assert "unified" not in _call_session_usage(agent)

    def test_omits_unified_when_agent_is_none(self):
        assert "unified" not in _call_session_usage(None)

    def test_on_overage_surfaces_true(self):
        headers = dict(LIVE_HEADERS)
        headers["anthropic-ratelimit-unified-overage-utilization"] = "0.13"
        u = _call_session_usage(_agent_with(headers))["unified"]
        assert u["on_overage"] is True
        assert u["overage_percent"] == 13.0

    def test_percentages_are_rounded_for_display(self):
        headers = dict(LIVE_HEADERS)
        headers["anthropic-ratelimit-unified-5h-utilization"] = "0.27777"
        u = _call_session_usage(_agent_with(headers))["unified"]
        assert u["five_hour_percent"] == 27.8

    def test_payload_is_json_serializable(self):
        import json

        json.dumps(_call_session_usage(_agent_with(LIVE_HEADERS)))

    def test_does_not_disturb_base_usage_fields(self):
        payload = _call_session_usage(_agent_with(LIVE_HEADERS))
        assert payload["calls"] == 1
        assert payload["total"] == 15

    def test_broken_agent_does_not_break_the_rpc(self):
        # Fail-open contract: a raising accessor must not take down session.usage.
        class Exploding:
            def get_unified_usage_state(self):
                raise RuntimeError("boom")

        payload = _call_session_usage(Exploding())
        assert payload["calls"] == 1
        assert "unified" not in payload
