"""The per-turn usage payload must carry the unified subscription block.

Regression guard for the gap that shipped a non-rendering statusbar item: the
`unified` block was added to the `session.usage` RPC only, but the desktop and
TUI statusbars render from `_get_usage()` — the payload attached to streaming
turn events. `session.usage` is reached only on a legacy resume fallback, so the
item never received data and stayed permanently hidden.

Any future usage-payload consumer should be checked against _get_usage(), not
just the RPC.
"""

from types import SimpleNamespace

LIVE_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.33",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.43",
    "anthropic-ratelimit-unified-overage-utilization": "0.0",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
    "anthropic-ratelimit-unified-status": "allowed",
}


def _agent(headers=LIVE_HEADERS):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._unified_usage_state = None
    agent._rate_limit_state = None
    agent.provider = "anthropic"
    agent.model = "claude-opus-4-5"
    if headers is not None:
        agent._capture_unified_usage(SimpleNamespace(headers=headers))
    return agent


def _usage(agent):
    from tui_gateway.server import _get_usage

    return _get_usage(agent)


class TestGetUsageUnifiedBlock:
    def test_streaming_usage_payload_includes_unified(self):
        # THE regression: this is the payload the statusbar actually renders.
        u = _usage(_agent())["unified"]
        assert u["five_hour_percent"] == 33.0
        assert u["seven_day_percent"] == 43.0
        assert u["on_overage"] is False

    def test_omitted_when_no_unified_headers_seen(self):
        assert "unified" not in _usage(_agent(headers=None))

    def test_omitted_for_non_anthropic_provider(self):
        assert "unified" not in _usage(_agent({"x-ratelimit-limit-requests": "100"}))

    def test_omitted_when_agent_is_none(self):
        assert "unified" not in _usage(None)

    def test_on_overage_surfaces(self):
        headers = dict(LIVE_HEADERS)
        headers["anthropic-ratelimit-unified-overage-utilization"] = "0.09"
        u = _usage(_agent(headers))["unified"]
        assert u["on_overage"] is True
        assert u["overage_percent"] == 9.0

    def test_base_usage_fields_intact(self):
        payload = _usage(_agent())
        for key in ("input", "output", "total", "calls", "model"):
            assert key in payload, key

    def test_payload_is_json_serializable(self):
        import json

        json.dumps(_usage(_agent()))

    def test_broken_accessor_does_not_break_the_turn(self):
        # Fail-open: a raising accessor must never take down the usage payload
        # that every streaming turn depends on.
        class Exploding:
            model = "m"

            def get_unified_usage_state(self):
                raise RuntimeError("boom")

        payload = _usage(Exploding())
        assert "unified" not in payload
        assert "input" in payload


class TestBothPathsAgree:
    """_get_usage() and the session.usage RPC must project identically."""

    def test_streaming_and_rpc_blocks_match(self):
        from unittest.mock import patch as mockpatch

        from tui_gateway import server

        agent = _agent()
        streaming = _usage(agent)["unified"]

        handler = server._methods["session.usage"]
        with mockpatch.object(server, "_sess_nowait", return_value=({"agent": agent}, None)), \
             mockpatch.object(server, "_session_usage_snapshot",
                              return_value={"calls": 1, "input": 1, "output": 1, "total": 2}):
            rpc = handler(1, {"session_id": "s"})["result"]["unified"]

        assert streaming == rpc
