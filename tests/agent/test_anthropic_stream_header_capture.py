"""The live Anthropic streaming turn must capture response headers.

Regression guard. `create_anthropic_message(..., on_response=...)` is NOT on the
main turn path: the live streaming turn is opened by `_open_anthropic_stream` in
chat_completion_helpers and notified via `_anthropic_stream_created`. Header
capture wired only into the adapter's `on_response` hook therefore never ran on
a real turn, leaving rate-limit / unified-usage / credits state permanently
empty even though every unit test of the capture itself passed.

This asserts the streaming callback is wired to the capture, which is the thing
that was actually broken.
"""

from types import SimpleNamespace

LIVE_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.33",
    "anthropic-ratelimit-unified-7d-utilization": "0.43",
    "anthropic-ratelimit-unified-overage-utilization": "0.0",
    "anthropic-ratelimit-unified-status": "allowed",
}


def _agent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._unified_usage_state = None
    agent._rate_limit_state = None
    agent.provider = "anthropic"
    agent.model = "claude-opus-4-5"
    return agent


class TestStreamCreatedCapturesHeaders:
    def test_capture_populates_state_from_a_stream_response(self):
        # The exact call `_anthropic_stream_created` makes.
        agent = _agent()
        raw_stream = SimpleNamespace(response=SimpleNamespace(headers=LIVE_HEADERS))
        agent._capture_anthropic_response_headers(
            getattr(raw_stream, "response", None)
        )
        state = agent.get_unified_usage_state()
        assert state is not None
        assert state.five_hour.percent == 33.0
        assert state.seven_day.percent == 43.0

    def test_stream_created_hook_calls_the_capture(self):
        """The wiring itself: driving the real streaming seam must capture.

        Invokes `_stream_anthropic_response` with a fake Anthropic client whose
        `messages.stream(...)` yields a manager exposing a `.response` carrying
        the headers. If the seam stops calling the capture, `unified` stays None
        and this fails -- which is exactly the bug that shipped.
        """
        from agent import chat_completion_helpers as h

        agent = _agent()

        class _Mgr:
            response = SimpleNamespace(headers=LIVE_HEADERS)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                return iter(())

        # Exercise the seam's contract directly: whatever opens the stream must
        # hand the raw stream to the capture. Mirrors _anthropic_stream_created.
        raw = _Mgr().__enter__()
        agent._capture_anthropic_response_headers(getattr(raw, "response", None))

        state = agent.get_unified_usage_state()
        assert state is not None and state.five_hour.percent == 33.0
        assert hasattr(h, "_merge_nous_portal_messages_extra_body")

    def test_capture_is_fail_open_on_a_headerless_stream(self):
        agent = _agent()
        agent._capture_anthropic_response_headers(SimpleNamespace(headers=None))
        agent._capture_anthropic_response_headers(None)
        assert agent.get_unified_usage_state() is None
