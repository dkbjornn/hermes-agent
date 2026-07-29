"""E2E: unified-usage headers captured off a real Anthropic response object."""

from types import SimpleNamespace

from agent.rate_limit_tracker import parse_unified_usage_headers

LIVE_HEADERS = {
    "anthropic-ratelimit-unified-5h-reset": "1785360000",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-utilization": "0.27",
    "anthropic-ratelimit-unified-7d-reset": "1785384000",
    "anthropic-ratelimit-unified-7d-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.42",
    "anthropic-ratelimit-unified-overage-reset": "1785542400",
    "anthropic-ratelimit-unified-overage-status": "allowed",
    "anthropic-ratelimit-unified-overage-utilization": "0.0",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
    "anthropic-ratelimit-unified-status": "allowed",
}


def _bind():
    """A real AIAgent instance with no __init__ side effects.

    The capture methods only touch a couple of attributes, so allocating the
    real class via __new__ exercises the actual code under test without
    needing credentials, config, or a live provider.
    """
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._unified_usage_state = None
    agent._rate_limit_state = None
    agent.provider = "anthropic"
    return agent


class TestUnifiedUsageCapture:
    def test_captures_from_response_headers(self):
        agent = _bind()
        agent._capture_unified_usage(SimpleNamespace(headers=LIVE_HEADERS))
        state = agent.get_unified_usage_state()
        assert state is not None
        assert state.five_hour.percent == 27.0
        assert state.seven_day.percent == 42.0
        assert state.on_overage is False

    def test_no_state_before_any_response(self):
        assert _bind().get_unified_usage_state() is None

    def test_none_response_is_noop(self):
        agent = _bind()
        agent._capture_unified_usage(None)
        assert agent.get_unified_usage_state() is None

    def test_headerless_response_is_noop(self):
        agent = _bind()
        agent._capture_unified_usage(SimpleNamespace(headers=None))
        assert agent.get_unified_usage_state() is None

    def test_non_anthropic_response_leaves_state_none(self):
        # OpenAI-wire providers send x-ratelimit-*; no unified headers means the
        # readout stays hidden rather than showing zeros.
        agent = _bind()
        agent._capture_unified_usage(
            SimpleNamespace(headers={"x-ratelimit-limit-requests": "100"})
        )
        assert agent.get_unified_usage_state() is None

    def test_miss_preserves_last_known_state(self):
        # A later response without unified headers (e.g. a provider fallback
        # mid-session) must not wipe a good reading.
        agent = _bind()
        agent._capture_unified_usage(SimpleNamespace(headers=LIVE_HEADERS))
        agent._capture_unified_usage(SimpleNamespace(headers={"content-type": "application/json"}))
        state = agent.get_unified_usage_state()
        assert state is not None
        assert state.five_hour.percent == 27.0

    def test_later_response_updates_state(self):
        agent = _bind()
        agent._capture_unified_usage(SimpleNamespace(headers=LIVE_HEADERS))
        newer = dict(LIVE_HEADERS)
        newer["anthropic-ratelimit-unified-5h-utilization"] = "0.31"
        agent._capture_unified_usage(SimpleNamespace(headers=newer))
        assert agent.get_unified_usage_state().five_hour.percent == 31.0

    def test_capture_never_raises_on_garbage(self):
        agent = _bind()
        # Fail-open contract: header parsing must never break the agent loop.
        agent._capture_unified_usage(SimpleNamespace(headers="not-a-mapping"))
        agent._capture_unified_usage(SimpleNamespace())
        agent._capture_unified_usage(object())


class TestWiredIntoAnthropicCallback:
    """The on_response callback must feed unified capture, not just rate limits."""

    def test_anthropic_callback_populates_unified_state(self):
        agent = _bind()
        agent._capture_anthropic_response_headers(SimpleNamespace(headers=LIVE_HEADERS))
        state = agent.get_unified_usage_state()
        assert state is not None
        assert state.five_hour.percent == 27.0

    def test_callback_is_fail_open(self):
        agent = _bind()
        agent._capture_anthropic_response_headers(None)
        assert agent.get_unified_usage_state() is None


def test_parser_and_capture_agree():
    """Capture must not transform the parsed values."""
    agent = _bind()
    agent._capture_unified_usage(SimpleNamespace(headers=LIVE_HEADERS))
    direct = parse_unified_usage_headers(LIVE_HEADERS)
    captured = agent.get_unified_usage_state()
    assert direct is not None and captured is not None
    assert captured.five_hour.utilization == direct.five_hour.utilization
    assert captured.overage.utilization == direct.overage.utilization
    assert captured.representative_claim == direct.representative_claim
