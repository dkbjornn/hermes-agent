"""Overage detection must not depend on the lagging utilization counter.

Regression guard from a real miss: with the 5h plan bucket at 100% and Anthropic
reporting `overage-in-use: true`, the statusbar showed no badge because
`on_overage` keyed solely off `overage-utilization`, which still read 0.0. The
user was paying for extra usage with no indication — the exact failure the
readout exists to prevent.
"""

from agent.rate_limit_tracker import parse_unified_usage_headers

# Captured verbatim from a live api.anthropic.com response on a Max plan at the
# moment the 5-hour bucket filled.
PLAN_EXHAUSTED_HEADERS = {
    "anthropic-ratelimit-unified-5h-reset": "1785360000",
    "anthropic-ratelimit-unified-5h-status": "rejected",
    "anthropic-ratelimit-unified-5h-surpassed-threshold": "1.0",
    "anthropic-ratelimit-unified-5h-utilization": "1.0",
    "anthropic-ratelimit-unified-7d-reset": "1785384000",
    "anthropic-ratelimit-unified-7d-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.51",
    "anthropic-ratelimit-unified-fallback-percentage": "0.5",
    "anthropic-ratelimit-unified-overage-in-use": "true",
    "anthropic-ratelimit-unified-overage-reset": "1785542400",
    "anthropic-ratelimit-unified-overage-status": "allowed",
    "anthropic-ratelimit-unified-overage-utilization": "0.0",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
    "anthropic-ratelimit-unified-reset": "1785360000",
    "anthropic-ratelimit-unified-status": "rejected",
    "anthropic-ratelimit-unified-upgrade-paths": "overage",
}

HEALTHY_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.83",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.49",
    "anthropic-ratelimit-unified-overage-in-use": "false",
    "anthropic-ratelimit-unified-overage-utilization": "0.0",
    "anthropic-ratelimit-unified-status": "allowed",
}


class TestOverageInUseFlag:
    def test_flag_wins_when_utilization_still_zero(self):
        """THE regression: paying for extra usage with overage_utilization 0.0."""
        state = parse_unified_usage_headers(PLAN_EXHAUSTED_HEADERS)
        assert state is not None
        assert state.overage.utilization == 0.0  # the lagging counter
        assert state.overage_in_use is True      # the authoritative flag
        assert state.on_overage is True          # must report paid usage

    def test_healthy_session_is_not_on_overage(self):
        state = parse_unified_usage_headers(HEALTHY_HEADERS)
        assert state is not None
        assert state.overage_in_use is False
        assert state.on_overage is False

    def test_utilization_still_triggers_without_the_flag(self):
        # Fallback path: spend reported with no flag present.
        headers = {
            "anthropic-ratelimit-unified-5h-utilization": "1.0",
            "anthropic-ratelimit-unified-overage-utilization": "0.12",
        }
        state = parse_unified_usage_headers(headers)
        assert state is not None
        assert state.overage_in_use is False
        assert state.on_overage is True

    def test_flag_parsing_is_case_and_form_tolerant(self):
        for raw in ("true", "TRUE", "True", "1", "yes"):
            headers = dict(HEALTHY_HEADERS)
            headers["anthropic-ratelimit-unified-overage-in-use"] = raw
            assert parse_unified_usage_headers(headers).overage_in_use is True, raw
        for raw in ("false", "FALSE", "0", "no", ""):
            headers = dict(HEALTHY_HEADERS)
            headers["anthropic-ratelimit-unified-overage-in-use"] = raw
            assert parse_unified_usage_headers(headers).overage_in_use is False, raw

    def test_missing_flag_defaults_false(self):
        headers = {k: v for k, v in HEALTHY_HEADERS.items() if "overage-in-use" not in k}
        assert parse_unified_usage_headers(headers).overage_in_use is False


class TestPlanExhausted:
    def test_rejected_status_marks_plan_exhausted(self):
        state = parse_unified_usage_headers(PLAN_EXHAUSTED_HEADERS)
        assert state.plan_exhausted is True

    def test_healthy_plan_is_not_exhausted(self):
        assert parse_unified_usage_headers(HEALTHY_HEADERS).plan_exhausted is False


class TestPayloadProjection:
    def test_get_usage_surfaces_the_flag(self):
        from types import SimpleNamespace

        from run_agent import AIAgent
        from tui_gateway.server import _get_usage

        agent = AIAgent.__new__(AIAgent)
        agent._unified_usage_state = None
        agent._rate_limit_state = None
        agent.provider = "anthropic"
        agent.model = "claude-opus-5"
        agent._capture_anthropic_response_headers(
            SimpleNamespace(headers=PLAN_EXHAUSTED_HEADERS)
        )

        u = _get_usage(agent)["unified"]
        assert u["on_overage"] is True
        assert u["overage_in_use"] is True
        assert u["plan_exhausted"] is True
        assert u["five_hour_percent"] == 100.0
