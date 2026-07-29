"""Tests for Anthropic unified-usage (OAuth subscription) header parsing."""

import time

from agent.rate_limit_tracker import (
    UnifiedUsageState,
    parse_rate_limit_headers,
    parse_unified_usage_headers,
)

# Captured verbatim from a live api.anthropic.com/v1/messages response on an
# OAuth (Max plan) credential.
LIVE_HEADERS = {
    "anthropic-organization-id": "8eddcea0-e06f-4f23-86db-8dcd6db0d9d3",
    "anthropic-ratelimit-unified-5h-reset": "1785360000",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-utilization": "0.27",
    "anthropic-ratelimit-unified-7d-reset": "1785384000",
    "anthropic-ratelimit-unified-7d-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.42",
    "anthropic-ratelimit-unified-fallback-percentage": "0.5",
    "anthropic-ratelimit-unified-overage-reset": "1785542400",
    "anthropic-ratelimit-unified-overage-status": "allowed",
    "anthropic-ratelimit-unified-overage-utilization": "0.0",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
    "anthropic-ratelimit-unified-reset": "1785360000",
    "anthropic-ratelimit-unified-status": "allowed",
}


class TestParseUnifiedUsageHeaders:
    def test_parses_live_headers(self):
        state = parse_unified_usage_headers(LIVE_HEADERS)
        assert state is not None
        assert state.five_hour.utilization == 0.27
        assert state.seven_day.utilization == 0.42
        assert state.overage.utilization == 0.0
        assert state.status == "allowed"
        assert state.representative_claim == "five_hour"
        assert state.has_data

    def test_percent_is_hundred_scaled(self):
        state = parse_unified_usage_headers(LIVE_HEADERS)
        assert state is not None
        # Wire sends 0..1; display wants 0..100. Guards against a double-scale
        # regression in either direction.
        assert state.five_hour.percent == 27.0
        assert state.seven_day.percent == 42.0

    def test_on_overage_false_when_plan_is_paying(self):
        state = parse_unified_usage_headers(LIVE_HEADERS)
        assert state is not None
        assert state.on_overage is False

    def test_on_overage_true_when_metered_pool_consumed(self):
        headers = dict(LIVE_HEADERS)
        headers["anthropic-ratelimit-unified-overage-utilization"] = "0.13"
        state = parse_unified_usage_headers(headers)
        assert state is not None
        assert state.on_overage is True
        assert state.overage.percent == 13.0

    def test_returns_none_without_unified_headers(self):
        # API-key Anthropic traffic and every non-Anthropic provider: callers
        # must be able to hide the readout rather than render zeros.
        assert parse_unified_usage_headers({}) is None
        assert parse_unified_usage_headers({"x-ratelimit-limit-requests": "100"}) is None

    def test_header_lookup_is_case_insensitive(self):
        # HTTP header names are case-insensitive (RFC 7230); servers and proxies
        # do not agree on casing.
        shouted = {k.upper(): v for k, v in LIVE_HEADERS.items()}
        state = parse_unified_usage_headers(shouted)
        assert state is not None
        assert state.five_hour.utilization == 0.27

    def test_malformed_values_degrade_to_zero(self):
        headers = dict(LIVE_HEADERS)
        headers["anthropic-ratelimit-unified-5h-utilization"] = "not-a-number"
        state = parse_unified_usage_headers(headers)
        assert state is not None
        assert state.five_hour.utilization == 0.0
        # A bad value in one window must not poison the others.
        assert state.seven_day.utilization == 0.42

    def test_partial_headers_do_not_raise(self):
        state = parse_unified_usage_headers(
            {"anthropic-ratelimit-unified-5h-utilization": "0.5"}
        )
        assert state is not None
        assert state.five_hour.utilization == 0.5
        assert state.seven_day.utilization == 0.0
        assert state.overage.utilization == 0.0


class TestSecondsUntilReset:
    def test_future_reset_is_positive(self):
        state = parse_unified_usage_headers(
            {"anthropic-ratelimit-unified-5h-reset": str(int(time.time()) + 600)}
        )
        assert state is not None
        assert 0 < state.five_hour.seconds_until_reset <= 600

    def test_past_reset_clamps_to_zero(self):
        state = parse_unified_usage_headers(
            {"anthropic-ratelimit-unified-5h-reset": str(int(time.time()) - 600)}
        )
        assert state is not None
        assert state.five_hour.seconds_until_reset == 0.0

    def test_missing_reset_is_zero(self):
        assert UnifiedUsageState().five_hour.seconds_until_reset == 0.0


class TestCoexistenceWithPortalSchema:
    """The two header families are independent and must not interfere."""

    def test_unified_headers_do_not_satisfy_portal_parser(self):
        # parse_rate_limit_headers keys off `x-ratelimit-*`; unified headers use
        # a different prefix and must not produce a bogus all-zero state.
        assert parse_rate_limit_headers(LIVE_HEADERS) is None

    def test_portal_headers_do_not_satisfy_unified_parser(self):
        portal = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        assert parse_unified_usage_headers(portal) is None

    def test_both_parse_when_both_present(self):
        merged = {**LIVE_HEADERS, "x-ratelimit-limit-requests": "100",
                  "x-ratelimit-remaining-requests": "99"}
        unified = parse_unified_usage_headers(merged)
        portal = parse_rate_limit_headers(merged, provider="anthropic")
        assert unified is not None and unified.five_hour.utilization == 0.27
        assert portal is not None and portal.requests_min.limit == 100
