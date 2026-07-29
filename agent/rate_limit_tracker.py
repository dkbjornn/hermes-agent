"""Rate limit tracking for inference API responses.

Captures x-ratelimit-* headers from provider responses and provides
formatted display for the /usage slash command.  Currently supports
the Nous Portal header format (also used by OpenRouter and OpenAI-compatible
APIs that follow the same convention).

Header schema (12 headers total):
    x-ratelimit-limit-requests          RPM cap
    x-ratelimit-limit-requests-1h       RPH cap
    x-ratelimit-limit-tokens            TPM cap
    x-ratelimit-limit-tokens-1h         TPH cap
    x-ratelimit-remaining-requests      requests left in minute window
    x-ratelimit-remaining-requests-1h   requests left in hour window
    x-ratelimit-remaining-tokens        tokens left in minute window
    x-ratelimit-remaining-tokens-1h     tokens left in hour window
    x-ratelimit-reset-requests          seconds until minute request window resets
    x-ratelimit-reset-requests-1h       seconds until hour request window resets
    x-ratelimit-reset-tokens            seconds until minute token window resets
    x-ratelimit-reset-tokens-1h         seconds until hour token window resets
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class RateLimitBucket:
    """One rate-limit window (e.g. requests per minute)."""

    limit: int = 0
    remaining: int = 0
    reset_seconds: float = 0.0
    captured_at: float = 0.0  # time.time() when this was captured

    @property
    def used(self) -> int:
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        if self.limit <= 0:
            return 0.0
        return (self.used / self.limit) * 100.0

    @property
    def remaining_seconds_now(self) -> float:
        """Estimated seconds remaining until reset, adjusted for elapsed time."""
        elapsed = time.time() - self.captured_at
        return max(0.0, self.reset_seconds - elapsed)


@dataclass
class UnifiedUsageWindow:
    """One Anthropic unified-usage window (plan bucket or overage pool).

    ``utilization`` is a 0..1 fraction as sent on the wire; ``percent`` exposes
    the 0..100 form callers display.
    """

    utilization: float = 0.0
    status: str = ""
    reset_epoch: float = 0.0

    @property
    def percent(self) -> float:
        return self.utilization * 100.0

    @property
    def seconds_until_reset(self) -> float:
        if self.reset_epoch <= 0:
            return 0.0
        return max(0.0, self.reset_epoch - time.time())


@dataclass
class UnifiedUsageState:
    """Anthropic OAuth (subscription) usage state.

    Parsed from ``anthropic-ratelimit-unified-*`` response headers, which ride
    along on every Messages response for OAuth-authenticated (Pro/Max plan)
    traffic. This is the only reliable read on whether a request billed to the
    subscription or to the metered overage pool: ``/api/oauth/usage`` is a
    separate, aggressively rate-limited endpoint.

    ``overage`` is the metered "extra usage" pool — non-zero utilization there
    means the plan bucket was bypassed and the user is paying per token.
    """

    five_hour: UnifiedUsageWindow = field(default_factory=UnifiedUsageWindow)
    seven_day: UnifiedUsageWindow = field(default_factory=UnifiedUsageWindow)
    overage: UnifiedUsageWindow = field(default_factory=UnifiedUsageWindow)
    status: str = ""
    representative_claim: str = ""
    # Anthropic's authoritative "this request drew on extra usage" flag.
    # Distinct from ``overage.utilization``, which can still read 0.0 on the
    # first requests after the plan bucket fills — the flag flips immediately,
    # the utilization number lags. Keying the UI off utilization alone silently
    # under-reports paid usage, so this is the primary signal.
    overage_in_use: bool = False
    captured_at: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def on_overage(self) -> bool:
        """True when metered extra usage is being consumed.

        ``overage_in_use`` is authoritative and leads the utilization counter;
        the utilization check is a fallback for responses that report spend
        without the flag.
        """
        return self.overage_in_use or self.overage.utilization > 0

    @property
    def plan_exhausted(self) -> bool:
        """True when a plan window is full and requests no longer bill to plan."""
        return self.status == "rejected" or self.five_hour.status == "rejected"


@dataclass
class RateLimitState:
    """Full rate-limit state parsed from response headers."""

    requests_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    requests_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    captured_at: float = 0.0  # when the headers were captured
    provider: str = ""

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        if not self.has_data:
            return float("inf")
        return time.time() - self.captured_at


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_unified_usage_headers(
    headers: Mapping[str, str],
) -> Optional[UnifiedUsageState]:
    """Parse ``anthropic-ratelimit-unified-*`` headers into a UnifiedUsageState.

    These headers appear on Anthropic Messages responses for OAuth
    (subscription) traffic and report plan-bucket utilization plus the metered
    overage pool. Returns None when no unified headers are present — i.e. on
    API-key traffic and on every non-Anthropic provider — so callers can hide
    the readout rather than render zeros.
    """
    lowered = {k.lower(): v for k, v in headers.items()}

    prefix = "anthropic-ratelimit-unified-"
    if not any(k.startswith(prefix) for k in lowered):
        return None

    def _window(tag: str) -> UnifiedUsageWindow:
        return UnifiedUsageWindow(
            utilization=_safe_float(lowered.get(f"{prefix}{tag}-utilization")),
            status=str(lowered.get(f"{prefix}{tag}-status") or ""),
            reset_epoch=_safe_float(lowered.get(f"{prefix}{tag}-reset")),
        )

    def _truthy(tag: str) -> bool:
        return str(lowered.get(f"{prefix}{tag}", "")).strip().lower() in (
            "true", "1", "yes",
        )

    return UnifiedUsageState(
        five_hour=_window("5h"),
        seven_day=_window("7d"),
        overage=_window("overage"),
        status=str(lowered.get(f"{prefix}status") or ""),
        representative_claim=str(lowered.get(f"{prefix}representative-claim") or ""),
        overage_in_use=_truthy("overage-in-use"),
        captured_at=time.time(),
    )


def parse_rate_limit_headers(
    headers: Mapping[str, str],
    provider: str = "",
) -> Optional[RateLimitState]:
    """Parse x-ratelimit-* headers into a RateLimitState.

    Returns None if no rate limit headers are present.
    """
    # Normalize to lowercase so lookups work regardless of how the server
    # capitalises headers (HTTP header names are case-insensitive per RFC 7230).
    lowered = {k.lower(): v for k, v in headers.items()}

    # Quick check: at least one rate limit header must exist
    has_any = any(k.startswith("x-ratelimit-") for k in lowered)
    if not has_any:
        return None

    now = time.time()

    def _bucket(resource: str, suffix: str = "") -> RateLimitBucket:
        # e.g. resource="requests", suffix="" -> per-minute
        #      resource="tokens", suffix="-1h" -> per-hour
        tag = f"{resource}{suffix}"
        return RateLimitBucket(
            limit=_safe_int(lowered.get(f"x-ratelimit-limit-{tag}")),
            remaining=_safe_int(lowered.get(f"x-ratelimit-remaining-{tag}")),
            reset_seconds=_safe_float(lowered.get(f"x-ratelimit-reset-{tag}")),
            captured_at=now,
        )

    return RateLimitState(
        requests_min=_bucket("requests"),
        requests_hour=_bucket("requests", "-1h"),
        tokens_min=_bucket("tokens"),
        tokens_hour=_bucket("tokens", "-1h"),
        captured_at=now,
        provider=provider,
    )


# ── Formatting ──────────────────────────────────────────────────────────


def _fmt_count(n: int) -> str:
    """Human-friendly number: 7999856 -> '8.0M', 33599 -> '33.6K', 799 -> '799'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_seconds(seconds: float) -> str:
    """Seconds -> human-friendly duration: '58s', '2m 14s', '58m 57s', '1h 2m'."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    h, remainder = divmod(s, 3600)
    m = remainder // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar: [████████░░░░░░░░░░░░] 40%."""
    filled = int(pct / 100.0 * width)
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


def _bucket_line(label: str, bucket: RateLimitBucket, label_width: int = 14) -> str:
    """Format one bucket as a single line."""
    if bucket.limit <= 0:
        return f"  {label:<{label_width}}  (no data)"

    pct = bucket.usage_pct
    used = _fmt_count(bucket.used)
    limit = _fmt_count(bucket.limit)
    remaining = _fmt_count(bucket.remaining)
    reset = _fmt_seconds(bucket.remaining_seconds_now)

    bar = _bar(pct)
    return f"  {label:<{label_width}} {bar} {pct:5.1f}%  {used}/{limit} used  ({remaining} left, resets in {reset})"


def format_rate_limit_display(state: RateLimitState) -> str:
    """Format rate limit state for terminal/chat display."""
    if not state.has_data:
        return "No rate limit data yet — make an API request first."

    age = state.age_seconds
    if age < 5:
        freshness = "just now"
    elif age < 60:
        freshness = f"{int(age)}s ago"
    else:
        freshness = f"{_fmt_seconds(age)} ago"

    provider_label = state.provider.title() if state.provider else "Provider"

    lines = [
        f"{provider_label} Rate Limits (captured {freshness}):",
        "",
        _bucket_line("Requests/min", state.requests_min),
        _bucket_line("Requests/hr", state.requests_hour),
        "",
        _bucket_line("Tokens/min", state.tokens_min),
        _bucket_line("Tokens/hr", state.tokens_hour),
    ]

    # Add warnings if any bucket is getting hot
    warnings = []
    for label, bucket in [
        ("requests/min", state.requests_min),
        ("requests/hr", state.requests_hour),
        ("tokens/min", state.tokens_min),
        ("tokens/hr", state.tokens_hour),
    ]:
        if bucket.limit > 0 and bucket.usage_pct >= 80:
            reset = _fmt_seconds(bucket.remaining_seconds_now)
            warnings.append(f"  ⚠ {label} at {bucket.usage_pct:.0f}% — resets in {reset}")

    if warnings:
        lines.append("")
        lines.extend(warnings)

    return "\n".join(lines)


def format_rate_limit_compact(state: RateLimitState) -> str:
    """One-line compact summary for status bars / gateway messages."""
    if not state.has_data:
        return "No rate limit data."

    rm = state.requests_min
    tm = state.tokens_min
    rh = state.requests_hour
    th = state.tokens_hour

    parts = []
    if rm.limit > 0:
        parts.append(f"RPM: {rm.remaining}/{rm.limit}")
    if rh.limit > 0:
        parts.append(f"RPH: {_fmt_count(rh.remaining)}/{_fmt_count(rh.limit)} (resets {_fmt_seconds(rh.remaining_seconds_now)})")
    if tm.limit > 0:
        parts.append(f"TPM: {_fmt_count(tm.remaining)}/{_fmt_count(tm.limit)}")
    if th.limit > 0:
        parts.append(f"TPH: {_fmt_count(th.remaining)}/{_fmt_count(th.limit)} (resets {_fmt_seconds(th.remaining_seconds_now)})")

    return " | ".join(parts)
