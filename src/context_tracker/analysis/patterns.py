"""Cross-session pattern detection and trend analysis."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session as DbSession

from context_tracker.db import HookEventRecord, SessionRecord

logger = logging.getLogger(__name__)


@dataclass
class SessionPattern:
    name: str
    description: str
    evidence: str
    confidence: float  # 0-1
    actionable: str


@dataclass
class TrendAnalysis:
    metric: str  # "cost", "turns", "efficiency", "error_rate"
    direction: str  # "improving" | "stable" | "degrading"
    magnitude: float  # percentage change
    period: str  # e.g., "last 30 days"
    data_points: int
    values: list[float] = field(default_factory=list)  # for sparkline


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp string to a timezone-aware datetime, or None."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, AttributeError):
        return None


def _linear_regression(values: list[float]) -> tuple[float, float]:
    """Simple linear regression on index-based x values.

    Returns (slope, intercept).
    """
    n = len(values)
    if n < 2:
        return 0.0, values[0] if values else 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0, y_mean
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _rolling_average(values: list[float], window: int = 5) -> list[float]:
    """Compute a rolling average with the given window size."""
    if not values:
        return []
    result: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result


def _classify_direction(
    values: list[float],
    lower_is_better: bool = True,
) -> tuple[str, float]:
    """Classify trend direction and magnitude from a value series.

    Returns (direction, magnitude_pct).
    """
    if len(values) < 2:
        return "stable", 0.0
    slope, _ = _linear_regression(values)
    first_val = values[0] if values[0] != 0 else 1.0
    magnitude_pct = abs(slope * len(values) / abs(first_val)) * 100

    # Threshold: < 5% change is stable
    if magnitude_pct < 5.0:
        return "stable", magnitude_pct

    if lower_is_better:
        direction = "improving" if slope < 0 else "degrading"
    else:
        direction = "improving" if slope > 0 else "degrading"
    return direction, magnitude_pct


def analyze_patterns(db_session: DbSession, min_sessions: int = 5) -> list[SessionPattern]:
    """Identify cross-session patterns."""
    records: list[SessionRecord] = db_session.query(SessionRecord).order_by(SessionRecord.started_at).all()
    if len(records) < min_sessions:
        return []

    patterns: list[SessionPattern] = []

    # 1. Session length sweet spot
    _detect_session_length_sweet_spot(records, patterns)

    # 2. Time-of-day patterns
    _detect_time_of_day_patterns(records, patterns)

    # 3. Error rate trend
    _detect_error_rate_trend(db_session, records, patterns)

    # 4. Cost trajectory
    _detect_cost_trajectory(records, patterns)

    # 5. Tool preference shifts
    _detect_tool_preference_shifts(db_session, records, patterns)

    return patterns


def _detect_session_length_sweet_spot(
    records: list[SessionRecord],
    patterns: list[SessionPattern],
) -> None:
    """Group sessions by turn count buckets and find optimal efficiency."""
    buckets: dict[str, list[float]] = {
        "1-10": [],
        "11-20": [],
        "21-30": [],
        "30+": [],
    }
    bucket_order = ["1-10", "11-20", "21-30", "30+"]

    for rec in records:
        turns = int(rec.total_turns or 0)
        cost = float(rec.total_cost_usd or 0.0)
        if turns == 0:
            continue
        cost_per_turn = cost / turns
        if turns <= 10:
            buckets["1-10"].append(cost_per_turn)
        elif turns <= 20:
            buckets["11-20"].append(cost_per_turn)
        elif turns <= 30:
            buckets["21-30"].append(cost_per_turn)
        else:
            buckets["30+"].append(cost_per_turn)

    # Find bucket with lowest avg cost/turn (only consider buckets with data)
    bucket_avgs: dict[str, float] = {}
    for name, vals in buckets.items():
        if vals:
            bucket_avgs[name] = sum(vals) / len(vals)

    if len(bucket_avgs) < 2:
        return

    best_bucket = min(bucket_avgs, key=lambda k: bucket_avgs[k])
    worst_bucket = max(bucket_avgs, key=lambda k: bucket_avgs[k])

    if best_bucket == worst_bucket:
        return

    best_avg = bucket_avgs[best_bucket]
    worst_avg = bucket_avgs[worst_bucket]
    pct_more = ((worst_avg - best_avg) / best_avg * 100) if best_avg > 0 else 0

    if pct_more < 10:
        return

    evidence_parts = []
    for b in bucket_order:
        if b in bucket_avgs:
            evidence_parts.append(f"{b} turns: ${bucket_avgs[b]:.3f}/turn (n={len(buckets[b])})")

    patterns.append(
        SessionPattern(
            name="session_length_sweet_spot",
            description=(
                f"Your efficiency peaks at {best_bucket} turns - "
                f"sessions beyond that cost {pct_more:.0f}% more per turn"
            ),
            evidence="; ".join(evidence_parts),
            confidence=min(0.9, 0.5 + len(records) * 0.02),
            actionable=(
                f"Aim for {best_bucket} turns per session. Consider splitting longer tasks into multiple sessions."
            ),
        )
    )


def _detect_time_of_day_patterns(
    records: list[SessionRecord],
    patterns: list[SessionPattern],
) -> None:
    """Compare avg cost for morning/afternoon/evening sessions."""
    period_costs: dict[str, list[float]] = {
        "morning": [],  # 6-12
        "afternoon": [],  # 12-18
        "evening": [],  # 18-24
        "night": [],  # 0-6
    }

    for rec in records:
        dt = _parse_iso(str(rec.started_at) if rec.started_at else None)
        if dt is None:
            continue
        cost = float(rec.total_cost_usd or 0.0)
        hour = dt.hour
        if 6 <= hour < 12:
            period_costs["morning"].append(cost)
        elif 12 <= hour < 18:
            period_costs["afternoon"].append(cost)
        elif 18 <= hour < 24:
            period_costs["evening"].append(cost)
        else:
            period_costs["night"].append(cost)

    # Calculate averages for periods with enough data
    period_avgs: dict[str, float] = {}
    for period, costs in period_costs.items():
        if len(costs) >= 2:
            period_avgs[period] = sum(costs) / len(costs)

    if len(period_avgs) < 2:
        return

    best_period = min(period_avgs, key=lambda k: period_avgs[k])
    worst_period = max(period_avgs, key=lambda k: period_avgs[k])

    if best_period == worst_period:
        return

    best_avg = period_avgs[best_period]
    worst_avg = period_avgs[worst_period]
    pct_diff = ((worst_avg - best_avg) / best_avg * 100) if best_avg > 0 else 0

    if pct_diff < 20:
        return

    evidence_parts = []
    for period, avg in sorted(period_avgs.items(), key=lambda x: x[1]):
        evidence_parts.append(f"{period}: avg ${avg:.2f} (n={len(period_costs[period])})")

    patterns.append(
        SessionPattern(
            name="time_of_day",
            description=(
                f"{worst_period.capitalize()} sessions cost {pct_diff:.0f}% more than {best_period} sessions on average"
            ),
            evidence="; ".join(evidence_parts),
            confidence=min(0.8, 0.4 + sum(len(v) for v in period_costs.values()) * 0.02),
            actionable=(
                f"Your most cost-efficient sessions happen in the {best_period}. "
                f"Consider scheduling complex tasks during that time."
            ),
        )
    )


def _detect_error_rate_trend(
    db_session: DbSession,
    records: list[SessionRecord],
    patterns: list[SessionPattern],
) -> None:
    """Track tool failure rate over time."""
    error_rates: list[float] = []

    for rec in records:
        total_tool_events = (
            db_session.query(HookEventRecord)
            .filter(
                HookEventRecord.session_id == rec.session_id,
                HookEventRecord.event_type.in_(["post_tool_use", "post_tool_use_failure"]),
            )
            .count()
        )
        failure_events = (
            db_session.query(HookEventRecord)
            .filter(
                HookEventRecord.session_id == rec.session_id,
                HookEventRecord.event_type == "post_tool_use_failure",
            )
            .count()
        )
        if total_tool_events > 0:
            error_rates.append(failure_events / total_tool_events)

    if len(error_rates) < 3:
        return

    direction, magnitude = _classify_direction(error_rates, lower_is_better=True)
    if direction == "stable":
        return

    avg_rate = sum(error_rates) / len(error_rates)
    recent_avg = sum(error_rates[-3:]) / min(3, len(error_rates))

    patterns.append(
        SessionPattern(
            name="error_rate_trend",
            description=(f"Tool error rate is {direction} (recent: {recent_avg:.1%}, overall: {avg_rate:.1%})"),
            evidence=(f"Tracked across {len(error_rates)} sessions, {magnitude:.1f}% change"),
            confidence=min(0.85, 0.5 + len(error_rates) * 0.03),
            actionable=(
                "Error rate is increasing - review recent tool failures for systemic issues"
                if direction == "degrading"
                else "Error rate is decreasing - your workflow is becoming more reliable"
            ),
        )
    )


def _detect_cost_trajectory(
    records: list[SessionRecord],
    patterns: list[SessionPattern],
) -> None:
    """Simple linear regression on session costs over time."""
    costs = [float(rec.total_cost_usd or 0.0) for rec in records if float(rec.total_cost_usd or 0.0) > 0]
    if len(costs) < 3:
        return

    slope, _ = _linear_regression(costs)
    avg_cost = sum(costs) / len(costs)

    # Compute magnitude as pct change from first to predicted last value
    first_val = costs[0] if costs[0] != 0 else 1.0
    predicted_change = slope * len(costs)
    magnitude_pct = abs(predicted_change / abs(first_val)) * 100

    if magnitude_pct < 10:
        return

    direction = "increasing" if slope > 0 else "decreasing"

    patterns.append(
        SessionPattern(
            name="cost_trajectory",
            description=(
                f"Session costs are {direction} - avg ${avg_cost:.2f}, trend shows {magnitude_pct:.0f}% change"
            ),
            evidence=(f"Linear regression across {len(costs)} sessions, slope=${slope:.3f}/session"),
            confidence=min(0.85, 0.5 + len(costs) * 0.02),
            actionable=(
                "Costs are rising - consider shorter sessions or reviewing context usage"
                if slope > 0
                else "Costs are declining - your usage patterns are becoming more efficient"
            ),
        )
    )


def _detect_tool_preference_shifts(
    db_session: DbSession,
    records: list[SessionRecord],
    patterns: list[SessionPattern],
) -> None:
    """Compare tool usage distribution across recent vs older sessions."""
    if len(records) < 6:
        return

    midpoint = len(records) // 2
    older_records = records[:midpoint]
    newer_records = records[midpoint:]

    def _get_tool_counts(session_records: list[SessionRecord]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for rec in session_records:
            events = (
                db_session.query(HookEventRecord)
                .filter(
                    HookEventRecord.session_id == rec.session_id,
                    HookEventRecord.tool_name.isnot(None),
                )
                .all()
            )
            for evt in events:
                if evt.tool_name:
                    counter[str(evt.tool_name)] += 1
        return counter

    older_counts = _get_tool_counts(older_records)
    newer_counts = _get_tool_counts(newer_records)

    if not older_counts or not newer_counts:
        return

    # Normalize to proportions
    older_total = sum(older_counts.values())
    newer_total = sum(newer_counts.values())

    if older_total == 0 or newer_total == 0:
        return

    all_tools = set(older_counts.keys()) | set(newer_counts.keys())
    significant_shifts: list[tuple[str, float, float]] = []

    for tool in all_tools:
        older_pct = older_counts.get(tool, 0) / older_total
        newer_pct = newer_counts.get(tool, 0) / newer_total
        shift = newer_pct - older_pct
        if abs(shift) > 0.05:  # > 5% shift
            significant_shifts.append((tool, older_pct, newer_pct))

    if not significant_shifts:
        return

    # Sort by absolute shift magnitude
    significant_shifts.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
    top_shifts = significant_shifts[:3]

    evidence_parts = []
    for tool, old_pct, new_pct in top_shifts:
        direction = "increased" if new_pct > old_pct else "decreased"
        evidence_parts.append(f"{tool}: {direction} from {old_pct:.0%} to {new_pct:.0%}")

    patterns.append(
        SessionPattern(
            name="tool_preference_shift",
            description=(f"Tool usage patterns have shifted across your last {len(newer_records)} sessions"),
            evidence="; ".join(evidence_parts),
            confidence=min(0.75, 0.4 + len(records) * 0.02),
            actionable=(
                "Review whether these tool shifts align with your workflow goals. "
                "Heavy reliance on specific tools may indicate areas for optimization."
            ),
        )
    )


def analyze_trends(db_session: DbSession, period_days: int = 30) -> list[TrendAnalysis]:
    """Compute metric trends over a time period."""
    cutoff = datetime.now(tz=UTC) - timedelta(days=period_days)

    records: list[SessionRecord] = (
        db_session.query(SessionRecord)
        .filter(SessionRecord.started_at.isnot(None))
        .order_by(SessionRecord.started_at)
        .all()
    )

    # Filter to records within the period
    period_records: list[SessionRecord] = []
    for rec in records:
        dt = _parse_iso(str(rec.started_at) if rec.started_at else None)
        if dt and dt >= cutoff:
            period_records.append(rec)

    # Fall back to all records if fewer than 3 within the period
    if len(period_records) < 3:
        period_records = records

    if len(period_records) < 2:
        return []

    period_label = f"last {period_days} days"
    trends: list[TrendAnalysis] = []

    # Cost trend
    cost_values = [float(rec.total_cost_usd or 0.0) for rec in period_records]
    cost_rolling = _rolling_average(cost_values)
    cost_dir, cost_mag = _classify_direction(cost_rolling, lower_is_better=True)
    trends.append(
        TrendAnalysis(
            metric="cost",
            direction=cost_dir,
            magnitude=round(cost_mag, 1),
            period=period_label,
            data_points=len(cost_values),
            values=cost_rolling,
        )
    )

    # Turns trend
    turn_values = [float(rec.total_turns or 0) for rec in period_records]
    turn_rolling = _rolling_average(turn_values)
    turn_dir, turn_mag = _classify_direction(turn_rolling, lower_is_better=False)
    trends.append(
        TrendAnalysis(
            metric="turns",
            direction=turn_dir,
            magnitude=round(turn_mag, 1),
            period=period_label,
            data_points=len(turn_values),
            values=turn_rolling,
        )
    )

    # Efficiency (cost per turn) - lower is better
    efficiency_values: list[float] = []
    for rec in period_records:
        turns = int(rec.total_turns or 0)
        cost = float(rec.total_cost_usd or 0.0)
        if turns > 0:
            efficiency_values.append(cost / turns)
    if efficiency_values:
        eff_rolling = _rolling_average(efficiency_values)
        eff_dir, eff_mag = _classify_direction(eff_rolling, lower_is_better=True)
        trends.append(
            TrendAnalysis(
                metric="efficiency",
                direction=eff_dir,
                magnitude=round(eff_mag, 1),
                period=period_label,
                data_points=len(efficiency_values),
                values=eff_rolling,
            )
        )

    # Error rate trend
    error_rates: list[float] = []
    for rec in period_records:
        total_tool = (
            db_session.query(HookEventRecord)
            .filter(
                HookEventRecord.session_id == rec.session_id,
                HookEventRecord.event_type.in_(["post_tool_use", "post_tool_use_failure"]),
            )
            .count()
        )
        failures = (
            db_session.query(HookEventRecord)
            .filter(
                HookEventRecord.session_id == rec.session_id,
                HookEventRecord.event_type == "post_tool_use_failure",
            )
            .count()
        )
        if total_tool > 0:
            error_rates.append(failures / total_tool)

    if error_rates:
        err_rolling = _rolling_average(error_rates)
        err_dir, err_mag = _classify_direction(err_rolling, lower_is_better=True)
        trends.append(
            TrendAnalysis(
                metric="error_rate",
                direction=err_dir,
                magnitude=round(err_mag, 1),
                period=period_label,
                data_points=len(error_rates),
                values=err_rolling,
            )
        )

    return trends
