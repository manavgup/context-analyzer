"""Team benchmarks — anonymous efficiency comparison across developers.

Exports anonymized aggregate metrics and compares against imported team data.
No identifying info (session IDs, file paths, prompts) is ever exported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from context_tracker.db import HookEventRecord, SessionRecord

MIN_TEAM_SIZE = 3  # Privacy threshold for comparisons


@dataclass
class AnonymizedMetrics:
    """Anonymized aggregate metrics for a single developer over a period."""

    period_start: str  # ISO date
    period_end: str
    alias: str  # user-chosen display name
    session_count: int
    avg_cost_per_session: float
    avg_turns_per_session: float
    avg_context_peak: int
    tool_distribution: dict[str, float]  # tool_name -> percentage of total calls
    error_rate: float  # failure_events / total_tool_events
    avg_cost_per_turn: float
    compact_frequency: float  # compactions per session


@dataclass
class TeamComparison:
    """Comparison result of your metrics against the team."""

    your_metrics: AnonymizedMetrics
    team_metrics: list[AnonymizedMetrics]
    rankings: dict[str, int]  # metric -> your rank (1 = best)
    total_members: int
    insights: list[str]


def export_metrics(
    db_session: DbSession,
    period_days: int = 30,
    alias: str = "anonymous",
) -> AnonymizedMetrics:
    """Compute and export anonymized aggregate metrics.

    Queries SessionRecord and HookEventRecord for the given period,
    then computes averages and distributions. The result contains NO
    identifying info (no session IDs, file paths, or prompts).
    """
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=period_days)
    cutoff_iso = cutoff.isoformat()

    # Query sessions within the period
    sessions = db_session.query(SessionRecord).filter(SessionRecord.started_at >= cutoff_iso).all()

    if not sessions:
        return AnonymizedMetrics(
            period_start=cutoff.date().isoformat(),
            period_end=now.date().isoformat(),
            alias=alias,
            session_count=0,
            avg_cost_per_session=0.0,
            avg_turns_per_session=0.0,
            avg_context_peak=0,
            tool_distribution={},
            error_rate=0.0,
            avg_cost_per_turn=0.0,
            compact_frequency=0.0,
        )

    session_count = len(sessions)
    session_ids = [s.session_id for s in sessions]

    total_cost = sum(s.total_cost_usd or 0.0 for s in sessions)
    total_turns = sum(s.total_turns or 0 for s in sessions)
    total_peak = sum(s.peak_context_tokens or 0 for s in sessions)

    avg_cost_per_session = total_cost / session_count
    avg_turns_per_session = total_turns / session_count
    avg_context_peak = int(total_peak / session_count)
    avg_cost_per_turn = total_cost / total_turns if total_turns > 0 else 0.0

    # Aggregate hook events for tool distribution and error rate
    hook_events = db_session.query(HookEventRecord).filter(HookEventRecord.session_id.in_(session_ids)).all()

    tool_counts: dict[str, int] = {}
    total_tool_events = 0
    failure_events = 0
    compact_events = 0

    for evt in hook_events:
        if evt.event_type == "pre_compact":
            compact_events += 1

        if evt.tool_name:
            tn = str(evt.tool_name)
            tool_counts[tn] = tool_counts.get(tn, 0) + 1
            total_tool_events += 1

        if evt.event_type in ("tool_error", "tool_failure") or (evt.error_length and evt.error_length > 0):
            failure_events += 1

    # Compute tool distribution as percentages
    tool_distribution: dict[str, float] = {}
    if total_tool_events > 0:
        for tool_name, count in tool_counts.items():
            tool_distribution[tool_name] = round(count / total_tool_events * 100, 2)

    error_rate = failure_events / total_tool_events if total_tool_events > 0 else 0.0
    compact_frequency = compact_events / session_count

    return AnonymizedMetrics(
        period_start=cutoff.date().isoformat(),
        period_end=now.date().isoformat(),
        alias=alias,
        session_count=session_count,
        avg_cost_per_session=round(avg_cost_per_session, 4),
        avg_turns_per_session=round(avg_turns_per_session, 2),
        avg_context_peak=avg_context_peak,
        tool_distribution=tool_distribution,
        error_rate=round(error_rate, 4),
        avg_cost_per_turn=round(avg_cost_per_turn, 4),
        compact_frequency=round(compact_frequency, 2),
    )


def export_metrics_to_json(metrics: AnonymizedMetrics) -> str:
    """Serialize AnonymizedMetrics to a JSON string."""
    return json.dumps(asdict(metrics), indent=2)


def import_metrics(file_path: Path) -> AnonymizedMetrics:
    """Import a team member's exported metrics from a JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return _dict_to_metrics(data)


def import_metrics_from_dict(data: dict) -> AnonymizedMetrics:
    """Import a team member's exported metrics from a dict (parsed JSON)."""
    return _dict_to_metrics(data)


def _dict_to_metrics(data: dict) -> AnonymizedMetrics:
    """Convert a dict to AnonymizedMetrics, validating required fields."""
    required_fields = {
        "period_start",
        "period_end",
        "alias",
        "session_count",
        "avg_cost_per_session",
        "avg_turns_per_session",
        "avg_context_peak",
        "tool_distribution",
        "error_rate",
        "avg_cost_per_turn",
        "compact_frequency",
    }
    missing = required_fields - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return AnonymizedMetrics(
        period_start=str(data["period_start"]),
        period_end=str(data["period_end"]),
        alias=str(data["alias"]),
        session_count=int(data["session_count"]),
        avg_cost_per_session=float(data["avg_cost_per_session"]),
        avg_turns_per_session=float(data["avg_turns_per_session"]),
        avg_context_peak=int(data["avg_context_peak"]),
        tool_distribution={str(k): float(v) for k, v in data["tool_distribution"].items()},
        error_rate=float(data["error_rate"]),
        avg_cost_per_turn=float(data["avg_cost_per_turn"]),
        compact_frequency=float(data["compact_frequency"]),
    )


def compare_with_team(
    yours: AnonymizedMetrics,
    team: list[AnonymizedMetrics],
) -> TeamComparison:
    """Compare your metrics against team and generate insights.

    Requires at least MIN_TEAM_SIZE team members for privacy.
    """
    if len(team) < MIN_TEAM_SIZE:
        raise ValueError(f"Need at least {MIN_TEAM_SIZE} team members for comparison, got {len(team)}")

    # All members including you for ranking
    all_members = [yours, *team]
    total_members = len(all_members)

    # Ranking: lower is better for cost/error, higher is better for cache hits
    # Rank definitions: rank 1 = best
    rankings: dict[str, int] = {}

    # Cost per session: lower is better
    sorted_cost = sorted(all_members, key=lambda m: m.avg_cost_per_session)
    rankings["avg_cost_per_session"] = sorted_cost.index(yours) + 1

    # Cost per turn: lower is better
    sorted_cpt = sorted(all_members, key=lambda m: m.avg_cost_per_turn)
    rankings["avg_cost_per_turn"] = sorted_cpt.index(yours) + 1

    # Error rate: lower is better
    sorted_err = sorted(all_members, key=lambda m: m.error_rate)
    rankings["error_rate"] = sorted_err.index(yours) + 1

    # Context peak: lower is better (more efficient)
    sorted_peak = sorted(all_members, key=lambda m: m.avg_context_peak)
    rankings["avg_context_peak"] = sorted_peak.index(yours) + 1

    # Turns per session: lower could mean more efficient, or less work
    sorted_turns = sorted(all_members, key=lambda m: m.avg_turns_per_session)
    rankings["avg_turns_per_session"] = sorted_turns.index(yours) + 1

    # Compact frequency: lower is better (fewer compactions needed)
    sorted_compact = sorted(all_members, key=lambda m: m.compact_frequency)
    rankings["compact_frequency"] = sorted_compact.index(yours) + 1

    # Generate insights
    insights = _generate_insights(yours, team, rankings, total_members)

    return TeamComparison(
        your_metrics=yours,
        team_metrics=team,
        rankings=rankings,
        total_members=total_members,
        insights=insights,
    )


def _generate_insights(
    yours: AnonymizedMetrics,
    team: list[AnonymizedMetrics],
    rankings: dict[str, int],
    total_members: int,
) -> list[str]:
    """Generate actionable insights from comparison data."""
    insights: list[str] = []

    # Team averages
    team_avg_cost = sum(m.avg_cost_per_session for m in team) / len(team)
    team_avg_error = sum(m.error_rate for m in team) / len(team)
    team_avg_peak = sum(m.avg_context_peak for m in team) / len(team)
    team_avg_compact = sum(m.compact_frequency for m in team) / len(team)

    # Cost insight
    if team_avg_cost > 0:
        cost_diff_pct = ((yours.avg_cost_per_session - team_avg_cost) / team_avg_cost) * 100
        if abs(cost_diff_pct) >= 5:
            direction = "more" if cost_diff_pct > 0 else "less"
            insights.append(
                f"Your sessions cost {abs(cost_diff_pct):.0f}% {direction} than team average "
                f"(${yours.avg_cost_per_session:.2f} vs ${team_avg_cost:.2f})"
            )

    # Error rate insight
    if team_avg_error > 0:
        insights.append(
            f"Your error rate is {yours.error_rate * 100:.1f}% vs team average of {team_avg_error * 100:.1f}%"
        )
    elif yours.error_rate > 0:
        insights.append(f"Your error rate is {yours.error_rate * 100:.1f}% while team average is 0%")

    # Context peak insight
    if team_avg_peak > 0:
        peak_diff_pct = ((yours.avg_context_peak - team_avg_peak) / team_avg_peak) * 100
        if abs(peak_diff_pct) >= 10:
            direction = "higher" if peak_diff_pct > 0 else "lower"
            insights.append(
                f"Your average context peak is {abs(peak_diff_pct):.0f}% {direction} than team "
                f"({yours.avg_context_peak:,} vs {int(team_avg_peak):,} tokens)"
            )

    # Compact frequency insight
    if yours.compact_frequency > team_avg_compact + 0.5:
        insights.append(
            f"You trigger {yours.compact_frequency:.1f} compactions per session vs team average of "
            f"{team_avg_compact:.1f} -- consider breaking tasks into smaller sessions"
        )

    # Tool distribution insights
    team_tool_agg: dict[str, float] = {}
    for m in team:
        for tool, pct in m.tool_distribution.items():
            team_tool_agg[tool] = team_tool_agg.get(tool, 0.0) + pct
    team_tool_avg: dict[str, float] = {t: v / len(team) for t, v in team_tool_agg.items()}

    for tool, your_pct in yours.tool_distribution.items():
        team_pct = team_tool_avg.get(tool, 0.0)
        diff = your_pct - team_pct
        if abs(diff) >= 15:
            direction = "more" if diff > 0 else "less"
            suggestion = ""
            if tool == "Bash" and diff > 0:
                suggestion = " -- consider using Read/Write tools for file operations"
            elif tool == "Read" and diff > 0:
                suggestion = " -- look for opportunities to batch file reads"
            elif tool == "Edit" and diff < 0:
                suggestion = " -- targeted edits can be more efficient than full rewrites"
            insights.append(f"You use {tool} {abs(diff):.0f}% {direction} than teammates{suggestion}")

    # Ranking insights
    for metric, rank in rankings.items():
        percentile = (total_members - rank) / (total_members - 1) * 100 if total_members > 1 else 100
        if percentile >= 75:
            label = metric.replace("_", " ").replace("avg ", "").title()
            insights.append(f"Top {100 - percentile:.0f}% in {label}")

    return insights
