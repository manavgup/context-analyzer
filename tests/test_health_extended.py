"""Extended tests for analysis/health.py — build_health_signals, generate_recommendations."""

from context_tracker.analysis.config import HealthConfig
from context_tracker.analysis.health import (
    HealthSignals,
    build_health_signals,
    compute_turn_cost,
    generate_recommendations,
)
from context_tracker.analysis.models import (
    ApiCall,
    BlockType,
    ContentStore,
    ContextBlock,
    ConversationTurn,
    TurnSnapshot,
)


def _block(
    block_id: str,
    turn: int,
    block_type: BlockType = BlockType.TOOL_RESULT,
    resource: str | None = None,
    resource_type: str | None = None,
    size_chars: int = 400,
    is_pinned: bool = False,
    tool_name: str | None = None,
    is_error: bool = False,
) -> ContextBlock:
    return ContextBlock(
        block_id=block_id,
        turn_entered=turn,
        api_call_entered=0,
        epoch_entered=0,
        block_type=block_type,
        resource=resource,
        resource_type=resource_type,
        size_chars=size_chars,
        size_tokens_est=size_chars // 4,
        content_hash=f"hash-{block_id}",
        is_pinned=is_pinned,
        tool_name=tool_name,
        is_error=is_error,
    )


def _snapshot(
    turn: int,
    block_ids: list[str],
    blocks_entered: list[str] | None = None,
    input_tokens: int = 5000,
    output_tokens: int = 500,
    cache_read: int = 4000,
    cache_creation: int = 500,
    compaction: bool = False,
    epoch: int = 0,
) -> TurnSnapshot:
    return TurnSnapshot(
        turn_number=turn,
        timestamp=None,
        epoch=epoch,
        block_ids=block_ids,
        block_states=[],
        blocks_entered_ids=blocks_entered or block_ids,
        blocks_exited_ids=[],
        total_tokens_est=input_tokens + cache_read + cache_creation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        actual_context_tokens=input_tokens + cache_read + cache_creation,
        compaction_detected=compaction,
        api_call_count=1,
    )


def test_build_health_signals_basic():
    """Build health signals from a simple session."""
    b_sys = _block("sys", 1, BlockType.SYSTEM, is_pinned=True, size_chars=2000)
    b_user = _block("u1", 1, BlockType.USER_PROMPT, size_chars=200)
    b_assistant = _block("a1", 1, BlockType.ASSISTANT_TEXT, size_chars=500)
    b_tool = _block("t1", 2, BlockType.TOOL_RESULT, resource="/src/a.py", tool_name="Read", size_chars=1000)

    block_registry = {b.block_id: b for b in [b_sys, b_user, b_assistant, b_tool]}

    content_store = ContentStore()
    content_store.add("sys", "system prompt" * 50)
    content_store.add("u1", "Fix the bug")
    content_store.add("a1", "I will read /src/a.py to investigate")
    content_store.add("t1", "file content" * 100)

    turns = [
        ConversationTurn(
            turn_number=1,
            timestamp="2026-06-01T10:00:00Z",
            user_prompt_text="Fix the bug in /src/a.py",
            api_calls=[
                ApiCall(
                    api_call_index=0,
                    conversation_turn=1,
                    input_tokens=5000,
                    output_tokens=500,
                    cache_read_tokens=4000,
                    cache_creation_tokens=500,
                )
            ],
        ),
        ConversationTurn(
            turn_number=2,
            timestamp="2026-06-01T10:01:00Z",
            user_prompt_text="That looks good, continue",
            api_calls=[
                ApiCall(
                    api_call_index=1,
                    conversation_turn=2,
                    input_tokens=6000,
                    output_tokens=600,
                    cache_read_tokens=5000,
                    cache_creation_tokens=400,
                )
            ],
        ),
    ]

    snapshots = [
        _snapshot(1, ["sys", "u1", "a1"]),
        _snapshot(2, ["sys", "u1", "a1", "t1"], blocks_entered=["t1"]),
    ]

    signals = build_health_signals(
        turns=turns,
        snapshots=snapshots,
        block_registry=block_registry,
        content_store=content_store,
        model="claude-opus-4-6",
    )

    assert signals.turn_number == 2
    assert 0.0 <= signals.dead_weight_ratio <= 1.0
    assert 0.0 <= signals.context_utilization <= 1.0
    assert 0.0 <= signals.cache_efficiency <= 1.0
    assert signals.compaction_count == 0
    assert signals.cost_cumulative > 0


def test_build_health_signals_with_compaction():
    """Sessions with compaction epochs."""
    b_sys = _block("sys", 1, BlockType.SYSTEM, is_pinned=True, size_chars=2000)
    b_summary = _block("cs1", 3, BlockType.COMPACTION_SUMMARY, size_chars=500)

    block_registry = {b.block_id: b for b in [b_sys, b_summary]}
    content_store = ContentStore()
    content_store.add("sys", "system")
    content_store.add("cs1", "compaction summary")

    turns = [
        ConversationTurn(
            turn_number=1,
            timestamp=None,
            user_prompt_text="do stuff",
            api_calls=[
                ApiCall(
                    api_call_index=0,
                    conversation_turn=1,
                    input_tokens=5000,
                    output_tokens=500,
                    cache_read_tokens=4000,
                    cache_creation_tokens=500,
                )
            ],
        ),
        ConversationTurn(
            turn_number=2,
            timestamp=None,
            user_prompt_text="more stuff",
            api_calls=[
                ApiCall(
                    api_call_index=1,
                    conversation_turn=2,
                    input_tokens=6000,
                    output_tokens=600,
                    cache_read_tokens=5000,
                    cache_creation_tokens=400,
                )
            ],
        ),
        ConversationTurn(
            turn_number=3,
            timestamp=None,
            user_prompt_text="after compaction",
            api_calls=[
                ApiCall(
                    api_call_index=2,
                    conversation_turn=3,
                    input_tokens=3000,
                    output_tokens=400,
                    cache_read_tokens=2000,
                    cache_creation_tokens=300,
                )
            ],
        ),
    ]

    snapshots = [
        _snapshot(1, ["sys"]),
        _snapshot(2, ["sys"], blocks_entered=[], compaction=True, epoch=1),
        _snapshot(3, ["sys", "cs1"], blocks_entered=["cs1"], epoch=1),
    ]

    signals = build_health_signals(
        turns=turns,
        snapshots=snapshots,
        block_registry=block_registry,
        content_store=content_store,
        model="claude-opus-4-6",
    )

    assert signals.compaction_count == 1
    # Final non-compaction snapshot is turn 3
    assert signals.turn_number == 3


def test_build_health_signals_with_errors():
    """Session with tool errors — tests error_rate and error_rate_spike."""
    blocks = [
        _block("sys", 1, BlockType.SYSTEM, is_pinned=True),
    ]
    # Add tool results (some with errors) across many turns
    for i in range(1, 12):
        blocks.append(
            _block(
                f"tr{i}",
                i,
                BlockType.TOOL_RESULT,
                tool_name="Bash",
                is_error=(i >= 9),  # Errors in turns 9, 10, 11
                size_chars=200,
            )
        )
    block_registry = {b.block_id: b for b in blocks}
    content_store = ContentStore()
    for b in blocks:
        content_store.add(b.block_id, "content")

    turns = [
        ConversationTurn(
            turn_number=i,
            timestamp=None,
            user_prompt_text=f"turn {i}",
            api_calls=[
                ApiCall(
                    api_call_index=i - 1,
                    conversation_turn=i,
                    input_tokens=5000,
                    output_tokens=500,
                    cache_read_tokens=4000,
                    cache_creation_tokens=500,
                )
            ],
        )
        for i in range(1, 12)
    ]

    all_block_ids = [b.block_id for b in blocks]
    snapshots = [
        _snapshot(
            i,
            all_block_ids[: i + 1],
            blocks_entered=[all_block_ids[i]],
        )
        for i in range(1, 12)
    ]

    signals = build_health_signals(
        turns=turns,
        snapshots=snapshots,
        block_registry=block_registry,
        content_store=content_store,
        model="claude-opus-4-6",
    )

    assert signals.error_rate > 0
    # Recent turns have higher error rate than average
    assert signals.error_rate_spike >= 0


def test_build_health_signals_with_repeated_reads():
    """Session with repeated file reads."""
    blocks = [
        _block("sys", 1, BlockType.SYSTEM, is_pinned=True),
    ]
    # Read the same file 4 times
    for i in range(1, 5):
        blocks.append(
            _block(
                f"read{i}",
                i,
                BlockType.TOOL_RESULT,
                tool_name="Read",
                resource="/src/server.py",
                size_chars=2000,
            )
        )
    block_registry = {b.block_id: b for b in blocks}
    content_store = ContentStore()
    for b in blocks:
        content_store.add(b.block_id, "content")

    turns = [
        ConversationTurn(
            turn_number=i,
            timestamp=None,
            user_prompt_text=f"turn {i}",
            api_calls=[
                ApiCall(
                    api_call_index=i - 1,
                    conversation_turn=i,
                    input_tokens=5000,
                    output_tokens=500,
                    cache_read_tokens=4000,
                    cache_creation_tokens=500,
                )
            ],
        )
        for i in range(1, 5)
    ]

    all_ids = [b.block_id for b in blocks]
    snapshots = [_snapshot(i, all_ids[: i + 1], blocks_entered=[all_ids[i]]) for i in range(1, 5)]

    signals = build_health_signals(
        turns=turns,
        snapshots=snapshots,
        block_registry=block_registry,
        content_store=content_store,
        model="claude-opus-4-6",
    )

    assert len(signals.repeated_reads) > 0
    assert "/src/server.py" in signals.repeated_reads


def test_build_health_signals_with_edit_churn():
    """Session with repeated edits to the same file."""
    blocks = [
        _block("sys", 1, BlockType.SYSTEM, is_pinned=True),
    ]
    # Edit the same file 4 times in a row
    for i in range(1, 5):
        blocks.append(
            _block(
                f"edit{i}",
                i,
                BlockType.TOOL_RESULT,
                tool_name="Edit",
                resource="/src/hooks.py",
                size_chars=500,
            )
        )
    block_registry = {b.block_id: b for b in blocks}
    content_store = ContentStore()
    for b in blocks:
        content_store.add(b.block_id, "edit content")

    turns = [
        ConversationTurn(
            turn_number=i,
            timestamp=None,
            user_prompt_text=f"turn {i}",
            api_calls=[
                ApiCall(
                    api_call_index=i - 1,
                    conversation_turn=i,
                    input_tokens=3000,
                    output_tokens=300,
                    cache_read_tokens=2000,
                    cache_creation_tokens=200,
                )
            ],
        )
        for i in range(1, 5)
    ]

    all_ids = [b.block_id for b in blocks]
    snapshots = [_snapshot(i, all_ids[: i + 1], blocks_entered=[all_ids[i]]) for i in range(1, 5)]

    signals = build_health_signals(
        turns=turns,
        snapshots=snapshots,
        block_registry=block_registry,
        content_store=content_store,
        model="claude-opus-4-6",
    )

    # edit_churn should detect /src/hooks.py edited 4 times
    assert "/src/hooks.py" in signals.edit_churn


def test_build_health_signals_output_inflation():
    """Session where recent output tokens are much larger than average."""
    blocks = [_block("sys", 1, BlockType.SYSTEM, is_pinned=True)]
    block_registry = {b.block_id: b for b in blocks}
    content_store = ContentStore()
    content_store.add("sys", "system")

    turns = [
        ConversationTurn(
            turn_number=i,
            timestamp=None,
            user_prompt_text=f"turn {i}",
            api_calls=[
                ApiCall(
                    api_call_index=i - 1,
                    conversation_turn=i,
                    input_tokens=5000,
                    output_tokens=200 if i < 8 else 2000,  # Big spike in recent turns
                    cache_read_tokens=4000,
                    cache_creation_tokens=500,
                )
            ],
        )
        for i in range(1, 11)
    ]

    snapshots = [
        _snapshot(
            i,
            ["sys"],
            blocks_entered=(["sys"] if i == 1 else []),
            output_tokens=200 if i < 8 else 2000,
        )
        for i in range(1, 11)
    ]

    signals = build_health_signals(
        turns=turns,
        snapshots=snapshots,
        block_registry=block_registry,
        content_store=content_store,
        model="claude-opus-4-6",
    )

    assert signals.output_inflation > 0


def test_build_health_signals_large_model_window():
    """When peak context exceeds model window, auto-detect larger variant."""
    b_sys = _block("sys", 1, BlockType.SYSTEM, is_pinned=True, size_chars=2000)
    block_registry = {"sys": b_sys}
    content_store = ContentStore()
    content_store.add("sys", "x" * 2000)

    turns = [
        ConversationTurn(
            turn_number=1,
            timestamp=None,
            user_prompt_text="test",
            api_calls=[
                ApiCall(
                    api_call_index=0,
                    conversation_turn=1,
                    input_tokens=300000,
                    output_tokens=500,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                )
            ],
        ),
    ]

    snapshots = [
        TurnSnapshot(
            turn_number=1,
            timestamp=None,
            epoch=0,
            block_ids=["sys"],
            block_states=[],
            blocks_entered_ids=["sys"],
            blocks_exited_ids=[],
            total_tokens_est=300000,
            input_tokens=300000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            actual_context_tokens=300000,
        ),
    ]

    signals = build_health_signals(
        turns=turns,
        snapshots=snapshots,
        block_registry=block_registry,
        content_store=content_store,
        model="claude-opus-4-6",
    )

    # Context utilization should be calculated against 1M window, not 200K
    assert signals.context_utilization < 0.5  # 300K / 1M = 0.3


# --- generate_recommendations tests ---


def test_recommendations_high_dead_weight():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.55,
        context_utilization=0.4,
        cache_efficiency=0.9,
        cache_efficiency_trend=0.1,
        repeated_reads={},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    snapshots = [
        TurnSnapshot(
            turn_number=50,
            timestamp=None,
            epoch=0,
            block_ids=[],
            block_states=[],
            blocks_entered_ids=[],
            blocks_exited_ids=[],
            total_tokens_est=100000,
            input_tokens=50000,
            output_tokens=500,
            cache_read_tokens=40000,
            cache_creation_tokens=10000,
            actual_context_tokens=100000,
        ),
    ]

    recs = generate_recommendations(
        signals=signals,
        block_registry={},
        snapshots=snapshots,
        config=config,
    )

    codes = [r["code"] for r in recs]
    assert "HIGH_DEAD_WEIGHT" in codes
    dw_rec = next(r for r in recs if r["code"] == "HIGH_DEAD_WEIGHT")
    assert dw_rec["priority"] == "critical"
    assert dw_rec["tokens_recoverable"] > 0


def test_recommendations_context_near_limit():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=80,
        dead_weight_ratio=0.1,
        context_utilization=0.92,
        cache_efficiency=0.9,
        cache_efficiency_trend=0.1,
        repeated_reads={},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    recs = generate_recommendations(signals=signals, block_registry={}, snapshots=[], config=config)
    codes = [r["code"] for r in recs]
    assert "CONTEXT_NEAR_LIMIT" in codes
    cnl = next(r for r in recs if r["code"] == "CONTEXT_NEAR_LIMIT")
    assert cnl["priority"] == "critical"


def test_recommendations_context_warning():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=80,
        dead_weight_ratio=0.1,
        context_utilization=0.80,
        cache_efficiency=0.9,
        cache_efficiency_trend=0.1,
        repeated_reads={},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    recs = generate_recommendations(signals=signals, block_registry={}, snapshots=[], config=config)
    codes = [r["code"] for r in recs]
    assert "CONTEXT_NEAR_LIMIT" in codes
    cnl = next(r for r in recs if r["code"] == "CONTEXT_NEAR_LIMIT")
    assert cnl["priority"] == "warning"


def test_recommendations_repeated_reads():
    config = HealthConfig()
    # Make a block for the resource
    b = _block("tr1", 5, BlockType.TOOL_RESULT, resource="/src/server.py", tool_name="Read", size_chars=4000)
    block_registry = {"tr1": b}

    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.1,
        context_utilization=0.3,
        cache_efficiency=0.9,
        cache_efficiency_trend=0.1,
        repeated_reads={"/src/server.py": 5},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    snapshots = [
        TurnSnapshot(
            turn_number=50,
            timestamp=None,
            epoch=0,
            block_ids=["tr1"],
            block_states=[],
            blocks_entered_ids=["tr1"],
            blocks_exited_ids=[],
            total_tokens_est=10000,
            input_tokens=5000,
            output_tokens=500,
            cache_read_tokens=4000,
            cache_creation_tokens=1000,
            actual_context_tokens=10000,
        ),
    ]

    recs = generate_recommendations(signals=signals, block_registry=block_registry, snapshots=snapshots, config=config)
    codes = [r["code"] for r in recs]
    assert "REPEATED_READS" in codes
    rr = next(r for r in recs if r["code"] == "REPEATED_READS")
    assert rr["priority"] == "critical"  # 5 >= repeated_read_critical


def test_recommendations_cache_churn():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.1,
        context_utilization=0.3,
        cache_efficiency=0.5,
        cache_efficiency_trend=0.7,
        repeated_reads={},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    recs = generate_recommendations(signals=signals, block_registry={}, snapshots=[], config=config)
    codes = [r["code"] for r in recs]
    assert "CACHE_CHURN" in codes
    cc = next(r for r in recs if r["code"] == "CACHE_CHURN")
    assert cc["priority"] == "critical"


def test_recommendations_high_error_rate():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.1,
        context_utilization=0.3,
        cache_efficiency=0.9,
        cache_efficiency_trend=0.1,
        repeated_reads={},
        error_rate=0.35,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    recs = generate_recommendations(signals=signals, block_registry={}, snapshots=[], config=config)
    codes = [r["code"] for r in recs]
    assert "HIGH_ERROR_RATE" in codes
    er = next(r for r in recs if r["code"] == "HIGH_ERROR_RATE")
    assert er["priority"] == "critical"


def test_recommendations_output_inflation():
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.1,
        context_utilization=0.3,
        cache_efficiency=0.9,
        cache_efficiency_trend=0.1,
        repeated_reads={},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.9,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    recs = generate_recommendations(signals=signals, block_registry={}, snapshots=[], config=config)
    codes = [r["code"] for r in recs]
    assert "OUTPUT_INFLATION" in codes


def test_recommendations_superseded_reads():
    """Test SUPERSEDED_READS recommendation when >3 blocks are superseded."""
    config = HealthConfig()
    # Create blocks with same resource to trigger superseded detection
    blocks = []
    for i in range(5):
        blocks.append(
            _block(
                f"r{i}",
                i + 1,
                BlockType.TOOL_RESULT,
                resource="/src/server.py",
                resource_type="file",
                tool_name="Read",
                size_chars=2000,
            )
        )
    block_registry = {b.block_id: b for b in blocks}

    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.1,
        context_utilization=0.3,
        cache_efficiency=0.9,
        cache_efficiency_trend=0.1,
        repeated_reads={},
        error_rate=0.01,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    all_ids = [b.block_id for b in blocks]
    snapshots = [
        TurnSnapshot(
            turn_number=50,
            timestamp=None,
            epoch=0,
            block_ids=all_ids,
            block_states=[],
            blocks_entered_ids=all_ids,
            blocks_exited_ids=[],
            total_tokens_est=10000,
            input_tokens=5000,
            output_tokens=500,
            cache_read_tokens=4000,
            cache_creation_tokens=1000,
            actual_context_tokens=10000,
        ),
    ]

    recs = generate_recommendations(signals=signals, block_registry=block_registry, snapshots=snapshots, config=config)
    codes = [r["code"] for r in recs]
    assert "SUPERSEDED_READS" in codes


def test_recommendations_healthy_session():
    """A healthy session should have no recommendations."""
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=10,
        dead_weight_ratio=0.05,
        context_utilization=0.15,
        cache_efficiency=0.97,
        cache_efficiency_trend=0.05,
        repeated_reads={},
        error_rate=0.0,
        error_rate_spike=0.0,
        output_inflation=0.0,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.01,
        cost_cumulative=0.5,
    )

    recs = generate_recommendations(signals=signals, block_registry={}, snapshots=[], config=config)
    assert len(recs) == 0


def test_recommendations_sorted_by_priority():
    """Critical recommendations come before warnings."""
    config = HealthConfig()
    signals = HealthSignals(
        turn_number=50,
        dead_weight_ratio=0.55,
        context_utilization=0.92,
        cache_efficiency=0.5,
        cache_efficiency_trend=0.7,
        repeated_reads={},
        error_rate=0.35,
        error_rate_spike=0.0,
        output_inflation=0.9,
        edit_churn=[],
        compaction_count=0,
        cost_this_turn=0.02,
        cost_cumulative=1.0,
    )

    snapshots = [
        TurnSnapshot(
            turn_number=50,
            timestamp=None,
            epoch=0,
            block_ids=[],
            block_states=[],
            blocks_entered_ids=[],
            blocks_exited_ids=[],
            total_tokens_est=100000,
            input_tokens=50000,
            output_tokens=500,
            cache_read_tokens=40000,
            cache_creation_tokens=10000,
            actual_context_tokens=100000,
        ),
    ]

    recs = generate_recommendations(signals=signals, block_registry={}, snapshots=snapshots, config=config)
    priorities = [r["priority"] for r in recs]
    # All criticals should come before any warnings
    first_warning = next((i for i, p in enumerate(priorities) if p == "warning"), len(priorities))
    last_critical = max((i for i, p in enumerate(priorities) if p == "critical"), default=-1)
    assert last_critical < first_warning


def test_compute_turn_cost_default_model():
    """Unknown model should use _default pricing."""
    api_call = ApiCall(
        api_call_index=0,
        conversation_turn=1,
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    cost = compute_turn_cost(api_call, "unknown-model")
    assert cost > 0
