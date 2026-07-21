#!/usr/bin/env python3
"""Stage 0 of the Referee Play (#94): retrospective compression audit.

THIN SHIM — the audit core moved into the installable package at
``src/context_tracker/headroom_audit.py`` so that the one-command
reproduction path (``context-tracker audit-headroom``) ships in the wheel
(``experiments/`` does not). This file re-exports the full public surface so
the committed reproduction docs and existing invocations stay valid:

    python experiments/headroom/retrospective.py \
        --db experiments/headroom/scratch/corpus.db \
        --projects-dir ~/.claude/projects \
        --limit 3 --out report.md

See context_tracker.headroom_audit for the methodology documentation
(residency-weighted math, both profiles, offline hardening, read-only
guarantees).
"""

from __future__ import annotations

from context_tracker.headroom_audit import (  # noqa: F401
    RATE_CACHE_CREATION,
    RATE_CACHE_READ,
    RATE_INPUT,
    RATE_OUTPUT,
    SUSPICIOUS_RATIO,
    CorpusResult,
    GroupStat,
    SessionResult,
    _extract_compressed,
    _fmt_tok,
    _tool_result_text,
    analyze_session,
    bucket_tool,
    build_report,
    build_scratch_corpus,
    check_audit_dependencies,
    classify_content,
    find_transcripts,
    input_side_cost,
    iter_transcript_items,
    load_sessions,
    load_tool_result_blocks,
    main,
    make_headroom_compressor,
    make_tiktoken_counter,
    open_db_readonly,
    run_audit,
    run_audit_headroom,
    tool_use_id_from_block,
)

if __name__ == "__main__":
    raise SystemExit(main())
