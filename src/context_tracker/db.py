"""SQLite persistence layer for session data."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Float, ForeignKey, Integer, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

DEFAULT_DB_DIR = Path.home() / ".context-analyzer"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "analyzer.db"

# Known source agents for a session.
AGENT_CLAUDE_CODE = "claude-code"
AGENT_CODEX = "codex"


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id = Column(Text, primary_key=True)
    agent = Column(Text, nullable=False, default=AGENT_CLAUDE_CODE, server_default=AGENT_CLAUDE_CODE)
    project_path = Column(Text, nullable=True)
    started_at = Column(Text, nullable=True)  # ISO timestamp
    ended_at = Column(Text, nullable=True)
    model = Column(Text, nullable=True)
    total_turns = Column(Integer, default=0)
    total_api_calls = Column(Integer, default=0)
    total_blocks = Column(Integer, default=0)
    peak_context_tokens = Column(Integer, default=0)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cache_read = Column(Integer, default=0)
    total_cache_creation = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    health_score = Column(Float, nullable=True)
    source_mtime = Column(Float, default=0.0)  # mtime of source JSONL, for staleness check

    turns = relationship("TurnRecord", back_populates="session", cascade="all, delete-orphan")
    api_calls = relationship("ApiCallRecord", back_populates="session", cascade="all, delete-orphan")
    blocks = relationship("BlockRecord", back_populates="session", cascade="all, delete-orphan")
    hook_events = relationship("HookEventRecord", back_populates="session", cascade="all, delete-orphan")
    subagents = relationship("SubagentRecord", back_populates="session", cascade="all, delete-orphan")
    workflow_runs = relationship(
        "WorkflowRunRecord",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    tool_result_offloads = relationship(
        "ToolResultOffloadRecord",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class TurnRecord(Base):
    __tablename__ = "turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, ForeignKey("sessions.session_id"), nullable=False)
    turn_number = Column(Integer, nullable=False)
    first_api_call = Column(Integer, nullable=True)  # index into api_calls
    last_api_call = Column(Integer, nullable=True)
    prompt_preview = Column(Text, nullable=True)  # first 200 chars of user prompt

    session = relationship("SessionRecord", back_populates="turns")


class ApiCallRecord(Base):
    __tablename__ = "api_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, ForeignKey("sessions.session_id"), nullable=False)
    call_index = Column(Integer, nullable=False)  # 0-based
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cache_read = Column(Integer, default=0)
    cache_creation = Column(Integer, default=0)
    system_tokens = Column(Integer, default=0)  # estimated system prefix
    working_tokens = Column(Integer, default=0)  # working set tokens

    session = relationship("SessionRecord", back_populates="api_calls")


class BlockRecord(Base):
    __tablename__ = "blocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, ForeignKey("sessions.session_id"), nullable=False)
    block_id = Column(Text, nullable=False)  # e.g. "t5-tool_call-toolu_01ABC"
    block_type = Column(Text, nullable=False)  # system, user, assistant, tool_call, tool_result, etc.
    label = Column(Text, nullable=True)
    tokens = Column(Integer, default=0)
    enter_turn = Column(Integer, nullable=True)  # API call index when block entered
    exit_turn = Column(Integer, nullable=True)  # API call index when block exited (null = still present)
    cached = Column(Integer, default=0)  # 1 if served from prefix cache
    ref = Column(Integer, default=0)  # 1 if block is still referenced
    content_preview = Column(Text, nullable=True)  # first 500 chars for display

    session = relationship("SessionRecord", back_populates="blocks")


class HookEventRecord(Base):
    __tablename__ = "hook_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, ForeignKey("sessions.session_id"), nullable=False)
    event_type = Column(Text, nullable=False)  # post_tool_use, pre_compact, session_start, etc.
    timestamp = Column(Text, nullable=True)  # ISO timestamp from hook
    tool_name = Column(Text, nullable=True)  # for tool events
    tool_use_id = Column(Text, nullable=True)
    payload_chars = Column(Integer, default=0)  # input + output size for tool events
    error_length = Column(Integer, default=0)  # for failure events
    metadata_json = Column(Text, nullable=True)  # JSON blob for event-specific fields

    session = relationship("SessionRecord", back_populates="hook_events")


class WorkflowRunRecord(Base):
    """A multi-agent workflow run (subagents/workflows/wf_<runid>/)."""

    __tablename__ = "workflow_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wf_id = Column(Text, nullable=False, unique=True)  # e.g. "wf_e95c637f-433"
    session_id = Column(Text, ForeignKey("sessions.session_id"), nullable=False)
    name = Column(Text, nullable=True)
    started_at = Column(Text, nullable=True)
    ended_at = Column(Text, nullable=True)

    session = relationship("SessionRecord", back_populates="workflow_runs")
    subagents = relationship("SubagentRecord", back_populates="workflow")


class SubagentRecord(Base):
    __tablename__ = "subagents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, ForeignKey("sessions.session_id"), nullable=False)
    agent_id = Column(Text, nullable=False)
    agent_type = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    peak_resident = Column(Integer, default=0)
    total_cache_read = Column(Integer, default=0)
    total_api_calls = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    # Multi-agent workflow grouping. NULL for plain Task subagents.
    workflow_id = Column(Integer, ForeignKey("workflow_runs.id"), nullable=True)
    phase = Column(Text, nullable=True)  # journal "key" grouping the agent into a phase
    label = Column(Text, nullable=True)  # human-readable label (e.g. result dimension)

    session = relationship("SessionRecord", back_populates="subagents")
    workflow = relationship("WorkflowRunRecord", back_populates="subagents")
    api_calls = relationship("SubagentApiCallRecord", back_populates="subagent", cascade="all, delete-orphan")


class SubagentApiCallRecord(Base):
    """Per-API-call token data for a subagent's own context window."""

    __tablename__ = "subagent_api_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subagent_id = Column(Integer, ForeignKey("subagents.id"), nullable=False)
    session_id = Column(Text, nullable=False)  # denormalized for fast queries
    call_index = Column(Integer, nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cache_read = Column(Integer, default=0)
    cache_creation = Column(Integer, default=0)

    subagent = relationship("SubagentRecord", back_populates="api_calls")


class ToolResultOffloadRecord(Base):
    """Tool results that were offloaded to disk (too large for inline transcript)."""

    __tablename__ = "tool_result_offloads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, ForeignKey("sessions.session_id"), nullable=False)
    filename = Column(Text, nullable=False)  # e.g. "bdjqb7345.txt"
    size_bytes = Column(Integer, default=0)
    content_preview = Column(Text, nullable=True)  # first 500 chars

    session = relationship("SessionRecord", back_populates="tool_result_offloads")


def _migrate_schema(engine: Engine) -> None:
    """Apply lightweight in-place migrations for pre-existing databases.

    ``Base.metadata.create_all`` only creates missing tables; columns added
    to existing tables need an explicit ALTER TABLE.
    """
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(sessions)"))}
        if cols and "agent" not in cols:
            conn.execute(text(f"ALTER TABLE sessions ADD COLUMN agent TEXT NOT NULL DEFAULT '{AGENT_CLAUDE_CODE}'"))
            conn.commit()


def get_engine(db_path: Path = DEFAULT_DB_PATH) -> Engine:
    """Create SQLAlchemy engine. Creates the DB directory and tables if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    _migrate_schema(engine)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine: Engine | None = None, db_path: Path = DEFAULT_DB_PATH) -> sessionmaker:  # type: ignore[type-arg]
    """Get a sessionmaker bound to an engine."""
    if engine is None:
        engine = get_engine(db_path)
    return sessionmaker(bind=engine)
