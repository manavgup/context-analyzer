"""SQLite persistence layer for session data."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, Float, ForeignKey, Integer, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

DEFAULT_DB_DIR = Path.home() / ".context-analyzer"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "analyzer.db"


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id = Column(Text, primary_key=True)
    project_path = Column(Text, nullable=True)
    started_at = Column(Text, nullable=True)  # ISO timestamp
    ended_at = Column(Text, nullable=True)
    model = Column(Text, nullable=True)
    total_turns = Column(Integer, default=0)
    total_api_calls = Column(Integer, default=0)
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


def get_engine(db_path: Path = DEFAULT_DB_PATH):
    """Create SQLAlchemy engine. Creates the DB directory and tables if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None, db_path: Path = DEFAULT_DB_PATH):
    """Get a sessionmaker bound to an engine."""
    if engine is None:
        engine = get_engine(db_path)
    return sessionmaker(bind=engine)
