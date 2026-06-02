"""Tests for the SQLite persistence layer (db.py)."""

from context_tracker.db import (
    ApiCallRecord,
    Base,
    SessionRecord,
    TurnRecord,
    get_engine,
    get_session_factory,
)


class TestCreateTables:
    def test_create_tables_without_error(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        # Verify tables exist by inspecting metadata
        assert "sessions" in Base.metadata.tables
        assert "turns" in Base.metadata.tables
        assert "api_calls" in Base.metadata.tables
        engine.dispose()

    def test_db_file_created(self, tmp_path):
        db_path = tmp_path / "subdir" / "test.db"
        engine = get_engine(db_path)
        assert db_path.exists()
        engine.dispose()


class TestInsertSession:
    def test_insert_and_read_back(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        with factory() as db:
            rec = SessionRecord(
                session_id="sess-001",
                model="claude-opus-4-6",
                total_api_calls=5,
                peak_context_tokens=100000,
                total_input_tokens=500,
                total_output_tokens=1000,
                total_cache_read=8000,
                total_cache_creation=9000,
                total_cost_usd=0.1234,
                source_mtime=1700000000.0,
            )
            db.add(rec)
            db.commit()

        with factory() as db:
            loaded = db.get(SessionRecord, "sess-001")
            assert loaded is not None
            assert loaded.model == "claude-opus-4-6"
            assert loaded.total_api_calls == 5
            assert loaded.peak_context_tokens == 100000
            assert loaded.total_input_tokens == 500
            assert loaded.total_output_tokens == 1000
            assert loaded.total_cache_read == 8000
            assert loaded.total_cache_creation == 9000
            assert loaded.total_cost_usd == 0.1234
            assert loaded.source_mtime == 1700000000.0

        engine.dispose()

    def test_nullable_fields_default_to_none(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        with factory() as db:
            rec = SessionRecord(session_id="sess-002")
            db.add(rec)
            db.commit()

        with factory() as db:
            loaded = db.get(SessionRecord, "sess-002")
            assert loaded is not None
            assert loaded.model is None
            assert loaded.project_path is None
            assert loaded.health_score is None

        engine.dispose()


class TestInsertWithApiCalls:
    def test_api_calls_relationship(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        with factory() as db:
            session = SessionRecord(session_id="sess-003", total_api_calls=2)
            db.add(session)
            db.add(
                ApiCallRecord(
                    session_id="sess-003",
                    call_index=0,
                    input_tokens=100,
                    output_tokens=200,
                    cache_read=5000,
                    cache_creation=8000,
                )
            )
            db.add(
                ApiCallRecord(
                    session_id="sess-003",
                    call_index=1,
                    input_tokens=150,
                    output_tokens=300,
                    cache_read=8000,
                    cache_creation=500,
                )
            )
            db.commit()

        with factory() as db:
            loaded = db.get(SessionRecord, "sess-003")
            assert loaded is not None
            assert len(loaded.api_calls) == 2
            calls = sorted(loaded.api_calls, key=lambda c: c.call_index)
            assert calls[0].call_index == 0
            assert calls[0].input_tokens == 100
            assert calls[1].call_index == 1
            assert calls[1].output_tokens == 300

        engine.dispose()


class TestInsertWithTurns:
    def test_turns_relationship(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        with factory() as db:
            session = SessionRecord(session_id="sess-004", total_turns=2)
            db.add(session)
            db.add(
                TurnRecord(
                    session_id="sess-004",
                    turn_number=1,
                    first_api_call=0,
                    last_api_call=2,
                    prompt_preview="Fix the bug in server.py",
                )
            )
            db.add(
                TurnRecord(
                    session_id="sess-004",
                    turn_number=2,
                    first_api_call=3,
                    last_api_call=5,
                    prompt_preview="Now add tests",
                )
            )
            db.commit()

        with factory() as db:
            loaded = db.get(SessionRecord, "sess-004")
            assert loaded is not None
            assert len(loaded.turns) == 2
            turns = sorted(loaded.turns, key=lambda t: t.turn_number)
            assert turns[0].prompt_preview == "Fix the bug in server.py"
            assert turns[1].first_api_call == 3

        engine.dispose()


class TestCascadeDelete:
    def test_delete_session_cascades_to_children(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine)

        with factory() as db:
            session = SessionRecord(session_id="sess-005")
            db.add(session)
            db.add(ApiCallRecord(session_id="sess-005", call_index=0))
            db.add(ApiCallRecord(session_id="sess-005", call_index=1))
            db.add(TurnRecord(session_id="sess-005", turn_number=1))
            db.commit()

        # Verify children exist
        with factory() as db:
            loaded = db.get(SessionRecord, "sess-005")
            assert len(loaded.api_calls) == 2
            assert len(loaded.turns) == 1

        # Delete session
        with factory() as db:
            loaded = db.get(SessionRecord, "sess-005")
            db.delete(loaded)
            db.commit()

        # Verify everything is gone
        with factory() as db:
            assert db.get(SessionRecord, "sess-005") is None
            from sqlalchemy import select

            api_calls = db.execute(
                select(ApiCallRecord).where(ApiCallRecord.session_id == "sess-005")
            ).scalars().all()
            assert len(api_calls) == 0

            turns = db.execute(
                select(TurnRecord).where(TurnRecord.session_id == "sess-005")
            ).scalars().all()
            assert len(turns) == 0

        engine.dispose()


class TestGetSessionFactory:
    def test_factory_without_engine(self, tmp_path):
        db_path = tmp_path / "test.db"
        factory = get_session_factory(db_path=db_path)
        with factory() as db:
            rec = SessionRecord(session_id="sess-006")
            db.add(rec)
            db.commit()

        with factory() as db:
            loaded = db.get(SessionRecord, "sess-006")
            assert loaded is not None

    def test_factory_with_engine(self, tmp_path):
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        factory = get_session_factory(engine=engine)
        with factory() as db:
            rec = SessionRecord(session_id="sess-007")
            db.add(rec)
            db.commit()

        with factory() as db:
            loaded = db.get(SessionRecord, "sess-007")
            assert loaded is not None

        engine.dispose()
