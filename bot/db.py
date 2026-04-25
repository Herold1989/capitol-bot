from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # Singleton live account state for the contribution-aware paper strategy.
    """
    CREATE TABLE IF NOT EXISTS account_state (
        account_id INTEGER PRIMARY KEY CHECK (account_id = 1),
        cash DOUBLE PRECISION NOT NULL,
        benchmark_units DOUBLE PRECISION NOT NULL,
        benchmark_symbol TEXT NOT NULL,
        state_version INTEGER NOT NULL DEFAULT 2,
        strategy_contributed DOUBLE PRECISION NOT NULL DEFAULT 0,
        benchmark_contributed DOUBLE PRECISION NOT NULL DEFAULT 0,
        last_strategy_contribution_date DATE,
        last_benchmark_contribution_date DATE,
        last_run_date DATE,
        benchmark_start_date DATE,
        benchmark_start_price DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # Immutable summary row for each distinct paper-run day.
    """
    CREATE TABLE IF NOT EXISTS paper_runs (
        id BIGSERIAL PRIMARY KEY,
        run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        run_date DATE NOT NULL,
        command TEXT NOT NULL,
        contribution_applied BOOLEAN NOT NULL DEFAULT FALSE,
        rebalance_executed BOOLEAN NOT NULL,
        benchmark_symbol TEXT NOT NULL,
        benchmark_price DOUBLE PRECISION,
        cash DOUBLE PRECISION NOT NULL,
        market_value DOUBLE PRECISION NOT NULL,
        portfolio_value DOUBLE PRECISION NOT NULL,
        benchmark_value DOUBLE PRECISION,
        strategy_contribution_amount DOUBLE PRECISION NOT NULL DEFAULT 0,
        benchmark_contribution_amount DOUBLE PRECISION NOT NULL DEFAULT 0,
        total_contributed_strategy DOUBLE PRECISION NOT NULL DEFAULT 0,
        total_contributed_benchmark DOUBLE PRECISION NOT NULL DEFAULT 0,
        action_summary TEXT,
        action_reason TEXT,
        metrics JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    # Resolution audit log for raw disclosure symbols.
    """
    CREATE TABLE IF NOT EXISTS symbol_resolutions (
        id BIGSERIAL PRIMARY KEY,
        run_id BIGINT NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
        raw_symbol TEXT NOT NULL,
        resolved_symbol TEXT,
        provider TEXT,
        status TEXT NOT NULL,
        reason TEXT NOT NULL,
        history_rows INTEGER NOT NULL DEFAULT 0
    )
    """,
    # Target holdings emitted by the simplified daily strategy.
    """
    CREATE TABLE IF NOT EXISTS target_positions (
        id BIGSERIAL PRIMARY KEY,
        run_id BIGINT NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
        symbol TEXT NOT NULL,
        weight DOUBLE PRECISION NOT NULL,
        score DOUBLE PRECISION NOT NULL,
        sector TEXT
    )
    """,
    # Live holdings carried from one daily paper run to the next.
    """
    CREATE TABLE IF NOT EXISTS paper_positions (
        symbol TEXT PRIMARY KEY,
        quantity DOUBLE PRECISION NOT NULL,
        avg_cost DOUBLE PRECISION NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    # Executed simulated trades, with a short machine-generated reason string.
    """
    CREATE TABLE IF NOT EXISTS paper_trades (
        id BIGSERIAL PRIMARY KEY,
        run_id BIGINT NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
        traded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity DOUBLE PRECISION NOT NULL,
        price DOUBLE PRECISION NOT NULL,
        gross_notional DOUBLE PRECISION NOT NULL,
        fees DOUBLE PRECISION NOT NULL,
        target_weight DOUBLE PRECISION,
        reason TEXT
    )
    """,
    # End-of-run holdings snapshot for reporting.
    """
    CREATE TABLE IF NOT EXISTS position_snapshots (
        id BIGSERIAL PRIMARY KEY,
        run_id BIGINT NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
        symbol TEXT NOT NULL,
        asset_name TEXT,
        quantity DOUBLE PRECISION NOT NULL,
        price DOUBLE PRECISION NOT NULL,
        market_value DOUBLE PRECISION NOT NULL,
        target_weight DOUBLE PRECISION,
        score DOUBLE PRECISION,
        sector TEXT
    )
    """,
    # Explicit cash-flow ledger for initial funding and monthly contributions.
    """
    CREATE TABLE IF NOT EXISTS cash_flows (
        id BIGSERIAL PRIMARY KEY,
        run_id BIGINT NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
        flow_date DATE NOT NULL,
        flow_type TEXT NOT NULL,
        amount DOUBLE PRECISION NOT NULL,
        applies_to TEXT NOT NULL
    )
    """,
    # Decision log for buys, sells, holds, and skipped candidates.
    """
    CREATE TABLE IF NOT EXISTS decision_log (
        id BIGSERIAL PRIMARY KEY,
        run_id BIGINT NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
        symbol TEXT,
        decision TEXT NOT NULL,
        reason TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS target_diffs (
        id BIGSERIAL PRIMARY KEY,
        run_id BIGINT NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
        symbol TEXT NOT NULL,
        change_type TEXT NOT NULL,
        previous_rank INTEGER,
        current_rank INTEGER,
        previous_score DOUBLE PRECISION,
        current_score DOUBLE PRECISION,
        previous_weight DOUBLE PRECISION,
        current_weight DOUBLE PRECISION,
        reason TEXT
    )
    """,
    # Additive migrations for existing deployments created by the older bot.
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS benchmark_symbol TEXT
    """,
    """
    ALTER TABLE position_snapshots
    ADD COLUMN IF NOT EXISTS asset_name TEXT
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS state_version INTEGER NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS strategy_contributed DOUBLE PRECISION NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS benchmark_contributed DOUBLE PRECISION NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS last_strategy_contribution_date DATE
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS last_benchmark_contribution_date DATE
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS last_run_date DATE
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS benchmark_start_date DATE
    """,
    """
    ALTER TABLE account_state
    ADD COLUMN IF NOT EXISTS benchmark_start_price DOUBLE PRECISION
    """,
    """
    ALTER TABLE paper_runs
    ADD COLUMN IF NOT EXISTS contribution_applied BOOLEAN NOT NULL DEFAULT FALSE
    """,
    """
    ALTER TABLE paper_runs
    ADD COLUMN IF NOT EXISTS strategy_contribution_amount DOUBLE PRECISION NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE paper_runs
    ADD COLUMN IF NOT EXISTS benchmark_contribution_amount DOUBLE PRECISION NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE paper_runs
    ADD COLUMN IF NOT EXISTS total_contributed_strategy DOUBLE PRECISION NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE paper_runs
    ADD COLUMN IF NOT EXISTS total_contributed_benchmark DOUBLE PRECISION NOT NULL DEFAULT 0
    """,
    """
    ALTER TABLE paper_runs
    ADD COLUMN IF NOT EXISTS action_summary TEXT
    """,
    """
    ALTER TABLE paper_runs
    ADD COLUMN IF NOT EXISTS action_reason TEXT
    """,
    """
    ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS reason TEXT
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS paper_runs_run_date_command_idx
    ON paper_runs (run_date, command)
    """,
]


def connect(db_url: str) -> psycopg.Connection:
    # Use dict rows everywhere so the rest of the code can address result
    # columns by name instead of unpacking positional tuples.
    return psycopg.connect(db_url, row_factory=dict_row)


def ensure_schema(conn: psycopg.Connection) -> None:
    # Apply the schema bootstrap and additive migrations idempotently on each run.
    for index, statement in enumerate(SCHEMA_STATEMENTS, start=1):
        conn.execute(statement)
        conn.execute(
            """
            INSERT INTO schema_migrations (version)
            VALUES (%s)
            ON CONFLICT (version) DO NOTHING
            """,
            (f"bootstrap_{index:03d}",),
        )
    conn.commit()
