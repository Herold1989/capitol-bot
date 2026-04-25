from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import psycopg
from psycopg import errors
from psycopg.types.json import Jsonb

from bot.config import AppConfig
from bot.db import ensure_schema
from bot.strategy import CapitolStrategy

CURRENT_ACCOUNT_STATE_VERSION = 2


@dataclass(slots=True)
class ContributionEvent:
    # Explicit ledger row for initial funding and monthly contribution events.
    flow_date: date
    flow_type: str
    amount: float
    applies_to: str


@dataclass(slots=True)
class DecisionRecord:
    # Human-readable explanation of why the bot bought, sold, held, or skipped.
    symbol: str | None
    decision: str
    reason: str


@dataclass(slots=True)
class ExecutedTrade:
    # One simulated fill written to paper_trades.
    symbol: str
    side: str
    quantity: float
    price: float
    gross_notional: float
    fees: float
    target_weight: float | None
    reason: str


@dataclass(slots=True)
class PlannedOrder:
    # Pre-execution order plan used to turn targets into simulated fills.
    symbol: str
    side: str
    quantity: float
    exec_price: float
    gross_notional: float
    fees: float
    target_weight: float | None
    reason: str


@dataclass(slots=True)
class AccountStateSnapshot:
    # In-memory representation of the singleton account_state row.
    cash: float
    benchmark_units: float
    benchmark_symbol: str
    state_version: int
    strategy_contributed: float
    benchmark_contributed: float
    last_strategy_contribution_date: date | None
    last_benchmark_contribution_date: date | None
    last_run_date: date | None
    benchmark_start_date: date
    benchmark_start_price: float


@dataclass(slots=True)
class PaperRunResult:
    # Summary returned to the CLI after the daily paper run finishes or is reused.
    run_id: int
    run_date: date
    already_ran_today: bool
    contribution_applied: bool
    strategy_contribution_amount: float
    benchmark_contribution_amount: float
    total_contributed_strategy: float
    total_contributed_benchmark: float
    rebalance_executed: bool
    cash: float
    market_value: float
    portfolio_value: float
    benchmark_price: float
    benchmark_value: float
    trades: list[ExecutedTrade]
    decisions: list[DecisionRecord]
    targets: pd.DataFrame
    signal_frame: pd.DataFrame
    action_summary: str
    action_reason: str
    target_diff: pd.DataFrame


def should_skip_duplicate_run(run_date: date, last_run_date: date | None) -> bool:
    # The VPS scheduler should be able to invoke the command repeatedly without
    # creating duplicate same-day contributions or trades.
    return last_run_date == run_date


def is_monthly_contribution_due(run_date: date, last_contribution_date: date | None, contribution_day: int) -> bool:
    # The first eligible run on or after contribution_day receives the monthly
    # contribution, and later runs in that same month do nothing.
    if run_date.day < contribution_day:
        return False
    if last_contribution_date is None:
        return True
    return (last_contribution_date.year, last_contribution_date.month) != (run_date.year, run_date.month)


def build_target_diff(previous_targets: pd.DataFrame, current_targets: pd.DataFrame, signal_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "change_type",
        "previous_rank",
        "current_rank",
        "previous_score",
        "current_score",
        "previous_weight",
        "current_weight",
        "reason",
    ]
    if previous_targets.empty and current_targets.empty:
        return pd.DataFrame(columns=columns)
    previous = previous_targets.copy()
    current = current_targets.copy()
    if "rank" not in previous.columns and not previous.empty:
        previous["rank"] = range(1, len(previous) + 1)
    if "rank" not in current.columns and not current.empty:
        current = current.sort_values(["score", "weight"], ascending=[False, False]).reset_index(drop=True)
        current["rank"] = range(1, len(current) + 1)
    signal_lookup = signal_frame.set_index("symbol").to_dict("index") if not signal_frame.empty else {}
    previous_lookup = previous.set_index("symbol").to_dict("index") if not previous.empty else {}
    current_lookup = current.set_index("symbol").to_dict("index") if not current.empty else {}
    strongest_new = max(
        (float(row.get("score") or 0.0) for symbol, row in current_lookup.items() if symbol not in previous_lookup),
        default=None,
    )
    rows: list[dict[str, object]] = []
    for symbol in sorted(set(previous_lookup) | set(current_lookup)):
        old = previous_lookup.get(symbol)
        new = current_lookup.get(symbol)
        if old and new:
            change_type = "retained"
            reason = "target retained"
        elif new:
            change_type = "new"
            reason = str(new.get("reason") or "new target entered active set")
        else:
            change_type = "removed"
            signal_reason = str(signal_lookup.get(symbol, {}).get("sell_reason") or "")
            reason = signal_reason or "removed from active target set"
            if strongest_new is not None:
                reason = f"{reason}; displaced by candidate score {strongest_new:.2f}"
        rows.append(
            {
                "symbol": symbol,
                "change_type": change_type,
                "previous_rank": old.get("rank") if old else None,
                "current_rank": new.get("rank") if new else None,
                "previous_score": old.get("score") if old else None,
                "current_score": new.get("score") if new else None,
                "previous_weight": old.get("weight") if old else None,
                "current_weight": new.get("weight") if new else None,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns)


class PaperTrader:
    def __init__(self, conn: psycopg.Connection, config: AppConfig, apply_schema: bool = True) -> None:
        self.conn = conn
        self.config = config
        # Ensure tables exist before any account or ledger state is touched.
        if apply_schema:
            ensure_schema(self.conn)

    def run(
        self,
        disclosures: pd.DataFrame,
        prices: pd.DataFrame,
        symbol_resolution: pd.DataFrame,
        strategy: CapitolStrategy,
        benchmark_symbol: str,
        command: str,
        run_metadata: dict[str, object] | None = None,
        dry_run: bool = False,
    ) -> PaperRunResult:
        # The paper-run operates on the latest available market date in the
        # fetched price history because that is the close the VPS run can audit.
        if prices.empty:
            raise ValueError("Price history is required for paper-run")
        normalized_prices = prices.copy()
        normalized_prices["date"] = pd.to_datetime(normalized_prices["date"]).dt.normalize()
        benchmark_history = (
            normalized_prices.loc[normalized_prices["symbol"] == benchmark_symbol, ["date", "close"]]
            .sort_values("date")
            .dropna(subset=["date", "close"])
            .copy()
        )
        latest_prices = (
            normalized_prices.sort_values("date")
            .groupby("symbol")
            .tail(1)
            .set_index("symbol")["close"]
            .astype(float)
        )
        if benchmark_symbol not in latest_prices.index:
            raise ValueError(f"Benchmark price unavailable for {benchmark_symbol}")

        run_date = pd.Timestamp(normalized_prices["date"].max()).date()
        benchmark_price = float(latest_prices[benchmark_symbol])
        benchmark_start_date = pd.Timestamp(self.config.strategy.benchmark_start_date).date()
        benchmark_start_row = benchmark_history[benchmark_history["date"] >= pd.Timestamp(benchmark_start_date)].head(1)
        if benchmark_start_row.empty:
            raise ValueError(f"Benchmark history unavailable on or after configured benchmark_start_date {benchmark_start_date}")
        benchmark_start_trade_date = benchmark_start_row.iloc[0]["date"].date()
        benchmark_start_price = float(benchmark_start_row.iloc[0]["close"])

        portfolio_start_date = pd.Timestamp(self.config.strategy.portfolio_start_date).date()
        previous_run_day_ts = normalized_prices.loc[normalized_prices["date"] < pd.Timestamp(run_date), "date"].max()
        previous_run_day = previous_run_day_ts.date() if pd.notna(previous_run_day_ts) else None

        account_state, had_existing_state, reset_live_positions = self._load_or_initialize_account_state(
            run_date,
            benchmark_symbol,
            benchmark_start_trade_date,
            benchmark_start_price,
        )
        if not dry_run and should_skip_duplicate_run(run_date, account_state.last_run_date):
            existing = self._load_existing_run_result(run_date, command)
            if existing is not None:
                return existing

        if reset_live_positions and not dry_run:
            # A legacy allocator account should not carry its old live holdings
            # into the simplified contribution-aware strategy.
            self.conn.execute("DELETE FROM paper_positions")
        if (not had_existing_state or reset_live_positions) and previous_run_day is not None and previous_run_day >= portfolio_start_date:
            seeded_state = strategy.simulate_walk_forward(
                disclosures=disclosures,
                prices=normalized_prices,
                benchmark_symbol=benchmark_symbol,
                end_date=previous_run_day,
            )
            account_state = AccountStateSnapshot(
                cash=float(seeded_state.cash),
                benchmark_units=float(seeded_state.benchmark_units),
                benchmark_symbol=benchmark_symbol,
                state_version=CURRENT_ACCOUNT_STATE_VERSION,
                strategy_contributed=float(seeded_state.total_contributed_strategy),
                benchmark_contributed=float(seeded_state.total_contributed_benchmark),
                last_strategy_contribution_date=seeded_state.last_strategy_contribution_date,
                last_benchmark_contribution_date=seeded_state.last_benchmark_contribution_date,
                last_run_date=previous_run_day,
                benchmark_start_date=benchmark_start_trade_date,
                benchmark_start_price=benchmark_start_price,
            )
            positions = {
                symbol: {
                    "quantity": float(state["quantity"]),
                    "avg_cost": float(state["avg_cost"]),
                }
                for symbol, state in seeded_state.positions.items()
            }
        else:
            positions = self._load_positions()
        signal_frame = strategy.build_signal_frame(disclosures, normalized_prices, as_of_date=run_date)
        targets = strategy.build_targets(signal_frame, positions, account_state.cash, as_of_date=run_date)
        previous_targets = self._load_previous_targets(run_date, command)
        target_diff = build_target_diff(previous_targets, targets, signal_frame)
        contribution_events: list[ContributionEvent] = []
        if account_state.last_run_date is None and account_state.strategy_contributed == float(self.config.execution.initial_cash):
            contribution_events.extend(
                [
                    ContributionEvent(
                        flow_date=run_date,
                        flow_type="initial_funding",
                        amount=float(self.config.execution.initial_cash),
                        applies_to="strategy",
                    ),
                    ContributionEvent(
                        flow_date=account_state.benchmark_start_date,
                        flow_type="initial_funding",
                        amount=float(self.config.execution.initial_cash),
                        applies_to="benchmark",
                    ),
                ]
            )
        contribution_due = is_monthly_contribution_due(
            run_date,
            account_state.last_strategy_contribution_date,
            self.config.execution.contribution_day,
        )
        if contribution_due:
            strategy_contribution = self.apply_monthly_contribution_if_due(run_date, account_state, contribution_events)
            benchmark_contribution = self.apply_benchmark_contribution_if_due(
                run_date,
                benchmark_price,
                account_state,
                contribution_events,
            )
        else:
            strategy_contribution = 0.0
            benchmark_contribution = 0.0

        held_symbols = {symbol for symbol, state in positions.items() if float(state.get("quantity", 0.0)) > 1e-8}
        target_symbols = set(targets["symbol"].tolist()) if not targets.empty else set()
        tiny_cleanup_symbols = {
            symbol
            for symbol, state in positions.items()
            if symbol in latest_prices.index and (float(state["quantity"]) * float(latest_prices[symbol])) < self.config.execution.min_trade_dollars
        }

        # Daily checks do not imply daily trades. Rebalance only when the active
        # book changes, new contribution cash arrives, or a tiny orphan can be cleaned up.
        should_rebalance = bool(strategy_contribution > 0 and target_symbols)
        should_rebalance = should_rebalance or held_symbols != target_symbols
        should_rebalance = should_rebalance or bool(tiny_cleanup_symbols - target_symbols)
        if held_symbols == target_symbols and not tiny_cleanup_symbols:
            max_drift = self._max_weight_drift(positions, targets, latest_prices, account_state.cash)
            should_rebalance = should_rebalance or max_drift >= float(self.config.strategy.min_rebalance_drift)

        planned_orders: list[PlannedOrder] = []
        trades: list[ExecutedTrade] = []
        if should_rebalance:
            planned_orders = self.generate_rebalance_orders(
                current_positions=positions,
                targets=targets,
                latest_prices=latest_prices,
                available_cash=account_state.cash,
                signal_frame=signal_frame,
            )
            trades, account_state.cash = self._apply_orders(planned_orders, positions, account_state.cash)

        market_value = self._market_value(positions, latest_prices)
        portfolio_value = account_state.cash + market_value
        benchmark_value = account_state.benchmark_units * benchmark_price

        decisions = self._build_decision_log(
            signal_frame=signal_frame,
            targets=targets,
            current_positions=positions,
            planned_orders=planned_orders,
            trades=trades,
            strategy_contribution_amount=strategy_contribution,
        )
        action_summary, action_reason = self._summarize_actions(trades, decisions, strategy_contribution)

        metrics = {
            "total_contributions_strategy": float(account_state.strategy_contributed),
            "total_contributions_benchmark": float(account_state.benchmark_contributed),
            "net_profit_strategy": float(portfolio_value - account_state.strategy_contributed),
            "net_profit_benchmark": float(benchmark_value - account_state.benchmark_contributed),
            "relative_performance_vs_benchmark": float((portfolio_value / benchmark_value) - 1.0) if benchmark_value > 0 else 0.0,
            "trade_count": float(len(trades)),
        }
        if run_metadata:
            metrics["routine"] = run_metadata
        metrics["dry_run"] = dry_run

        if dry_run:
            return PaperRunResult(
                run_id=0,
                run_date=run_date,
                already_ran_today=False,
                contribution_applied=contribution_applied,
                strategy_contribution_amount=strategy_contribution,
                benchmark_contribution_amount=benchmark_contribution,
                total_contributed_strategy=account_state.strategy_contributed,
                total_contributed_benchmark=account_state.benchmark_contributed,
                rebalance_executed=bool(trades),
                cash=account_state.cash,
                market_value=market_value,
                portfolio_value=portfolio_value,
                benchmark_price=benchmark_price,
                benchmark_value=benchmark_value,
                trades=trades,
                decisions=decisions,
                targets=targets,
                signal_frame=signal_frame,
                action_summary=f"DRY RUN: {action_summary}",
                action_reason=action_reason,
                target_diff=target_diff,
            )

        contribution_applied = bool(strategy_contribution > 0 or benchmark_contribution > 0)
        run_row = self.conn.execute(
            """
            INSERT INTO paper_runs (
                run_date, command, contribution_applied, rebalance_executed, benchmark_symbol, benchmark_price,
                cash, market_value, portfolio_value, benchmark_value,
                strategy_contribution_amount, benchmark_contribution_amount,
                total_contributed_strategy, total_contributed_benchmark,
                action_summary, action_reason, metrics
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_date, command) DO NOTHING
            RETURNING id
            """,
            (
                run_date,
                command,
                contribution_applied,
                bool(trades),
                benchmark_symbol,
                benchmark_price,
                account_state.cash,
                market_value,
                portfolio_value,
                benchmark_value,
                strategy_contribution,
                benchmark_contribution,
                account_state.strategy_contributed,
                account_state.benchmark_contributed,
                action_summary,
                action_reason,
                Jsonb(metrics),
            ),
        ).fetchone()
        if run_row is None:
            existing = self._load_existing_run_result(run_date, command)
            if existing is not None:
                return existing
            raise RuntimeError(f"paper run for {run_date} and {command} already exists but could not be loaded")
        run_id = int(run_row["id"])

        for event in contribution_events:
            self.conn.execute(
                """
                INSERT INTO cash_flows (run_id, flow_date, flow_type, amount, applies_to)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, event.flow_date, event.flow_type, event.amount, event.applies_to),
            )

        if not symbol_resolution.empty:
            # Symbol resolution outcomes remain useful for auditing skipped Capitol symbols.
            for row in symbol_resolution.to_dict("records"):
                self.conn.execute(
                    """
                    INSERT INTO symbol_resolutions (
                        run_id, raw_symbol, resolved_symbol, provider, status, reason, history_rows
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        row.get("raw_symbol"),
                        row.get("resolved_symbol"),
                        row.get("provider"),
                        row.get("status"),
                        row.get("reason"),
                        int(row.get("history_rows") or 0),
                    ),
                )

        if not targets.empty:
            for row in targets.to_dict("records"):
                self.conn.execute(
                    """
                    INSERT INTO target_positions (run_id, symbol, weight, score, sector)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (run_id, row["symbol"], row["weight"], row["score"], None),
                )

        if not target_diff.empty:
            for row in target_diff.to_dict("records"):
                self.conn.execute(
                    """
                    INSERT INTO target_diffs (
                        run_id, symbol, change_type, previous_rank, current_rank,
                        previous_score, current_score, previous_weight, current_weight, reason
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        row["symbol"],
                        row["change_type"],
                        row.get("previous_rank"),
                        row.get("current_rank"),
                        row.get("previous_score"),
                        row.get("current_score"),
                        row.get("previous_weight"),
                        row.get("current_weight"),
                        row.get("reason"),
                    ),
                )

        for trade in trades:
            self.conn.execute(
                """
                INSERT INTO paper_trades (
                    run_id, symbol, side, quantity, price, gross_notional, fees, target_weight, reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.gross_notional,
                    trade.fees,
                    trade.target_weight,
                    trade.reason,
                ),
            )

        for decision in decisions:
            self.conn.execute(
                """
                INSERT INTO decision_log (run_id, symbol, decision, reason)
                VALUES (%s, %s, %s, %s)
                """,
                (run_id, decision.symbol, decision.decision, decision.reason),
            )

        target_lookup = targets.set_index("symbol").to_dict("index") if not targets.empty else {}
        signal_lookup = signal_frame.set_index("symbol").to_dict("index") if not signal_frame.empty else {}
        for symbol, state in positions.items():
            if symbol not in latest_prices.index:
                continue
            target_row = target_lookup.get(symbol, {})
            signal_row = signal_lookup.get(symbol, {})
            price = float(latest_prices[symbol])
            quantity = float(state["quantity"])
            self.conn.execute(
                """
                INSERT INTO position_snapshots (
                    run_id, symbol, asset_name, quantity, price, market_value, target_weight, score, sector
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    symbol,
                    signal_row.get("asset_name"),
                    quantity,
                    price,
                    quantity * price,
                    target_row.get("weight"),
                    target_row.get("score", signal_row.get("score")),
                    None,
                ),
            )

        # Replace the live holdings table with the post-run state.
        self.conn.execute("DELETE FROM paper_positions")
        for symbol, state in positions.items():
            if float(state["quantity"]) <= 1e-8:
                continue
            self.conn.execute(
                """
                INSERT INTO paper_positions (symbol, quantity, avg_cost)
                VALUES (%s, %s, %s)
                """,
                (symbol, float(state["quantity"]), float(state["avg_cost"])),
            )

        account_state.last_run_date = run_date
        self.conn.execute(
            """
            INSERT INTO account_state (
                account_id, cash, benchmark_units, benchmark_symbol,
                state_version,
                strategy_contributed, benchmark_contributed,
                last_strategy_contribution_date, last_benchmark_contribution_date,
                last_run_date, benchmark_start_date, benchmark_start_price
            )
            VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (account_id)
            DO UPDATE SET
                cash = EXCLUDED.cash,
                benchmark_units = EXCLUDED.benchmark_units,
                benchmark_symbol = EXCLUDED.benchmark_symbol,
                state_version = EXCLUDED.state_version,
                strategy_contributed = EXCLUDED.strategy_contributed,
                benchmark_contributed = EXCLUDED.benchmark_contributed,
                last_strategy_contribution_date = EXCLUDED.last_strategy_contribution_date,
                last_benchmark_contribution_date = EXCLUDED.last_benchmark_contribution_date,
                last_run_date = EXCLUDED.last_run_date,
                benchmark_start_date = EXCLUDED.benchmark_start_date,
                benchmark_start_price = EXCLUDED.benchmark_start_price,
                updated_at = NOW()
            """,
            (
                account_state.cash,
                account_state.benchmark_units,
                account_state.benchmark_symbol,
                account_state.state_version,
                account_state.strategy_contributed,
                account_state.benchmark_contributed,
                account_state.last_strategy_contribution_date,
                account_state.last_benchmark_contribution_date,
                account_state.last_run_date,
                account_state.benchmark_start_date,
                account_state.benchmark_start_price,
            ),
        )
        self.conn.commit()

        return PaperRunResult(
            run_id=run_id,
            run_date=run_date,
            already_ran_today=False,
            contribution_applied=contribution_applied,
            strategy_contribution_amount=strategy_contribution,
            benchmark_contribution_amount=benchmark_contribution,
            total_contributed_strategy=account_state.strategy_contributed,
            total_contributed_benchmark=account_state.benchmark_contributed,
            rebalance_executed=bool(trades),
            cash=account_state.cash,
            market_value=market_value,
            portfolio_value=portfolio_value,
            benchmark_price=benchmark_price,
            benchmark_value=benchmark_value,
            trades=trades,
            decisions=decisions,
            targets=targets,
            signal_frame=signal_frame,
            action_summary=action_summary,
            action_reason=action_reason,
            target_diff=target_diff,
        )

    def apply_monthly_contribution_if_due(
        self,
        run_date: date,
        account_state: AccountStateSnapshot,
        contribution_events: list[ContributionEvent],
    ) -> float:
        # Strategy-side monthly cash arrives once per month and stays as cash
        # until the current signal set justifies deploying it.
        if not is_monthly_contribution_due(
            run_date,
            account_state.last_strategy_contribution_date,
            self.config.execution.contribution_day,
        ):
            return 0.0
        amount = float(self.config.execution.monthly_contribution)
        account_state.cash += amount
        account_state.strategy_contributed += amount
        account_state.last_strategy_contribution_date = run_date
        contribution_events.append(
            ContributionEvent(
                flow_date=run_date,
                flow_type="monthly_contribution",
                amount=amount,
                applies_to="strategy",
            )
        )
        return amount

    def apply_benchmark_contribution_if_due(
        self,
        run_date: date,
        benchmark_price: float,
        account_state: AccountStateSnapshot,
        contribution_events: list[ContributionEvent],
    ) -> float:
        # The benchmark must receive the same external cash flow schedule as the
        # strategy so the comparison is contribution matched.
        if not is_monthly_contribution_due(
            run_date,
            account_state.last_benchmark_contribution_date,
            self.config.execution.contribution_day,
        ):
            return 0.0
        amount = float(self.config.execution.monthly_contribution)
        account_state.benchmark_units += amount / benchmark_price
        account_state.benchmark_contributed += amount
        account_state.last_benchmark_contribution_date = run_date
        contribution_events.append(
            ContributionEvent(
                flow_date=run_date,
                flow_type="benchmark_contribution",
                amount=amount,
                applies_to="benchmark",
            )
        )
        return amount

    def generate_rebalance_orders(
        self,
        current_positions: dict[str, dict[str, float]],
        targets: pd.DataFrame,
        latest_prices: pd.Series,
        available_cash: float,
        signal_frame: pd.DataFrame | None = None,
    ) -> list[PlannedOrder]:
        # Orders are derived from the current total portfolio value and the
        # target slot weights, with sells planned first and buys clipped to cash.
        target_weights = targets.set_index("symbol")["weight"].to_dict() if not targets.empty else {}
        target_lookup = targets.set_index("symbol").to_dict("index") if not targets.empty else {}
        signal_lookup = signal_frame.set_index("symbol").to_dict("index") if signal_frame is not None and not signal_frame.empty else {}

        market_value = self._market_value(current_positions, latest_prices)
        portfolio_value = available_cash + market_value
        target_values = {symbol: portfolio_value * float(weight) for symbol, weight in target_weights.items()}

        planned_orders: list[PlannedOrder] = []
        simulated_cash = float(available_cash)
        projected_positions = {
            symbol: {"quantity": float(state["quantity"]), "avg_cost": float(state["avg_cost"])}
            for symbol, state in current_positions.items()
        }

        universe = sorted(set(projected_positions) | set(target_weights))

        # First handle sells so the generated buy set can safely spend the freed cash.
        for symbol in universe:
            if symbol not in latest_prices.index:
                continue
            price = float(latest_prices[symbol])
            current_qty = float(projected_positions.get(symbol, {}).get("quantity", 0.0))
            if current_qty <= 1e-8:
                continue
            target_value = float(target_values.get(symbol, 0.0))
            target_qty = target_value / price if price > 0 else 0.0
            if current_qty <= target_qty + 1e-8:
                continue

            delta_qty = current_qty - target_qty
            gross_notional = delta_qty * price
            target_weight = target_weights.get(symbol)
            if target_weight and gross_notional < self.config.execution.min_trade_dollars:
                continue
            reason = self._sell_reason(symbol, signal_lookup, set(target_weights))
            exec_price = price * (1 - self.config.execution.slippage_bps / 10_000.0)
            fees = gross_notional * (self.config.execution.commission_bps / 10_000.0)
            planned_orders.append(
                PlannedOrder(
                    symbol=symbol,
                    side="sell",
                    quantity=delta_qty,
                    exec_price=exec_price,
                    gross_notional=gross_notional,
                    fees=fees,
                    target_weight=target_weight,
                    reason=reason,
                )
            )
            simulated_cash += delta_qty * exec_price - fees
            new_qty = current_qty - delta_qty
            if new_qty <= 1e-8:
                projected_positions.pop(symbol, None)
            else:
                projected_positions[symbol]["quantity"] = new_qty

        for symbol in universe:
            if symbol not in latest_prices.index or symbol not in target_weights:
                continue
            price = float(latest_prices[symbol])
            current_qty = float(projected_positions.get(symbol, {}).get("quantity", 0.0))
            target_value = float(target_values.get(symbol, 0.0))
            target_qty = target_value / price if price > 0 else 0.0
            if target_qty <= current_qty + 1e-8:
                continue

            desired_qty = target_qty - current_qty
            desired_notional = desired_qty * price
            if desired_notional < self.config.execution.min_trade_dollars:
                continue

            exec_price = price * (1 + self.config.execution.slippage_bps / 10_000.0)
            max_affordable_qty = (
                simulated_cash / (exec_price * (1 + self.config.execution.commission_bps / 10_000.0))
                if exec_price > 0
                else 0.0
            )
            trade_qty = min(desired_qty, max_affordable_qty)
            gross_notional = trade_qty * price
            if trade_qty <= 1e-8 or gross_notional < self.config.execution.min_trade_dollars:
                continue

            fees = gross_notional * (self.config.execution.commission_bps / 10_000.0)
            simulated_cash -= trade_qty * exec_price + fees
            projected_positions[symbol] = {
                "quantity": current_qty + trade_qty,
                "avg_cost": float(projected_positions.get(symbol, {}).get("avg_cost", exec_price)),
            }
            planned_orders.append(
                PlannedOrder(
                    symbol=symbol,
                    side="buy",
                    quantity=trade_qty,
                    exec_price=exec_price,
                    gross_notional=gross_notional,
                    fees=fees,
                    target_weight=float(target_lookup[symbol]["weight"]),
                    reason=str(target_lookup[symbol].get("reason") or "target slot buy"),
                )
            )

        return planned_orders

    def _apply_orders(
        self,
        planned_orders: list[PlannedOrder],
        positions: dict[str, dict[str, float]],
        cash: float,
    ) -> tuple[list[ExecutedTrade], float]:
        # Apply sells first, then buys, updating cash and average cost bases in-place.
        executed: list[ExecutedTrade] = []

        for order in planned_orders:
            if order.side != "sell":
                continue
            current_qty = float(positions.get(order.symbol, {}).get("quantity", 0.0))
            trade_qty = min(order.quantity, current_qty)
            if trade_qty <= 1e-8:
                continue
            cash += trade_qty * order.exec_price - order.fees
            new_qty = current_qty - trade_qty
            if new_qty <= 1e-8:
                positions.pop(order.symbol, None)
            else:
                positions[order.symbol]["quantity"] = new_qty
            executed.append(
                ExecutedTrade(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=trade_qty,
                    price=order.exec_price,
                    gross_notional=trade_qty * (order.gross_notional / order.quantity) if order.quantity > 0 else order.gross_notional,
                    fees=order.fees,
                    target_weight=order.target_weight,
                    reason=order.reason,
                )
            )

        for order in planned_orders:
            if order.side != "buy":
                continue
            max_affordable_qty = (
                cash / (order.exec_price * (1 + self.config.execution.commission_bps / 10_000.0))
                if order.exec_price > 0
                else 0.0
            )
            trade_qty = min(order.quantity, max_affordable_qty)
            gross_notional = trade_qty * (order.gross_notional / order.quantity) if order.quantity > 0 else 0.0
            if trade_qty <= 1e-8 or gross_notional < self.config.execution.min_trade_dollars:
                continue
            fees = gross_notional * (self.config.execution.commission_bps / 10_000.0)
            total_cost = trade_qty * order.exec_price + fees
            if total_cost > cash + 1e-9:
                continue
            current_qty = float(positions.get(order.symbol, {}).get("quantity", 0.0))
            current_avg = float(positions.get(order.symbol, {}).get("avg_cost", 0.0))
            new_qty = current_qty + trade_qty
            new_avg = ((current_qty * current_avg) + (trade_qty * order.exec_price)) / new_qty if new_qty > 0 else 0.0
            positions[order.symbol] = {"quantity": new_qty, "avg_cost": new_avg}
            cash -= total_cost
            executed.append(
                ExecutedTrade(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=trade_qty,
                    price=order.exec_price,
                    gross_notional=gross_notional,
                    fees=fees,
                    target_weight=order.target_weight,
                    reason=order.reason,
                )
            )

        return executed, cash

    def _load_or_initialize_account_state(
        self,
        run_date: date,
        benchmark_symbol: str,
        benchmark_start_trade_date: date,
        benchmark_start_price: float,
    ) -> tuple[AccountStateSnapshot, bool, bool]:
        # Fresh deployments start with $5,000 cash and a matching benchmark lot;
        # existing deployments continue from the persisted singleton account row.
        try:
            row = self.conn.execute(
                """
                SELECT cash, benchmark_units, benchmark_symbol, state_version, strategy_contributed, benchmark_contributed,
                       last_strategy_contribution_date, last_benchmark_contribution_date, last_run_date,
                       benchmark_start_date, benchmark_start_price
                FROM account_state
                WHERE account_id = 1
                """
            ).fetchone()
        except errors.UndefinedTable:
            self.conn.rollback()
            row = None
        if row is None or int(row.get("state_version") or 0) < CURRENT_ACCOUNT_STATE_VERSION:
            return (
                AccountStateSnapshot(
                    cash=float(self.config.execution.initial_cash),
                    benchmark_units=float(self.config.execution.initial_cash / benchmark_start_price)
                    if run_date >= benchmark_start_trade_date
                    else 0.0,
                    benchmark_symbol=benchmark_symbol,
                    state_version=CURRENT_ACCOUNT_STATE_VERSION,
                    strategy_contributed=float(self.config.execution.initial_cash),
                    benchmark_contributed=float(self.config.execution.initial_cash)
                    if run_date >= benchmark_start_trade_date
                    else 0.0,
                    # Initial funding occupies the opening month's contribution slot.
                    last_strategy_contribution_date=run_date,
                    last_benchmark_contribution_date=benchmark_start_trade_date if run_date >= benchmark_start_trade_date else None,
                    last_run_date=None,
                    benchmark_start_date=benchmark_start_trade_date,
                    benchmark_start_price=benchmark_start_price,
                ),
                row is not None,
                row is not None,
            )

        snapshot = AccountStateSnapshot(
            cash=float(row["cash"]),
            benchmark_units=float(row["benchmark_units"]),
            benchmark_symbol=str(row["benchmark_symbol"] or benchmark_symbol),
            state_version=int(row["state_version"] or CURRENT_ACCOUNT_STATE_VERSION),
            strategy_contributed=float(row["strategy_contributed"] or 0.0),
            benchmark_contributed=float(row["benchmark_contributed"] or 0.0),
            last_strategy_contribution_date=row["last_strategy_contribution_date"],
            last_benchmark_contribution_date=row["last_benchmark_contribution_date"],
            last_run_date=row["last_run_date"],
            benchmark_start_date=row["benchmark_start_date"] or run_date,
            benchmark_start_price=float(row["benchmark_start_price"] or benchmark_start_price),
        )

        # If the configured benchmark symbol changes, rebuild units from the
        # accumulated benchmark contributions at today's benchmark price.
        if snapshot.benchmark_symbol != benchmark_symbol:
            benchmark_base = snapshot.benchmark_contributed or float(self.config.execution.initial_cash)
            snapshot.benchmark_units = benchmark_base / benchmark_start_price if benchmark_start_price > 0 else 0.0
            snapshot.benchmark_symbol = benchmark_symbol
            snapshot.benchmark_start_date = benchmark_start_trade_date
            snapshot.benchmark_start_price = benchmark_start_price
        return snapshot, True, False

    def _load_positions(self) -> dict[str, dict[str, float]]:
        # The live book is small, so a simple dict keyed by symbol is enough.
        try:
            rows = self.conn.execute(
                "SELECT symbol, quantity, avg_cost FROM paper_positions WHERE quantity > 0"
            ).fetchall()
        except errors.UndefinedTable:
            self.conn.rollback()
            rows = []
        return {
            str(row["symbol"]): {
                "quantity": float(row["quantity"]),
                "avg_cost": float(row["avg_cost"]),
            }
            for row in rows
        }

    def _load_existing_run_result(self, run_date: date, command: str) -> PaperRunResult | None:
        # Duplicate same-day invocations reuse the already-persisted result
        # instead of inserting a second run row with no-op side effects.
        run = self.conn.execute(
            """
            SELECT id, run_date, contribution_applied, strategy_contribution_amount, benchmark_contribution_amount,
                   total_contributed_strategy, total_contributed_benchmark, rebalance_executed,
                   cash, market_value, portfolio_value, benchmark_price, benchmark_value,
                   action_summary, action_reason
            FROM paper_runs
            WHERE run_date = %s AND command = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (run_date, command),
        ).fetchone()
        if run is None:
            return None

        run_id = int(run["id"])
        trades = [
            ExecutedTrade(
                symbol=str(row["symbol"]),
                side=str(row["side"]),
                quantity=float(row["quantity"]),
                price=float(row["price"]),
                gross_notional=float(row["gross_notional"]),
                fees=float(row["fees"]),
                target_weight=float(row["target_weight"]) if row["target_weight"] is not None else None,
                reason=str(row["reason"] or ""),
            )
            for row in self.conn.execute(
                """
                SELECT symbol, side, quantity, price, gross_notional, fees, target_weight, reason
                FROM paper_trades
                WHERE run_id = %s
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        ]
        decisions = [
            DecisionRecord(symbol=row["symbol"], decision=str(row["decision"]), reason=str(row["reason"]))
            for row in self.conn.execute(
                """
                SELECT symbol, decision, reason
                FROM decision_log
                WHERE run_id = %s
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        ]
        targets = pd.DataFrame(
            self.conn.execute(
                """
                SELECT symbol, weight, score
                FROM target_positions
                WHERE run_id = %s
                ORDER BY weight DESC, symbol
                """,
                (run_id,),
            ).fetchall()
        )
        target_diff = pd.DataFrame(
            self.conn.execute(
                """
                SELECT symbol, change_type, previous_rank, current_rank, previous_score,
                       current_score, previous_weight, current_weight, reason
                FROM target_diffs
                WHERE run_id = %s
                ORDER BY change_type, symbol
                """,
                (run_id,),
            ).fetchall()
        )

        return PaperRunResult(
            run_id=run_id,
            run_date=run["run_date"],
            already_ran_today=True,
            contribution_applied=bool(run["contribution_applied"]),
            strategy_contribution_amount=float(run["strategy_contribution_amount"]),
            benchmark_contribution_amount=float(run["benchmark_contribution_amount"]),
            total_contributed_strategy=float(run["total_contributed_strategy"]),
            total_contributed_benchmark=float(run["total_contributed_benchmark"]),
            rebalance_executed=bool(run["rebalance_executed"]),
            cash=float(run["cash"]),
            market_value=float(run["market_value"]),
            portfolio_value=float(run["portfolio_value"]),
            benchmark_price=float(run["benchmark_price"]),
            benchmark_value=float(run["benchmark_value"]),
            trades=trades,
            decisions=decisions,
            targets=targets,
            signal_frame=pd.DataFrame(),
            action_summary=str(run["action_summary"] or ""),
            action_reason=str(run["action_reason"] or ""),
            target_diff=target_diff,
        )

    @staticmethod
    def _market_value(positions: dict[str, dict[str, float]], latest_prices: pd.Series) -> float:
        # Mark the current live holdings to the latest fetched close prices.
        return sum(
            float(state["quantity"]) * float(latest_prices[symbol])
            for symbol, state in positions.items()
            if symbol in latest_prices.index
        )

    def _max_weight_drift(
        self,
        positions: dict[str, dict[str, float]],
        targets: pd.DataFrame,
        latest_prices: pd.Series,
        cash: float,
    ) -> float:
        if targets.empty:
            return 0.0
        portfolio_value = cash + self._market_value(positions, latest_prices)
        if portfolio_value <= 0:
            return 0.0
        target_weights = targets.set_index("symbol")["weight"].to_dict()
        symbols = set(target_weights) | set(positions)
        drift = 0.0
        for symbol in symbols:
            current_value = (
                float(positions.get(symbol, {}).get("quantity", 0.0)) * float(latest_prices[symbol])
                if symbol in latest_prices.index
                else 0.0
            )
            current_weight = current_value / portfolio_value
            drift = max(drift, abs(current_weight - float(target_weights.get(symbol, 0.0))))
        return drift

    def _load_previous_targets(self, run_date: date, command: str) -> pd.DataFrame:
        try:
            rows = self.conn.execute(
                """
                WITH previous_run AS (
                    SELECT id
                    FROM paper_runs
                    WHERE run_date < %s AND command = %s
                    ORDER BY run_date DESC, id DESC
                    LIMIT 1
                )
                SELECT tp.symbol, tp.weight, tp.score
                FROM previous_run pr
                JOIN target_positions tp ON tp.run_id = pr.id
                ORDER BY tp.score DESC, tp.weight DESC, tp.symbol
                """,
                (run_date, command),
            ).fetchall()
        except errors.UndefinedTable:
            self.conn.rollback()
            rows = []
        if not rows:
            return pd.DataFrame(columns=["symbol", "weight", "score", "rank"])
        frame = pd.DataFrame(rows).drop_duplicates("symbol", keep="first").reset_index(drop=True)
        frame["rank"] = range(1, len(frame) + 1)
        return frame

    @staticmethod
    def _sell_reason(symbol: str, signal_lookup: dict[str, dict[str, object]], target_symbols: set[str]) -> str:
        # Sell reasons mirror the simplified strategy rules so the daily report
        # can explain exits without inspecting raw internals.
        row = signal_lookup.get(symbol)
        if row is None:
            return "symbol no longer has a valid current Capitol disclosure signal"
        if float(row.get("net_notional") or 0.0) <= 0:
            return "net disclosure flow turned negative"
        if not bool(row.get("above_moving_average")):
            return "latest close fell below 50d MA"
        if symbol not in target_symbols:
            return "replaced by a stronger top-3 disclosure candidate"
        return "clean up non-target position"

    def _build_decision_log(
        self,
        signal_frame: pd.DataFrame,
        targets: pd.DataFrame,
        current_positions: dict[str, dict[str, float]],
        planned_orders: list[PlannedOrder],
        trades: list[ExecutedTrade],
        strategy_contribution_amount: float,
    ) -> list[DecisionRecord]:
        # The decision log records both actions and explicit inaction so daily
        # runs are auditable after the fact.
        decisions: list[DecisionRecord] = []
        trade_symbols = {trade.symbol for trade in trades}
        for trade in trades:
            decisions.append(DecisionRecord(symbol=trade.symbol, decision=trade.side, reason=trade.reason))

        target_lookup = targets.set_index("symbol").to_dict("index") if not targets.empty else {}
        for symbol, row in target_lookup.items():
            if symbol in trade_symbols:
                continue
            decisions.append(
                DecisionRecord(
                    symbol=symbol,
                    decision="hold",
                    reason=str(row.get("reason") or "active target unchanged"),
                )
            )

        if not decisions:
            if strategy_contribution_amount > 0 and targets.empty:
                decisions.append(
                    DecisionRecord(
                        symbol=None,
                        decision="hold",
                        reason="monthly contribution added but no valid buy signal exists, so cash was retained",
                    )
                )
            elif targets.empty and current_positions:
                decisions.append(
                    DecisionRecord(
                        symbol=None,
                        decision="hold",
                        reason="no valid top-3 buy signals are available today",
                    )
                )
            else:
                decisions.append(
                    DecisionRecord(
                        symbol=None,
                        decision="hold",
                        reason="signal set and holdings were unchanged, so no rebalance was necessary",
                    )
                )

        # Add skips for notable candidates that almost qualified but did not.
        if not signal_frame.empty:
            for _, row in signal_frame.head(5).iterrows():
                symbol = str(row["symbol"])
                if symbol in trade_symbols or symbol in target_lookup:
                    continue
                decisions.append(
                    DecisionRecord(
                        symbol=symbol,
                        decision="skip",
                        reason=self._sell_reason(symbol, signal_frame.set_index("symbol").to_dict("index"), set(target_lookup)),
                    )
                )
        return decisions

    @staticmethod
    def _summarize_actions(
        trades: list[ExecutedTrade],
        decisions: list[DecisionRecord],
        strategy_contribution_amount: float,
    ) -> tuple[str, str]:
        # The short report needs one deterministic headline action and one reason line.
        if trades:
            first_trade = trades[0]
            summary = f"{first_trade.side.upper()} {first_trade.symbol} ${first_trade.gross_notional:,.2f}"
            return summary, first_trade.reason
        if strategy_contribution_amount > 0:
            return "No trade", "monthly contribution was added but current signals did not require deployment"
        first_decision = decisions[0] if decisions else DecisionRecord(symbol=None, decision="hold", reason="no action recorded")
        return "No trade", first_decision.reason
