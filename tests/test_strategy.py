import pandas as pd

from bot.capitol_scraper import CapitolTradesScraper
from bot.config import AppConfig, DataConfig, ExecutionConfig, StrategyConfig
from bot.paper_trader import (
    AccountStateSnapshot,
    PaperTrader,
    is_monthly_contribution_due,
    should_skip_duplicate_run,
)
from bot.reporting import DailyReport, render_daily_message
from bot.symbol_resolver import SymbolResolver
from bot.strategy import CapitolStrategy


def _build_config(**strategy_overrides) -> AppConfig:
    strategy_defaults = {
        "start_date": "2024-01-01",
        "portfolio_start_date": "2024-01-01",
        "benchmark_start_date": "2024-01-01",
        "moving_average_days": 3,
        "max_positions": 3,
        "min_buy_disclosures": 2,
        "allow_cash_buffer": False,
    }
    strategy_defaults.update(strategy_overrides)
    strategy = StrategyConfig(**strategy_defaults)
    return AppConfig(strategy=strategy, data=DataConfig(), execution=ExecutionConfig())


def _build_trader(config: AppConfig | None = None) -> PaperTrader:
    # These tests exercise pure helper methods, so a DB connection is unnecessary.
    trader = PaperTrader.__new__(PaperTrader)
    trader.config = config or _build_config()
    trader.conn = None
    return trader


def _build_signal_test_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    rows = []
    aaa = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    bbb = [10.0, 10.0, 10.0, 9.0, 8.0, 7.0]
    ccc = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
    for idx, current in enumerate(dates):
        rows.append({"date": current, "symbol": "AAA", "close": aaa[idx]})
        rows.append({"date": current, "symbol": "BBB", "close": bbb[idx]})
        rows.append({"date": current, "symbol": "CCC", "close": ccc[idx]})
        rows.append({"date": current, "symbol": "SPY", "close": 100.0 + idx})
    return pd.DataFrame(rows)


def test_monthly_contribution_applied_exactly_once() -> None:
    # Strategy-side monthly cash should only be added once per calendar month.
    trader = _build_trader()
    state = AccountStateSnapshot(
        cash=5000.0,
        benchmark_units=50.0,
        benchmark_symbol="IE00B5BMR087",
        state_version=2,
        strategy_contributed=5000.0,
        benchmark_contributed=5000.0,
        last_strategy_contribution_date=pd.Timestamp("2026-03-15").date(),
        last_benchmark_contribution_date=pd.Timestamp("2026-03-15").date(),
        last_run_date=None,
        benchmark_start_date=pd.Timestamp("2026-03-15").date(),
        benchmark_start_price=100.0,
    )
    events = []

    first = trader.apply_monthly_contribution_if_due(pd.Timestamp("2026-04-01").date(), state, events)
    second = trader.apply_monthly_contribution_if_due(pd.Timestamp("2026-04-01").date(), state, events)

    assert first == 250.0
    assert second == 0.0
    assert state.cash == 5250.0
    assert state.strategy_contributed == 5250.0
    assert len(events) == 1


def test_benchmark_contribution_applied_exactly_once() -> None:
    # Benchmark units should accumulate once on the first eligible run of the month.
    trader = _build_trader()
    state = AccountStateSnapshot(
        cash=5000.0,
        benchmark_units=50.0,
        benchmark_symbol="IE00B5BMR087",
        state_version=2,
        strategy_contributed=5000.0,
        benchmark_contributed=5000.0,
        last_strategy_contribution_date=pd.Timestamp("2026-03-15").date(),
        last_benchmark_contribution_date=pd.Timestamp("2026-03-15").date(),
        last_run_date=None,
        benchmark_start_date=pd.Timestamp("2026-03-15").date(),
        benchmark_start_price=100.0,
    )
    events = []

    first = trader.apply_benchmark_contribution_if_due(pd.Timestamp("2026-04-01").date(), 100.0, state, events)
    second = trader.apply_benchmark_contribution_if_due(pd.Timestamp("2026-04-01").date(), 100.0, state, events)

    assert first == 250.0
    assert second == 0.0
    assert state.benchmark_units == 52.5
    assert state.benchmark_contributed == 5250.0
    assert len(events) == 1


def test_no_duplicate_run_side_effects_same_day() -> None:
    # A second same-day invocation should be recognized as a duplicate daily run.
    run_date = pd.Timestamp("2026-04-22").date()
    assert should_skip_duplicate_run(run_date, run_date) is True
    assert should_skip_duplicate_run(run_date, pd.Timestamp("2026-04-21").date()) is False
    assert should_skip_duplicate_run(run_date, None) is False


def test_buy_signal_requires_positive_flow_min_buy_count_and_price_above_ma() -> None:
    # Only symbols that satisfy every simple rule should become buyable.
    config = _build_config()
    strategy = CapitolStrategy(config)
    disclosures = pd.DataFrame(
        [
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-06", "transaction_type": "buy", "signed_notional": 1200.0, "notional_mid": 1200.0},
            {"ticker": "BBB", "asset_name": "BBB", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "BBB", "asset_name": "BBB", "published_at": "2024-01-06", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "CCC", "asset_name": "CCC", "published_at": "2024-01-06", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
        ]
    )

    signal_frame = strategy.build_signal_frame(disclosures, _build_signal_test_prices(), as_of_date="2024-01-06")
    buyable = signal_frame.set_index("symbol")

    assert bool(buyable.loc["AAA", "is_buyable"]) is True
    assert bool(buyable.loc["BBB", "is_buyable"]) is False
    assert bool(buyable.loc["CCC", "is_buyable"]) is False


def test_sell_signal_triggers_on_negative_flow_or_price_below_ma() -> None:
    # Negative net flow and a broken trend should both produce explicit sell reasons.
    config = _build_config()
    strategy = CapitolStrategy(config)
    disclosures = pd.DataFrame(
        [
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-06", "transaction_type": "sell", "signed_notional": -1500.0, "notional_mid": 1500.0},
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 500.0, "notional_mid": 500.0},
            {"ticker": "BBB", "asset_name": "BBB", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "BBB", "asset_name": "BBB", "published_at": "2024-01-06", "transaction_type": "buy", "signed_notional": 900.0, "notional_mid": 900.0},
        ]
    )

    signal_frame = strategy.build_signal_frame(disclosures, _build_signal_test_prices(), as_of_date="2024-01-06").set_index("symbol")

    assert signal_frame.loc["AAA", "sell_reason"] == "net disclosure flow turned negative"
    assert signal_frame.loc["BBB", "sell_reason"] == "latest close fell below 50d MA"


def test_backtest_uses_only_disclosures_published_up_to_each_date() -> None:
    # The walk-forward simulator must not buy a name before its disclosure is published.
    config = _build_config(moving_average_days=2)
    strategy = CapitolStrategy(config)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = pd.DataFrame(
        [
            *[
                {"date": current, "symbol": "AAA", "close": 10.0 + idx}
                for idx, current in enumerate(dates)
            ],
            *[
                {"date": current, "symbol": "SPY", "close": 100.0 + idx}
                for idx, current in enumerate(dates)
            ],
        ]
    )
    disclosures = pd.DataFrame(
        [
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 1200.0, "notional_mid": 1200.0},
        ]
    )

    curve, _ = strategy.backtest_walk_forward(disclosures, prices, "SPY")
    curve = curve.set_index("date")

    assert curve.loc[pd.Timestamp("2024-01-04"), "market_value"] == 0.0
    assert curve.loc[pd.Timestamp("2024-01-05"), "market_value"] > 0.0


def test_backtest_respects_benchmark_start_date() -> None:
    # The benchmark should remain at zero before its configured start date.
    config = _build_config(start_date="2025-12-29", benchmark_start_date="2026-01-01", moving_average_days=2)
    strategy = CapitolStrategy(config)
    dates = pd.date_range("2025-12-29", periods=6, freq="D")
    prices = pd.DataFrame(
        [
            *[
                {"date": current, "symbol": "AAA", "close": 10.0 + idx}
                for idx, current in enumerate(dates)
            ],
            *[
                {"date": current, "symbol": "SPY", "close": 100.0 + idx}
                for idx, current in enumerate(dates)
            ],
        ]
    )

    curve, _ = strategy.backtest_walk_forward(pd.DataFrame(), prices, "SPY")
    curve = curve.set_index("date")

    assert curve.loc[pd.Timestamp("2025-12-31"), "benchmark_value"] == 0.0
    assert curve.loc[pd.Timestamp("2026-01-01"), "benchmark_value"] > 0.0


def test_simulation_seeds_strategy_and_benchmark_from_same_start_month() -> None:
    # A January portfolio start should accumulate both strategy and benchmark contributions before the first live run.
    config = _build_config(
        start_date="2025-11-01",
        portfolio_start_date="2026-01-01",
        benchmark_start_date="2026-01-01",
        moving_average_days=2,
    )
    strategy = CapitolStrategy(config)
    prices = pd.DataFrame(
        [
            {"date": "2025-12-31", "symbol": "AAA", "close": 10.0},
            {"date": "2026-01-01", "symbol": "AAA", "close": 11.0},
            {"date": "2026-01-02", "symbol": "AAA", "close": 12.0},
            {"date": "2026-02-02", "symbol": "AAA", "close": 13.0},
            {"date": "2026-03-02", "symbol": "AAA", "close": 14.0},
            {"date": "2026-04-01", "symbol": "AAA", "close": 15.0},
            {"date": "2025-12-31", "symbol": "SPY", "close": 100.0},
            {"date": "2026-01-01", "symbol": "SPY", "close": 101.0},
            {"date": "2026-01-02", "symbol": "SPY", "close": 102.0},
            {"date": "2026-02-02", "symbol": "SPY", "close": 110.0},
            {"date": "2026-03-02", "symbol": "SPY", "close": 120.0},
            {"date": "2026-04-01", "symbol": "SPY", "close": 125.0},
        ]
    )
    disclosures = pd.DataFrame(
        [
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2025-12-31", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2026-01-01", "transaction_type": "buy", "signed_notional": 1200.0, "notional_mid": 1200.0},
        ]
    )
    simulation = strategy.simulate_walk_forward(disclosures, prices, "SPY", end_date="2026-04-01")

    assert simulation.total_contributed_strategy == 5750.0
    assert simulation.total_contributed_benchmark == 5750.0
    assert simulation.last_strategy_contribution_date == pd.Timestamp("2026-04-01").date()
    assert simulation.last_benchmark_contribution_date == pd.Timestamp("2026-04-01").date()
    january_curve = simulation.curve.set_index("date")
    assert january_curve.loc[pd.Timestamp("2026-01-01"), "market_value"] > 0.0
    assert simulation.metrics["trade_count"] > 0.0
    assert simulation.cash >= 0.0


def test_initial_seed_targets_use_top_prior_year_bought_names() -> None:
    # The January seed should start from the most-bought Capitol names from the configured prior year.
    config = _build_config(start_date="2025-01-01", portfolio_start_date="2026-01-01", initial_selection_year=2025, initial_top_n=2)
    strategy = CapitolStrategy(config)
    prices = pd.DataFrame(
        [
            {"date": "2026-01-02", "symbol": "AAA", "close": 10.0},
            {"date": "2026-01-02", "symbol": "BBB", "close": 20.0},
            {"date": "2026-01-02", "symbol": "CCC", "close": 30.0},
        ]
    )
    disclosures = pd.DataFrame(
        [
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2025-01-10", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2025-06-10", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "BBB", "asset_name": "BBB", "published_at": "2025-03-10", "transaction_type": "buy", "signed_notional": 900.0, "notional_mid": 900.0},
            {"ticker": "CCC", "asset_name": "CCC", "published_at": "2025-04-10", "transaction_type": "buy", "signed_notional": 800.0, "notional_mid": 800.0},
        ]
    )

    targets = strategy.build_initial_seed_targets(disclosures, prices, "2026-01-02")

    assert list(targets["symbol"]) == ["AAA", "BBB"]
    assert round(float(targets["weight"].sum()), 6) == 1.0


def test_capitol_scraper_parses_structured_nextjs_trade_payload() -> None:
    # The live Capitol Trades page now embeds trade rows in a JSON `data` array.
    # This regression test keeps the scraper aligned with that structure.
    scraper = CapitolTradesScraper(user_agent="capitol-paper-bot-test")
    html = """
    <html><body>
    <script>
    self.__next_f.push([1,"..."]);
    </script>
    <script>
    {"data":[{"_txId":1,"chamber":"house","issuer":{"issuerName":"NVIDIA Corporation","issuerTicker":"NVDA:US","sector":"information-technology"},"politician":{"firstName":"Nancy","lastName":"Pelosi","nickname":null,"party":"democrat","_stateId":"ca","chamber":"house"},"pubDate":"2026-04-22T13:05:02Z","reportingGap":28,"txDate":"2026-03-24","txType":"buy","value":32500}]}
    </script>
    </body></html>
    """

    disclosures = list(scraper._parse_trade_listing(html))

    assert len(disclosures) == 1
    assert disclosures[0].ticker == "NVDA"
    assert disclosures[0].asset_name == "NVIDIA Corporation"
    assert disclosures[0].politician == "Nancy Pelosi"
    assert disclosures[0].published_at == pd.Timestamp("2026-04-22").date()
    assert disclosures[0].traded_at == pd.Timestamp("2026-03-24").date()
    assert disclosures[0].notional_mid == 32500.0


def test_portfolio_never_spends_more_cash_than_available() -> None:
    # Buy sizing must be clipped so cash never goes negative after fees and slippage.
    config = _build_config()
    trader = _build_trader(config)
    latest_prices = pd.Series({"AAA": 300.0, "BBB": 100.0})
    current_positions = {"AAA": {"quantity": 1.0, "avg_cost": 300.0}}
    targets = pd.DataFrame(
        [
            {"symbol": "AAA", "weight": 1 / 3, "score": 10.0, "reason": "hold target"},
            {"symbol": "BBB", "weight": 1 / 3, "score": 9.0, "reason": "buy candidate"},
        ]
    )

    orders = trader.generate_rebalance_orders(
        current_positions=current_positions,
        targets=targets,
        latest_prices=latest_prices,
        available_cash=10.0,
        signal_frame=pd.DataFrame(),
    )
    positions = {"AAA": {"quantity": 1.0, "avg_cost": 300.0}}
    _, cash = trader._apply_orders(orders, positions, 10.0)

    assert cash >= -1e-9


def test_signal_targets_fully_allocate_when_fewer_than_three_names() -> None:
    # With cash buffering disabled, active names should split the portfolio rather than leaving empty slots in cash.
    config = _build_config()
    strategy = CapitolStrategy(config)
    disclosures = pd.DataFrame(
        [
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-06", "transaction_type": "buy", "signed_notional": 1200.0, "notional_mid": 1200.0},
            {"ticker": "CCC", "asset_name": "CCC", "published_at": "2024-01-05", "transaction_type": "buy", "signed_notional": 900.0, "notional_mid": 900.0},
            {"ticker": "CCC", "asset_name": "CCC", "published_at": "2024-01-06", "transaction_type": "buy", "signed_notional": 950.0, "notional_mid": 950.0},
        ]
    )

    targets = strategy.build_targets(
        strategy.build_signal_frame(disclosures, _build_signal_test_prices(), as_of_date="2024-01-06"),
        current_positions=None,
        available_cash=5000.0,
        as_of_date="2024-01-06",
    )

    assert len(targets) == 2
    assert round(float(targets["weight"].sum()), 6) == 1.0


def test_existing_position_is_retained_without_fresh_disclosures_if_trend_holds() -> None:
    # Existing holdings should not be forced to cash just because recent Capitol disclosures have gone stale.
    config = _build_config(start_date="2024-01-01", portfolio_start_date="2024-01-01", moving_average_days=2)
    strategy = CapitolStrategy(config)
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "symbol": "AAA", "close": 10.0},
            {"date": "2024-01-02", "symbol": "AAA", "close": 11.0},
            {"date": "2024-03-01", "symbol": "AAA", "close": 15.0},
        ]
    )
    disclosures = pd.DataFrame(
        [
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-01", "transaction_type": "buy", "signed_notional": 1000.0, "notional_mid": 1000.0},
            {"ticker": "AAA", "asset_name": "AAA", "published_at": "2024-01-02", "transaction_type": "buy", "signed_notional": 1200.0, "notional_mid": 1200.0},
        ]
    )

    signal_frame = strategy.build_signal_frame(disclosures, prices, as_of_date="2024-03-01")
    targets = strategy.build_targets(
        signal_frame,
        current_positions={"AAA": {"quantity": 10.0, "avg_cost": 12.0}},
        available_cash=0.0,
        as_of_date="2024-03-01",
    )

    assert list(targets["symbol"]) == ["AAA"]


def test_symbol_resolver_normalizes_share_class_symbols() -> None:
    # Share-class aliases should map to the provider's expected punctuation.
    resolver = SymbolResolver()
    assert resolver.candidate_symbols("BRK/B")[0] == "BRK-B"


def test_symbol_resolver_maps_benchmark_isin_to_cspx_l() -> None:
    # The benchmark ISIN must resolve to a tradable market-data symbol.
    resolver = SymbolResolver()
    assert resolver.candidate_symbols("IE00B5BMR087")[0] == "CSPX.L"


def test_symbol_resolver_skips_mutual_fund_like_symbols() -> None:
    # Mutual-fund-style symbols remain out of scope for the paper bot.
    resolver = SymbolResolver()
    assert resolver.skip_reason("JTSXX") == "mutual_fund_like"


def test_symbol_resolver_skips_xsp_index_option() -> None:
    # Known index/derivative symbols must remain filtered out.
    resolver = SymbolResolver()
    assert resolver.skip_reason("XSP") == "index_or_derivative"


def test_render_daily_message_includes_contribution_trade_and_reason() -> None:
    # The short daily report should state contribution, trade summary, and reason.
    report = DailyReport(
        run_id=1,
        run_date="2026-04-22",
        contribution_applied=True,
        strategy_contribution_amount=250.0,
        benchmark_contribution_amount=250.0,
        total_contributed_strategy=6250.0,
        total_contributed_benchmark=6250.0,
        portfolio_value=6412.10,
        benchmark_value=6380.44,
        benchmark_symbol="IE00B5BMR087",
        cash=172.0,
        market_value=6240.10,
        rebalance_executed=True,
        action_summary="BUY NVDA $250.00",
        action_reason="new net-positive Capitol disclosure flow and price above 50d MA",
        routine_status={"reason": "fresh"},
        top_holdings=pd.DataFrame(
            [
                {"symbol": "NVDA", "asset_name": "NVIDIA Corporation", "market_value": 2180.0},
                {"symbol": "MSFT", "asset_name": "Microsoft Corp", "market_value": 2060.0},
            ]
        ),
        recent_trades=pd.DataFrame(),
        skipped_symbols=pd.DataFrame(),
    )

    message = render_daily_message(report)

    assert "Contribution applied: yes" in message
    assert "Benchmark ETF $6,380.44 on $6,250.00 contributed" in message
    assert "Capitol stocks $6,240.10 on $6,250.00 contributed" in message
    assert "Capitol total $6,412.10" in message
    assert "Cash balance: $172.00" in message
    assert "Holdings: NVDA (NVIDIA Corporation) $2,180.00, MSFT (Microsoft Corp) $2,060.00, cash $172.00" in message
    assert "Trades today: BUY NVDA $250.00" in message
    assert "Reason: new net-positive Capitol disclosure flow and price above 50d MA" in message
