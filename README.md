# Capitol Paper Bot

Dockerized daily paper-trading bot using Capitol Trades disclosures and free market data.

## What It Does

- Pulls recent published Capitol Trades disclosures from `capitoltrades.com`
- Resolves tradable equity and plain-ETF symbols and filters known bad symbols such as `XSP`
- Applies a simple daily rule set:
  - look back 45 calendar days by `published_at`
  - require positive net disclosure flow
  - require at least 2 buy disclosures
  - require latest close above the 50-day moving average
  - keep only the top 3 names by disclosure score
- Starts with `$5,000` cash and adds `$250` on the first eligible run of each month
- Tracks a contribution-matched benchmark that starts with `$5,000` in `IE00B5BMR087` and also adds `$250` monthly
- Persists state, cash flows, trades, decisions, and benchmark history in Postgres
- Writes flat artifacts and a short text report to `data/`
- Supports dry-run paper execution that computes targets and orders without writing paper state
- Writes a daily run manifest and target diff audit artifact

## Strategy Design

The strategy is intentionally conservative and audit-friendly.

Daily signal rules:

1. Use only disclosures whose `published_at` is within the last 45 calendar days.
2. Weight newer disclosures more than older disclosures.
3. Score each symbol as:
   `recency_weighted_net_notional + 0.25 * buy_count - 0.25 * sell_count`
4. A symbol is buyable only if:
   - 45-day net flow is positive
   - buy count is at least 2
   - latest close is above the 50-day moving average
  - it ranks in the top 3 names
5. Penalize late filings, track politician-level forward-return reliability, and separate `buy_score` from `sell_risk`.
6. Hold at most 3 equal slots and allow cash when fewer than 3 names qualify.
7. Avoid churn unless target membership changes, a contribution arrives, or existing holdings drift by at least `min_rebalance_drift`.
8. Require replacement candidates to clear current holdings by `min_replacement_score_advantage` before displacing them.
9. Exit an existing holding when its drawdown from average cost exceeds `stop_loss_pct`.
10. On monthly contribution days, keep the new cash idle if no valid buy signal exists.

This is a paper bot only. It does not connect to a broker.

## Ports

The compose stack avoids common defaults:

- Postgres is published on `55432`

## Run

```bash
cd /Users/herold/Desktop/capitol-paper-bot
docker compose build
docker compose up -d db
docker compose run --rm bot
```

The default command is now the scheduled paper-trading workflow. For a one-shot backtest:

```bash
docker compose run --rm bot python -m bot.cli --config config/strategy.yaml backtest
```

For a no-write deployment check:

```bash
docker compose run --rm bot python -m bot.cli --config config/strategy.yaml --dry-run paper-run
```

To regenerate the short iMessage-style status text from the latest stored run:

```bash
docker compose run --rm bot python -m bot.cli --config config/strategy.yaml daily-report
```

## Output files

- `data/disclosures.csv`
- `data/prices.csv`
- `data/symbol_resolution.csv`
- `data/targets.csv`
- `data/target_diff.csv`
- `data/equity_curve.csv`
- `data/backtest_report.md`
- `data/backtest_yearly_returns.csv`
- `data/backtest_trades.csv`
- `data/benchmark_report.csv`
- `data/run_manifest.json`
- `data/portfolio_history.csv`
- `data/portfolio_chart.svg`
- `data/daily_message.txt`

Paper-trading state is stored in Postgres tables:

- `account_state`
- `paper_runs`
- `symbol_resolutions`
- `target_positions`
- `paper_positions`
- `paper_trades`
- `position_snapshots`
- `cash_flows`
- `decision_log`
- `target_diffs`
- `schema_migrations`

## Developer Commands

All developer commands run through Docker:

```bash
make test
make health
make paper-run
make dry-run
make backtest
```

## Alerts

Set `DISCORD_WEBHOOK_URL` to send Discord alerts for the daily run summary, executed trades, stale data, provider errors, and disclosure-feed warnings.

Optional: set `DISCORD_USER_ID` to your numeric Discord user ID if you want the bot to mention you. A handle such as `herold1989` is not enough for Discord mentions.

Local Discord smoke test:

```bash
docker compose run --rm bot python -m bot.cli --config config/strategy.yaml discord-test
```

Coolify deployment notes are in `docs/coolify.md`.

## Notes

- Capitol Trades changes its frontend periodically, so the scraper is intentionally isolated in one module.
- The bot now resolves and filters raw Capitol Trades tickers before requesting prices, which is necessary because Capitol Trades symbols do not always map cleanly to free market-data providers.
- The benchmark is configured as ISIN `IE00B5BMR087` and resolved to `CSPX.L`, the USD Yahoo Finance listing of the iShares Core S&P 500 UCITS ETF. This keeps the benchmark aligned with the ISIN you provided while avoiding a currency mismatch versus the USD portfolio.
- `XSP` is explicitly filtered out because it is a Cboe Mini-SPX index options product, not a stock. Source: [Cboe XSP](https://www.cboe.com/tradable-products/sp-500/xsp-options)
- The intended VPS automation command is:
  `docker compose run --rm bot python -m bot.cli --config config/strategy.yaml paper-run`
- The daily run is idempotent per market date: same-day duplicate runs reuse the stored result and do not re-apply contributions or trades.
- Every `paper-run` writes a short message to `data/daily_message.txt` so the latest status can be delivered externally.
