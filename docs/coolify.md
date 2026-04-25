# Coolify Deployment

This setup does not require a domain. Coolify only needs to run the Docker Compose app and schedule the bot command.

## Services

- `db`: Postgres state store
- `bot`: one-shot paper-trading container
- `data/`: mounted artifact directory for reports, CSVs, and charts

## Environment Variables

Set these in Coolify:

```text
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DISCORD_USER_ID=
```

`DISCORD_USER_ID` is optional. Discord mentions require the numeric user ID, not the visible handle `herold1989`.

## Scheduled Command

Run this once per market day after the US close:

```bash
docker compose run --rm bot python -m bot.cli --config config/strategy.yaml paper-run
```

For a no-write smoke test:

```bash
docker compose run --rm bot python -m bot.cli --config config/strategy.yaml --dry-run paper-run
```

## Outputs

The bot stores state in Postgres and writes artifacts under `data/`, including:

- `daily_message.txt`
- `run_manifest.json`
- `routine_status.json`
- `target_diff.csv`
- `portfolio_history.csv`
- `portfolio_chart.svg`

On a successful paper run, the bot posts the daily message to Discord. It also posts warning alerts for stale data, disclosure-feed warnings, provider errors, and trade-executed events.
