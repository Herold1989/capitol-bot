# Coolify Deployment

This setup does not require a domain. Coolify only needs to run the Docker Compose app and schedule the bot command.

## Services

- `db`: Postgres state store
- `bot`: one-shot paper-trading container
- `data/`: mounted artifact directory for reports, CSVs, and charts

The image includes a fallback config at `/app/default_config/strategy.yaml`. If Coolify creates an empty config volume at `/app/config`, the bot will still start using that fallback. If you want to override strategy settings in production, mount a real `strategy.yaml` at `/app/config/strategy.yaml`.

## Environment Variables

Set these in Coolify:

```text
POSTGRES_DB=capitol_bot
POSTGRES_USER=capitol_bot
POSTGRES_PASSWORD=<long random password>
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DISCORD_USER_ID=
```

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
- `portfolio_chart.png`

On a successful paper run, the bot posts the daily message and `portfolio_chart.png` to Discord. It also posts warning alerts for stale data, disclosure-feed warnings, provider errors, and trade-executed events.
