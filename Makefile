COMPOSE ?= docker compose
BOT = $(COMPOSE) run --rm bot

.PHONY: test health paper-run dry-run backtest

test:
	$(COMPOSE) build bot
	$(BOT) python -m pytest

health:
	$(COMPOSE) up -d db
	$(BOT) python -m bot.cli --config config/strategy.yaml health-check

paper-run:
	$(COMPOSE) up -d db
	$(BOT) python -m bot.cli --config config/strategy.yaml paper-run

dry-run:
	$(COMPOSE) up -d db
	$(BOT) python -m bot.cli --config config/strategy.yaml --dry-run paper-run

backtest:
	$(BOT) python -m bot.cli --config config/strategy.yaml backtest
