from __future__ import annotations

import os

import requests


def send_alert(subject: str, body: str) -> None:
    webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    requests.post(webhook_url, json={"text": f"{subject}\n{body}"}, timeout=10).raise_for_status()
