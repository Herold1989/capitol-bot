from __future__ import annotations

import os

import requests


def send_alert(subject: str, body: str, raise_on_error: bool = False) -> bool:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("ALERT_WEBHOOK_URL")
    if not webhook_url:
        if raise_on_error:
            raise RuntimeError("DISCORD_WEBHOOK_URL or ALERT_WEBHOOK_URL is not set")
        return False
    mention = ""
    user_id = os.environ.get("DISCORD_USER_ID")
    if user_id:
        mention = f"<@{user_id}> "
    payload = {
        "content": f"{mention}**{subject}**\n{body}",
        "allowed_mentions": {"users": [user_id]} if user_id else {"parse": []},
    }
    try:
        requests.post(webhook_url, json=payload, timeout=10).raise_for_status()
        return True
    except requests.RequestException:
        # Alert delivery must never fail the trading run or DB write path.
        if raise_on_error:
            raise
        return False
