
# ── Redis helpers ──────────────────────────────────────────────────────────
#pour self.py & trading.py


def idx_key(user_code: str) -> str:
    return f"trade:{user_code}:index"


def pos_key(user_code: str, isin: str) -> str:
    return f"trade:{user_code}:pos:{isin}"


def import_ts_key(user_code: str) -> str:
    return f"trade:{user_code}:last_import_ts"


def price_cache_key(isin: str) -> str:
    return f"trade:price_cache:{isin}"


def alert_queue_key(user_code: str) -> str:
    return f"trade:{user_code}:pending_alerts"