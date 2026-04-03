"""
migrate_convlog_keys.py
=======================
One-shot migration: rename Redis conversation log keys from
  episodic:{user_code}:conversations
to
  convlog:{user_code}

Safe to run multiple times (skips keys that don't exist or are already migrated).

Usage (inside the jarvis-api container or with Redis accessible):
  python scripts/migrate_convlog_keys.py
"""

import sys
import os

# Allow running from repo root or from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jarvis-core", "src"))

from config import USERS, REDIS_URL
import redis

def main():
    r = redis.from_url(REDIS_URL, decode_responses=True)

    if not USERS:
        print("No users found in config — check USERS_LIST path.")
        sys.exit(1)

    migrated = 0
    skipped  = 0

    for user_code in USERS:
        old_key = f"episodic:{user_code}:conversations"
        new_key = f"convlog:{user_code}"

        if not r.exists(old_key):
            print(f"  [{user_code}] old key not found — skip")
            skipped += 1
            continue

        if r.exists(new_key):
            print(f"  [{user_code}] new key already exists — skip (no overwrite)")
            skipped += 1
            continue

        r.rename(old_key, new_key)
        count = r.zcard(new_key)
        print(f"  [{user_code}] migrated → convlog:{user_code} ({count} entries)")
        migrated += 1

    print(f"\nDone: {migrated} migrated, {skipped} skipped.")

if __name__ == "__main__":
    main()
