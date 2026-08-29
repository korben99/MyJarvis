#!/usr/bin/env python3
"""
Jarvis — One-time Google OAuth token generator
===============================================
Run this ONCE on the host machine (not inside Docker) to obtain
the refresh token needed by the Jarvis API container.

Requirements (host only, never added to the container):
    pip install google-auth-oauthlib

Usage:
    # First user / admin setup (outputs global GOOGLE_REFRESH_TOKEN)
    python3 generate_google_token.py

    # Per-user token (outputs GOOGLE_REFRESH_TOKEN_<CODE>)
    python3 generate_google_token.py --user ALICE1

The script will open a browser window for Google sign-in.
Sign in with the account of the user specified via --user.

The client_secret.json file is NOT needed by the container — only the
values printed at the end.
"""

import argparse
import sys

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("ERROR: google-auth-oauthlib not installed.")
    print("Run: pip install google-auth-oauthlib")
    sys.exit(1)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

CLIENT_SECRET_FILE = "/opt/jarvis/scripts/client_secret.json"

parser = argparse.ArgumentParser(description="Generate a Google OAuth refresh token for Jarvis.")
parser.add_argument("--user", metavar="CODE", default="",
                    help="User code (e.g. ALICE1). Outputs GOOGLE_REFRESH_TOKEN_<CODE>.")
args = parser.parse_args()

user_code = args.user.strip().upper()

print("Jarvis — Google OAuth token generator")
print("=" * 40)
if user_code:
    print(f"User: {user_code}")
print(f"Scopes: {', '.join(SCOPES)}")
print()

try:
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
except FileNotFoundError:
    print(f"ERROR: '{CLIENT_SECRET_FILE}' not found.")
    print("Download it from Google Cloud Console > APIs & Services > Credentials.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print()
print("=" * 40)
print("SUCCESS — Add to /opt/jarvis/.env:")
print("=" * 40)
if user_code:
    print(f"GOOGLE_REFRESH_TOKEN_{user_code}={creds.refresh_token}")
    print()
    print(f"Also ensure users_list.json has  \"google\": true  for {user_code}")
    print(f"and docker-compose.yml forwards  GOOGLE_REFRESH_TOKEN_{user_code}  to the container.")
else:
    print(f"GOOGLE_CLIENT_ID={creds.client_id}")
    print(f"GOOGLE_CLIENT_SECRET={creds.client_secret}")
    print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
    print("GOOGLE_CALENDAR_ID=primary")
print()
print("Do NOT commit .env or client_secret.json to version control.")
