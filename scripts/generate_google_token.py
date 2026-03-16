#!/usr/bin/env python3
"""
Jarvis — One-time Google OAuth token generator
===============================================
Run this ONCE on the host machine (not inside Docker) to obtain
the refresh token needed by the Jarvis API container.

Requirements (host only, never added to the container):
    pip install google-auth-oauthlib

Usage:
    python3 generate_google_token.py

The script will open a browser window for Google sign-in, then print
the three values to add to /opt/jarvis/.env.

The client_secret.json file is NOT needed by the container — only the
three values printed at the end.
"""

import json
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
    "https://www.googleapis.com/auth/calendar.readonly",
]

CLIENT_SECRET_FILE = "scripts/client_secret.json"

print("Jarvis — Google OAuth token generator")
print("=" * 40)
print(f"Scopes: {', '.join(SCOPES)}")
print()

try:
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
except FileNotFoundError:
    print(f"ERROR: '{CLIENT_SECRET_FILE}' not found in the current directory.")
    print("Download it from Google Cloud Console > APIs & Services > Credentials.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print()
print("=" * 40)
print("SUCCESS — Add these lines to /opt/jarvis/.env:")
print("=" * 40)
print(f"GOOGLE_CLIENT_ID={creds.client_id}")
print(f"GOOGLE_CLIENT_SECRET={creds.client_secret}")
print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
print("GOOGLE_CALENDAR_ID=primary")
print()
print("The client_secret.json file is no longer needed after this step.")
print("Do NOT commit the .env file or client_secret.json to version control.")
