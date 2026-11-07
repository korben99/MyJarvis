# Google OAuth setup

This guide connects a Google account (Gmail + Calendar) to Jarvis. Each user goes through
the procedure **once**.

---

## User codes

The codes below are the placeholders used throughout this documentation. Yours are whatever
you put in `users_list.json`.

| User  | Code   | Email             |
|-------|--------|-------------------|
| Alice | ALICE1 | alice@example.com |
| Bob   | BOB2   | bob@example.com   |
| Carol | CAROL1 | carol@example.com |

---

## Prerequisites

- Be physically at the machine hosting Jarvis — the script opens a local browser on an
  ephemeral port
- `google-auth-oauthlib` available to the interpreter you run the script with. It is in
  `/opt/jarvis/venv`, not in the system `python3` — hence the venv interpreter in step 1.
- `client_secret.json` present in `/opt/jarvis/scripts/`, downloaded from Google Cloud
  Console → APIs & Services → Credentials → your OAuth Client ID → *Download JSON*
- Nothing to prepare on the user's own Google account: the publishing status lives in the
  Cloud project, not in their settings. They only have to click through the *Google hasn't
  verified this app* screen (*Advanced* → *Go to … (unsafe)*), which an unverified app
  shows for good.
- Google Cloud Console → *Audience* → *Publishing status* must read **In production**. In
  *Testing*, external accounts must additionally be listed as *Test users*, and their
  refresh tokens expire **every 7 days** — the connection then breaks weekly for no
  visible reason. A token generated before the switch to production keeps that expiry, so
  switch first, generate after.

---

## Procedure

### Step 1 — Run the script

On the **host**, with the venv interpreter:

```bash
cd /opt/jarvis
./venv/bin/python scripts/generate_google_token.py --user BOB2
```

Replace `BOB2` with the code of the user concerned.

### Step 2 — Authorise in the browser

1. A browser opens on the Google consent page.
2. **Sign in with that user's Google account** (`bob@example.com`).
3. Accept every requested permission (Gmail read, Gmail send, Calendar).
4. The window closes on its own. The terminal prints the token.

### Step 3 — Copy the token into `.env`

The script prints something like:

```
SUCCESS — Add to /opt/jarvis/.env:
========================================
GOOGLE_REFRESH_TOKEN_BOB2=1//04xXXXXXXXXXXXXXXXXXXXXX...
```

Open `/opt/jarvis/.env` and uncomment / fill in the matching line:

```bash
# Before:
# GOOGLE_REFRESH_TOKEN_BOB2=<token for Bob — generated via GOOGLE.md procedure>

# After:
GOOGLE_REFRESH_TOKEN_BOB2=1//04xXXXXXXXXXXXXXXXXXXXXX...
```

### Step 4 — Enable in `users_list.json`

In `/opt/jarvis/jarvis-core/JarvisData/users_list.json`, add `"google": true` to that user's
entry:

```json
{
  "code": "BOB2",
  ...
  "briefing_enabled": true,
  "trading": false,
  "google": true
}
```

Do steps 3 and 4 **before** restarting. `config.py` builds `GOOGLE_USER_TOKENS` by reading
`GOOGLE_REFRESH_TOKEN_<CODE>` only for the codes already in `USER_GOOGLE`: the token
without the flag loads nothing, and the flag without the token loads nothing either. A
restart between the two costs a second model load for no result.

### Step 5 — Restart and verify

The Jarvis API runs **natively under launchd**, not as a Docker container — restarting the
`jarvis-api` container will not pick up the new token, because there is no such container.

```bash
jarvis-restart
```

**A restart alone proves nothing.** `google_services._get_credentials` refreshes lazily, on
the first Gmail or Calendar call for that user — startup logs stay silent on the subject,
whether the token works or not. The check requires provoking a call.

Lightest way: have the user ask Jarvis a mail question **from their own account** (Open
WebUI signed in with their address, or the iOS app with their code), then read the journal:

```bash
grep BOB2 /opt/jarvis/logs/jarvis-service.log | tail -5
```

| Line | Meaning |
|------|---------|
| `Google access token refreshed for BOB2` | Connected. |
| `Google credential refresh failed for BOB2: RefreshError` | Token invalid or a scope declined at the consent screen — redo step 1. |
| nothing | No call reached Google. Either the question was not routed to Gmail, or the code is absent from `USER_GOOGLE` (step 4 missing, or no restart since). |

End-to-end check covering Calendar **and** Gmail in one call:

```bash
curl -s -X POST http://localhost:8000/briefing/generate/BOB2 \
  -H "Authorization: Bearer BOB2" | head -c 400
```

This endpoint does not only generate, it **delivers**: the user gets the briefing by mail
and as a push notification. Off-hours, prefer the chat question.

From there on, the morning briefing carries that user's own calendar and mail.

---

## Security

- The token is stored **only** in `.env` on the host — never in the code, never in a Docker
  image.
- Each user reaches exclusively **their own** Gmail and Calendar; no cross-access is
  possible.
- To revoke: [myaccount.google.com/permissions](https://myaccount.google.com/permissions) →
  *Jarvis* → *Remove access*.
- Never commit `.env` or `client_secret.json`.
