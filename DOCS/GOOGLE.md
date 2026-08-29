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

- Be physically at the machine hosting Jarvis — the script opens a local browser
- `google-auth-oauthlib` installed on the host:
  ```bash
  pip install google-auth-oauthlib
  ```
- `client_secret.json` present in `/opt/jarvis/scripts/`, downloaded from Google Cloud
  Console → APIs & Services → Credentials → your OAuth Client ID → *Download JSON*

---

## Procedure

### Step 1 — Run the script

On the **host**:

```bash
python3 /opt/jarvis/scripts/generate_google_token.py --user BOB2
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

### Step 5 — Restart and verify

The Jarvis API runs **natively under launchd**, not as a Docker container — restarting the
`jarvis-api` container will not pick up the new token, because there is no such container.

```bash
jarvis-restart
tail -30 /opt/jarvis/logs/jarvis-service.log
```

You should see:

```
Google access token refreshed for BOB2
```

And in the next morning's briefing, that user's calendar and mail will be their own.

---

## Security

- The token is stored **only** in `.env` on the host — never in the code, never in a Docker
  image.
- Each user reaches exclusively **their own** Gmail and Calendar; no cross-access is
  possible.
- To revoke: [myaccount.google.com/permissions](https://myaccount.google.com/permissions) →
  *Jarvis* → *Remove access*.
- Never commit `.env` or `client_secret.json`.
