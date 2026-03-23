# Jarvis — Google OAuth Setup

Ce guide explique comment connecter un compte Google (Gmail + Agenda) à Jarvis.
Chaque utilisateur effectue la procédure **une seule fois**.

---

## Codes utilisateur

| Utilisateur | Code     | Email                            |
|-------------|----------|----------------------------------|
| Sébastien   | KORBEN99 | sebastien.viou@gmail.com         |
| Hélène      | AQWZSX   | helene.viou@gmail.com            |
| Mathilde    | ZSXEDC   | mathilde.rachelle.viou@gmail.com |

---

## Prérequis

- Être physiquement devant la machine qui héberge Jarvis (le script ouvre un navigateur en local)
- `google-auth-oauthlib` installé sur le host :
  ```bash
  pip install google-auth-oauthlib
  ```
- Le fichier `client_secret.json` présent dans `/opt/jarvis/scripts/`
  (téléchargeable depuis Google Cloud Console → APIs & Services → Credentials → ton OAuth Client ID → "Download JSON")

---

## Procédure

### Étape 1 — Lancer le script

Sur le **host** , exécute :

```bash
python3 /opt/jarvis/scripts/generate_google_token.py --user AQWZSX
```

Remplace `AQWZSX` par le code de l'utilisatrice concernée.

### Étape 2 — Autoriser l'accès dans le navigateur

1. Un navigateur s'ouvre automatiquement sur la page Google.
2. **Connecte-toi avec le compte Google de l'utilisatrice** (helene.viou@gmail.com ou mathilde.rachelle.viou@gmail.com).
3. Accepte toutes les permissions demandées (lecture Gmail, envoi Gmail, Agenda).
4. La fenêtre se ferme toute seule. Le terminal affiche le token.

### Étape 3 — Copier le token dans `.env`

Le script affiche quelque chose comme :

```
SUCCESS — Add to /opt/jarvis/.env:
========================================
GOOGLE_REFRESH_TOKEN_AQWZSX=1//04xXXXXXXXXXXXXXXXXXXXXX...
```

Ouvre `/opt/jarvis/.env` et décommente / remplis la ligne correspondante :

```bash
# Avant :
# GOOGLE_REFRESH_TOKEN_AQWZSX=<token for Hélène — generated via GOOGLE.md procedure>

# Après :
GOOGLE_REFRESH_TOKEN_AQWZSX=1//04xXXXXXXXXXXXXXXXXXXXXX...
```

### Étape 4 — Activer dans `users_list.json`

Dans `/opt/jarvis/jarvis-core/JarvisData/users_list.json`, ajoute `"google": true` à l'entrée de l'utilisatrice :

```json
{
  "code": "AQWZSX",
  ...
  "briefing_enabled": true,
  "trading": false,
  "google": true
}
```

### Étape 5 — Redémarrer et vérifier

```bash
cd /opt/jarvis
docker compose restart jarvis-api
docker logs jarvis-api --tail 30
```

Dans les logs tu dois voir :
```
Google access token refreshed for AQWZSX
```

Et dans le briefing du lendemain matin, l'agenda et les emails d'Hélène seront les siens.

---

## Sécurité

- Le token est stocké **uniquement** dans `.env` sur le host, jamais dans le code ni dans l'image Docker.
- Chaque utilisateur accède exclusivement à **son propre** Gmail et Agenda — aucun accès croisé possible.
- Pour révoquer l'accès : [myaccount.google.com/permissions](https://myaccount.google.com/permissions) → "Jarvis" → "Supprimer l'accès".
- Ne jamais committer `.env` ou `client_secret.json` dans git.
