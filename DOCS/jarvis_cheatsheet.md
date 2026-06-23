# Project Jarvis — Cheat Sheet

## SSH & Access

```bash
ssh jarvis@192.168.1.XXX                  # SSH into mini PC
```

| Service | URL |
|---------|-----|
| Open WebUI | http://192.168.1.XXX:3000 |
| Qdrant dashboard | http://192.168.1.XXX:6333/dashboard |
| Jarvis API | http://192.168.1.XXX:8000 |
| Tailscale (anywhere) | http://100.64.x.x:3000 |

## CRON

crontab -l
crontab -e

Automatique save: /usr/local/bin/backup-jarvis-rsync.sh
Automatique SMB mounting:
Automatique document raging: 
---

korben@jarvis ~ % more .zshrc 
alias tailscale="/Applications/Tailscale.app/Contents/MacOS/Tailscale"

alias jarvis-stop='launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist'
alias jarvis-start='launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist'
alias jarvis-reload='launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist; launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist'
korben@jarvis ~ % 


## Docker — Daily Commands

```bash
cd /opt/jarvis

docker compose up -d                      # Start all services
docker compose down                       # Stop all services
docker compose restart open-webui         # Restart one service
docker compose restart jarvis-api         # Restart API after editing main.py
docker compose ps                         # Check running services
docker compose logs -f jarvis-webui       # Follow logs (Ctrl+C to exit)
docker compose logs -f jarvis-api         # Follow API logs
```

## Docker — Updates
```bash
docker compose pull                       # Pull latest images
docker compose up -d                      # Restart with new images
docker image prune -f                     # Clean old images
```

## Docker — Rebuild (after Dockerfile changes)
```bash
docker compose up -d --build jarvis-api   # Rebuild + restart API
```
---

## Jarvis Status
```bash
/opt/jarvis/jarvis-status.sh              # Full status check
```
---

## RAG — Document Indexing + SMB SHARE

```bash
source /opt/jarvis/venv/bin/activate      # Activate Python venv

# Index all documents (first time or full re-index)
python3 /opt/jarvis/scripts/upload-to-openwebui.py

# Search (no LLM needed, free)
python3 /opt/jarvis/scripts/search-qdrant.py "your query here"


How it works:
  1. Drop files into the SMB share from any computer
  2. Every 15 minutes, the cron job checks for new files
  3. New files are automatically chunked, embedded, and indexed in Qdrant
  4. Next time you ask Jarvis a question, it searches the new content

Manual commands:
  /opt/jarvis/scripts/cron-index.sh          # Force index now
  crontab -l                                   # View cron jobs
  tail -f /opt/jarvis/logs/cron-index.log     # Watch indexing log

Connect from Mac:
  Finder → Go → Connect to Server
  smb://192.168.1.53/JarvisDocuments
  User: jarvis / Password: (what you just set)

Connect from Windows:
  Explorer → \\192.168.1.53\JarvisDocuments

```
## RAG — From your Mac

```bash
# Sync documents to mini PC
rsync -avz --progress ~/JarvisDocuments/ jarvis@192.168.1.XXX:/opt/jarvis/data/

# Or add aliases to ~/.zshrc:
alias jarvis-sync='rsync -avz --progress ~/JarvisDocuments/ jarvis@192.168.1.XXX:/opt/jarvis/data/'
alias jarvis-index='ssh jarvis@192.168.1.XXX "/opt/jarvis/venv/bin/python3 /opt/jarvis/scripts/index-documents.py"'
alias jarvis-update='jarvis-sync && jarvis-index'
```

## RAG — Qdrant

```bash
# Check collection stats
curl -s http://localhost:6333/collections/open-webui_knowledge | python3 -m json.tool

# List all collections
curl -s http://localhost:6333/collections | python3 -m json.tool

# Delete a collection (careful!)
curl -X DELETE http://localhost:6333/collections/jarvis-knowledge
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `/opt/jarvis/.env` | API keys, model config |
| `/opt/jarvis/docker-compose.yml` | All services |
| `/opt/jarvis/jarvis-core/main.py` | Jarvis API code (mounted, restart to apply) |
| `/opt/jarvis/jarvis-status.sh` | Status check script |
| `/opt/jarvis/scripts/index-documents.py` | Full document indexer |
| `/opt/jarvis/scripts/index-new.py` | Incremental indexer |
| `/opt/jarvis/scripts/search-qdrant.py` | Search test tool |
| `/opt/jarvis/scripts/reindex.sh` | Wipe + rebuild index |
| `/opt/jarvis/data/` | Your documents |
| `/opt/jarvis/logs/` | Logs |

## Edit Config

```bash
nano /opt/jarvis/.env                     # API keys, model
nano /opt/jarvis/docker-compose.yml       # Services
nano /opt/jarvis/jarvis-core/main.py      # API code
```

After editing `.env` or docker-compose:

```bash
docker compose up -d                      # Recreates affected services
```

After editing main.py:

```bash
docker compose restart jarvis-api         # Just restart, no rebuild
```

---
## DEBUG JARVIS-API
docker compose logs -f jarvis-api
docker exec -it jarvis-api
KEYS "*"
KEYS user:*
KEYS chat:*
KEYS episodic:*
KEYS jarvis:*
LRANGE chat:KORBEN99:iphone-main 0 -1
HGETALL user:KORBEN99:profile
ZRANGE episodic:KORBEN99:conversations 0 -1
---

## DEBUG JARVIS-REDIS
docker exec -it jarvis-redis redis-cli

## View what Jarvis knows about you
curl http://localhost:8000/memory/profile | python3 -m json.tool

# View emotional state
curl http://localhost:8000/memory/emotional-state

# View recent conversations
curl "http://localhost:8000/memory/recent?hours=24"

# View Jarvis self-knowledge
curl http://localhost:8000/memory/self

# View today's reflection
cat /opt/jarvis/data/reflections/*.json | python3 -m json.tool

# Reset all memory (careful!)
curl -X DELETE http://localhost:8000/memory/reset

# Force nightly reflection
source /opt/jarvis/.env
OPENAI_API_KEY=$OPENAI_API_KEY /opt/jarvis/venv/bin/python3 /opt/jarvis/scripts/nightly-reflection.py

# View reflection log
tail -f /opt/jarvis/logs/reflection.log

## Costs

| Item | Cost |
|------|------|
| OpenAI gpt-4o-mini | ~$5-15/month |
| OpenAI gpt-4o (complex queries) | ~$0.01-0.03/message |
| Claude API (optional) | ~$10-30/month |
| RunPod A40 (when available) | $0.39/hr |
| N150 electricity | ~4€/month |
| Mac Studio M2 Ultra (future) | ~3,000€ one-time, then $0 |



  Étapes                                                                                                                                                                                                                                                                              
  1. Préparer le bundle ID (une seule fois)
                                                                                                                                                                                                                                                                             
  Dans le Developer Portal (https://developer.apple.com) → Identifiers → + → App ID → enregistre ton bundle ID (ex. com.viou.jarvis). Si l'app en a déjà un, rien à changer.                                                                                                 
   
  2. Créer l'app dans App Store Connect                                                                                                                                                                                                                                      
                  
  → https://appstoreconnect.apple.com → Mes apps → + → Nouvelle app → renseigne le bundle ID. Pas besoin de la soumettre à l'App Store.                                                                                                                                      
   
  3. Archiver dans Xcode                                                                                                                                                                                                                                                     
                  
  Product → Archive                                                                                                                                                                                                                                                          
  Une fois l'archive créée → Organizer s'ouvre automatiquement.
                                                                                                                                                                                                                                                                             
  4. Uploader vers App Store Connect
                                                                                                                                                                                                                                                                             
  Dans l'Organizer → Distribute App → App Store Connect → Upload.                                                                                                                                                                                                            
  Xcode gère la signature automatiquement si tu as le bon compte connecté (Xcode → Settings → Accounts).
                                                                                                                                                                                                                                                                             
  5. Activer TestFlight                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                             
  Dans App Store Connect → ton app → TestFlight → attends 5-10 min que le build soit traité → Tests internes → ajoute-toi comme testeur (et les membres de la famille si besoin).                                                                                            
                  
  6. Installer sur l'iPhone                                                                                                                                                                                                                                                  
                  
  - Installe l'app TestFlight depuis l'App Store                                                                                                                                                                                                                             
  - Accepte l'invitation par email, ou utilise le lien de test interne
  - Installe directement depuis TestFlight — Developer Mode non requis    
  
Pour la famille (Hélène, Mathilde)
                                                                                                                                                                                                                                                                             
  Dans TestFlight → Tests externes → ajoute leurs emails. Elles installent TestFlight, acceptent l'invitation, c'est tout. Aucune manipulation technique de leur côté.



  Checklist APNs — étapes manuelles                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                             
  1. Apple Developer Portal (5 min)                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                             
  1. Va sur https://developer.apple.com → Certificates, Identifiers & Profiles → Keys                                                                                                                                                                                        
  2. + → Nom : "Jarvis APNs" → coche Apple Push Notifications service (APNs) → Continue → Register                                                                                                                                                                           
  3. Download → copie le fichier .p8 dans /opt/jarvis/apns_key.p8    
44KTB6547P                                                                                                                                                                                                        
  4. chmod 600 /opt/jarvis/jarvis_apn_key.p8                                                                                                                                                                                                                                       
  5. Note le Key ID (10 chars, ex: AB12CD34EF) et ton Team ID (visible en haut à droite du portail)                                                                                                                                                                          
                                                                                                                                                                                                                                                                             
  2. Fichier .env                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                             
  Décommente et remplis le bloc APNs :                                                                                                                                                                                                                                       
  APNS_KEY_ID=AB12CD34EF          # ton Key ID
  APNS_TEAM_ID=XXXXXXXXXX         # ton Team ID
  APNS_BUNDLE_ID=com.sebastienviou.JarvisApp                                                                                                                                                                                                                                 
  APNS_KEY_PATH=/opt/jarvis/apns_key.p8     
  APNS_ENV=production                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                             
  3. Xcode (2 clics)                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                             
  1. Sélectionne la target JarvisApp → onglet Signing & Capabilities                                                                                                                                                                                                         
  2. + → cherche Push Notifications → double-clic
                                                                                                                                                                                                                                                                             
  C'est tout — le signing automatique génère l'entitlement aps-environment dans le provisioning profile.                                                                                                                                                                     
                                                                                                                                                                                                                                                                             
  4. Dépendance serveur                                                                                                                                                                                                                                                      
                  
  pip install "httpx[http2]"
  (ou ajoute-le dans requirements.txt si tu en as un)                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                             
  5. Redémarre jarvis-core                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                             
  Après avoir rempli .env, redémarre le serveur pour charger les nouvelles variables.                                                                                                                                                                                        
  
  ---                                                                                                                                                                                                                                                                        
  Après ça, au premier lancement de l'app depuis TestFlight, iOS demandera la permission de notifications → AppDelegate reçoit le token → NotificationService l'envoie au backend → queue_push l'utilise pour les APNs en direct, avec le polling Redis en fallback.
