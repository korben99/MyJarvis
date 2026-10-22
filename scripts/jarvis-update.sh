#!/usr/bin/env bash
# Jarvis updater — déploie le dernier tag poussé sur origin.
#
#   jarvis-update.sh             (défaut --check) compare le tag déployé au dernier tag
#                                distant et rend son verdict. N'écrit rien, ne redémarre
#                                rien. Code de sortie 10 si une mise à jour attend.
#   jarvis-update.sh --apply     déploie ce tag : install.sh, redémarrage, vérification
#                                de la version réellement en ligne, retour en arrière
#                                automatique si l'API ne revient pas.
#
# Le déclencheur est un TAG, jamais une branche : `main` porte des états de travail, et un
# updater branché dessus les livre tels quels. Côté mainteneur, publier une version est
# donc `git tag <version> && git push origin <version>`.
#
# Variables d'environnement :
#   JARVIS_URL              API à interroger (défaut http://localhost:8000)
#   JARVIS_UPDATE_TIMEOUT   secondes d'attente du retour en ligne (défaut 420)
#   JARVIS_TAG_PATTERN      motif des tags candidats (défaut *)
set -euo pipefail

JARVIS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$JARVIS_HOME"

JARVIS_URL="${JARVIS_URL:-http://localhost:8000}"
READY_TIMEOUT="${JARVIS_UPDATE_TIMEOUT:-420}"
TAG_PATTERN="${JARVIS_TAG_PATTERN:-*}"
LOCK_DIR="$JARVIS_HOME/logs/.update.lock"
SNAP_DIR="$JARVIS_HOME/logs/update"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  [ok] %s\n' "$1"; }
info() { printf '  [..] %s\n' "$1"; }
warn() { printf '  [!!] %s\n' "$1"; }

MODE="${1:---check}"
case "$MODE" in
    --check|--apply) ;;
    *) echo "usage: $(basename "$0") [--check|--apply]" >&2; exit 2 ;;
esac

# ── Inspection du dépôt ───────────────────────────────────────────────────────

command -v git >/dev/null 2>&1 || { warn "git introuvable"; exit 1; }
git -C "$JARVIS_HOME" rev-parse --git-dir >/dev/null 2>&1 \
    || { warn "$JARVIS_HOME n'est pas un dépôt git — mise à jour impossible"; exit 1; }

# Version en place : le tag exact si HEAD est posé dessus, sinon la description longue.
# C'est la même valeur que celle exposée par /status, à ceci près qu'elle décrit les
# fichiers sur disque quand /status décrit le code chargé en mémoire. L'écart entre les
# deux est exactement ce qu'on veut détecter.
version_disque() {
    git describe --tags --always --dirty 2>/dev/null || echo "inconnue"
}

# Une réponse de /status vaut readiness, et pas seulement « processus lancé » : le
# préchargement des modèles tourne dans le lifespan FastAPI, donc rien n'écoute avant sa fin.
api_repond() {
    curl -fsS --connect-timeout 2 --max-time 10 -o /dev/null "$JARVIS_URL/status" 2>/dev/null
}

# Version réellement en ligne, lue sur l'API. Le processus ne recharge pas à chaud : tant
# qu'il n'a pas redémarré, il annonce l'ancien tag même si le disque porte le nouveau.
# Rend une chaîne vide sur un code antérieur à l'exposition du champ.
version_en_ligne() {
    local corps
    corps="$(curl -fsS --connect-timeout 2 --max-time 10 "$JARVIS_URL/status" 2>/dev/null)" || return 1
    printf '%s' "$corps" | "$JARVIS_HOME/venv/bin/python" -c \
        'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null
}

DEPLOYEE="$(version_disque)"

info "récupération des tags depuis origin"
if ! git fetch --tags --prune --quiet origin 2>/dev/null; then
    warn "origin injoignable — vérifie le réseau et l'accès au dépôt"
    exit 1
fi

# Tri par version et non par date de création : `git tag <nom>` produit un tag léger, qui
# n'a pas de date propre — son « creatordate » est celle du commit pointé, donc tagger un
# vieux commit le classerait en queue de liste.
CIBLE="$(git tag --list "$TAG_PATTERN" --sort=-v:refname | head -n1)"
if [[ -z "$CIBLE" ]]; then
    warn "aucun tag ne correspond à « $TAG_PATTERN » — rien à déployer"
    exit 1
fi

bold "Jarvis update — $JARVIS_HOME"
ok "version déployée : $DEPLOYEE"
ok "dernier tag       : $CIBLE"

EN_LIGNE="$(version_en_ligne || true)"
if [[ -n "$EN_LIGNE" && "$EN_LIGNE" != "$DEPLOYEE" ]]; then
    warn "l'API annonce « $EN_LIGNE » alors que le disque porte « $DEPLOYEE » — redémarrage en attente"
fi

# Comparaison sur le commit et non sur le nom : un tag déplacé sur le même commit n'est pas
# une mise à jour, et HEAD détaché sur le bon commit non plus.
SHA_CIBLE="$(git rev-parse "$CIBLE^{commit}")"
SHA_HEAD="$(git rev-parse HEAD)"
if [[ "$SHA_CIBLE" == "$SHA_HEAD" ]]; then
    ok "déjà à jour"
    exit 0
fi

# ── Pré-vol : ce que la mise à jour va remuer ─────────────────────────────────

bold "Contenu de la mise à jour"
git log --oneline --no-decorate "$SHA_HEAD..$SHA_CIBLE" 2>/dev/null | head -n 20 \
    || info "historique non linéaire — voir git log $CIBLE"

CHANGES="$(git diff --name-only "$SHA_HEAD" "$SHA_CIBLE" 2>/dev/null || true)"

if grep -q '^requirements.txt$' <<<"$CHANGES"; then
    warn "requirements.txt change — le venv va bouger, et un retour en arrière du code ne le rétablit pas"
fi

if grep -q '^docker-compose.yml$' <<<"$CHANGES"; then
    warn "docker-compose.yml change — relire avant de recréer les conteneurs"
fi

MIGRATIONS="$(grep '^scripts/migrate_' <<<"$CHANGES" || true)"
if [[ -n "$MIGRATIONS" ]]; then
    warn "script(s) de migration modifié(s), à lancer à la main après le déploiement :"
    printf '       %s\n' $MIGRATIONS
fi

# Clés de configuration nouvelles. install.sh ne touche jamais à un .env existant : sans ce
# relevé, une clé ajoutée en amont reste absente et `config.py` applique son défaut en
# silence — la fonctionnalité livrée ne s'allume donc jamais, sans la moindre erreur.
# Le .env.example lu est celui de la CIBLE, pas celui du disque : c'est un pré-vol.
if [[ -f .env ]]; then
    cles_env() { grep -oE '^#? *[A-Z][A-Z0-9_]*=' | tr -d '# ' | tr -d '='; }
    NOUVELLES="$(comm -13 \
        <(cles_env < .env | sort -u) \
        <(git show "$CIBLE:.env.example" 2>/dev/null | cles_env | sort -u))"
    if [[ -n "$NOUVELLES" ]]; then
        NB_NOUVELLES="$(grep -c . <<<"$NOUVELLES")"
        warn "$NB_NOUVELLES clé(s) absente(s) de ton .env (défaut appliqué en silence si tu ne les poses pas) :"
        printf '       %s\n' $(head -n 20 <<<"$NOUVELLES")
        if (( NB_NOUVELLES > 20 )); then
            info "… et $((NB_NOUVELLES - 20)) autres — liste complète : git show $CIBLE:.env.example"
        fi
    fi
fi

if [[ "$MODE" == "--check" ]]; then
    printf '\n'
    info "pour déployer : $(basename "$0") --apply"
    exit 10
fi

# ── Déploiement ───────────────────────────────────────────────────────────────

# Verrou par mkdir : atomique, et un updater lancé par cron ne doit pas pouvoir doubler
# celui qui recharge déjà le service. Le dossier parent est créé à part — `mkdir` échoue
# aussi sur un parent absent, ce qui ferait annoncer un déploiement en cours sur un dépôt
# fraîchement cloné, où logs/ n'existe pas encore.
mkdir -p "$(dirname "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    warn "une mise à jour est déjà en cours ($LOCK_DIR) — supprimer ce dossier si ce n'est pas le cas"
    exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

# Un arbre sale fait échouer le checkout à mi-chemin, ou emporte un correctif local non
# commité. On s'arrête avant d'avoir rien touché. Les fichiers d'exploitation (.env, clés,
# données) sont gitignorés : ils ne comptent pas comme sale et survivent au déploiement.
if [[ -n "$(git status --porcelain)" ]]; then
    warn "arbre de travail modifié — commite, remise ou annule avant de déployer :"
    git status --short | head -n 20
    exit 1
fi

mkdir -p "$SNAP_DIR"
HORO="$(date +%Y%m%d_%H%M%S)"
FREEZE="$SNAP_DIR/$HORO-pip-freeze.txt"
RETOUR="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"

bold "1/5 Point de retour"
"$JARVIS_HOME/venv/bin/python" -m pip freeze > "$FREEZE" 2>/dev/null \
    && ok "venv relevé dans $FREEZE" \
    || warn "relevé du venv impossible — retour en arrière des dépendances non couvert"
ok "code réversible vers $RETOUR"

bold "2/5 Bascule sur $CIBLE"
git -c advice.detachedHead=false checkout --quiet "$CIBLE"
ok "disque sur $(version_disque)"

bold "3/5 install.sh"
JARVIS_NONINTERACTIVE=1 bash "$JARVIS_HOME/install.sh"

bold "4/5 Redémarrage"
"$JARVIS_HOME/scripts/jarvis-launchd.sh" restart
info "chargement du modèle en cours — jusqu'à ${READY_TIMEOUT}s"

attendre_api() {
    local fin=$((SECONDS + READY_TIMEOUT))
    while (( SECONDS < fin )); do
        api_repond && return 0
        sleep 5
    done
    return 1
}

bold "5/5 Vérification"
ATTENDUE="$(version_disque)"
ECHEC=""
if ! attendre_api; then
    ECHEC="l'API n'est pas revenue en ${READY_TIMEOUT}s"
else
    VUE="$(version_en_ligne || true)"
    if [[ -z "$VUE" ]]; then
        # Le champ n'existe pas avant la version qui l'introduit : sur un tag antérieur, le
        # retour en ligne est la seule preuve disponible.
        warn "version non exposée par ce tag — vérification limitée au retour en ligne"
        ok "en ligne"
    elif [[ "$VUE" == "$ATTENDUE" ]]; then
        ok "en ligne en version $VUE"
    else
        ECHEC="l'API annonce « $VUE » au lieu de « $ATTENDUE » — redémarrage sans effet"
    fi
fi

if [[ -z "$ECHEC" ]]; then
    [[ -n "$MIGRATIONS" ]] && warn "pense aux migrations listées plus haut"
    exit 0
fi

# ── Retour en arrière ─────────────────────────────────────────────────────────

warn "$ECHEC"
warn "retour sur $RETOUR"

git -c advice.detachedHead=false checkout --quiet "$RETOUR"
JARVIS_NONINTERACTIVE=1 bash "$JARVIS_HOME/install.sh" || warn "install.sh a échoué pendant le retour"
"$JARVIS_HOME/scripts/jarvis-launchd.sh" restart

if attendre_api; then
    ok "revenu en ligne en version $(version_en_ligne || echo inconnue)"
else
    warn "l'API ne revient pas non plus sur $RETOUR — voir logs/ et ./jarvis-status.sh"
fi

# Le retour du code ne défait pas un pip install : si la mise à jour a monté une dépendance,
# c'est ce relevé qui la redescend.
warn "dépendances d'avant la mise à jour : $FREEZE"
warn "   restauration : ./venv/bin/python -m pip install -r $FREEZE"
exit 1
