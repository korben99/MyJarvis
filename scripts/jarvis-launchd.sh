#!/usr/bin/env bash
# Jarvis launchd manager — 100% idempotent.
#
#   jarvis-launchd.sh install   converge le plist (l'écrit seulement si changé, ne charge pas)
#   jarvis-launchd.sh start     charge le service s'il n'est pas déjà chargé
#   jarvis-launchd.sh stop      décharge le service s'il est chargé
#   jarvis-launchd.sh restart   bootout + bootstrap (ne casse pas si non chargé)
#   jarvis-launchd.sh status    état du service
#   jarvis-launchd.sh uninstall bootout + suppression du plist
set -euo pipefail

JARVIS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LABEL="com.jarvis.api"
UID_NUM="$(id -u)"
DOMAIN="gui/$UID_NUM"
SERVICE="$DOMAIN/$LABEL"
TEMPLATE="$JARVIS_HOME/DOCS/examples/com.jarvis.api.plist.template"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

TMP_PLIST=""
cleanup() { [[ -n "$TMP_PLIST" ]] && rm -f "$TMP_PLIST"; return 0; }
trap cleanup EXIT

is_loaded() { launchctl print "$SERVICE" >/dev/null 2>&1; }

# bootout par LABEL et non par chemin : la forme chemin échoue (code 5) dès que le fichier
# a disparu, ce qui bloquait uninstall avant même son rm.
unload() { launchctl bootout "$SERVICE" >/dev/null 2>&1 || true; }

render_plist() {
    local extra_path=""
    [[ -d "/Applications/OrbStack.app" ]] && extra_path="$HOME/.orbstack/bin:"
    sed -e "s#__JARVIS_HOME__#$JARVIS_HOME#g" \
        -e "s#__EXTRA_PATH__#$extra_path#g" \
        "$TEMPLATE"
}

cmd_install() {
    mkdir -p "$(dirname "$PLIST_DEST")"
    TMP_PLIST="$(mktemp)"
    render_plist > "$TMP_PLIST"

    # Validation avant écriture : ce fichier pilote un service qui tourne. Un template
    # cassé ou un rendu sed foireux ne doit jamais remplacer un plist fonctionnel.
    if ! plutil -lint "$TMP_PLIST" >/dev/null 2>&1; then
        echo "erreur — le plist rendu est invalide, $PLIST_DEST laissé intact" >&2
        exit 1
    fi

    if diff -q "$TMP_PLIST" "$PLIST_DEST" >/dev/null 2>&1; then
        echo "ok — $PLIST_DEST déjà à jour"
        return
    fi

    # Sauvegarde de la version en place : install écrase désormais sans demander (le
    # garde « ne touche pas si le fichier existe » a été retiré d'install.sh), et un plist
    # ajusté à la main est vite perdu.
    if [[ -f "$PLIST_DEST" ]]; then
        cp "$PLIST_DEST" "$PLIST_DEST.bak"
        echo "sauvegarde — $PLIST_DEST.bak"
    fi
    cp "$TMP_PLIST" "$PLIST_DEST"
    chmod 644 "$PLIST_DEST"
    echo "installé — $PLIST_DEST mis à jour"
    if is_loaded; then
        echo "      le service tourne encore sur l'ancienne config → lancez : jarvis-restart"
    fi
}

require_plist() {
    if [[ ! -f "$PLIST_DEST" ]]; then
        echo "erreur — $PLIST_DEST absent, lancez d'abord : jarvis-install" >&2
        exit 1
    fi
}

cmd_start() {
    require_plist
    if is_loaded; then
        echo "ok — $LABEL déjà chargé"
    else
        launchctl bootstrap "$DOMAIN" "$PLIST_DEST"
        echo "démarré — $LABEL"
    fi
}

cmd_stop() {
    if is_loaded; then
        unload
        echo "arrêté — $LABEL"
    else
        echo "ok — $LABEL déjà arrêté"
    fi
}

# bootout rend parfois la main avant que launchd ait réellement libéré le label ; un
# bootstrap immédiat échoue alors avec le code 5 « déjà chargé » — exactement l'erreur
# que ce script est censé faire disparaître. On attend la libération effective.
# (kickstart -k serait atomique mais NE RELIT PAS le plist : inutilisable après install.)
wait_unloaded() {
    local i=0
    while is_loaded && (( i < 50 )); do
        sleep 0.1
        i=$((i + 1))
    done
    if is_loaded; then
        echo "erreur — $LABEL toujours chargé après 5s, abandon" >&2
        exit 1
    fi
}

cmd_restart() {
    require_plist
    if is_loaded; then
        unload
        wait_unloaded
    fi
    launchctl bootstrap "$DOMAIN" "$PLIST_DEST"
    echo "redémarré — $LABEL"
}

cmd_status() {
    if is_loaded; then
        echo "chargé — $LABEL ($PLIST_DEST)"
        launchctl print "$SERVICE" | grep -E '^\s*state =' | sed 's/^/  /'
    else
        echo "non chargé — $LABEL"
    fi
}

cmd_uninstall() {
    if is_loaded; then
        unload
        echo "déchargé — $LABEL"
    else
        echo "ok — $LABEL déjà déchargé"
    fi
    rm -f "$PLIST_DEST"
    echo "supprimé — $PLIST_DEST"
}

# Les lignes d'aide sont marquées par le préfixe "#   " : pas de plage de lignes en dur,
# qui se désynchronise au premier ajout dans l'en-tête.
usage() {
    grep -E '^#   ' "$0" | sed 's/^#   //'
}

case "${1:-}" in
    install)   cmd_install ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    uninstall) cmd_uninstall ;;
    help|-h|--help|"") usage ;;
    *) echo "sous-commande inconnue : $1" >&2; usage >&2; exit 1 ;;
esac
