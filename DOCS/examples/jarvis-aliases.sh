# Jarvis launchd shortcuts — installed into your shell rc by install.sh.
# Toutes les commandes sont idempotentes (sûres à rejouer).
#
# Résolution du chemin du script : BASH_SOURCE n'existe QUE en bash. Ce fichier étant
# sourcé depuis ~/.zshrc, s'appuyer dessus seul donnait un chemin vide sous zsh, donc
# "//scripts/jarvis-launchd.sh" et six alias morts. ${(%):-%x} est l'équivalent zsh ;
# il n'est évalué que si BASH_SOURCE est absent, donc bash ne le voit jamais.
_jarvis_aliases_src="${BASH_SOURCE[0]:-${(%):-%x}}"
JARVIS_LAUNCHD="$(cd "$(dirname "$_jarvis_aliases_src")/../.." && pwd)/scripts/jarvis-launchd.sh"
unset _jarvis_aliases_src

alias jarvis-install="$JARVIS_LAUNCHD install"
alias jarvis-start="$JARVIS_LAUNCHD start"
alias jarvis-stop="$JARVIS_LAUNCHD stop"
alias jarvis-restart="$JARVIS_LAUNCHD restart"
alias jarvis-reload="$JARVIS_LAUNCHD restart"
alias jarvis-status="$JARVIS_LAUNCHD status"

# Éditeur de configuration : ouvre la page locale (aucun serveur, aucun script).
#
# Le chemin du .env est copié dans le presse-papier au passage : macOS masque les fichiers
# commençant par un point dans les boîtes de dialogue, et le plus court est d'y faire
# ⌘⇧G puis ⌘V plutôt que d'aller chercher le raccourci d'affichage des fichiers cachés.
jarvis-config() {
  local racine
  racine="$(cd "$(dirname "$JARVIS_LAUNCHD")/.." && pwd)"
  printf '%s' "$racine/.env" | pbcopy 2>/dev/null \
    && echo "chemin du .env copié — dans le sélecteur : ⌘⇧G puis ⌘V"
  open "$racine/configuration.html"
}
