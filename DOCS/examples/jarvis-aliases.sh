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
