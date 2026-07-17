# Jarvis launchd shortcuts — installed into your shell rc by install.sh.
alias jarvis-stop='launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist'
alias jarvis-start='launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist'
alias jarvis-reload='launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist; launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jarvis.api.plist'
