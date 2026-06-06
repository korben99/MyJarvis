# Jarvis iOS App — On-Device Speech Processing

```
┌─────────────────────────────────┐
│  iPhone (A-series Neural Engine)│
│                                 │
│  AirPods → Mic                  │
│      ↓                          │
│  WhisperKit STT (on-device)     │  ← No audio leaves iPhone
│      ↓                          │
│  TEXT query                     │
│      ↓ WiFi / Tailscale         │
│  ┌──────────────────────────┐   │
│  │  Mac Mini M4 Pro         │   │
│  │  • Jarvis API (port 8000)│   │
│  │  • Qdrant (RAG, port 6333│   │
│  │  • Redis (mem, port 6379)│   │
│  │  • Local MLX inference   │   │
│  │    Qwen3.6-35B (no cloud)│   │
│  └────────────┬─────────────┘   │
│               ↓                  │
│  TEXT response (streaming SSE)  │
│      ↓                          │
│  iOS TTS (on-device, instant)   │  ← No audio from network
│      ↓                          │
│  AirPods → Speaker              │
└─────────────────────────────────┘
```

All LLM inference runs locally on the Mac Mini M4 Pro via MLX — no external API, no cloud.

## Swift Project Files

```
JarvisApp/
├── JarvisApp.swift            # App entry point, lifecycle, notification wiring
├── AppSettings.swift          # @AppStorage persistent settings (URLs, user code, voice, whisper model)
├── Models.swift               # Shared data models (ChatMessage, ChatRequest, etc.)
├── JarvisAPI.swift            # API client: streaming SSE chat, history, polling
├── SpeechEngine.swift         # WhisperKit STT + AVSpeech TTS
├── WakeWordEngine.swift       # On-device wake word detection ("Hey Jarvis")
├── NotificationService.swift  # Push polling (BGAppRefreshTask + foreground timer)
├── ContentView.swift          # Root tab navigation
├── ChatView.swift             # Text chat with streaming
├── VoiceView.swift            # Voice mode UI
└── SettingsView.swift         # Server URLs, user code, whisper model, voice settings
```

## Key Design Decisions

**Session ID** — hardcoded to `iphone-main` in `AppSettings.swift` (`let sessionID = "iphone-main"`). Not user-configurable. Jarvis uses this fixed key to inject proactive push messages into the conversation history.

**Network routing** — the app probes `localServerURL` first (2 s timeout), then falls back to `vpnServerURL` (Tailscale). Resolved URL displayed in the status bar, reused until a failure triggers a fresh probe.

**Streaming** — responses arrive as SSE events. Thinking tokens (`{"think": "..."}`) are stripped from display but drive a progress indicator.

## Xcode Setup

### Required Capabilities (manual steps in Xcode)

1. **Signing & Capabilities** → **Background Modes** → tick **Background App Refresh**
2. **Info.plist** → add key `BGTaskSchedulerPermittedIdentifiers` (Array) → item `fr.jarvis.push-poll`

### Info.plist permissions

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Jarvis needs microphone access for voice commands</string>
<key>NSLocalNetworkUsageDescription</key>
<string>Jarvis connects to your local AI server</string>
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used for on-device wake word detection</string>
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

### Package Dependencies

- **WhisperKit** — `https://github.com/argmaxinc/whisperkit` (Up to Next Major from 0.9.0)

## WhisperKit Model Selection

| Model | Size | Transcription | Accuracy | Recommendation |
|-------|------|--------------|----------|----------------|
| tiny | ~75 MB | ~0.2 s/sentence | Basic | Dev only |
| **base** | ~140 MB | **~0.5 s/sentence** | Good | Default |
| **small** | ~460 MB | **~1.0 s/sentence** | Very good | Best quality |
| medium | ~1.5 GB | ~3 s/sentence | Excellent | Too slow |

Recommended: `base` for daily use, `small` for accented speech or noisy environments.

## Settings

| Setting | Description |
|---------|-------------|
| Local Server URL | Direct LAN IP (e.g. `http://192.168.1.50:8000`) |
| VPN / Tailscale URL | Tailscale IP for remote access |
| User Code | Authentication code (e.g. `KORBEN99`) |
| User Name | Display name |
| Whisper Model | `tiny` / `base` / `small` |
| Speak responses | Toggle AVSpeech TTS output |
| Speech rate | AVSpeech rate (0.3–0.7) |
| Wake Word | Enable "Hey Jarvis" on-device detection |
| Language | `fr` (French) or `en` (English) — affects STT + TTS + wake word |

## Troubleshooting

**WhisperKit model won't download:**
- Needs internet for first download only (cached after)
- Check available storage (`base` = 140 MB, `small` = 460 MB)

**Slow response after transcription:**
- Transcription is instant (~0.5 s) — the delay is network + LLM prefill (~3–5 s TTFT on Mac Mini M4 Pro)

**Push notifications not appearing:**
- Verify Background App Refresh is enabled in iOS Settings → Jarvis
- Check `fr.jarvis.push-poll` is in `BGTaskSchedulerPermittedIdentifiers`
- Device token must be registered: the app calls `POST /device/register` on launch

**AirPods not working for recording:**
- iOS routes audio automatically when AirPods are connected
- If issues: Settings → Bluetooth → forget and re-pair AirPods
