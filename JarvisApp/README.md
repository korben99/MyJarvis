# Jarvis iOS App v3 — On-Device Speech Processing

```
┌─────────────────────────────────┐
│  iPhone 13 (A15 Neural Engine)  │
│                                 │
│  AirPods → Mic                  │
│      ↓                          │
│  WhisperKit STT (on-device)     │  ← No audio leaves iPhone
│      ↓                          │
│  TEXT query                     │
│      ↓ WiFi / Tailscale         │
│  ┌──────────────────┐           │
│  │   Mini PC (N150) │           │
│  │   • Jarvis API   │           │
│  │   • Qdrant (RAG) │           │
│  │   • Redis (mem)  │           │
│  └────────┬─────────┘           │
│           ↓ Internet             │
│  ┌──────────────────┐           │
│  │  RunPod RTX 5090 │           │
│  │  Llama 70B       │           │
│  └────────┬─────────┘           │
│           ↓                      │
│  TEXT response                  │
│      ↓                          │
│  iOS TTS (on-device, instant)   │  ← No audio from network
│      ↓                          │
│  AirPods → Speaker              │
└─────────────────────────────────┘
```

## What Changed from v2

| Component | v2 (old) | v3 (new) |
|-----------|----------|----------|
| STT | Mini PC Whisper (CPU, ~1.5s) | **iPhone WhisperKit (Neural Engine, ~0.5s)** |
| TTS | Mini PC Piper or iOS fallback | **iPhone iOS TTS (instant)** |
| Audio over network | Yes (WAV upload to mini PC) | **No — only text crosses the network** |
| Whisper Docker container | Required on mini PC | **Not needed — saves RAM** |
| Piper Docker container | Required on mini PC | **Not needed — saves RAM** |
| Mini PC API | /chat + /voice endpoints | **Only /chat (simpler)** |
| Privacy | Audio went to mini PC | **Audio never leaves iPhone** |
| Latency | ~4-6 seconds end-to-end | **~1.5-3 seconds** |

## iPhone 13 (A15 Bionic) WhisperKit Performance

| Model | Size | Download | Transcription Speed | Accuracy | RAM Usage |
|-------|------|----------|-------------------|----------|-----------|
| tiny | ~75 MB | ~10s | ~0.2s/sentence | Basic | ~200 MB |
| **base** | ~140 MB | ~20s | **~0.5s/sentence** | Good | ~350 MB |
| **small** | ~460 MB | ~60s | **~1.0s/sentence** | Very good | ~700 MB |
| medium | ~1.5 GB | ~3min | ~3s/sentence | Excellent | ~2 GB |
| large-v3 | ~3 GB | ~6min | Not recommended | Best | Too much RAM |

**Recommendation for iPhone 13**: Start with `base`, upgrade to `small` if you need better accuracy (especially for accented speech or noisy environments). The `small` model still fits comfortably in 4GB RAM.

## Setup Guide

### Step 1: Create Xcode Project

1. Open Xcode 15+ → File → New → Project
2. Select **App** (iOS)
3. Product Name: **JarvisApp**
4. Interface: **SwiftUI**, Language: **Swift**
5. Minimum deployment: **iOS 17.0**

### Step 2: Add WhisperKit Package

1. File → Add Package Dependencies
2. Enter URL: `https://github.com/argmaxinc/whisperkit`
3. Version: **Up to Next Major** from `0.9.0`
4. When prompted, select the **WhisperKit** library product
5. Click Add Package

### Step 3: Add Source Files

Copy all `.swift` files from `ios-app/JarvisApp/` into your Xcode project:

- `JarvisApp.swift` (replace default)
- `AppSettings.swift`
- `Models.swift`
- `JarvisAPI.swift`
- `SpeechEngine.swift` ← NEW: WhisperKit + TTS engine
- `ContentView.swift`
- `ChatView.swift`
- `VoiceView.swift`
- `SettingsView.swift`

### Step 4: Configure Info.plist

Add these keys:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Jarvis needs microphone access for voice commands</string>
<key>NSLocalNetworkUsageDescription</key>
<string>Jarvis connects to your local AI server</string>
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

### Step 5: Build and Run

1. Connect iPhone 13 via USB or wireless debugging
2. Select your iPhone as build target
3. ⌘R to build and run
4. On first launch, WhisperKit will download the `base` model (~140MB, once)
5. Go to Settings → enter your mini PC IP
6. Start chatting or speaking!

## Mini PC API (Simplified)

Since the iPhone handles all audio, the mini PC API is now text-only:

```bash
# Add to your docker-compose.yml
cp mini-pc-api/ /opt/jarvis/jarvis-core/
docker compose up -d --build jarvis-api

# Test
curl http://MINI_PC_IP:8000/status
curl -X POST http://MINI_PC_IP:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Jarvis", "stream": false}'
```

### Updated docker-compose.yml

You can now **remove** the Whisper and Piper containers from your compose file, freeing ~3-4 GB RAM on the N150:

```yaml
# REMOVE these services (no longer needed):
# whisper:
#   image: onerahmet/openai-whisper-asr-webservice:latest-cpu
# piper:
#   image: rhasspy/wyoming-piper:latest
```

## Project Files

```
jarvis-ios-v3/
├── README.md                          # This file
├── mini-pc-api/
│   ├── main.py                        # Simplified text-only API
│   └── Dockerfile
└── ios-app/
    └── JarvisApp/
        ├── JarvisApp.swift            # App entry, initializes WhisperKit
        ├── AppSettings.swift          # Settings with Whisper model selection
        ├── Models.swift               # Data models (simplified)
        ├── JarvisAPI.swift            # Text-only API client
        ├── SpeechEngine.swift         # WhisperKit STT + iOS TTS (on-device)
        ├── ContentView.swift          # Tab navigation
        ├── ChatView.swift             # Text chat with streaming
        ├── VoiceView.swift            # Voice with on-device processing
        └── SettingsView.swift         # Config + Whisper model picker
```

## Troubleshooting

**WhisperKit model won't download:**
- Needs internet for first download only (cached after)
- Check available storage (base = 140MB, small = 460MB)
- Try `tiny` model first to verify everything works

**Transcription is inaccurate:**
- Switch from `base` to `small` in Settings
- Speak clearly and closer to the mic
- Reduce background noise
- WhisperKit works best with 1+ seconds of speech

**Slow response after transcription:**
- Transcription is instant (~0.5s) — the delay is network + LLM
- Check RunPod pod is running
- Check mini PC API connection in Settings

**AirPods not working for recording:**
- iOS routes audio automatically when AirPods are connected
- If issues: Settings → Bluetooth → forget and re-pair AirPods
- The app sets `.allowBluetooth` on the audio session

**Build error: "No such module 'WhisperKit'":**
- File → Packages → Reset Package Caches
- Product → Clean Build Folder (⇧⌘K)
- Verify package was added with WhisperKit product selected
