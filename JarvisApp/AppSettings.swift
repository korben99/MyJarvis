//
//  AppSettings.swift
//  JarvisApp
//
//  Created by sebastien viou on 09/03/2026.
//



// AppSettings.swift
// Persistent settings for server connection and on-device ML

import SwiftUI
import Combine

class AppSettings: ObservableObject {
    // Server
    @AppStorage("localServerURL") var localServerURL: String = "http://192.168.1.50:8000"
    @AppStorage("vpnServerURL")   var vpnServerURL:   String = "http://100.94.97.27:8000"
    let sessionID: String = "iphone-main"

    // Identity
    @AppStorage("userName") var userName: String = ""
    @AppStorage("userCode") var userCode: String = ""

    // On-device Whisper (WhisperKit)
    // iPhone 13 (A15, 4GB RAM):
    //   "tiny"  — fastest, basic accuracy (~75MB)
    //   "base"  — good balance (~140MB)           ← recommended
    //   "small" — best accuracy for A15 (~460MB)  ← best quality
    // Avoid "large-v3" on iPhone 13 (too much RAM)
    @AppStorage("whisperModel") var whisperModel: String = "base"

    // Voice
    @AppStorage("useVoiceResponse") var useVoiceResponse: Bool = true
    @AppStorage("voiceRate") var voiceRate: Double = 0.52  // AVSpeech rate (0.0-1.0)

    // Language
    @AppStorage("language") var language: String = "en"  // "en" or "fr"

    // Wake Word
    @AppStorage("wakeWordEnabled") var wakeWordEnabled: Bool = false
}
