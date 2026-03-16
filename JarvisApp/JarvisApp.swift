// JarvisApp.swift
// Project Jarvis — iOS Client v3
// On-device STT (WhisperKit) + TTS + Remote LLM

import SwiftUI
import Combine

@main
struct JarvisApp: App {
    @StateObject private var settings = AppSettings()
    @StateObject private var api = JarvisAPI()
    @StateObject private var speechEngine = SpeechEngine()
    @StateObject private var wakeWord = WakeWordEngine()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .environmentObject(api)
                .environmentObject(speechEngine)
                .environmentObject(wakeWord)
                .onAppear {
                    api.configure(localURL: settings.localServerURL, vpnURL: settings.vpnServerURL, sessionID: settings.sessionID)
                }
                .onChange(of: settings.localServerURL) { _, _ in
                    api.configure(localURL: settings.localServerURL, vpnURL: settings.vpnServerURL, sessionID: settings.sessionID)
                }
                .onChange(of: settings.vpnServerURL) { _, _ in
                    api.configure(localURL: settings.localServerURL, vpnURL: settings.vpnServerURL, sessionID: settings.sessionID)
                }
                .onChange(of: settings.wakeWordEnabled) { _, enabled in
                    if enabled { wakeWord.start(language: settings.language) } else { wakeWord.stop() }
                }
                .onChange(of: settings.language) { _, lang in
                    speechEngine.language = lang
                    if settings.wakeWordEnabled {
                        wakeWord.stop()
                        wakeWord.start(language: lang)
                    }
                }
                .task {
                    await speechEngine.setup(model: settings.whisperModel)
                    speechEngine.language = settings.language
                    if settings.wakeWordEnabled { wakeWord.start(language: settings.language) }
                }
        }
    }
}
