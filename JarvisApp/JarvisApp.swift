// JarvisApp.swift
// Project Jarvis — iOS Client v3
// On-device STT (WhisperKit) + TTS + Remote LLM

import SwiftUI
import Combine
import BackgroundTasks

@main
struct JarvisApp: App {
    @StateObject private var settings    = AppSettings()
    @StateObject private var api         = JarvisAPI()
    @StateObject private var speechEngine = SpeechEngine()
    @StateObject private var wakeWord    = WakeWordEngine()
    @Environment(\.scenePhase) private var scenePhase

    private let notifications = NotificationService.shared

    init() {
        // Register the BGAppRefreshTask identifier before the first scene loads.
        NotificationService.registerBackgroundTask()
    }

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
                    syncNotificationService()
                }
                .onChange(of: settings.vpnServerURL) { _, _ in
                    api.configure(localURL: settings.localServerURL, vpnURL: settings.vpnServerURL, sessionID: settings.sessionID)
                    syncNotificationService()
                }
                .onChange(of: settings.userCode) { _, _ in
                    // userCode is the polling identity — sync immediately so
                    // NotificationService doesn't keep polling the old endpoint.
                    syncNotificationService()
                }
                .onChange(of: settings.sessionID) { _, _ in
                    // Keep api.sessionID consistent with settings — without this,
                    // the session only updates when the Done button is pressed.
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
                // Sync api.resolvedURL → NotificationService after connection resolves
                .onChange(of: api.connectionState) { _, _ in
                    syncNotificationService()
                }
                .task {
                    await speechEngine.setup(model: settings.whisperModel)
                    speechEngine.language = settings.language
                    if settings.wakeWordEnabled { wakeWord.start(language: settings.language) }
                    // Request notification permissions once on first launch
                    await notifications.requestPermissions()
                }
        }
        // Pause all audio when the app goes to background; resume on foreground.
        // Without this, AVAudioEngine + SFSpeechRecognizer keep the mic and Neural Engine
        // running indefinitely, draining battery and heating the device.
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                wakeWord.pause()
                speechEngine.deactivateAudioSession()
                notifications.stopForegroundPolling()
                notifications.scheduleBackgroundRefresh()
            case .active:
                if settings.wakeWordEnabled { wakeWord.resume() }
                syncNotificationService()
                notifications.startForegroundPolling()
                // Immediately poll + reload conversation on foreground so messages
                // pushed by jarvis-core (added to session history while the app was
                // backgrounded or suspended) appear in the chat UI without a restart.
                Task {
                    await notifications.pollAndDeliver()
                    await api.loadConversation()
                }
            default:
                break
            }
        }
    }

    // MARK: - Helpers

    /// Copy the resolved server URL and user code into NotificationService,
    /// then (re-)register the device with the backend.
    private func syncNotificationService() {
        notifications.userCode       = settings.userCode
        notifications.resolvedURL    = api.resolvedURL
        notifications.localServerURL = settings.localServerURL
        notifications.vpnServerURL   = settings.vpnServerURL
        // Persist the resolved URL so a cold background launch (BGAppRefreshTask after the
        // app was killed) can still poll without waiting for JarvisAPI to reconnect.
        if !api.resolvedURL.isEmpty {
            UserDefaults.standard.set(api.resolvedURL, forKey: "lastResolvedURL")
        }
        guard !notifications.resolvedURL.isEmpty, !notifications.userCode.isEmpty else { return }
        Task { await notifications.registerDevice() }
    }

    // Expose resolvedURL from JarvisAPI (needed by syncNotificationService)
    // We declare resolvedURL as internal in JarvisAPI — see below.
}
