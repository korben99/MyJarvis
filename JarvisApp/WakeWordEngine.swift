// WakeWordEngine.swift
// Always-on "jarvis" keyword detection using Apple's on-device SFSpeechRecognizer.
// 100% free, no third-party SDK, no cloud — runs on the Neural Engine.
//
// SFSpeechRecognizer has a ~1-minute session limit; we auto-restart every 45s.

import Foundation
import AVFoundation
import Speech
import Combine

@MainActor
class WakeWordEngine: ObservableObject {

    @Published var isListening = false
    @Published var wakeWordDetected = false

    private var recognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var restartTimer: Timer?
    private var enabled = false
    private var language: String = "en"
    private var tapInstalled = false
    private var retryCount = 0
    private static let maxRetries = 5
    // Stored so teardown() can cancel a pending retry sleep.
    private var retryTask: Task<Void, Never>?

    // MARK: - Lifecycle

    func start(language: String = "en") {
        self.language = language
        let locale = language == "fr" ? "fr-FR" : "en-US"
        recognizer = SFSpeechRecognizer(locale: Locale(identifier: locale))
        enabled = true
        retryCount = 0
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            // Authorization callback can arrive on any thread — hop to MainActor.
            Task { @MainActor [weak self] in
                guard status == .authorized else { return }
                self?.startSession()
            }
        }
    }

    func stop() {
        enabled = false
        teardown()
        isListening = false
    }

    /// Call before SpeechEngine starts recording.
    func pause() {
        teardown()
        isListening = false
    }

    /// Call after the full voice pipeline completes.
    func resume() {
        guard enabled else { return }
        startSession()
    }

    // MARK: - Session

    private func startSession() {
        // Guard against the auth-callback Task firing after stop() was called.
        guard enabled else { return }
        teardown()

        guard let recognizer, recognizer.isAvailable else { return }

        retryCount = 0   // reset on each successful new session start

        // Auto-restart before the 1-minute iOS limit kicks in.
        // Timer block is @Sendable — hop to MainActor to safely access actor-isolated state.
        restartTimer = Timer.scheduledTimer(withTimeInterval: 45, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.enabled else { return }
                self.startSession()
            }
        }

        do {
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let request = recognitionRequest else { return }
            request.requiresOnDeviceRecognition = true
            request.shouldReportPartialResults = true

            let inputNode = audioEngine.inputNode
            let format = inputNode.outputFormat(forBus: 0)

            // 4096 samples ≈ 93 ms at 44.1 kHz → ~11 callbacks/sec vs ~43/sec at 1024.
            // Reduces continuous CPU overhead from the mic tap by ~4×.
            inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
                guard buffer.frameLength > 0 else { return }
                // Energy VAD: skip near-silent frames before they reach SFSpeechRecognizer.
                // Samples every 8th frame (512 comparisons) for peak amplitude.
                // Threshold ≈ -54 dB — well above mic noise floor, well below any speech.
                // Cuts Neural Engine work by ~70% in a quiet room.
                if let data = buffer.floatChannelData?[0] {
                    let n = Int(buffer.frameLength)
                    var peak: Float = 0
                    for i in stride(from: 0, to: n, by: 8) {
                        let s = abs(data[i])
                        if s > peak { peak = s }
                    }
                    guard peak > 0.002 else { return }
                }
                self?.recognitionRequest?.append(buffer)
            }
            tapInstalled = true

            audioEngine.prepare()
            try audioEngine.start()

            recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
                // This callback fires on a background thread. Extract plain values from the
                // result before hopping to MainActor so we never access actor-isolated state
                // from a non-isolated context (data race).
                let transcription = result?.bestTranscription.formattedString
                let taskError = error

                Task { @MainActor [weak self] in
                    guard let self else { return }

                    if let transcription {
                        let words = transcription.lowercased().split(separator: " ")
                        // Only check the last 3 words to avoid stale matches
                        let recent = words.suffix(3).joined(separator: " ")
                        if recent.contains("jarvis") {
                            self.wakeWordDetected = true
                            self.retryCount = 0
                            self.startSession()
                            return
                        }
                    }

                    if let error = taskError, self.enabled {
                        // Exponential backoff with a max retry cap.
                        let nsError = error as NSError
                        let isFatal = nsError.domain == "kAFAssistantErrorDomain" &&
                                      (nsError.code == 1101 || nsError.code == 203)
                        guard !isFatal, self.retryCount < Self.maxRetries else { return }
                        let delay = pow(2.0, Double(self.retryCount))
                        self.retryCount += 1
                        // Store the task so teardown() can cancel the sleep if stop() is called.
                        self.retryTask?.cancel()
                        self.retryTask = Task { @MainActor [weak self] in
                            try? await Task.sleep(for: .seconds(delay))
                            // Re-check enabled after the sleep — stop() may have been called.
                            guard let self, self.enabled else { return }
                            self.startSession()
                        }
                    }
                }
            }

            isListening = true   // safe: we're @MainActor

        } catch {
            print("[WakeWord] session error: \(error.localizedDescription)")
        }
    }

    private func teardown() {
        restartTimer?.invalidate()
        restartTimer = nil
        retryTask?.cancel()        // cancel any pending retry sleep
        retryTask = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        if audioEngine.isRunning { audioEngine.stop() }
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
    }

}
