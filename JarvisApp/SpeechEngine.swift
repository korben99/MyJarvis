// SpeechEngine.swift
// Jarvis iOS — Speech Engine
// STT: WhisperKit (on-device Neural Engine)
// TTS: AVSpeechSynthesizer

import Foundation
import AVFoundation
import WhisperKit
import Combine
import CoreML

@MainActor
class SpeechEngine: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {

    // MARK: - Published State

    @Published var isModelLoaded = false
    @Published var isModelLoading = false
    @Published var modelLoadProgress: String = ""

    @Published var isRecording = false
    @Published var isTranscribing = false
    @Published var isSpeaking = false

    @Published var recordingDuration: TimeInterval = 0
    @Published var currentTranscription: String = ""
    @Published var silenceProgress: Double = 0   // 0→1 as silence builds up
    @Published var shouldAutoStop = false

    var language: String = "en"   // "en" or "fr", set from AppSettings

    // MARK: - Private

    private var whisperKit: WhisperKit?
    private var audioRecorder: AVAudioRecorder?
    private var recordingURL: URL?
    private var timer: Timer?

    private var silenceDuration: TimeInterval = 0
    private let silenceThreshold: Float = -35.0    // dB — below this = silence
    private let silenceTimeout: TimeInterval = 1.5  // seconds of silence before auto-send
    private let minRecordingTime: TimeInterval = 0.8 // don't auto-stop before this

    private let synthesizer = AVSpeechSynthesizer()
    private let audioSession = AVAudioSession.sharedInstance()

    // MARK: - Init

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    // MARK: - Setup Whisper

    func setup(model: String = "base") async {

        guard !isModelLoaded && !isModelLoading else { return }

        isModelLoading = true
        modelLoadProgress = "Loading \(model) model..."

        do {

            let config = WhisperKitConfig(
                model: "openai_whisper-\(model)",
                computeOptions: ModelComputeOptions(
                    audioEncoderCompute: .all,
                    textDecoderCompute: .all
                )
            )

            whisperKit = try await WhisperKit(config)

            isModelLoaded = true
            modelLoadProgress = "Ready"

        } catch {
            modelLoadProgress = "Error: \(error.localizedDescription)"
        }

        isModelLoading = false
    }

    func switchModel(to model: String) async {
        whisperKit = nil
        isModelLoaded = false
        await setup(model: model)
    }

    // MARK: - Audio Session

    private func configureAudioSession() throws {

        try audioSession.setCategory(
            .playAndRecord,
            mode: .default,
            options: [.defaultToSpeaker, .allowBluetoothHFP, .allowBluetoothA2DP]
        )

        try audioSession.setActive(true)
    }

    // MARK: - Recording

    func startRecording() {

        Task {

            let granted = await requestMicrophonePermission()

            guard granted else {
                print("Microphone permission denied")
                return
            }

            do {
                try configureAudioSession()
            } catch {
                print("Audio session error:", error)
                return
            }

            stopSpeaking()
            currentTranscription = ""

            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("jarvis_\(UUID().uuidString).wav")

            recordingURL = url

            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 16000.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsFloatKey: false
            ]

            do {

                audioRecorder = try AVAudioRecorder(url: url, settings: settings)
                audioRecorder?.isMeteringEnabled = true
                audioRecorder?.record()

                isRecording = true
                recordingDuration = 0
                silenceDuration = 0
                silenceProgress = 0
                shouldAutoStop = false

                // Use Timer(timeInterval:) + RunLoop.add to avoid double-scheduling:
                // scheduledTimer already adds to .default; we want .common only.
                let t = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
                    // Timer fires on RunLoop.main (.common mode), so we're always on the
                    // main actor here. assumeIsolated asserts that at runtime and satisfies
                    // the Swift concurrency checker without a Task hop.
                    MainActor.assumeIsolated {
                        guard let self else { return }
                        self.recordingDuration += 0.1
                        self.checkSilence()
                    }
                }
                RunLoop.main.add(t, forMode: .common)
                timer = t

            } catch {
                print("Recording error:", error)
            }
        }
    }

    // MARK: - VAD (Voice Activity Detection)

    private func checkSilence() {
        guard recordingDuration >= minRecordingTime, !shouldAutoStop else { return }

        audioRecorder?.updateMeters()
        let power = audioRecorder?.averagePower(forChannel: 0) ?? 0

        if power < silenceThreshold {
            silenceDuration += 0.1
            silenceProgress = min(silenceDuration / silenceTimeout, 1.0)
            if silenceDuration >= silenceTimeout {
                shouldAutoStop = true
            }
        } else {
            silenceDuration = 0
            silenceProgress = 0
        }
    }

    // MARK: - Stop + Transcribe

    func stopAndTranscribe() async -> String? {

        audioRecorder?.stop()

        isRecording = false
        shouldAutoStop = false
        silenceDuration = 0
        silenceProgress = 0

        timer?.invalidate()
        timer = nil

        guard let url = recordingURL else { return nil }

        // Always clean up the temp file, regardless of which path we exit through.
        defer { try? FileManager.default.removeItem(at: url) }

        guard let whisper = whisperKit else {
            currentTranscription = "Model not loaded"
            return nil
        }

        isTranscribing = true
        defer { isTranscribing = false }

        do {

            let options = DecodingOptions(
                verbose: false,
                task: .transcribe,
                language: language,
                temperature: 0.0,
                temperatureFallbackCount: 0,  // biggest speedup: skip retry loop
                sampleLength: 224,
                usePrefillPrompt: true,
                usePrefillCache: true,
                skipSpecialTokens: true,
                withoutTimestamps: true,
                wordTimestamps: false,
                suppressBlank: true,
                compressionRatioThreshold: nil,
                logProbThreshold: nil,
                noSpeechThreshold: nil
            )
            let results = try await whisper.transcribe(audioPath: url.path, decodeOptions: options)

            // WhisperKit returns an array of TranscriptionResult; join their texts
            let text = results
                .map { $0.text }
                .joined(separator: " ")
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)

            currentTranscription = text

            return text.isEmpty ? nil : text

        } catch {

            currentTranscription = "Transcription error"
            return nil
        }
    }

    // MARK: - TTS

    func speak(_ text: String, rate: Float = 0.52) {

        synthesizer.stopSpeaking(at: .immediate)

        let utterance = AVSpeechUtterance(string: text)

        utterance.voice = Self.bestVoice(for: language)
        utterance.rate = rate
        utterance.pitchMultiplier = 1.0

        isSpeaking = true

        synthesizer.speak(utterance)
    }

    func stopSpeaking() {

        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }

        isSpeaking = false
    }

    // MARK: - Speech Delegate

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in
            self.isSpeaking = false
        }
    }

    // MARK: - Voice selection

    /// Returns the best available voice for the given language: premium > enhanced > default.
    static func bestVoice(for language: String) -> AVSpeechSynthesisVoice? {
        let locale = language == "fr" ? "fr-FR" : "en-US"
        let all = AVSpeechSynthesisVoice.speechVoices()
        let matching = all.filter { $0.language.hasPrefix(language) }
        return matching.first(where: { $0.quality == .premium })
            ?? matching.first(where: { $0.quality == .enhanced })
            ?? AVSpeechSynthesisVoice(language: locale)
    }

    // MARK: - Permissions

    func requestMicrophonePermission() async -> Bool {

        await withCheckedContinuation { continuation in

            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        }
    }
}

