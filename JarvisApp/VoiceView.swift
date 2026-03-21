// VoiceView.swift
// Voice conversation with on-device STT (WhisperKit) and TTS.
//
// Flow: AirPods → iPhone mic → WhisperKit (Neural Engine) → text
//       → Mini PC → RunPod LLM → text → iOS TTS → AirPods
//
// Only TEXT crosses the network. Audio never leaves the iPhone.

import SwiftUI
import Combine

struct VoiceView: View {
    @EnvironmentObject var api: JarvisAPI
    @EnvironmentObject var speech: SpeechEngine
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var wakeWord: WakeWordEngine
    @State private var showSettings = false
    @State private var hasMicPermission = false
    @State private var lastResponse = ""
    @State private var pulseScale: CGFloat = 1.0
    @State private var isProcessingVoice = false
    // Monotonically increasing — used as sensoryFeedback trigger so haptic fires
    // only on detection, not on the subsequent wakeWordDetected = false reset.
    @State private var wakeWordHapticCount = 0

    // Derived state for the orb
    private var currentPhase: Phase {
        if speech.isRecording { return .recording }
        if speech.isTranscribing { return .transcribing }
        if api.isProcessing { return .thinking }
        if speech.isSpeaking { return .speaking }
        if wakeWord.isListening { return .wakeListening }
        return .idle
    }

    private enum Phase {
        case idle, wakeListening, recording, transcribing, thinking, speaking

        var color: Color {
            switch self {
            case .idle:          return .jarvisBlue
            case .wakeListening: return .jarvisIndigo
            case .recording:     return .jarvisRed
            case .transcribing:  return .jarvisAmber
            case .thinking:      return .jarvisAmber
            case .speaking:      return .jarvisGreen
            }
        }

        var icon: String {
            switch self {
            case .idle:          return "mic.fill"
            case .wakeListening: return "ear.fill"
            case .recording:     return "waveform"
            case .transcribing:  return "text.magnifyingglass"
            case .thinking:      return "brain"
            case .speaking:      return "speaker.wave.2.fill"
            }
        }

        var label: String {
            switch self {
            case .idle:          return "Tap to speak"
            case .wakeListening: return "Say \"Hey Jarvis\"..."
            case .recording:     return "Listening... tap to send"
            case .transcribing:  return "Transcribing on device..."
            case .thinking:      return "Thinking..."
            case .speaking:      return "Speaking..."
            }
        }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.jarvisBg.ignoresSafeArea()

                VStack(spacing: 0) {
                    statusHeader.padding(.top, 16)
                    Spacer()
                    centerOrb
                    conversationArea.padding(.horizontal, 24)
                    Spacer()
                    actionButton.padding(.bottom, 40)
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Text("JARVIS").font(.system(size: 16, weight: .bold, design: .monospaced))
                        .foregroundColor(.jarvisBlue)
                        .fixedSize()
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: {
                        Image(systemName: "gear").foregroundColor(.gray)
                    }
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .task { hasMicPermission = await speech.requestMicrophonePermission() }
            .onChange(of: speech.shouldAutoStop) { _, should in
                guard should, speech.isRecording, !isProcessingVoice else { return }
                isProcessingVoice = true
                Task { await processVoice() }
            }
            .onChange(of: wakeWord.wakeWordDetected) { _, detected in
                guard detected else { return }
                wakeWordHapticCount += 1        // increment before reset so trigger fires once
                wakeWord.wakeWordDetected = false
                guard canInteract else { return }
                lastResponse = ""
                wakeWord.pause()
                speech.startRecording()
            }
            .sensoryFeedback(.impact(flexibility: .solid, intensity: 0.8), trigger: wakeWordHapticCount)
            .onChange(of: speech.isSpeaking) { _, isSpeaking in
                if !isSpeaking { wakeWord.resume() }
            }
        }
    }

    // MARK: - Status Header

    private var statusHeader: some View {
        VStack(spacing: 6) {
            // Connection
            HStack(spacing: 6) {
                Circle()
                    .fill(api.connectionState.isConnected ? .green : .red)
                    .frame(width: 6, height: 6)
                Text(api.connectionState.statusText)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundColor(.gray)
            }

            // WhisperKit status
            HStack(spacing: 6) {
                Circle()
                    .fill(speech.isModelLoaded ? .green : (speech.isModelLoading ? .orange : .red))
                    .frame(width: 5, height: 5)
                Text(speech.isModelLoaded ? "Whisper \(settings.whisperModel) ready" : speech.modelLoadProgress)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.gray.opacity(0.7))
            }

            // Wake word status
            if settings.wakeWordEnabled {
                HStack(spacing: 6) {
                    Circle()
                        .fill(wakeWord.isListening ? Color.jarvisIndigo : .gray)
                        .frame(width: 5, height: 5)
                    Text(wakeWord.isListening ? "Say \"Hey Jarvis\"" : "Wake word inactive")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.gray.opacity(0.7))
                }
            }

            if !hasMicPermission {
                Text("Microphone permission required")
                    .font(.system(size: 11)).foregroundColor(.orange)
            }
        }
    }

    // MARK: - Center Orb

    private var centerOrb: some View {
        ZStack {
            // Outer glow
            Circle()
                .fill(RadialGradient(
                    colors: [currentPhase.color.opacity(0.15), .clear],
                    center: .center, startRadius: 40, endRadius: 120
                ))
                .frame(width: 240, height: 240)
                .scaleEffect(pulseScale)

            // Ring
            Circle()
                .stroke(currentPhase.color.opacity(0.3), lineWidth: 1)
                .frame(width: 140, height: 140)
                .scaleEffect(pulseScale * 0.95)

            // Silence countdown ring
            if speech.isRecording && speech.silenceProgress > 0 {
                Circle()
                    .trim(from: 0, to: speech.silenceProgress)
                    .stroke(Color.jarvisRed, lineWidth: 3)
                    .frame(width: 140, height: 140)
                    .rotationEffect(.degrees(-90))
                    .animation(.linear(duration: 0.1), value: speech.silenceProgress)
            }

            // Inner orb
            Circle()
                .fill(RadialGradient(
                    colors: [currentPhase.color.opacity(0.6), currentPhase.color.opacity(0.2)],
                    center: .center, startRadius: 10, endRadius: 50
                ))
                .frame(width: 90, height: 90)

            // Core dot
            Circle().fill(currentPhase.color).frame(width: 16, height: 16)

            // State icon
            Image(systemName: currentPhase.icon)
                .font(.system(size: 26, weight: .light))
                .foregroundColor(.white.opacity(0.8))
                .symbolEffect(.variableColor.iterative.reversing,
                              isActive: currentPhase == .recording || currentPhase == .speaking)
        }
        .animation(.easeInOut(duration: 0.4), value: currentPhase.label)
        .onChange(of: speech.isRecording) { _, rec in
            withAnimation(rec
                ? .easeInOut(duration: 1.2).repeatForever(autoreverses: true)
                : .easeOut(duration: 0.3)) { pulseScale = rec ? 1.15 : 1.0 }
        }
        .onChange(of: api.isProcessing) { _, proc in
            withAnimation(proc
                ? .easeInOut(duration: 0.8).repeatForever(autoreverses: true)
                : .easeOut(duration: 0.3)) { pulseScale = proc ? 1.08 : 1.0 }
        }
    }

    // MARK: - Conversation Area

    private var conversationArea: some View {
        VStack(spacing: 14) {
            // Transcription
            if !speech.currentTranscription.isEmpty {
                VStack(spacing: 3) {
                    Text("YOU").font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundColor(.gray)
                    Text(speech.currentTranscription)
                        .font(.system(size: 15)).foregroundColor(.white.opacity(0.7)).multilineTextAlignment(.center)
                }
            }

            // Response
            if !lastResponse.isEmpty {
                VStack(spacing: 3) {
                    Text("JARVIS").font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundColor(.jarvisBlue)
                    Text(lastResponse)
                        .font(.system(size: 15)).foregroundColor(.white.opacity(0.9)).multilineTextAlignment(.center)
                        .lineLimit(6)
                }
            }

            // Duration / phase label
            if speech.isRecording {
                Text(String(format: "%.1fs", speech.recordingDuration))
                    .font(.system(size: 13, design: .monospaced)).foregroundColor(.jarvisRed)
            }

            Text(currentPhase.label)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 180)
    }

    // MARK: - Action Button

    private var actionButton: some View {
        VStack(spacing: 8) {
            Button(action: handleTap) {
                ZStack {
                    Circle()
                        .fill(speech.isRecording ? Color.jarvisRed : .jarvisCard)
                        .frame(width: 72, height: 72)
                    Circle()
                        .stroke(speech.isRecording ? Color.jarvisRed : .jarvisBlue, lineWidth: 2)
                        .frame(width: 72, height: 72)
                    Image(systemName: speech.isRecording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 24))
                        .foregroundColor(speech.isRecording ? .white : Color.jarvisBlue)
                }
            }
            .disabled(!canInteract)
            .opacity(canInteract ? 1.0 : 0.4)
        }
    }

    private var canInteract: Bool {
        hasMicPermission && speech.isModelLoaded && !speech.isTranscribing && !api.isProcessing
    }

    // MARK: - Voice Flow

    private func handleTap() {
        if speech.isRecording {
            // Stop recording → transcribe → query LLM → speak
            guard !isProcessingVoice else { return }
            isProcessingVoice = true
            Task { await processVoice() }
        } else if speech.isSpeaking {
            // Tap while speaking to stop
            speech.stopSpeaking()
        } else {
            // Start recording — pause wake word to free the mic
            lastResponse = ""
            wakeWord.pause()
            speech.startRecording()
        }
    }

    /// The full voice pipeline, executed on iPhone:
    /// 1. Stop recording
    /// 2. Transcribe on-device (WhisperKit, Neural Engine) — ~0.5s
    /// 3. Send TEXT to mini PC → RunPod LLM — ~1-2s
    /// 4. Speak response with iOS TTS — instant
    private func processVoice() async {
        defer { isProcessingVoice = false }

        // Step 1+2: Stop and transcribe locally
        guard let transcription = await speech.stopAndTranscribe() else {
            wakeWord.resume()
            return
        }

        // Step 3: Send text (not audio!) to mini PC → RunPod
        // voice_mode=true → server returns concise 2-3 sentence responses for TTS
        await api.sendMessage(transcription, voiceMode: true)

        // Step 4: Speak the response
        if let response = api.lastAssistantMessage, settings.useVoiceResponse {
            lastResponse = response
            speech.speak(response, rate: Float(settings.voiceRate))
        }

        // Resume wake word now if not speaking; otherwise onChange(isSpeaking) handles it
        if !speech.isSpeaking { wakeWord.resume() }
    }
}

