// SettingsView.swift
// Server connection, on-device ML, and voice settings

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var api: JarvisAPI
    @EnvironmentObject var speech: SpeechEngine
    @EnvironmentObject var wakeWord: WakeWordEngine
    @Environment(\.dismiss) var dismiss
    @State private var testResult: String?
    @State private var isTesting = false

    var body: some View {
        NavigationStack {
            Form {
                // Server
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("LOCAL NETWORK")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(.gray)
                        TextField("http://192.168.1.53:8000", text: $settings.localServerURL)
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                            .font(.system(size: 14, design: .monospaced))
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("VPN / TAILSCALE")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(.gray)
                        TextField("http://100.x.x.x:8000", text: $settings.vpnServerURL)
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                            .font(.system(size: 14, design: .monospaced))
                    }

                    HStack {
                        Button {
                            isTesting = true; testResult = nil
                            api.configure(localURL: settings.localServerURL, vpnURL: settings.vpnServerURL, sessionID: settings.sessionID)
                            Task {
                                await api.checkConnection()
                                isTesting = false
                                if api.connectionState.isConnected {
                                    testResult = "✓ \(api.activeNetwork.label)"
                                } else {
                                    testResult = "✗ Failed"
                                }
                            }
                        } label: {
                            HStack(spacing: 6) {
                                if isTesting { ProgressView().scaleEffect(0.7) }
                                Text("Test Connection")
                            }
                        }
                        .disabled(isTesting)
                        Spacer()
                        if let r = testResult {
                            Text(r).font(.system(size: 12, design: .monospaced))
                                .foregroundColor(r.contains("✓") ? .green : .red)
                        }
                    }
                } header: {
                    HStack {
                        Text("Server")
                        Spacer()
                        if api.activeNetwork != .unknown {
                            Label(api.activeNetwork.label, systemImage: api.activeNetwork.icon)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                } footer: {
                    Text("The app tries the local IP first (2 s timeout), then falls back to the VPN address automatically.")
                }

                // Language
                Section {
                    Picker("Language", selection: $settings.language) {
                        Text("🇬🇧 English").tag("en")
                        Text("🇫🇷 Français").tag("fr")
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Language")
                } footer: {
                    Text("Affects speech recognition (STT), voice synthesis (TTS), and wake word detection.")
                }

                // On-Device Speech (WhisperKit)
                Section {
                    Picker("Whisper Model", selection: $settings.whisperModel) {
                        Text("tiny — fastest, basic accuracy").tag("tiny")
                        Text("base — good balance (recommended)").tag("base")
                        Text("small — best for iPhone 13").tag("small")
                    }
                    .onChange(of: settings.whisperModel) { _, newModel in
                        Task { await speech.switchModel(to: newModel) }
                    }

                    HStack {
                        Text("Status")
                        Spacer()
                        HStack(spacing: 4) {
                            Circle()
                                .fill(speech.isModelLoaded ? .green : .orange)
                                .frame(width: 6, height: 6)
                            Text(speech.isModelLoaded ? "Ready" : speech.modelLoadProgress)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }

                    if !speech.isModelLoaded && !speech.isModelLoading {
                        Button("Load Model") {
                            Task { await speech.setup(model: settings.whisperModel) }
                        }
                    }
                } header: {
                    HStack {
                        Text("On-Device Speech")
                        Spacer()
                        Text("WhisperKit").font(.caption2).foregroundColor(.secondary)
                    }
                } footer: {
                    Text("Speech recognition runs entirely on your iPhone's Neural Engine. No audio is sent over the network. Model downloads once (~140MB for base) and is cached.")
                }

                // Wake Word
                Section {
                    Toggle("Enable \"Hey Jarvis\"", isOn: $settings.wakeWordEnabled)

                    HStack(spacing: 6) {
                        Circle()
                            .fill(wakeWord.isListening ? Color.jarvisIndigo : .gray)
                            .frame(width: 6, height: 6)
                        Text(wakeWord.isListening ? "Listening for \"Jarvis\"" : "Not active")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("Wake Word")
                } footer: {
                    Text("Uses Apple's on-device speech recognition — free, no account, no cloud. Just say \"Jarvis\" to activate. Grant Speech Recognition permission when prompted.")
                }

                // Voice Output
                Section {
                    Toggle("Speak responses aloud", isOn: $settings.useVoiceResponse)

                    VStack(alignment: .leading) {
                        Text("Speech rate: \(String(format: "%.2f", settings.voiceRate))")
                            .font(.system(size: 13))
                        Slider(value: $settings.voiceRate, in: 0.3...0.7, step: 0.02)
                    }
                } header: {
                    Text("Voice Output")
                } footer: {
                    Text("Uses iOS built-in voice synthesis. Plays through AirPods when connected.")
                }

                // Identity
                Section {
                    TextField("Your name", text: $settings.userName)
                    TextField("User Code", text: $settings.userCode)
                        .font(.system(size: 14, design: .monospaced))
                        .textInputAutocapitalization(.characters)
                        .autocorrectionDisabled(true)
                    
                    TextField("Session ID", text: $settings.sessionID)
                        .font(.system(size: 14, design: .monospaced))
                } header: {
                    Text("Identity")
                }

                // Architecture Info
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        infoRow("Speech → Text", "iPhone (WhisperKit, Neural Engine)")
                        infoRow("Text → Speech", "iPhone (iOS AVSpeech)")
                        infoRow("RAG / Memory", "Mini PC (Qdrant, Redis)")
                        infoRow("LLM Inference", "chatGPT4o-mini for now")
                    }
                    .font(.system(size: 12, design: .monospaced))
                } header: {
                    Text("Architecture")
                } footer: {
                    Text("Audio never leaves your iPhone. Only text queries are sent to the mini PC, which forwards them to RunPod for LLM inference.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        api.configure(localURL: settings.localServerURL, vpnURL: settings.vpnServerURL, sessionID: settings.sessionID)
                        dismiss()
                    }.fontWeight(.semibold)
                }
            }
        }
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top) {
            Text(label).foregroundColor(.gray).frame(width: 110, alignment: .leading)
            Text(value).foregroundColor(.white.opacity(0.8))
        }
    }
}
