// JarvisAPI.swift
// API client for text chat with the Jarvis mini PC bridge.
// Voice (STT/TTS) is handled entirely on-device by SpeechEngine.

import Foundation
import Combine
import UIKit

@MainActor
class JarvisAPI: ObservableObject {
    @Published var connectionState: ConnectionState = .disconnected
    @Published var messages: [ChatMessage] = []
    @Published var isProcessing = false
    @Published var activeNetwork: NetworkRoute = .unknown

    private var localServerURL = ""
    private var vpnServerURL   = ""
    private var resolvedURL    = ""
    private var sessionID      = "iphone-main"

    // URLSession with a short timeout used only for reachability probes
    private let probeSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 2
        config.timeoutIntervalForResource = 2
        return URLSession(configuration: config)
    }()

    // MARK: - Configuration

    private var configureTask: Task<Void, Never>?

    func configure(localURL: String, vpnURL: String, sessionID: String = "iphone-main") {
        self.localServerURL = localURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.vpnServerURL   = vpnURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.sessionID      = sessionID
        configureTask?.cancel()
        configureTask = Task {
            await checkConnection()
            guard !Task.isCancelled else { return }
            await loadConversation()
        }
    }

    // MARK: - Connection Check (with auto-routing)

    func checkConnection() async {
        connectionState = .connecting
        activeNetwork   = .unknown

        guard let (url, route) = await resolveActiveURL() else {
            connectionState = .error("No server — check IP & Local Network permission")
            activeNetwork   = .unknown
            return
        }

        resolvedURL   = url
        activeNetwork = route

        guard let statusURL = URL(string: "\(url)/status") else {
            connectionState = .error("Invalid URL")
            return
        }

        do {
            let (data, response) = try await URLSession.shared.data(from: statusURL)
            let httpCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard httpCode == 200 else {
                connectionState = .error("Server returned HTTP \(httpCode)")
                return
            }
            let status = try JSONDecoder().decode(StatusResponse.self, from: data)
            let llm = status.services?.activeLLM
            let modelName = llm?.model ?? status.status
            connectionState = .connected(model: modelName)
        } catch let decodeError as DecodingError {
            connectionState = .error("Bad /status format: \(decodeError)")
        } catch {
            connectionState = .error(error.localizedDescription)
        }
    }

    // MARK: - Text Chat (Streaming SSE)

    func sendMessage(_ text: String, image: UIImage? = nil, voiceMode: Bool = false) async {
        guard !text.isEmpty || image != nil else { return }

        // Compress image to JPEG base64 (≤ 1 MB target via 0.7 quality)
        let imageData = image?.jpegData(compressionQuality: 0.7)
        let imageBase64 = imageData.map { $0.base64EncodedString() }

        messages.append(ChatMessage(role: .user, content: text, imageData: imageData))
        let assistantMessage = ChatMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        let assistantID = assistantMessage.id

        isProcessing = true
        defer { isProcessing = false }

        // Re-resolve if we don't have a working URL yet
        if resolvedURL.isEmpty {
            if let (url, route) = await resolveActiveURL() {
                resolvedURL   = url
                activeNetwork = route
            }
        }

        guard let url = URL(string: "\(resolvedURL)/chat") else {
            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                messages[pos].content = "Error: no reachable server"
                messages[pos].isStreaming = false
            }
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONEncoder().encode(
            ChatRequest(message: text, session_id: sessionID,
                        user_code: UserDefaults.standard.string(forKey: "userCode"),
                        model: nil, stream: true, voice_mode: voiceMode,
                        image_base64: imageBase64)
        )

        do {
            let (bytes, _) = try await URLSession.shared.bytes(for: request)

            for try await line in bytes.lines {
                guard !line.isEmpty else { continue }
                guard line.hasPrefix("data: "),
                      let data = String(line.dropFirst(6)).data(using: .utf8),
                      let chunk = try? JSONDecoder().decode(StreamChunk.self, from: data)
                else { continue }

                if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                    if let content = chunk.content {
                        messages[pos].content += content
                    }
                    if chunk.done == true {
                        messages[pos].isStreaming = false
                    }
                }
            }
            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                messages[pos].isStreaming = false
            }
        } catch {
            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                messages[pos].content = "Error: \(error.localizedDescription)"
                messages[pos].isStreaming = false
            }
        }
    }

    // MARK: - Private helpers

    /// Tries local first (2 s timeout), then VPN. Returns the first reachable base URL.
    private func resolveActiveURL() async -> (url: String, route: NetworkRoute)? {
        let candidates: [(String, NetworkRoute)] = [
            (localServerURL, .local),
            (vpnServerURL,   .vpn)
        ]
        for (candidate, route) in candidates {
            guard !candidate.isEmpty,
                  let url = URL(string: "\(candidate)/status") else { continue }
            if (try? await probeSession.data(from: url)) != nil {
                return (candidate, route)
            }
        }
        return nil
    }

    var lastAssistantMessage: String? {
        messages.last(where: { $0.role == .assistant })?.content
    }

    func clearMessages() {
        messages.removeAll()
    }

    /// Clears local messages and server-side conversation history.
    func clearConversation() async {
        messages.removeAll()
        guard let userCode = UserDefaults.standard.string(forKey: "userCode"), !userCode.isEmpty else{ return }
        guard !resolvedURL.isEmpty,
              let url = URL(string: "\(resolvedURL)/conversations/\(userCode)/\(sessionID)") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        _ = try? await URLSession.shared.data(for: req)
    }
    
    func loadConversation() async {
        guard !resolvedURL.isEmpty else { return }

        let userCode = UserDefaults.standard.string(forKey: "userCode") ?? "default"

        guard let url = URL(string: "\(resolvedURL)/users/\(userCode)/history/\(sessionID)") else {
            return
        }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)

            let history = try JSONDecoder().decode([HistoryMessage].self, from: data)

            messages = history.map {
                ChatMessage(role: ChatMessage.Role(rawValue: $0.role) ?? .assistant,
                            content: $0.content)
            }

        } catch {
            print("History load error:", error)
        }
    }
}

enum NetworkRoute: Equatable {
    case local, vpn, unknown

    var label: String {
        switch self {
        case .local:   return "Local"
        case .vpn:     return "VPN"
        case .unknown: return "—"
        }
    }

    var icon: String {
        switch self {
        case .local:   return "wifi"
        case .vpn:     return "network.badge.shield.half.filled"
        case .unknown: return "questionmark.circle"
        }
    }
}

