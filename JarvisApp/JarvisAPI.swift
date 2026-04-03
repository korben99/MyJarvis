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
    var resolvedURL            = ""   // internal — read by NotificationService
    private var sessionID      = "iphone-main"

    // URLSession with a short timeout used only for reachability probes
    private let probeSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 2
        config.timeoutIntervalForResource = 2
        return URLSession(configuration: config)
    }()

    // URLSession for non-streaming API calls (history, clear). 10 s request timeout
    // prevents these from hanging for the URLSession.shared default of 60 s.
    private let apiSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 10
        config.timeoutIntervalForResource = 30
        return URLSession(configuration: config)
    }()

    // URLSession for SSE streaming. URLSession.shared has a 7-day resource timeout —
    // a server that hangs mid-stream would hold the connection open indefinitely.
    // 120 s resource timeout kills a stalled stream well within any reasonable LLM latency.
    private let sseSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 30
        config.timeoutIntervalForResource = 120
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

        // resolveActiveURL() already fetches and validates /status (HTTP 200).
        // We reuse its response body directly — no second network round-trip needed.
        guard let (url, route, statusData) = await resolveActiveURL() else {
            // Guard against cancelled tasks (e.g. rapid configure() calls while the
            // user is typing in Settings) writing a spurious error to the UI.
            guard !Task.isCancelled else { return }
            connectionState = .error("No server — check IP & Local Network permission")
            activeNetwork   = .unknown
            return
        }
        guard !Task.isCancelled else { return }

        resolvedURL   = url
        activeNetwork = route

        do {
            let status = try JSONDecoder().decode(StatusResponse.self, from: statusData)
            let llm = status.services?.activeLLM
            connectionState = .connected(model: llm?.model ?? status.status)
        } catch {
            connectionState = .error("Bad /status format: \(error)")
        }
    }

    // MARK: - Text Chat (Streaming SSE)

    func sendMessage(_ text: String, image: UIImage? = nil, voiceMode: Bool = false) async {
        guard !text.isEmpty || image != nil else { return }

        // Compress to JPEG (≤ 1 MB target via 0.7 quality).
        let imageData = image?.jpegData(compressionQuality: 0.7)
        let imageBase64 = imageData.map { $0.base64EncodedString() }

        messages.append(ChatMessage(role: .user, content: text, imageData: imageData))
        let assistantMessage = ChatMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        let assistantID = assistantMessage.id

        isProcessing = true
        defer { isProcessing = false }

        // Re-resolve if we don't have a working URL (initial state, or cleared after a failure).
        if resolvedURL.isEmpty {
            if let (url, route, _) = await resolveActiveURL() {
                resolvedURL   = url
                activeNetwork = route
            }
        }

        // resolvedURL can be empty if the probe failed above; an empty prefix produces
        // URL(string: "/chat") which is non-nil, so we must guard explicitly.
        guard !resolvedURL.isEmpty, let url = URL(string: "\(resolvedURL)/chat") else {
            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                messages[pos].content = "Error: no reachable server"
                messages[pos].isStreaming = false
            }
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        guard let body = try? JSONEncoder().encode(
            ChatRequest(message: text, session_id: sessionID,
                        user_code: UserDefaults.standard.string(forKey: "userCode"),
                        model: nil, stream: true, voice_mode: voiceMode,
                        image_base64: imageBase64)
        ) else {
            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                messages[pos].content = "Error: failed to encode request"
                messages[pos].isStreaming = false
            }
            return
        }
        request.httpBody = body

        do {
            let (bytes, response) = try await sseSession.bytes(for: request)
            if let httpCode = (response as? HTTPURLResponse)?.statusCode, httpCode != 200 {
                if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                    messages[pos].content = "Error: server returned HTTP \(httpCode)"
                    messages[pos].isStreaming = false
                }
                return
            }

            // Accumulate tokens and flush to the UI at most ~30fps (every 33ms).
            // This cuts @Published emissions from ~25/sec to ~30/sec → fewer SwiftUI
            // re-renders while the chat list grows, dramatically improving scroll performance.
            var pendingContent = ""
            var lastFlush = ContinuousClock.now

            for try await line in bytes.lines {
                guard !line.isEmpty else { continue }
                guard line.hasPrefix("data: "),
                      let data = String(line.dropFirst(6)).data(using: .utf8),
                      let chunk = try? JSONDecoder().decode(StreamChunk.self, from: data)
                else { continue }

                if let content = chunk.content { pendingContent += content }

                let isDone = chunk.done == true
                let now = ContinuousClock.now
                if isDone || now - lastFlush >= .milliseconds(100) {
                    if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                        if !pendingContent.isEmpty {
                            messages[pos].content += pendingContent
                            pendingContent = ""
                        }
                        if isDone { messages[pos].isStreaming = false }
                    }
                    lastFlush = now
                }
            }
            // Flush any remaining buffered content and mark stream done.
            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                if !pendingContent.isEmpty { messages[pos].content += pendingContent }
                messages[pos].isStreaming = false
            }
        } catch {
            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                messages[pos].content = "Error: \(error.localizedDescription)"
                messages[pos].isStreaming = false
            }
            // Invalidate the cached URL so the next send triggers a fresh probe
            // rather than hammering a server we already know is unreachable.
            resolvedURL = ""
            connectionState = .error("Connection lost")
        }
    }

    // MARK: - Private helpers

    /// Probes local and VPN in parallel — if local is unreachable the VPN probe
    /// is already in-flight, so the result comes back immediately after the 2 s
    /// local timeout instead of after an additional 2 s.  Prefers local on a tie.
    private func resolveActiveURL() async -> (url: String, route: NetworkRoute, data: Data)? {
        async let localResult = probe(base: localServerURL, route: .local)
        async let vpnResult   = probe(base: vpnServerURL,   route: .vpn)
        if let r = await localResult { return r }
        if let r = await vpnResult   { return r }
        return nil
    }

    private func probe(base: String, route: NetworkRoute) async -> (url: String, route: NetworkRoute, data: Data)? {
        guard !base.isEmpty, let url = URL(string: "\(base)/status") else { return nil }
        guard let (data, _) = try? await probeSession.data(from: url) else { return nil }
        return (base, route, data)
    }

    var lastAssistantMessage: String? {
        messages.last(where: { $0.role == .assistant })?.content
    }

    /// Clears local messages and server-side conversation history.
    func clearConversation() async {
        messages.removeAll()
        guard let userCode = UserDefaults.standard.string(forKey: "userCode"), !userCode.isEmpty else { return }
        guard !resolvedURL.isEmpty,
              let url = URL(string: "\(resolvedURL)/conversations/\(userCode)/\(sessionID)") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "DELETE"
        _ = try? await apiSession.data(for: req)
    }

    func loadConversation() async {
        // Don't clobber an in-flight stream — isProcessing is true while SSE is active.
        guard !resolvedURL.isEmpty, !isProcessing else { return }

        let userCode = UserDefaults.standard.string(forKey: "userCode") ?? "default"

        guard let url = URL(string: "\(resolvedURL)/users/\(userCode)/history/\(sessionID)") else {
            return
        }

        do {
            let (data, _) = try await apiSession.data(from: url)
            let history = try JSONDecoder().decode([HistoryMessage].self, from: data)

            // Smart merge — avoid replacing every ChatMessage with a new UUID,
            // which would force SwiftUI to re-render every bubble unnecessarily.
            //
            // Case 1: nothing changed → no-op, zero re-renders.
            // Case 2: server has more messages (jarvis-core added some) and the
            //         existing local messages are a clean prefix → append only the
            //         new tail, preserving existing bubble identity.
            // Case 3: anything else (clear, edit, mismatch) → full replace.

            if history.count == messages.count,
               zip(messages, history).allSatisfy({ $0.role.rawValue == $1.role && $0.content == $1.content }) {
                return  // no change
            }

            if history.count > messages.count,
               zip(messages, history).allSatisfy({ $0.role.rawValue == $1.role && $0.content == $1.content }) {
                // Existing messages are a prefix of history — append only new ones.
                let newMessages = history.dropFirst(messages.count).map {
                    ChatMessage(role: ChatMessage.Role(rawValue: $0.role) ?? .assistant,
                                content: $0.content)
                }
                messages.append(contentsOf: newMessages)
                return
            }

            // Full replace fallback (conversation was cleared or history diverged).
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

