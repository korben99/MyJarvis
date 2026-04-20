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
        config.timeoutIntervalForRequest  = 20
        config.timeoutIntervalForResource = 60
        return URLSession(configuration: config)
    }()

    // URLSession for SSE streaming. URLSession.shared has a 7-day resource timeout —
    // a server that hangs mid-stream would hold the connection open indefinitely.
    //
    // timeoutIntervalForRequest  = time without receiving ANY byte (= TTFT budget).
    //   Worst case: 512-token think budget (~8 s) + large-context prefill (~12 s)
    //   + _infer_lock wait (~5 s) = ~25 s. 60 s gives a 35 s safety margin.
    // timeoutIntervalForResource = total stream duration cap.
    //   Longest plausible response: 2000-token briefing @ 60 tok/s (~33 s) + 25 s TTFT
    //   = ~58 s. 180 s kills truly stalled streams without cutting off legitimate ones.
    private let sseSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 60
        config.timeoutIntervalForResource = 180
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
            // Classify the error before touching connection state.
            //
            // Three distinct cases:
            // 1. timedOut              — server reachable but LLM slow → try recovery reload
            // 2. networkConnectionLost — TCP dropped mid-stream (app backgrounded, screen
            //                           locked, network handoff) → try recovery reload.
            //                           NOT the same as being offline: the connection was
            //                           established and the server may have finished.
            // 3. notConnectedToInternet / cannotConnectToHost / cannotFindHost
            //                         — server truly unreachable → surface error, clear URL.
            let urlErr = error as? URLError
            let isServerUnreachable = [
                URLError.Code.notConnectedToInternet,
                .cannotConnectToHost,
                .cannotFindHost,
            ].contains(urlErr?.code)

            if let pos = messages.firstIndex(where: { $0.id == assistantID }) {
                switch urlErr?.code {
                case .timedOut:
                    messages[pos].content = "⏱ Pas de réponse dans les délais — rechargement…"
                case .networkConnectionLost:
                    // Connection dropped (backgrounding, screen lock, network switch).
                    // Server-side the response was saved — reload will recover it.
                    messages[pos].content = "Connexion interrompue — rechargement…"
                case .notConnectedToInternet, .cannotConnectToHost, .cannotFindHost:
                    messages[pos].content = "Réseau inaccessible."
                default:
                    messages[pos].content = "Erreur : \(error.localizedDescription)"
                }
                messages[pos].isStreaming = false
            }

            if isServerUnreachable {
                // Server genuinely unreachable — clear cached URL so next send
                // triggers a fresh probe, and surface the disconnected state.
                resolvedURL = ""
                connectionState = .error("Serveur inaccessible")
            } else {
                // Timeout or connection-lost — server is probably still up and may
                // have finished generating the response. Reload to recover it.
                await loadConversation()
            }
        }
    }

    // MARK: - Private helpers

    /// Probes local and VPN in parallel and returns whichever responds first.
    /// Prefers local when both answer within the same probe window.
    /// Uses a TaskGroup so neither probe blocks the other — if local times out
    /// after 2 s the VPN result (already in-flight) is returned immediately.
    private func resolveActiveURL() async -> (url: String, route: NetworkRoute, data: Data)? {
        await withTaskGroup(of: (url: String, route: NetworkRoute, data: Data)?.self) { group in
            group.addTask { await self.probe(base: self.localServerURL, route: .local) }
            group.addTask { await self.probe(base: self.vpnServerURL,   route: .vpn) }

            var vpnFallback: (url: String, route: NetworkRoute, data: Data)? = nil
            for await result in group {
                guard let r = result else { continue }
                if r.route == .local {
                    group.cancelAll()
                    return r          // local always wins if it responds
                }
                vpnFallback = r       // keep VPN in case local times out
            }
            return vpnFallback
        }
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

