// Models.swift
// Data models for chat messages and API responses

import Foundation
import Combine

// MARK: - Chat Message

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let role: Role
    var content: String
    var imageData: Data?        // thumbnail shown in the user bubble (not persisted to server)
    let timestamp: Date
    var isStreaming: Bool

    enum Role: String {
        case user, assistant, system
    }

    init(role: Role, content: String, imageData: Data? = nil, isStreaming: Bool = false) {
        self.role = role
        self.content = content
        self.imageData = imageData
        self.timestamp = Date()
        self.isStreaming = isStreaming
    }

    static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
        lhs.id == rhs.id && lhs.content == rhs.content && lhs.isStreaming == rhs.isStreaming
    }
}

// MARK: History
struct HistoryMessage: Codable {
    let role: String
    let content: String
}

// MARK: - API Types

struct ChatRequest: Codable {
    let message: String
    let session_id: String
    let user_code: String?
    let model: String?
    let stream: Bool
    let voice_mode: Bool
    let image_base64: String?
}

struct StreamChunk: Codable {
    let content: String?
    let think:   String?
    let done:    Bool?
    let model:   String?
    let duration_ms: Int?
}

struct StatusResponse: Codable {
    let status: String
    let services: Services?

    struct Services: Codable {
        let openai: LLMStatus?
        let ollama: LLMStatus?
        let llm: LLMStatus?

        struct LLMStatus: Codable {
            let status: String?
            let model: String?
            let url: String?
        }

        var activeLLM: LLMStatus? { openai ?? ollama ?? llm }
    }
}

// MARK: - Connection State

enum ConnectionState: Equatable {
    case disconnected
    case connecting
    case connected(model: String)
    case error(String)

    var isConnected: Bool {
        if case .connected = self { return true }
        return false
    }

    var statusText: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected(let model): return model
        case .error(let msg): return msg
        }
    }
}

