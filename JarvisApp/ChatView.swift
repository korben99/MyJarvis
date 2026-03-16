// ChatView.swift
// Text chat with streaming responses

import SwiftUI
import Combine

struct ChatView: View {
    @EnvironmentObject var api: JarvisAPI
    @State private var inputText = ""
    @State private var showSettings = false
    @FocusState private var isInputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                HStack(spacing: 6) {
                    Circle()
                        .fill(api.connectionState.isConnected ? Color.green : Color.red)
                        .frame(width: 6, height: 6)
                    Text(api.connectionState.statusText)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundColor(.gray)
                    Spacer()
                    if api.isProcessing {
                        ProgressView().scaleEffect(0.6).tint(Color.jarvisBlue)
                    }
                }
                .padding(.horizontal, 16).padding(.vertical, 6)
                .background(Color.jarvisBar)

                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            if api.messages.isEmpty {
                                VStack(spacing: 16) {
                                    Spacer().frame(height: 80)
                                    Image(systemName: "sparkles")
                                        .font(.system(size: 40))
                                        .foregroundColor(.jarvisBlue.opacity(0.5))
                                    Text("Hello.")
                                        .font(.system(size: 28, weight: .light))
                                        .foregroundColor(.white.opacity(0.7))
                                    Text("What can I help you with?")
                                        .font(.system(size: 15)).foregroundColor(.gray)
                                }
                                .frame(maxWidth: .infinity).padding(.top, 60)
                            }
                            ForEach(api.messages) { msg in
                                MessageBubble(message: msg).id(msg.id)
                            }
                        }
                        .padding(.horizontal, 16).padding(.vertical, 12)
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onChange(of: api.messages.count) { _, _ in
                        scrollToBottom(proxy)
                    }
                    .onChange(of: api.isProcessing) { _, _ in
                        scrollToBottom(proxy)
                    }
                }
            }
            .background(Color.jarvisBg)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Text("I'm Jarvis").font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundColor(.jarvisAmber)
                        .fixedSize()
                }
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 12) {
                        Button { Task { await api.clearConversation() } } label: {
                            Image(systemName: "trash").foregroundColor(.gray)
                        }
                        Button { showSettings = true } label: {
                            Image(systemName: "gear").foregroundColor(.gray)
                        }
                    }
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .safeAreaInset(edge: .bottom) {
                HStack(spacing: 10) {
                    TextField("Message à Jarvis...", text: $inputText, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.system(size: 16))
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Color.jarvisCard)
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .lineLimit(1...5)
                        .focused($isInputFocused)
                        .onSubmit { send() }

                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 32))
                            .foregroundColor(inputText.isEmpty || api.isProcessing
                                ? .gray.opacity(0.3) : .jarvisBlue)
                    }
                    .disabled(inputText.isEmpty || api.isProcessing)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.jarvisBgDeep)
            }
        }

    }

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""
        Task { await api.sendMessage(text) }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        if let last = api.messages.last {
            withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(last.id, anchor: .bottom) }
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage
    private var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if isUser { Spacer(minLength: 48) }

            if !isUser {
                ZStack {
                    Circle().fill(Color.jarvisBlue.opacity(0.15)).frame(width: 28, height: 28)
                    Text("J").font(.system(size: 13, weight: .bold, design: .monospaced))
                        .foregroundColor(.jarvisBlue)
                }.padding(.top, 2)
            }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content + (message.isStreaming ? " ●" : ""))
                    .font(.system(size: 15))
                    .foregroundColor(isUser ? .white : .white.opacity(0.9))
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(isUser
                        ? Color.jarvisBlue.opacity(0.2)
                        : Color.jarvisCard)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .textSelection(.enabled)

                Text(timeString)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.gray.opacity(0.5))
            }

            if !isUser { Spacer(minLength: 48) }
        }
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        return f
    }()

    private var timeString: String {
        Self.timeFormatter.string(from: message.timestamp)
    }
}

