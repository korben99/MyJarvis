// ChatView.swift
// Text chat with streaming responses

import SwiftUI
import Combine
import PhotosUI
import UIKit

@MainActor
struct ChatView: View {
    @EnvironmentObject var api: JarvisAPI
    @State private var inputText = ""
    @State private var showSettings = false
    @State private var selectedPhotoItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @FocusState private var isInputFocused: Bool
    // When the user drags the scroll view during streaming, auto-scroll is
    // suspended so they can read earlier content. Resets when a new message
    // arrives or streaming ends.
    @State private var isAutoScrollEnabled = true

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
                                // .equatable() skips re-rendering bubbles whose message
                                // hasn't changed — critical when streaming updates the last
                                // bubble only, but previously caused all bubbles to re-draw.
                                MessageBubble(message: msg).equatable().id(msg.id)
                            }
                        }
                        .padding(.horizontal, 16).padding(.vertical, 12)
                    }
                    .scrollDismissesKeyboard(.interactively)
                    // Disable auto-scroll if the user manually drags during streaming,
                    // so they can read earlier content without being yanked back.
                    .simultaneousGesture(DragGesture().onChanged { _ in
                        if api.messages.last?.isStreaming == true { isAutoScrollEnabled = false }
                    })
                    .onChange(of: api.messages.count) { _, _ in
                        isAutoScrollEnabled = true   // new message: re-enable
                        scrollToBottom(proxy)
                    }
                    .onChange(of: api.isProcessing) { @MainActor _, isProcessing in
                        if !isProcessing { isAutoScrollEnabled = true }  // stream ended: re-enable
                        scrollToBottom(proxy)
                    }
                    // Scroll to bottom as streaming content grows — gated on
                    // isAutoScrollEnabled so a manual drag can pause it.
                    .onChange(of: api.messages.last?.content.count) { _, _ in
                        guard let last = api.messages.last, last.isStreaming, isAutoScrollEnabled else { return }
                        proxy.scrollTo(last.id, anchor: .bottom)
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
                        .disabled(api.isProcessing)
                        Button { showSettings = true } label: {
                            Image(systemName: "gear").foregroundColor(.gray)
                        }
                    }
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .safeAreaInset(edge: .bottom) {
                VStack(spacing: 0) {
                    // Image preview strip
                    if let img = selectedImage {
                        HStack {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 64, height: 64)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                                .overlay(alignment: .topTrailing) {
                                    Button { selectedImage = nil; selectedPhotoItem = nil } label: {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundColor(.white)
                                            .background(Color.black.opacity(0.5), in: Circle())
                                    }
                                    .offset(x: 6, y: -6)
                                }
                            Spacer()
                        }
                        .padding(.horizontal, 16)
                        .padding(.top, 8)
                    }

                    HStack(spacing: 10) {
                        let processing = api.isProcessing
                        // Photo picker button
                        PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                            Image(systemName: "photo")
                                .font(.system(size: 22))
                                .foregroundColor(processing ? .gray.opacity(0.3) : .gray)
                        }
                        .disabled(processing)
                        .onChange(of: selectedPhotoItem) { _, item in
                            Task {
                                if let data = try? await item?.loadTransferable(type: Data.self),
                                   let ui = UIImage(data: data) {
                                    selectedImage = ui
                                }
                            }
                        }

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
                                .foregroundColor(canSend ? .jarvisBlue : .gray.opacity(0.3))
                        }
                        .disabled(!canSend)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
                .background(Color.jarvisBgDeep)
            }
        }

    }

    private var canSend: Bool {
        (!inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || selectedImage != nil)
            && !api.isProcessing
    }

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend else { return }
        let image = selectedImage
        inputText = ""
        selectedImage = nil
        selectedPhotoItem = nil
        Task { await api.sendMessage(text, image: image) }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        if let last = api.messages.last {
            withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(last.id, anchor: .bottom) }
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View, Equatable {
    let message: ChatMessage
    // Cache the decoded UIImage so it isn't re-decoded from Data on every render.
    @State private var cachedImage: UIImage?

    private var isUser: Bool { message.role == .user }

    // Equatable: delegate to ChatMessage which compares id + content + isStreaming.
    static func == (lhs: MessageBubble, rhs: MessageBubble) -> Bool {
        lhs.message == rhs.message
    }

    /// Renders the message content as an AttributedString so that Markdown
    /// inline syntax (**bold**, *italic*, `code`, etc.) is displayed correctly.
    /// Falls back to plain text if parsing fails.
    private var renderedContent: AttributedString {
        let raw = message.content + (message.isStreaming ? " ●" : "")
        let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: raw, options: opts)) ?? AttributedString(raw)
    }

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
                // Image thumbnail (user messages only) — uses @State cache to avoid
                // re-decoding the JPEG on every render.
                if let ui = cachedImage {
                    Image(uiImage: ui)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: 200, maxHeight: 200)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }

                if !message.content.isEmpty || message.isStreaming {
                    Text(renderedContent)
                        .font(.system(size: 15))
                        .foregroundColor(isUser ? .white : .white.opacity(0.9))
                        .padding(.horizontal, 14).padding(.vertical, 10)
                        .background(isUser
                            ? Color.jarvisBlue.opacity(0.2)
                            : Color.jarvisCard)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        // Disable text selection while streaming — the selection overlay
                        // tracks character positions in the growing string, which is very
                        // expensive at 10fps for long responses.
                        .textSelectionIfEnabled(!message.isStreaming)
                }

                Text(timeString)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.gray.opacity(0.5))
            }
            .onAppear {
                if cachedImage == nil, let data = message.imageData {
                    cachedImage = UIImage(data: data)
                }
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

// MARK: - Helpers

private extension View {
    /// Applies .textSelection(.enabled) only when `enabled` is true.
    /// A ternary won't compile here because EnabledTextSelectability and
    /// DisabledTextSelectability are distinct concrete types, not the same generic.
    @ViewBuilder func textSelectionIfEnabled(_ enabled: Bool) -> some View {
        if enabled { self.textSelection(.enabled) } else { self }
    }
}

