// ContentView.swift

import SwiftUI
import Combine

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            ChatView()
                .tabItem { Image(systemName: "message.fill"); Text("Chat") }
                .tag(0)
            VoiceView()
                .tabItem { Image(systemName: "waveform"); Text("Voice") }
                .tag(1)
        }
        .tint(.jarvisBlue)
        .preferredColorScheme(.dark)
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let r = Double((int >> 16) & 0xFF) / 255
        let g = Double((int >> 8) & 0xFF) / 255
        let b = Double(int & 0xFF) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1)
    }

    // MARK: - App palette (parsed once, not on every render)
    static let jarvisBlue    = Color(hex: "3B82F6")
    static let jarvisRed     = Color(hex: "EF4444")
    static let jarvisAmber   = Color(hex: "F59E0B")
    static let jarvisGreen   = Color(hex: "10B981")
    static let jarvisIndigo  = Color(hex: "6366F1")
    static let jarvisBg      = Color(hex: "0A0A0F")
    static let jarvisBgDeep  = Color(hex: "0D0D14")
    static let jarvisCard    = Color(hex: "1A1A24")
    static let jarvisBar     = Color(hex: "111118")
}
