// ThinkingBanner.swift
// Transparent single-line ticker that scrolls Jarvis's raw <think> content
// while the model is reasoning. Appears above the input bar, disappears the
// moment the first visible token is emitted.

import SwiftUI

struct ThinkingBanner: View {

    let text: String

    // Gear rotation angle, driven by a repeating SwiftUI animation.
    @State private var gearAngle: Double = 0

    // Strip newlines and collapse runs of spaces — think blocks often contain
    // structured reasoning with newlines that would break single-line display.
    private var clean: String {
        text
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }

    // Show only the most recent characters so the ticker always reflects
    // current thinking rather than text that arrived many seconds ago.
    // At 100 tok/sec the full accumulation would scroll off-screen instantly.
    private var displayText: String {
        let window = 120
        return clean.count > window ? String(clean.suffix(window)) : clean
    }

    var body: some View {
        HStack(spacing: 0) {

            // ── Fixed left badge: rotating gear ───────────────────────────
            Image(systemName: "gearshape.fill")
                .font(.system(size: 10))
                .foregroundColor(.jarvisAmber.opacity(0.65))
                .rotationEffect(.degrees(gearAngle))
                .frame(width: 32)
                .onAppear {
                    withAnimation(.linear(duration: 3).repeatForever(autoreverses: false)) {
                        gearAngle = 360
                    }
                }

            // ── Thin separator ────────────────────────────────────────────
            Rectangle()
                .fill(Color.white.opacity(0.08))
                .frame(width: 1, height: 14)

            // ── Scrolling ticker ──────────────────────────────────────────
            // Text is end-anchored: always shows the latest tokens at the right
            // edge. When the text is short enough to fit, it starts from x=0
            // (beginning visible). As thinking grows, the view shifts left to
            // keep the newest content visible — no looping animation.
            GeometryReader { geo in
                let charW: CGFloat = 6.8
                let textWidth = CGFloat(displayText.count) * charW
                let xOff = min(0.0, geo.size.width - textWidth)

                Text(displayText)
                    .font(.system(size: 11, weight: .light, design: .monospaced))
                    .foregroundColor(.white.opacity(0.45))
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                    .frame(width: max(textWidth, 1), alignment: .leading)
                    .offset(x: xOff)
            }
            .clipped()
            .padding(.trailing, 10)
        }
        .frame(height: 28)
        .background(Color.jarvisBar)
    }
}
