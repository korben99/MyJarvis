// ThinkingBanner.swift
// Transparent single-line ticker that scrolls Jarvis's raw <think> content
// while the model is reasoning. Appears above the input bar, disappears the
// moment the first visible token is emitted.

import SwiftUI

struct ThinkingBanner: View {

    let text: String

    // Anchor for elapsed-time computation — reset on every appear.
    @State private var startTime: Date = .now
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
            // TimelineView drives the offset from wall-clock time so that:
            //   • text updates (new think chunks) never restart the animation
            //   • the scroll speed is constant regardless of text length
            GeometryReader { geo in
                TimelineView(.animation(minimumInterval: 1.0 / 30, paused: displayText.isEmpty)) { tl in
                    let elapsed = tl.date.timeIntervalSince(startTime)
                    // Fixed frame width = worst-case 120 chars × 6.8 pt (SF Mono 11 pt).
                    // Using a constant here prevents the cycle from changing as tokens
                    // arrive (displayText grows 0→120), which would cause phase jumps
                    // and make the container appear to resize.
                    let textW  = 120.0 * 6.8          // 816 pt — constant
                    let contW  = max(Double(geo.size.width), 1.0)  // guard against 0 on first pass
                    let cycle  = textW + contW
                    let speed  = 80.0
                    let phase  = (elapsed * speed).truncatingRemainder(dividingBy: cycle)
                    let xOff   = contW - phase

                    Text(displayText)
                        .font(.system(size: 11, weight: .light, design: .monospaced))
                        .foregroundColor(.white.opacity(0.45))
                        .lineLimit(1)
                        // Explicit fixed frame instead of fixedSize(horizontal:).
                        // fixedSize propagates the text's ideal width up through
                        // TimelineView → GeometryReader → HStack, making the
                        // clipping window appear to change size on each layout pass.
                        .frame(width: CGFloat(textW), alignment: .leading)
                        .offset(x: xOff)
                }
            }
            .clipped()
            .padding(.trailing, 10)
        }
        .frame(height: 28)
        .background(Color.jarvisBar)
        .onAppear { startTime = .now }
    }
}
