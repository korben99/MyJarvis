// NotificationService.swift
// Jarvis proactive push notifications — Phase 1 (polling, no APNs required)
//
// Strategy:
//   • Foreground: poll /device/pending/{userCode} every 15 min via a Timer.
//   • Background: BGAppRefreshTask polls once (~every 2h, iOS-throttled).
//   • All notifications are displayed as local UNUserNotificationCenter alerts.
//
// To enable background refresh in Xcode:
//   Signing & Capabilities → Background Modes → Background App Refresh
//   Info.plist → BGTaskSchedulerPermittedIdentifiers → ["fr.jarvis.push-poll"]

import Foundation
import UserNotifications
import BackgroundTasks
import UIKit

// MARK: - Background task identifier

let kJarvisPushPollTaskID = "fr.jarvis.push-poll"

// MARK: - NotificationService

@MainActor
final class NotificationService: ObservableObject {

    // Set by JarvisApp after configure() so this service knows where to call.
    var resolvedURL: String = ""
    var userCode:    String = ""

    private var pollTimer: Timer?
    private let pollInterval: TimeInterval = 15 * 60   // 15 min foreground interval

    // MARK: - Permissions

    /// Call once at startup. Requests UNUserNotification authorization.
    func requestPermissions() async {
        let center = UNUserNotificationCenter.current()
        _ = try? await center.requestAuthorization(options: [.alert, .sound, .badge])
    }

    // MARK: - Foreground polling

    /// Start a repeating timer that polls for pending pushes while the app is active.
    func startForegroundPolling() {
        stopForegroundPolling()
        pollTimer = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { await self.pollAndDeliver() }
        }
    }

    func stopForegroundPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    // MARK: - Background task registration

    /// Register the BGAppRefreshTask identifier. Call from app init before the first scene.
    static func registerBackgroundTask() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: kJarvisPushPollTaskID, using: nil) { task in
            guard let refreshTask = task as? BGAppRefreshTask else { return }
            Task { @MainActor in
                await NotificationService.shared.handleBackgroundTask(refreshTask)
            }
        }
    }

    /// Schedule the next background refresh. Call after each execution and on app background.
    func scheduleBackgroundRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: kJarvisPushPollTaskID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)   // iOS may delay this
        try? BGTaskScheduler.shared.submit(request)
    }

    // MARK: - Shared singleton (needed for the static BGTask handler)

    static let shared = NotificationService()

    // MARK: - Background task handler

    private func handleBackgroundTask(_ task: BGAppRefreshTask) async {
        // Schedule the next refresh immediately so iOS queues it
        scheduleBackgroundRefresh()

        task.expirationHandler = { task.setTaskCompleted(success: false) }

        await pollAndDeliver()
        task.setTaskCompleted(success: true)
    }

    // MARK: - Poll & deliver

    /// Call /device/pending/{userCode}, show a local notification for each message.
    func pollAndDeliver() async {
        guard !resolvedURL.isEmpty, !userCode.isEmpty else { return }
        guard let url = URL(string: "\(resolvedURL)/device/pending/\(userCode)") else { return }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let response  = try JSONDecoder().decode(PendingPushResponse.self, from: data)
            for push in response.messages {
                await deliver(push.message)
            }
        } catch {
            // Silent failure — network may be unavailable in background
        }
    }

    // MARK: - Device registration

    /// Register this device with the backend so the reflection cycle can queue pushes.
    /// Call once after the server URL is resolved (a token of "ios-polling" is used
    /// as a placeholder — APNs tokens are not needed for the polling strategy).
    func registerDevice() async {
        guard !resolvedURL.isEmpty, !userCode.isEmpty else { return }
        guard let url = URL(string: "\(resolvedURL)/device/register") else { return }

        let body = DeviceRegisterRequest(user_code: userCode, device_token: "ios-polling-\(userCode)")
        var request    = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody   = try? JSONEncoder().encode(body)
        _ = try? await URLSession.shared.data(for: request)
    }

    // MARK: - Local notification delivery

    private func deliver(_ message: String) async {
        let content           = UNMutableNotificationContent()
        content.title         = "Jarvis"
        content.body          = message
        content.sound         = .default

        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content:    content,
            trigger:    trigger
        )
        try? await UNUserNotificationCenter.current().add(request)
    }
}

// MARK: - Codable models

private struct PendingPushResponse: Decodable {
    let messages: [PushMessage]
    let count:    Int
}

private struct PushMessage: Decodable {
    let message:   String
    let queued_at: String
}

private struct DeviceRegisterRequest: Encodable {
    let user_code:    String
    let device_token: String
}
