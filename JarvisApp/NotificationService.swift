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
import Combine
import UserNotifications
import BackgroundTasks
import UIKit

// MARK: - NotificationService

@MainActor
final class NotificationService: NSObject, ObservableObject {

    // nonisolated: must be reachable from nonisolated contexts (registerBackgroundTask,
    // scheduleBackgroundRefresh) without a main-actor hop.
    nonisolated static let taskID = "fr.jarvis.push-poll"

    // Set by JarvisApp after configure() so this service knows where to call.
    var resolvedURL:    String = ""
    var userCode:       String = ""
    // Both candidate URLs — used to re-probe on a cold background launch.
    var localServerURL: String = ""
    var vpnServerURL:   String = ""

    private var pollTimer: Timer?
    private let pollInterval: TimeInterval = 30 * 60   // 30 min foreground interval
    private var isPolling = false   // prevents concurrent polls delivering duplicate notifications

    // Short-timeout session: background tasks have limited execution time (~30 s),
    // so we can't afford URLSession.shared's 60 s default request timeout.
    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 10
        config.timeoutIntervalForResource = 20
        return URLSession(configuration: config)
    }()

    // MARK: - Permissions

    /// Call once at startup. Requests UNUserNotification authorization
    /// and sets self as the notification center delegate so notifications
    /// are also displayed while the app is in the foreground.
    func requestPermissions() async {
        let center = UNUserNotificationCenter.current()
        center.delegate = self
        _ = try? await center.requestAuthorization(options: [.alert, .sound, .badge])
    }

    // MARK: - Foreground polling

    /// Start a repeating timer that polls for pending pushes while the app is active.
    func startForegroundPolling() {
        stopForegroundPolling()
        pollTimer = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                await self?.pollAndDeliver()
            }
        }
    }

    func stopForegroundPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    // MARK: - Background task registration

    /// Register the BGAppRefreshTask identifier. Call from app init before the first scene.
    /// nonisolated: BGTaskScheduler.register is thread-safe and this method holds no
    /// actor-isolated state, so it's safe to call from JarvisApp.init() without a Task hop.
    nonisolated static func registerBackgroundTask() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: Self.taskID, using: nil) { task in
            guard let refreshTask = task as? BGAppRefreshTask else { return }
            Task { @MainActor in
                await NotificationService.shared.handleBackgroundTask(refreshTask)
            }
        }
    }

    /// Schedule the next background refresh. Call after each execution and on app background.
    func scheduleBackgroundRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: Self.taskID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 30 * 60)   // iOS may delay this
        try? BGTaskScheduler.shared.submit(request)
    }

    // MARK: - Shared singleton (needed for the static BGTask handler)

    static let shared = NotificationService()

    // MARK: - Background task handler

    private func handleBackgroundTask(_ task: BGAppRefreshTask) async {
        // Schedule the next refresh immediately so iOS queues it
        scheduleBackgroundRefresh()

        // Use [weak task] to avoid a retain cycle: the task holds its expirationHandler,
        // and the handler must not hold a strong reference back to the task.
        task.expirationHandler = { [weak task] in task?.setTaskCompleted(success: false) }

        await pollAndDeliver()
        task.setTaskCompleted(success: true)
    }

    // MARK: - URL resolution

    /// Probes local and VPN in parallel — mirrors JarvisAPI.resolveActiveURL().
    /// On a cold background launch the in-memory URLs are empty (no UI ever rendered),
    /// so we fall back to the @AppStorage values persisted by AppSettings.
    private func resolveURL() async -> String? {
        let local = localServerURL.isEmpty
            ? (UserDefaults.standard.string(forKey: "localServerURL") ?? "")
            : localServerURL
        let vpn = vpnServerURL.isEmpty
            ? (UserDefaults.standard.string(forKey: "vpnServerURL") ?? "")
            : vpnServerURL
        async let localResult = probe(candidate: local)
        async let vpnResult   = probe(candidate: vpn)
        if let r = await localResult { return r }
        if let r = await vpnResult   { return r }
        return nil
    }

    /// 2 s timeout — matches JarvisAPI's probeSession and stays well inside the
    /// ~30 s BGAppRefreshTask execution budget.
    private func probe(candidate: String) async -> String? {
        guard !candidate.isEmpty, let url = URL(string: "\(candidate)/status") else { return nil }
        let request = URLRequest(url: url, timeoutInterval: 2)
        if (try? await session.data(for: request)) != nil { return candidate }
        return nil
    }

    // MARK: - Poll & deliver

    /// Call /device/pending/{userCode}, show a local notification for each message.
    func pollAndDeliver() async {
        guard !isPolling else { return }
        isPolling = true
        defer { isPolling = false }

        // On a cold background launch (app killed, woken by BGAppRefreshTask), JarvisAPI
        // never gets to call configure(), so resolvedURL/userCode are empty. Re-probe to
        // find a live server — tries local first, then VPN, matching JarvisAPI's routing.
        if resolvedURL.isEmpty {
            resolvedURL = await resolveURL()
                       ?? UserDefaults.standard.string(forKey: "lastResolvedURL")
                       ?? ""
        }
        if userCode.isEmpty {
            userCode = UserDefaults.standard.string(forKey: "userCode") ?? ""
        }
        guard !resolvedURL.isEmpty, !userCode.isEmpty else { return }
        guard let url = URL(string: "\(resolvedURL)/device/pending/\(userCode)") else { return }

        do {
            let (data, _) = try await session.data(from: url)
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
        _ = try? await session.data(for: request)
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

// MARK: - UNUserNotificationCenterDelegate

extension NotificationService: UNUserNotificationCenterDelegate {
    /// Called when a notification arrives while the app is in the foreground.
    /// Without this, iOS silently drops the notification instead of showing it.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
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
