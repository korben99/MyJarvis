// AppDelegate.swift
// Jarvis — APNs registration callbacks (SwiftUI @UIApplicationDelegateAdaptor)

import UIKit

final class AppDelegate: NSObject, UIApplicationDelegate {

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        application.registerForRemoteNotifications()
        return true
    }

    // Real APNs device token — 32-byte Data converted to lowercase hex string.
    func application(
        _ application: UIApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {
        let hex = deviceToken.map { String(format: "%02x", $0) }.joined()
        Task { @MainActor in
            await NotificationService.shared.setAPNsToken(hex)
        }
    }

    func application(
        _ application: UIApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {
        // APNs unavailable (simulator, entitlement missing…) — polling fallback active.
        print("[APNs] Registration failed: \(error.localizedDescription)")
    }

    // Called when a silent remote notification arrives in background.
    // Triggers an immediate poll so the message appears in the chat feed
    // without waiting for the next BGAppRefreshTask window.
    func application(
        _ application: UIApplication,
        didReceiveRemoteNotification userInfo: [AnyHashable: Any],
        fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void
    ) {
        Task { @MainActor in
            await NotificationService.shared.pollAndDeliver()
            completionHandler(.newData)
        }
    }
}
