//! Pure lifecycle state for a macOS capture backend.
//!
//! This module does not request microphone permission, choose a CoreAudio
//! device, or start an audio stream. It records the state implied by an
//! application-owned permission-resolution phase and the events emitted at
//! the [`MacOsCaptureEvent`] boundary.

use super::{CapturePoint, MacOsCaptureEvent, MacOsInputDevice};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureStatus {
    Idle,
    AwaitingPermission,
    Recording,
    Interrupted,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureLifecycleAction {
    BeginPermissionResolution,
    CancelPreparation,
    CaptureStarted,
    InputDeviceChanged,
    InputDeviceUnavailable,
    CaptureStopped,
    PacketDropped,
    CaptureQueueClosed,
    Fail,
}

impl fmt::Display for CaptureLifecycleAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::BeginPermissionResolution => "begin permission resolution",
            Self::CancelPreparation => "cancel capture preparation",
            Self::CaptureStarted => "process capture started",
            Self::InputDeviceChanged => "process input device changed",
            Self::InputDeviceUnavailable => "process input device unavailable",
            Self::CaptureStopped => "process capture stopped",
            Self::PacketDropped => "process packet dropped",
            Self::CaptureQueueClosed => "process capture queue closed",
            Self::Fail => "record a capture failure",
        };
        formatter.write_str(name)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureFailureCode {
    PermissionDenied,
    PermissionRestricted,
    NoInputDevice,
    StreamStartFailed,
    InputDeviceUnavailable,
    CaptureQueueClosed,
    External,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureLifecycleFailure {
    code: CaptureFailureCode,
    message: String,
    at: CapturePoint,
}

impl CaptureLifecycleFailure {
    pub fn code(&self) -> CaptureFailureCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn at(&self) -> &CapturePoint {
        &self.at
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureLifecycleError {
    InvalidTransition {
        status: CaptureStatus,
        action: CaptureLifecycleAction,
    },
    DeviceMismatch {
        expected_uid: String,
        received_uid: String,
    },
    EmptyFailureMessage,
}

impl fmt::Display for CaptureLifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition { status, action } => {
                write!(formatter, "cannot {action} while capture lifecycle is {status:?}")
            }
            Self::DeviceMismatch {
                expected_uid,
                received_uid,
            } => write!(
                formatter,
                "capture event device uid {received_uid:?} does not match selected device {expected_uid:?}"
            ),
            Self::EmptyFailureMessage => formatter.write_str("capture failure message must not be empty"),
        }
    }
}

impl std::error::Error for CaptureLifecycleError {}

/// A deterministic state machine for translating capture-boundary events into
/// UI- and policy-safe lifecycle state.
///
/// A selected device remains available after a normal stop so a caller can
/// show the last device used. A new successful `CaptureStarted` event clears
/// the last error. Events that cannot follow the current state return a typed
/// [`CaptureLifecycleError`] without mutating this value.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureLifecycle {
    status: CaptureStatus,
    selected_device: Option<MacOsInputDevice>,
    last_error: Option<CaptureLifecycleFailure>,
    transitioned_at: Option<CapturePoint>,
}

impl Default for CaptureLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureLifecycle {
    pub fn new() -> Self {
        Self {
            status: CaptureStatus::Idle,
            selected_device: None,
            last_error: None,
            transitioned_at: None,
        }
    }

    pub fn status(&self) -> CaptureStatus {
        self.status
    }

    pub fn selected_device(&self) -> Option<&MacOsInputDevice> {
        self.selected_device.as_ref()
    }

    pub fn last_error(&self) -> Option<&CaptureLifecycleFailure> {
        self.last_error.as_ref()
    }

    pub fn transitioned_at(&self) -> Option<&CapturePoint> {
        self.transitioned_at.as_ref()
    }

    /// Record that the caller has started resolving microphone permission.
    ///
    /// This does not prompt macOS or inspect a permission result. The caller
    /// must apply a later event or call [`Self::fail`] once it knows the
    /// outcome.
    pub fn begin_permission_resolution(
        &mut self,
        at: CapturePoint,
    ) -> Result<(), CaptureLifecycleError> {
        self.require_status(
            CaptureLifecycleAction::BeginPermissionResolution,
            &[
                CaptureStatus::Idle,
                CaptureStatus::Interrupted,
                CaptureStatus::Failed,
            ],
        )?;
        self.transition(CaptureStatus::AwaitingPermission, at);
        Ok(())
    }

    /// Return an unarmed staged start to idle without publishing a recording
    /// or failure. The caller has already stopped any prepared producer.
    pub fn cancel_preparation(&mut self, at: CapturePoint) -> Result<(), CaptureLifecycleError> {
        self.require_status(
            CaptureLifecycleAction::CancelPreparation,
            &[CaptureStatus::AwaitingPermission],
        )?;
        self.transition(CaptureStatus::Idle, at);
        Ok(())
    }

    /// Record an application-observed failure that is not represented by a
    /// [`MacOsCaptureEvent`].
    pub fn fail(
        &mut self,
        at: CapturePoint,
        message: impl Into<String>,
    ) -> Result<(), CaptureLifecycleError> {
        self.fail_with_code(at, CaptureFailureCode::External, message)
    }

    /// Record an application-observed failure with a stable code suitable for
    /// a compact UI projection and audit event.
    pub fn fail_with_code(
        &mut self,
        at: CapturePoint,
        code: CaptureFailureCode,
        message: impl Into<String>,
    ) -> Result<(), CaptureLifecycleError> {
        self.require_status(
            CaptureLifecycleAction::Fail,
            &[
                CaptureStatus::Idle,
                CaptureStatus::AwaitingPermission,
                CaptureStatus::Recording,
                CaptureStatus::Interrupted,
            ],
        )?;

        let message = message.into();
        if message.trim().is_empty() {
            return Err(CaptureLifecycleError::EmptyFailureMessage);
        }

        self.record_failure(code, message, at);
        Ok(())
    }

    /// Apply one event from [`MacOsCaptureAdapter`].
    pub fn apply(&mut self, event: MacOsCaptureEvent) -> Result<(), CaptureLifecycleError> {
        match event {
            MacOsCaptureEvent::CaptureStarted { device, at } => {
                self.require_status(
                    CaptureLifecycleAction::CaptureStarted,
                    &[
                        CaptureStatus::Idle,
                        CaptureStatus::AwaitingPermission,
                        CaptureStatus::Interrupted,
                    ],
                )?;
                self.selected_device = Some(device);
                self.last_error = None;
                self.transition(CaptureStatus::Recording, at);
                Ok(())
            }
            MacOsCaptureEvent::InputDeviceChanged {
                previous_device,
                current_device,
                at,
            } => {
                self.require_status(
                    CaptureLifecycleAction::InputDeviceChanged,
                    &[CaptureStatus::Recording],
                )?;
                self.require_selected_device(&previous_device)?;
                self.selected_device = Some(current_device);
                self.transition(CaptureStatus::Recording, at);
                Ok(())
            }
            MacOsCaptureEvent::InputDeviceUnavailable { device, at } => {
                self.require_status(
                    CaptureLifecycleAction::InputDeviceUnavailable,
                    &[CaptureStatus::Recording],
                )?;
                self.require_selected_device(&device)?;
                self.record_interruption(
                    CaptureFailureCode::InputDeviceUnavailable,
                    format!("input device {} is unavailable", device.name()),
                    at,
                );
                Ok(())
            }
            MacOsCaptureEvent::CaptureStopped { device, at } => {
                self.require_status(
                    CaptureLifecycleAction::CaptureStopped,
                    &[
                        CaptureStatus::Recording,
                        CaptureStatus::Interrupted,
                        CaptureStatus::Failed,
                    ],
                )?;
                self.require_selected_device(&device)?;
                self.transition(CaptureStatus::Idle, at);
                Ok(())
            }
            MacOsCaptureEvent::PacketDropped { device, .. } => {
                self.require_status(
                    CaptureLifecycleAction::PacketDropped,
                    &[CaptureStatus::Recording],
                )?;
                self.require_selected_device(&device)
            }
            MacOsCaptureEvent::CaptureQueueClosed { device, at, .. } => {
                self.require_status(
                    CaptureLifecycleAction::CaptureQueueClosed,
                    &[CaptureStatus::Recording],
                )?;
                self.require_selected_device(&device)?;
                self.record_interruption(
                    CaptureFailureCode::CaptureQueueClosed,
                    format!(
                        "capture queue closed while receiving from {}",
                        device.name()
                    ),
                    at,
                );
                Ok(())
            }
        }
    }

    fn require_status(
        &self,
        action: CaptureLifecycleAction,
        allowed: &[CaptureStatus],
    ) -> Result<(), CaptureLifecycleError> {
        if allowed.contains(&self.status) {
            Ok(())
        } else {
            Err(CaptureLifecycleError::InvalidTransition {
                status: self.status,
                action,
            })
        }
    }

    fn require_selected_device(
        &self,
        received: &MacOsInputDevice,
    ) -> Result<(), CaptureLifecycleError> {
        let Some(expected) = self.selected_device.as_ref() else {
            return Err(CaptureLifecycleError::InvalidTransition {
                status: self.status,
                action: CaptureLifecycleAction::CaptureStarted,
            });
        };
        if expected.uid() == received.uid() {
            return Ok(());
        }
        Err(CaptureLifecycleError::DeviceMismatch {
            expected_uid: expected.uid().to_owned(),
            received_uid: received.uid().to_owned(),
        })
    }

    fn record_failure(&mut self, code: CaptureFailureCode, message: String, at: CapturePoint) {
        self.last_error = Some(CaptureLifecycleFailure {
            code,
            message,
            at: at.clone(),
        });
        self.transition(CaptureStatus::Failed, at);
    }

    fn record_interruption(&mut self, code: CaptureFailureCode, message: String, at: CapturePoint) {
        self.last_error = Some(CaptureLifecycleFailure {
            code,
            message,
            at: at.clone(),
        });
        self.transition(CaptureStatus::Interrupted, at);
    }

    fn transition(&mut self, status: CaptureStatus, at: CapturePoint) {
        self.status = status;
        self.transitioned_at = Some(at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration, Utc};

    fn point(monotonic_ns: u64, milliseconds: i64) -> CapturePoint {
        CapturePoint {
            monotonic_ns,
            wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::milliseconds(milliseconds),
        }
    }

    fn device(uid: &str) -> MacOsInputDevice {
        MacOsInputDevice::new(uid, format!("{uid} microphone")).unwrap()
    }

    fn start_recording(lifecycle: &mut CaptureLifecycle, input: MacOsInputDevice) {
        lifecycle
            .apply(MacOsCaptureEvent::CaptureStarted {
                device: input,
                at: point(100, 100),
            })
            .unwrap();
    }

    #[test]
    fn records_permission_resolution_then_a_started_capture() {
        let mut lifecycle = CaptureLifecycle::new();
        let resolving_at = point(10, 10);
        lifecycle
            .begin_permission_resolution(resolving_at.clone())
            .unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::AwaitingPermission);
        assert_eq!(lifecycle.transitioned_at(), Some(&resolving_at));

        let input = device("built-in");
        let started_at = point(20, 20);
        lifecycle
            .apply(MacOsCaptureEvent::CaptureStarted {
                device: input.clone(),
                at: started_at.clone(),
            })
            .unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::Recording);
        assert_eq!(lifecycle.selected_device(), Some(&input));
        assert_eq!(lifecycle.last_error(), None);
        assert_eq!(lifecycle.transitioned_at(), Some(&started_at));
    }

    #[test]
    fn cancels_an_unarmed_preparation_without_publishing_recording() {
        let mut lifecycle = CaptureLifecycle::new();
        lifecycle
            .begin_permission_resolution(point(10, 10))
            .unwrap();

        let cancelled_at = point(20, 20);
        lifecycle.cancel_preparation(cancelled_at.clone()).unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::Idle);
        assert_eq!(lifecycle.transitioned_at(), Some(&cancelled_at));
        assert!(lifecycle.selected_device().is_none());
        assert!(lifecycle.last_error().is_none());
    }

    #[test]
    fn device_change_keeps_recording_and_updates_the_selected_device() {
        let mut lifecycle = CaptureLifecycle::new();
        let built_in = device("built-in");
        start_recording(&mut lifecycle, built_in.clone());
        let usb = device("usb-mic");
        let changed_at = point(200, 200);

        lifecycle
            .apply(MacOsCaptureEvent::InputDeviceChanged {
                previous_device: built_in,
                current_device: usb.clone(),
                at: changed_at.clone(),
            })
            .unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::Recording);
        assert_eq!(lifecycle.selected_device(), Some(&usb));
        assert_eq!(lifecycle.transitioned_at(), Some(&changed_at));
    }

    #[test]
    fn unavailable_device_and_closed_queue_are_interrupted_failures() {
        let mut lifecycle = CaptureLifecycle::new();
        let built_in = device("built-in");
        start_recording(&mut lifecycle, built_in.clone());
        let unavailable_at = point(200, 200);

        lifecycle
            .apply(MacOsCaptureEvent::InputDeviceUnavailable {
                device: built_in.clone(),
                at: unavailable_at.clone(),
            })
            .unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::Interrupted);
        assert_eq!(
            lifecycle.last_error().map(CaptureLifecycleFailure::code),
            Some(CaptureFailureCode::InputDeviceUnavailable)
        );
        assert_eq!(lifecycle.transitioned_at(), Some(&unavailable_at));

        start_recording(&mut lifecycle, built_in.clone());
        let closed_at = point(400, 400);
        lifecycle
            .apply(MacOsCaptureEvent::CaptureQueueClosed {
                device: built_in,
                at: closed_at.clone(),
                starting_sample_offset: 16_000,
            })
            .unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::Interrupted);
        assert_eq!(
            lifecycle.last_error().map(CaptureLifecycleFailure::code),
            Some(CaptureFailureCode::CaptureQueueClosed)
        );
        assert_eq!(lifecycle.transitioned_at(), Some(&closed_at));
    }

    #[test]
    fn stopped_capture_returns_to_idle_and_keeps_the_selected_device() {
        let mut lifecycle = CaptureLifecycle::new();
        let input = device("built-in");
        start_recording(&mut lifecycle, input.clone());
        let stopped_at = point(200, 200);

        lifecycle
            .apply(MacOsCaptureEvent::CaptureStopped {
                device: input.clone(),
                at: stopped_at.clone(),
            })
            .unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::Idle);
        assert_eq!(lifecycle.selected_device(), Some(&input));
        assert_eq!(lifecycle.last_error(), None);
        assert_eq!(lifecycle.transitioned_at(), Some(&stopped_at));
    }

    #[test]
    fn invalid_events_and_device_mismatches_return_typed_errors_without_mutation() {
        let mut lifecycle = CaptureLifecycle::new();
        let initial = lifecycle.clone();

        assert_eq!(
            lifecycle.apply(MacOsCaptureEvent::CaptureStopped {
                device: device("built-in"),
                at: point(10, 10),
            }),
            Err(CaptureLifecycleError::InvalidTransition {
                status: CaptureStatus::Idle,
                action: CaptureLifecycleAction::CaptureStopped,
            })
        );
        assert_eq!(lifecycle, initial);

        let built_in = device("built-in");
        start_recording(&mut lifecycle, built_in.clone());
        let recording = lifecycle.clone();
        assert_eq!(
            lifecycle.apply(MacOsCaptureEvent::InputDeviceUnavailable {
                device: device("usb-mic"),
                at: point(200, 200),
            }),
            Err(CaptureLifecycleError::DeviceMismatch {
                expected_uid: "built-in".to_owned(),
                received_uid: "usb-mic".to_owned(),
            })
        );
        assert_eq!(lifecycle, recording);
    }

    #[test]
    fn external_failures_are_recorded_without_claiming_a_permission_prompt() {
        let mut lifecycle = CaptureLifecycle::new();
        let failed_at = point(10, 10);

        lifecycle
            .fail(failed_at.clone(), "permission state could not be resolved")
            .unwrap();

        assert_eq!(lifecycle.status(), CaptureStatus::Failed);
        assert_eq!(
            lifecycle.last_error().map(CaptureLifecycleFailure::code),
            Some(CaptureFailureCode::External)
        );
        assert_eq!(
            lifecycle.last_error().map(CaptureLifecycleFailure::message),
            Some("permission state could not be resolved")
        );
        assert_eq!(lifecycle.transitioned_at(), Some(&failed_at));
    }
}
