mod capture;
mod clock;
#[cfg(target_os = "macos")]
mod cpal_input;
#[cfg(any(test, debug_assertions))]
mod development_mock;
mod lifecycle;
mod macos;
#[cfg(target_os = "macos")]
mod service;
#[cfg(any(test, debug_assertions))]
mod test_source;

pub use capture::{
    BoundedCaptureWriter, CaptureIngress, CaptureIngressPacket, CapturePacket, CaptureWriteResult,
};
pub use clock::{CaptureClock, CaptureGap, CaptureGapReason, CapturePoint};
#[cfg(target_os = "macos")]
pub use cpal_input::{
    capture_point_now, CpalInput, CpalInputFailure, CpalInputTelemetry, InputDeviceList,
    MicrophonePermission,
};
#[cfg(any(test, debug_assertions))]
pub use development_mock::{
    DevelopmentMockProgress, DevelopmentMockRunner, DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE,
};
pub use lifecycle::{
    CaptureFailureCode, CaptureLifecycle, CaptureLifecycleAction, CaptureLifecycleError,
    CaptureLifecycleFailure, CaptureStatus,
};
pub use macos::{
    MacOsCaptureAdapter, MacOsCaptureCallbackSink, MacOsCaptureError, MacOsCaptureEvent,
    MacOsInputCallback, MacOsInputDevice,
};
#[cfg(target_os = "macos")]
pub use service::{
    CaptureIssue, CaptureIssueCode, CaptureMeter, CaptureProjection, CaptureService, CaptureStart,
};
#[cfg(any(test, debug_assertions))]
pub use test_source::{CaptureSource, TestCaptureSource};
