mod capture;
mod clock;
#[cfg(target_os = "macos")]
mod cpal_input;
#[cfg(any(test, debug_assertions))]
mod development_mock;
mod dispatcher;
mod lifecycle;
mod macos;
mod native_runtime;
#[cfg(target_os = "macos")]
mod service;
#[cfg(any(test, debug_assertions))]
mod test_source;

pub use capture::{
    BoundedCaptureWriter, CaptureIngress, CaptureIngressPacket, CapturePacket, CaptureWriteResult,
    MAX_CAPTURE_SAMPLES_PER_PACKET,
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
pub use dispatcher::{
    AsrBridgeConfig, AsrJob, AsrJobMetadata, AsrOutcome, AsrQueueMetrics, CaptureDispatcher,
    DispatcherError, DispatcherMeter, DispatcherRuntime, DispatcherRuntimeId, DispatcherStatus,
    IngressPumpResult, OwnedOutcomeLease, OwnedOutcomeLeaseError, ShutdownDrainResult,
    WorkerPumpResult,
};
pub use lifecycle::{
    CaptureFailureCode, CaptureLifecycle, CaptureLifecycleAction, CaptureLifecycleError,
    CaptureLifecycleFailure, CaptureStatus,
};
pub use macos::{
    MacOsCaptureAdapter, MacOsCaptureCallbackSink, MacOsCaptureError, MacOsCaptureEvent,
    MacOsInputCallback, MacOsInputDevice,
};
pub use native_runtime::{
    NativeCaptureRuntime, NativeCaptureRuntimeConfig, NativeCaptureRuntimeError,
    NativeCaptureRuntimeSnapshot, NativeCaptureRuntimeStatus,
};
#[cfg(target_os = "macos")]
pub use service::{
    CaptureBridgeProjection, CaptureBridgeStatus, CaptureGapLease, CaptureIssue, CaptureIssueCode,
    CaptureMeter, CapturePreparation, CaptureProjection, CaptureService, CaptureStart,
};
#[cfg(any(test, debug_assertions))]
pub use test_source::{CaptureSource, TestCaptureSource};
