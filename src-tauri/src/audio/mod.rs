mod capture;
mod clock;
#[cfg(any(test, debug_assertions))]
mod development_mock;
mod lifecycle;
mod macos;
#[cfg(any(test, debug_assertions))]
mod test_source;

pub use capture::{BoundedCaptureWriter, CapturePacket, CaptureWriteResult};
pub use clock::{CaptureClock, CaptureGap, CaptureGapReason, CapturePoint};
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
#[cfg(any(test, debug_assertions))]
pub use test_source::{CaptureSource, TestCaptureSource};
