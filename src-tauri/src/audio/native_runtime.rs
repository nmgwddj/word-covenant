//! Native worker ownership for one bounded capture-to-inference bridge.
//!
//! The runtime has no CPAL stream, SQLite connection, Tauri handle, or
//! executable ASR model. It owns the only ingress consumer on a native thread
//! and exposes owned outcome leases so durable projection can happen after
//! releasing the dispatcher mutex.

use super::{
    AsrBridgeConfig, AsrJobClaim, AsrJobExecution, AsrJobLease, AsrQueueMetrics, CaptureClock,
    CaptureDispatcher, CaptureIngress, DispatcherError, DispatcherMeter, DispatcherRuntime,
    DispatcherStatus, IngressPumpResult, OwnedOutcomeLease, OwnedOutcomeLeaseError,
    ShutdownDrainResult, ShutdownPreparationResult, WorkerPumpResult,
};
use crate::inference::pipeline::{
    EnergyGatedSpeechDetector, EnergySpeechDetector, SpeechActivityDetector, SpeechPipelineConfig,
    SpeechSegmenter,
};
use crate::inference::{AsrEngine, InferenceError};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_IDLE_WAIT: Duration = Duration::from_millis(2);
pub const DEFAULT_SHUTDOWN_INFERENCE_ATTEMPT_LIMIT: usize = 1;
pub const MAX_SHUTDOWN_INFERENCE_ATTEMPT_LIMIT: usize = 256;

struct BoxedSpeechDetector {
    inner: Box<dyn SpeechActivityDetector>,
}

impl SpeechActivityDetector for BoxedSpeechDetector {
    fn is_speech(
        &mut self,
        frame: &crate::inference::InferenceAudioWindow,
    ) -> Result<bool, InferenceError> {
        self.inner.is_speech(frame)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

type NativeDispatcher =
    CaptureDispatcher<SpeechSegmenter<EnergyGatedSpeechDetector<BoxedSpeechDetector>>>;
type RuntimeControl = (Mutex<WorkerControl>, Condvar);

/// User-configurable local RMS threshold used to reject non-speech before it
/// reaches the speech segmenter or ASR worker.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechDetectionSettings {
    pub rms_threshold_dbfs: i8,
}

impl SpeechDetectionSettings {
    pub const MIN_RMS_THRESHOLD_DBFS: i8 = -60;
    pub const MAX_RMS_THRESHOLD_DBFS: i8 = 0;

    pub fn new(rms_threshold_dbfs: i8) -> Result<Self, String> {
        let settings = Self { rms_threshold_dbfs };
        settings.validate()?;
        Ok(settings)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(Self::MIN_RMS_THRESHOLD_DBFS..=Self::MAX_RMS_THRESHOLD_DBFS)
            .contains(&self.rms_threshold_dbfs)
        {
            return Err(format!(
                "speech RMS threshold must be between {} dBFS and {} dBFS",
                Self::MIN_RMS_THRESHOLD_DBFS,
                Self::MAX_RMS_THRESHOLD_DBFS
            ));
        }
        Ok(())
    }

    /// Convert a full-scale dB threshold to the normalized RMS value used by
    /// the native audio pipeline.
    pub fn normalized_rms_threshold(self) -> f32 {
        10_f32.powf(self.rms_threshold_dbfs as f32 / 20.0)
    }

    pub(crate) fn from_persisted_rms_threshold_dbfs(value: i64) -> Option<Self> {
        value
            .try_into()
            .ok()
            .and_then(|value| Self::new(value).ok())
    }
}

impl Default for SpeechDetectionSettings {
    fn default() -> Self {
        Self {
            rms_threshold_dbfs: -10,
        }
    }
}

/// Native-only VAD and ASR instances transferred into one capture runtime.
///
/// The VAD stays with the dispatcher thread. The ASR engine is moved into its
/// dedicated worker, so a non-`Sync` local model context never shares the
/// capture dispatcher mutex or crosses a Tauri boundary.
pub struct NativeInferenceEngines {
    vad_detector: Box<dyn SpeechActivityDetector>,
    asr_engine: Option<Box<dyn AsrEngine>>,
}

impl NativeInferenceEngines {
    pub fn new<D, A>(vad_detector: D, asr_engine: A) -> Self
    where
        D: SpeechActivityDetector + 'static,
        A: AsrEngine + 'static,
    {
        Self::from_boxed(Box::new(vad_detector), Some(Box::new(asr_engine)))
    }

    pub fn without_asr<D>(vad_detector: D) -> Self
    where
        D: SpeechActivityDetector + 'static,
    {
        Self::from_boxed(Box::new(vad_detector), None)
    }

    /// Accept boxed engines from a profile/runtime factory without exposing
    /// their implementation type to application state.
    pub fn from_boxed(
        vad_detector: Box<dyn SpeechActivityDetector>,
        asr_engine: Option<Box<dyn AsrEngine>>,
    ) -> Self {
        Self {
            vad_detector,
            asr_engine,
        }
    }
}

/// Local configuration for a parked native dispatcher worker.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeCaptureRuntimeConfig {
    pub bridge: AsrBridgeConfig,
    pub energy_threshold: f32,
    pub pipeline: SpeechPipelineConfig,
    pub idle_wait: Duration,
    /// At shutdown, at most this many pending jobs may invoke ASR after input
    /// sealing. Remaining jobs become auditable terminal gaps.
    pub shutdown_inference_attempt_limit: usize,
}

impl Default for NativeCaptureRuntimeConfig {
    fn default() -> Self {
        Self {
            bridge: AsrBridgeConfig::default(),
            // The default speech threshold is -10 dBFS. It is intentionally
            // conservative and can be tuned in local capture settings.
            energy_threshold: SpeechDetectionSettings::default().normalized_rms_threshold(),
            pipeline: SpeechPipelineConfig::default(),
            idle_wait: DEFAULT_IDLE_WAIT,
            shutdown_inference_attempt_limit: DEFAULT_SHUTDOWN_INFERENCE_ATTEMPT_LIMIT,
        }
    }
}

impl NativeCaptureRuntimeConfig {
    pub fn from_speech_detection_settings(
        settings: SpeechDetectionSettings,
    ) -> Result<Self, String> {
        settings.validate()?;
        Ok(Self {
            energy_threshold: settings.normalized_rms_threshold(),
            ..Self::default()
        })
    }

    fn validate(&self) -> Result<(), String> {
        self.bridge.validate()?;
        if !self.energy_threshold.is_finite() || self.energy_threshold < 0.0 {
            return Err(
                "native capture energy threshold must be a finite non-negative value".to_owned(),
            );
        }
        if self.idle_wait.is_zero() {
            return Err("native capture worker idle wait must be greater than zero".to_owned());
        }
        if self.shutdown_inference_attempt_limit == 0
            || self.shutdown_inference_attempt_limit > MAX_SHUTDOWN_INFERENCE_ATTEMPT_LIMIT
        {
            return Err(format!(
                "native shutdown inference attempt limit must be between 1 and {MAX_SHUTDOWN_INFERENCE_ATTEMPT_LIMIT}"
            ));
        }
        Ok(())
    }
}

/// Compact native state that can be projected without exposing PCM or text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCaptureRuntimeStatus {
    Parked,
    Armed,
    Closing,
    Drained,
}

/// A coherent, compact snapshot of native bridge state.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeCaptureRuntimeSnapshot {
    pub status: NativeCaptureRuntimeStatus,
    pub dispatcher_status: DispatcherStatus,
    pub armed: bool,
    pub shutdown_requested: bool,
    pub worker_finished: bool,
    pub meter: DispatcherMeter,
    pub metrics: AsrQueueMetrics,
}

#[derive(Debug)]
pub enum NativeCaptureRuntimeError {
    InvalidConfiguration(String),
    Dispatcher(DispatcherError),
    DispatcherLockPoisoned,
    CannotArm(DispatcherStatus),
    CannotAbortBeforeArm(DispatcherStatus),
    CannotJoinAfterAbort,
    ShutdownRequested,
    Outcome(OwnedOutcomeLeaseError),
    ThreadSpawn(String),
    WorkerPanicked,
}

impl fmt::Display for NativeCaptureRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::Dispatcher(error) => error.fmt(formatter),
            Self::DispatcherLockPoisoned => {
                formatter.write_str("native capture dispatcher lock is poisoned")
            }
            Self::CannotArm(status) => {
                write!(
                    formatter,
                    "cannot arm native capture runtime while dispatcher is {status:?}"
                )
            }
            Self::CannotAbortBeforeArm(status) => write!(
                formatter,
                "cannot abort native capture runtime before arm while dispatcher is {status:?}"
            ),
            Self::CannotJoinAfterAbort => formatter
                .write_str("native capture runtime can only join this way after an unarmed abort"),
            Self::ShutdownRequested => {
                formatter.write_str("native capture runtime shutdown has already been requested")
            }
            Self::Outcome(error) => error.fmt(formatter),
            Self::ThreadSpawn(message) => formatter.write_str(message),
            Self::WorkerPanicked => {
                formatter.write_str("native capture dispatcher worker panicked")
            }
        }
    }
}

impl std::error::Error for NativeCaptureRuntimeError {}

impl From<DispatcherError> for NativeCaptureRuntimeError {
    fn from(error: DispatcherError) -> Self {
        Self::Dispatcher(error)
    }
}

impl From<OwnedOutcomeLeaseError> for NativeCaptureRuntimeError {
    fn from(error: OwnedOutcomeLeaseError) -> Self {
        Self::Outcome(error)
    }
}

#[derive(Default)]
struct WorkerControl {
    armed: bool,
    shutdown_requested: bool,
    abort_before_arm_requested: bool,
    shutdown_input_prepared: bool,
    shutdown_asr_finished: bool,
    dispatcher_worker_finished: bool,
    asr_worker_finished: bool,
    signal_generation: u64,
}

#[derive(Clone, Copy)]
struct WorkerControlSnapshot {
    armed: bool,
    shutdown_requested: bool,
    abort_before_arm_requested: bool,
    shutdown_input_prepared: bool,
    shutdown_asr_finished: bool,
    dispatcher_worker_finished: bool,
    asr_worker_finished: bool,
    signal_generation: u64,
}

enum WorkerWait {
    Immediate,
    Idle,
    ExternalProgress,
    Drained,
}

/// One parked native dispatcher, its ingress thread, and its ASR worker.
///
/// Before [`Self::arm`] neither thread calls `CaptureIngress::try_consume`.
/// After arm, the dispatcher thread alone consumes ingress, updates the meter,
/// and segments speech. The ASR worker alone owns the local engine and never
/// retains the dispatcher mutex while it transcribes. No PCM crosses a Tauri
/// boundary.
pub struct NativeCaptureRuntime {
    dispatcher: Arc<Mutex<NativeDispatcher>>,
    control: Arc<RuntimeControl>,
    dispatcher_worker: Option<JoinHandle<()>>,
    asr_worker: Option<JoinHandle<()>>,
}

impl NativeCaptureRuntime {
    pub fn new(
        ingress: Arc<CaptureIngress>,
        runtime: DispatcherRuntime,
        clock: CaptureClock,
        config: NativeCaptureRuntimeConfig,
    ) -> Result<Self, NativeCaptureRuntimeError> {
        let detector = EnergySpeechDetector::new(config.energy_threshold)
            .map_err(NativeCaptureRuntimeError::InvalidConfiguration)?;
        Self::new_with_engines(
            ingress,
            runtime,
            clock,
            config,
            NativeInferenceEngines::without_asr(detector),
        )
    }

    /// Create a runtime with explicitly injected local VAD and ASR engines.
    ///
    /// The caller transfers ownership during construction. The VAD remains on
    /// the dispatcher thread while the ASR engine moves to the dedicated
    /// sequential worker, allowing a local engine to be `Send` but not
    /// `Sync`.
    pub fn new_with_engines(
        ingress: Arc<CaptureIngress>,
        runtime: DispatcherRuntime,
        clock: CaptureClock,
        config: NativeCaptureRuntimeConfig,
        engines: NativeInferenceEngines,
    ) -> Result<Self, NativeCaptureRuntimeError> {
        config
            .validate()
            .map_err(NativeCaptureRuntimeError::InvalidConfiguration)?;
        let NativeInferenceEngines {
            vad_detector,
            asr_engine,
        } = engines;
        // Apply the same local RMS floor to injected production VAD engines
        // that the no-engine fallback already receives. The wrapper calls the
        // VAD on every frame so stateful engines can observe silence.
        let vad_detector = EnergyGatedSpeechDetector::new(
            BoxedSpeechDetector {
                inner: vad_detector,
            },
            config.energy_threshold,
        )
        .map_err(NativeCaptureRuntimeError::InvalidConfiguration)?;
        let segmenter = SpeechSegmenter::new(
            runtime.session_id,
            clock.clone(),
            vad_detector,
            config.pipeline,
        )
        .map_err(NativeCaptureRuntimeError::InvalidConfiguration)?;
        let dispatcher = CaptureDispatcher::new(runtime, ingress, clock, segmenter, config.bridge)?;
        let dispatcher = Arc::new(Mutex::new(dispatcher));
        let control = Arc::new((Mutex::new(WorkerControl::default()), Condvar::new()));
        let worker_dispatcher = Arc::clone(&dispatcher);
        let worker_control = Arc::clone(&control);
        let idle_wait = config.idle_wait;
        let dispatcher_worker = thread::Builder::new()
            .name("word-covenant-native-dispatcher".to_owned())
            .spawn(move || run_dispatcher_worker(worker_dispatcher, worker_control, idle_wait))
            .map_err(|error| {
                NativeCaptureRuntimeError::ThreadSpawn(format!(
                    "could not start native capture dispatcher: {error}"
                ))
            })?;

        let asr_dispatcher = Arc::clone(&dispatcher);
        let asr_control = Arc::clone(&control);
        let shutdown_inference_attempt_limit = config.shutdown_inference_attempt_limit;
        let asr_worker = match thread::Builder::new()
            .name("word-covenant-native-asr".to_owned())
            .spawn(move || {
                run_asr_worker(
                    asr_dispatcher,
                    asr_control,
                    idle_wait,
                    asr_engine,
                    shutdown_inference_attempt_limit,
                )
            }) {
            Ok(worker) => worker,
            Err(error) => {
                request_unarmed_abort(&control);
                let _ = dispatcher_worker.join();
                return Err(NativeCaptureRuntimeError::ThreadSpawn(format!(
                    "could not start native ASR worker: {error}"
                )));
            }
        };

        Ok(Self {
            dispatcher,
            control,
            dispatcher_worker: Some(dispatcher_worker),
            asr_worker: Some(asr_worker),
        })
    }

    /// Permit the native worker to become the sole `CaptureIngress` consumer.
    pub fn arm(&self) -> Result<(), NativeCaptureRuntimeError> {
        let status = self.lock_dispatcher()?.status();
        if status != DispatcherStatus::Running {
            return Err(NativeCaptureRuntimeError::CannotArm(status));
        }

        let (mutex, condition) = &*self.control;
        let mut control = recover_mutex(mutex);
        if control.shutdown_requested {
            return Err(NativeCaptureRuntimeError::ShutdownRequested);
        }
        if !control.armed {
            // CPAL starts producing before the capture-start bundle is made
            // durable. Discard that pre-commit buffer without allowing it to
            // affect native meter, segmentation, outcomes, or audit evidence.
            self.lock_dispatcher()?
                .discard_pristine_ingress_before_arm()?;
            control.armed = true;
            signal_control(&mut control, condition);
        }
        Ok(())
    }

    /// Discard a prepared ingress after staged startup fails before arm.
    ///
    /// The caller must stop the producer first. This path never invokes the
    /// segmenter, so pre-commit PCM cannot become a transcript or inference
    /// gap. The caller must then call [`Self::join_after_abort`] after
    /// releasing any service mutex that owns this runtime.
    pub fn abort_before_arm(&self) -> Result<(), NativeCaptureRuntimeError> {
        let status = self.lock_dispatcher()?.status();
        if status != DispatcherStatus::Running {
            return Err(NativeCaptureRuntimeError::CannotAbortBeforeArm(status));
        }

        let (mutex, condition) = &*self.control;
        let mut control = recover_mutex(mutex);
        if control.armed {
            return Err(NativeCaptureRuntimeError::CannotAbortBeforeArm(status));
        }
        control.shutdown_requested = true;
        control.abort_before_arm_requested = true;
        signal_control(&mut control, condition);
        Ok(())
    }

    /// Fence new work and ask the worker to drain after the producer stops.
    ///
    /// The caller must stop CPAL before this call. The worker then drains
    /// ingress, seals the segmenter, and waits for result leases to be
    /// committed rather than polling while durable projection is pending.
    pub fn request_shutdown(&self) -> Result<(), NativeCaptureRuntimeError> {
        let (mutex, condition) = &*self.control;
        let mut control = recover_mutex(mutex);
        if !control.armed {
            control.shutdown_requested = true;
            control.abort_before_arm_requested = true;
            signal_control(&mut control, condition);
            return Ok(());
        }
        if !control.shutdown_requested {
            control.shutdown_requested = true;
            signal_control(&mut control, condition);
        } else {
            condition.notify_all();
        }
        drop(control);

        match self.dispatcher.lock() {
            Ok(mut dispatcher) => {
                dispatcher.begin_shutdown();
                Ok(())
            }
            Err(_) => Err(NativeCaptureRuntimeError::DispatcherLockPoisoned),
        }
    }

    pub fn is_drained(&self) -> Result<bool, NativeCaptureRuntimeError> {
        Ok(self.lock_dispatcher()?.status() == DispatcherStatus::Drained)
    }

    /// Join only after every result has reached a durable terminal state.
    ///
    /// Returns `false` without taking or blocking on the thread handle while
    /// the dispatcher is still closing. Callers should persist/acknowledge
    /// outstanding leases, then invoke this method again.
    pub fn join_if_drained(&mut self) -> Result<bool, NativeCaptureRuntimeError> {
        if !self.is_drained()? {
            return Ok(false);
        }
        if let Some(worker) = self.dispatcher_worker.take() {
            worker
                .join()
                .map_err(|_| NativeCaptureRuntimeError::WorkerPanicked)?;
        }
        if let Some(worker) = self.asr_worker.take() {
            worker
                .join()
                .map_err(|_| NativeCaptureRuntimeError::WorkerPanicked)?;
        }
        Ok(true)
    }

    /// Wait for an explicitly aborted parked worker, then join it.
    ///
    /// This is only valid after [`Self::abort_before_arm`] (or an equivalent
    /// unarmed shutdown) has been requested. It is deliberately separate from
    /// [`Self::join_if_drained`]: callers use it after releasing their service
    /// mutex, without polling for a pre-commit worker that cannot hold an
    /// outcome lease.
    pub fn join_after_abort(&mut self) -> Result<(), NativeCaptureRuntimeError> {
        let (mutex, condition) = &*self.control;
        let mut control = recover_mutex(mutex);
        if control.armed || !control.abort_before_arm_requested {
            return Err(NativeCaptureRuntimeError::CannotJoinAfterAbort);
        }
        while !workers_finished(&control) {
            control = match condition.wait(control) {
                Ok(control) => control,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        drop(control);

        if let Some(worker) = self.dispatcher_worker.take() {
            worker
                .join()
                .map_err(|_| NativeCaptureRuntimeError::WorkerPanicked)?;
        }
        if let Some(worker) = self.asr_worker.take() {
            worker
                .join()
                .map_err(|_| NativeCaptureRuntimeError::WorkerPanicked)?;
        }
        Ok(())
    }

    pub fn metrics(&self) -> Result<AsrQueueMetrics, NativeCaptureRuntimeError> {
        Ok(self.lock_dispatcher()?.metrics())
    }

    pub fn meter(&self) -> Result<DispatcherMeter, NativeCaptureRuntimeError> {
        Ok(self.lock_dispatcher()?.meter().clone())
    }

    pub fn runtime(&self) -> Result<DispatcherRuntime, NativeCaptureRuntimeError> {
        Ok(self.lock_dispatcher()?.runtime().clone())
    }

    pub fn snapshot(&self) -> Result<NativeCaptureRuntimeSnapshot, NativeCaptureRuntimeError> {
        let (dispatcher_status, meter, metrics) = {
            let dispatcher = self.lock_dispatcher()?;
            (
                dispatcher.status(),
                dispatcher.meter().clone(),
                dispatcher.metrics(),
            )
        };
        let control = control_snapshot(&self.control);
        Ok(NativeCaptureRuntimeSnapshot {
            status: runtime_status(dispatcher_status, control.armed),
            dispatcher_status,
            armed: control.armed,
            shutdown_requested: control.shutdown_requested,
            worker_finished: control.dispatcher_worker_finished && control.asr_worker_finished,
            meter,
            metrics,
        })
    }

    /// Claim an outcome without retaining the dispatcher mutex while SQLite
    /// and audit persistence execute in the caller.
    pub fn begin_owned_outcome(
        &self,
    ) -> Result<Option<OwnedOutcomeLease>, NativeCaptureRuntimeError> {
        let outcome = self.lock_dispatcher()?.begin_owned_outcome();
        if outcome.is_some() {
            wake_worker(&self.control);
        }
        Ok(outcome)
    }

    pub fn commit_owned_outcome(&self, token: u64) -> Result<(), NativeCaptureRuntimeError> {
        self.lock_dispatcher()?.commit_owned_outcome(token)?;
        wake_worker(&self.control);
        Ok(())
    }

    pub fn abort_owned_outcome(&self, token: u64) -> Result<(), NativeCaptureRuntimeError> {
        self.lock_dispatcher()?.abort_owned_outcome(token)?;
        wake_worker(&self.control);
        Ok(())
    }

    fn lock_dispatcher(
        &self,
    ) -> Result<MutexGuard<'_, NativeDispatcher>, NativeCaptureRuntimeError> {
        self.dispatcher
            .lock()
            .map_err(|_| NativeCaptureRuntimeError::DispatcherLockPoisoned)
    }
}

impl Drop for NativeCaptureRuntime {
    fn drop(&mut self) {
        let _ = self.request_shutdown();
        let is_drained = self.is_drained().unwrap_or(false);
        if is_drained {
            let _ = self.join_if_drained();
        }
    }
}

fn run_dispatcher_worker(
    dispatcher: Arc<Mutex<NativeDispatcher>>,
    control: Arc<RuntimeControl>,
    idle_wait: Duration,
) {
    loop {
        let snapshot = control_snapshot(&control);
        if !snapshot.armed && !snapshot.shutdown_requested {
            wait_for_signal(&control, snapshot.signal_generation);
            continue;
        }

        let wait = if snapshot.abort_before_arm_requested {
            drive_unarmed_abort_once(&dispatcher)
        } else if snapshot.shutdown_requested {
            drive_shutdown_dispatcher_once(&dispatcher, &control)
        } else {
            drive_running_dispatcher_once(&dispatcher, &control)
        };
        match wait {
            WorkerWait::Immediate => continue,
            WorkerWait::Idle => {
                wait_for_signal_or_timeout(&control, snapshot.signal_generation, idle_wait)
            }
            WorkerWait::ExternalProgress => wait_for_signal(&control, snapshot.signal_generation),
            WorkerWait::Drained => {
                mark_dispatcher_worker_finished(&control);
                return;
            }
        }
    }
}

fn run_asr_worker(
    dispatcher: Arc<Mutex<NativeDispatcher>>,
    control: Arc<RuntimeControl>,
    idle_wait: Duration,
    mut engine: Option<Box<dyn AsrEngine>>,
    mut shutdown_attempts_remaining: usize,
) {
    loop {
        let snapshot = control_snapshot(&control);
        if !snapshot.armed && !snapshot.shutdown_requested {
            wait_for_signal(&control, snapshot.signal_generation);
            continue;
        }
        if snapshot.abort_before_arm_requested {
            mark_asr_worker_finished(&control);
            return;
        }
        if snapshot.shutdown_requested && !snapshot.shutdown_input_prepared {
            wait_for_signal(&control, snapshot.signal_generation);
            continue;
        }
        if snapshot.shutdown_requested && shutdown_attempts_remaining == 0 {
            mark_shutdown_asr_finished(&control);
            mark_asr_worker_finished(&control);
            return;
        }

        let claim = {
            let mut dispatcher = recover_mutex(&dispatcher);
            dispatcher.claim_asr_job()
        };
        match claim {
            AsrJobClaim::Claimed(lease) => {
                let used_shutdown_attempt = snapshot.shutdown_requested;
                let execution = execute_asr_job(&mut engine, &lease);
                let completion = {
                    let mut dispatcher = recover_mutex(&dispatcher);
                    dispatcher
                        .complete_asr_job(lease, execution)
                        .expect("a native ASR worker completes its active job lease")
                };
                if used_shutdown_attempt {
                    shutdown_attempts_remaining = shutdown_attempts_remaining.saturating_sub(1);
                }
                wake_worker(&control);
                if used_shutdown_attempt && shutdown_attempts_remaining == 0 {
                    mark_shutdown_asr_finished(&control);
                    mark_asr_worker_finished(&control);
                    return;
                }
                match completion {
                    WorkerPumpResult::BlockedByResultQueue => {
                        wait_for_signal(&control, snapshot.signal_generation)
                    }
                    WorkerPumpResult::Processed
                    | WorkerPumpResult::DeliveredHeldOutcome
                    | WorkerPumpResult::NoJob
                    | WorkerPumpResult::InFlight => continue,
                }
            }
            AsrJobClaim::DeliveredHeldOutcome => continue,
            AsrJobClaim::BlockedByResultQueue | AsrJobClaim::InFlight => {
                wait_for_signal(&control, snapshot.signal_generation)
            }
            AsrJobClaim::NoJob => {
                if snapshot.shutdown_requested {
                    mark_shutdown_asr_finished(&control);
                    mark_asr_worker_finished(&control);
                    return;
                }
                wait_for_signal_or_timeout(&control, snapshot.signal_generation, idle_wait);
            }
        }
    }
}

fn execute_asr_job(
    engine: &mut Option<Box<dyn AsrEngine>>,
    lease: &AsrJobLease,
) -> AsrJobExecution {
    let Some(asr_engine) = engine.as_mut() else {
        return AsrJobExecution::EngineUnavailable;
    };
    let model_provenance = asr_engine.model_provenance().clone();
    let result = catch_unwind(AssertUnwindSafe(|| asr_engine.transcribe(lease.request())));
    match result {
        Ok(result) => AsrJobExecution::EngineResult {
            model_provenance,
            result,
        },
        Err(_) => {
            // An engine panic still needs a terminal outcome for the active
            // job. Retire the context so later jobs become explicit
            // local-engine-unavailable gaps instead of reusing a broken model.
            *engine = None;
            AsrJobExecution::EngineResult {
                model_provenance,
                result: Err(InferenceError::failed("local ASR engine panicked")),
            }
        }
    }
}

fn drive_running_dispatcher_once(
    dispatcher: &Arc<Mutex<NativeDispatcher>>,
    control: &Arc<RuntimeControl>,
) -> WorkerWait {
    let ingress = {
        let mut dispatcher = recover_mutex(dispatcher);
        dispatcher.pump_ingress_once()
    };

    if matches!(ingress, Ok(IngressPumpResult::BlockedByPendingEvent)) {
        return WorkerWait::ExternalProgress;
    }
    if matches!(ingress, Ok(IngressPumpResult::Drained)) {
        return WorkerWait::Drained;
    }
    if matches!(ingress, Ok(IngressPumpResult::Consumed)) {
        wake_worker(control);
        WorkerWait::Immediate
    } else {
        WorkerWait::Idle
    }
}

fn drive_shutdown_dispatcher_once(
    dispatcher: &Arc<Mutex<NativeDispatcher>>,
    control: &Arc<RuntimeControl>,
) -> WorkerWait {
    let (ingress, preparation) = {
        let mut dispatcher = recover_mutex(dispatcher);
        dispatcher.begin_shutdown();
        let ingress = dispatcher.pump_ingress_once();
        let preparation = match ingress {
            Ok(IngressPumpResult::BlockedByPendingEvent) => None,
            _ => Some(dispatcher.prepare_shutdown_for_inference_once()),
        };
        (ingress, preparation)
    };

    if matches!(ingress, Ok(IngressPumpResult::BlockedByPendingEvent)) {
        return WorkerWait::ExternalProgress;
    }
    if matches!(ingress, Ok(IngressPumpResult::Consumed)) {
        return WorkerWait::Immediate;
    }
    let Some(preparation) = preparation else {
        return WorkerWait::Idle;
    };
    match preparation {
        Ok(ShutdownPreparationResult::WaitingForIngress) => WorkerWait::Immediate,
        Ok(ShutdownPreparationResult::WaitingForPendingEvent) => WorkerWait::ExternalProgress,
        Ok(ShutdownPreparationResult::Drained) => WorkerWait::Drained,
        Ok(ShutdownPreparationResult::ReadyForInference) => {
            mark_shutdown_input_prepared(control);
            if !control_snapshot(control).shutdown_asr_finished {
                return WorkerWait::ExternalProgress;
            }
            drain_shutdown_once(dispatcher)
        }
        Err(_) => WorkerWait::Idle,
    }
}

fn drain_shutdown_once(dispatcher: &Arc<Mutex<NativeDispatcher>>) -> WorkerWait {
    let result = {
        let mut dispatcher = recover_mutex(dispatcher);
        dispatcher.drain_shutdown_once()
    };
    match result {
        Ok(ShutdownDrainResult::Drained) => WorkerWait::Drained,
        Ok(ShutdownDrainResult::WaitingForIngress) => WorkerWait::Immediate,
        Ok(
            ShutdownDrainResult::WaitingForPendingEvent
            | ShutdownDrainResult::WaitingForOutcomeDelivery
            | ShutdownDrainResult::AwaitingInference
            | ShutdownDrainResult::AwaitingOutcomeCommit,
        ) => WorkerWait::ExternalProgress,
        Err(_) => WorkerWait::Idle,
    }
}

fn drive_unarmed_abort_once(dispatcher: &Arc<Mutex<NativeDispatcher>>) -> WorkerWait {
    let mut dispatcher = recover_mutex(dispatcher);
    if dispatcher.abort_unarmed().is_err() {
        return WorkerWait::Idle;
    }
    match dispatcher.drain_shutdown_once() {
        Ok(ShutdownDrainResult::Drained) => WorkerWait::Drained,
        Ok(ShutdownDrainResult::WaitingForIngress) => WorkerWait::Immediate,
        Ok(
            ShutdownDrainResult::WaitingForPendingEvent
            | ShutdownDrainResult::WaitingForOutcomeDelivery
            | ShutdownDrainResult::AwaitingInference
            | ShutdownDrainResult::AwaitingOutcomeCommit,
        ) => WorkerWait::ExternalProgress,
        Err(_) => WorkerWait::Idle,
    }
}

fn runtime_status(dispatcher_status: DispatcherStatus, armed: bool) -> NativeCaptureRuntimeStatus {
    match dispatcher_status {
        DispatcherStatus::Running if armed => NativeCaptureRuntimeStatus::Armed,
        DispatcherStatus::Running => NativeCaptureRuntimeStatus::Parked,
        DispatcherStatus::Closing => NativeCaptureRuntimeStatus::Closing,
        DispatcherStatus::Drained => NativeCaptureRuntimeStatus::Drained,
    }
}

fn control_snapshot(control: &Arc<RuntimeControl>) -> WorkerControlSnapshot {
    let (mutex, _) = &**control;
    let control = recover_mutex(mutex);
    WorkerControlSnapshot {
        armed: control.armed,
        shutdown_requested: control.shutdown_requested,
        abort_before_arm_requested: control.abort_before_arm_requested,
        shutdown_input_prepared: control.shutdown_input_prepared,
        shutdown_asr_finished: control.shutdown_asr_finished,
        dispatcher_worker_finished: control.dispatcher_worker_finished,
        asr_worker_finished: control.asr_worker_finished,
        signal_generation: control.signal_generation,
    }
}

fn wake_worker(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    signal_control(&mut control, condition);
}

fn mark_shutdown_input_prepared(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    if !control.shutdown_input_prepared {
        control.shutdown_input_prepared = true;
        signal_control(&mut control, condition);
    }
}

fn mark_shutdown_asr_finished(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    if !control.shutdown_asr_finished {
        control.shutdown_asr_finished = true;
        signal_control(&mut control, condition);
    }
}

fn mark_dispatcher_worker_finished(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    if !control.dispatcher_worker_finished {
        control.dispatcher_worker_finished = true;
        signal_control(&mut control, condition);
    }
}

fn mark_asr_worker_finished(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    if !control.asr_worker_finished {
        control.asr_worker_finished = true;
        signal_control(&mut control, condition);
    }
}

fn request_unarmed_abort(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    control.shutdown_requested = true;
    control.abort_before_arm_requested = true;
    signal_control(&mut control, condition);
}

fn workers_finished(control: &WorkerControl) -> bool {
    control.dispatcher_worker_finished && control.asr_worker_finished
}

fn signal_control(control: &mut WorkerControl, condition: &Condvar) {
    control.signal_generation = control.signal_generation.wrapping_add(1);
    condition.notify_all();
}

fn wait_for_signal(control: &Arc<RuntimeControl>, observed_generation: u64) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    while control.signal_generation == observed_generation {
        control = match condition.wait(control) {
            Ok(control) => control,
            Err(poisoned) => poisoned.into_inner(),
        };
    }
}

fn wait_for_signal_or_timeout(
    control: &Arc<RuntimeControl>,
    observed_generation: u64,
    timeout: Duration,
) {
    let (mutex, condition) = &**control;
    let control = recover_mutex(mutex);
    if control.signal_generation == observed_generation {
        drop(match condition.wait_timeout(control, timeout) {
            Ok((control, _)) => control,
            Err(poisoned) => poisoned.into_inner().0,
        });
    }
}

fn recover_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{
        AsrOutcome, CapturePoint, CaptureWriteResult, DispatcherMeter, DispatcherRuntimeId,
    };
    use crate::inference::{
        AsrRequest, AsrResponse, InferenceEngine, InferenceExecutionScope, InferenceGapReason,
        ModelProvenance, TranscriptEmission, TranscriptEmissionKind, INFERENCE_SAMPLE_RATE_HZ,
    };
    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    };
    use std::time::Instant;
    use uuid::Uuid;

    #[test]
    fn speech_detection_settings_default_to_minus_10_dbfs() {
        assert_eq!(SpeechDetectionSettings::default().rms_threshold_dbfs, -10);
    }

    #[test]
    fn speech_detection_settings_accept_the_inclusive_dbfs_range() {
        assert_eq!(
            SpeechDetectionSettings::new(-60)
                .unwrap()
                .rms_threshold_dbfs,
            -60
        );
        assert_eq!(
            SpeechDetectionSettings::new(0).unwrap().rms_threshold_dbfs,
            0
        );
        assert!(SpeechDetectionSettings::new(-61).is_err());
        assert!(SpeechDetectionSettings::new(1).is_err());
    }

    #[test]
    fn speech_detection_settings_convert_minus_10_dbfs_to_normalized_rms() {
        let settings = SpeechDetectionSettings::default();
        let default_config = NativeCaptureRuntimeConfig::default();
        let config = NativeCaptureRuntimeConfig::from_speech_detection_settings(settings).unwrap();

        assert!((settings.normalized_rms_threshold() - 0.316_227_76).abs() < 0.000_001);
        assert!((default_config.energy_threshold - 0.316_227_76).abs() < 0.000_001);
        assert!((config.energy_threshold - 0.316_227_76).abs() < 0.000_001);
    }

    struct BlockingAsr {
        model: ModelProvenance,
        gate: Arc<(Mutex<BlockingAsrState>, Condvar)>,
    }

    struct ResetCountingDetector {
        resets: Arc<AtomicUsize>,
    }

    impl SpeechActivityDetector for ResetCountingDetector {
        fn is_speech(
            &mut self,
            _frame: &crate::inference::InferenceAudioWindow,
        ) -> Result<bool, InferenceError> {
            Ok(false)
        }

        fn reset(&mut self) {
            self.resets.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[derive(Default)]
    struct BlockingAsrState {
        started: bool,
        released: bool,
        calls: usize,
    }

    impl BlockingAsr {
        fn new(gate: Arc<(Mutex<BlockingAsrState>, Condvar)>) -> Self {
            Self {
                model: test_model(),
                gate,
            }
        }
    }

    impl InferenceEngine for BlockingAsr {
        fn model_provenance(&self) -> &ModelProvenance {
            &self.model
        }

        fn execution_scope(&self) -> InferenceExecutionScope {
            InferenceExecutionScope::OnDevice
        }
    }

    impl AsrEngine for BlockingAsr {
        fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError> {
            let (mutex, condition) = &*self.gate;
            let mut state = recover_mutex(mutex);
            state.started = true;
            state.calls = state.calls.saturating_add(1);
            condition.notify_all();
            while !state.released {
                state = match condition.wait(state) {
                    Ok(state) => state,
                    Err(poisoned) => poisoned.into_inner(),
                };
            }
            drop(state);
            test_response(request, &self.model)
        }
    }

    fn test_model() -> ModelProvenance {
        ModelProvenance::new("test", "native-runtime-asr", "v1", "b".repeat(64)).unwrap()
    }

    fn test_response(
        request: &AsrRequest,
        model: &ModelProvenance,
    ) -> Result<AsrResponse, InferenceError> {
        AsrResponse::new(
            request,
            model,
            vec![TranscriptEmission {
                utterance_key: format!("native-runtime-{}", request.audio.capture_start_ns()),
                capture_start_ns: request.audio.capture_start_ns(),
                capture_end_ns: request.audio.capture_end_ns(),
                text: "local speech".to_owned(),
                kind: TranscriptEmissionKind::Final,
                revision: 1,
                word_timings: Vec::new(),
                model_provenance: model.clone(),
            }],
        )
        .map_err(InferenceError::invalid)
    }

    fn anchor() -> CapturePoint {
        CapturePoint {
            monotonic_ns: 1_000,
            wall_clock: DateTime::<Utc>::UNIX_EPOCH + ChronoDuration::seconds(1),
        }
    }

    fn dispatcher_runtime() -> DispatcherRuntime {
        DispatcherRuntime::new(
            DispatcherRuntimeId::generate(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            anchor(),
        )
        .unwrap()
    }

    fn config() -> NativeCaptureRuntimeConfig {
        NativeCaptureRuntimeConfig {
            bridge: AsrBridgeConfig {
                job_queue_capacity: 2,
                result_queue_capacity: 2,
            },
            energy_threshold: 0.01,
            pipeline: SpeechPipelineConfig {
                pre_roll_frames: 0,
                hangover_frames: 0,
                maximum_window_frames: 4,
                minimum_speech_frames: 1,
                language: Some("zh".to_owned()),
                emit_partials: false,
            },
            idle_wait: Duration::from_millis(1),
            shutdown_inference_attempt_limit: 1,
        }
    }

    fn new_runtime() -> (NativeCaptureRuntime, Arc<CaptureIngress>) {
        let ingress = CaptureIngress::new(8, 160).unwrap();
        let runtime = NativeCaptureRuntime::new(
            Arc::clone(&ingress),
            dispatcher_runtime(),
            CaptureClock::new(anchor(), INFERENCE_SAMPLE_RATE_HZ).unwrap(),
            config(),
        )
        .unwrap();
        (runtime, ingress)
    }

    fn new_runtime_with_engine(
        engine: impl AsrEngine + 'static,
    ) -> (NativeCaptureRuntime, Arc<CaptureIngress>) {
        let ingress = CaptureIngress::new(8, 160).unwrap();
        let runtime = NativeCaptureRuntime::new_with_engines(
            Arc::clone(&ingress),
            dispatcher_runtime(),
            CaptureClock::new(anchor(), INFERENCE_SAMPLE_RATE_HZ).unwrap(),
            config(),
            NativeInferenceEngines::new(EnergySpeechDetector::new(0.01).unwrap(), engine),
        )
        .unwrap();
        (runtime, ingress)
    }

    fn write_speech_then_silence(ingress: &CaptureIngress) {
        assert_eq!(
            ingress.try_write(0, INFERENCE_SAMPLE_RATE_HZ, 1, &[0.5; 160]),
            CaptureWriteResult::Enqueued
        );
        assert_eq!(
            ingress.try_write(160, INFERENCE_SAMPLE_RATE_HZ, 1, &[0.0; 160]),
            CaptureWriteResult::Enqueued
        );
    }

    fn wait_for_outcome(runtime: &NativeCaptureRuntime) -> OwnedOutcomeLease {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(lease) = runtime.begin_owned_outcome().unwrap() {
                return lease;
            }
            assert!(
                Instant::now() < deadline,
                "native runtime did not produce an outcome"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn wait_for_drained(runtime: &NativeCaptureRuntime) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while !runtime.is_drained().unwrap() {
            assert!(
                Instant::now() < deadline,
                "native runtime did not drain after outcomes were committed"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn wait_for_ingress_packets(runtime: &NativeCaptureRuntime, minimum: u64) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if runtime.metrics().unwrap().ingress_packets_consumed >= minimum {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "dispatcher did not consume the expected ingress packet"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn wait_for_engine_start(gate: &Arc<(Mutex<BlockingAsrState>, Condvar)>) {
        let (mutex, condition) = &**gate;
        let mut state = recover_mutex(mutex);
        let deadline = Instant::now() + Duration::from_secs(1);
        while !state.started {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "ASR engine did not begin its blocked inference call"
            );
            state = match condition.wait_timeout(state, remaining) {
                Ok((state, _)) => state,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
    }

    fn release_engine(gate: &Arc<(Mutex<BlockingAsrState>, Condvar)>) {
        let (mutex, condition) = &**gate;
        let mut state = recover_mutex(mutex);
        state.released = true;
        condition.notify_all();
    }

    #[test]
    fn boxed_detector_forwards_a_discontinuity_reset_to_the_native_vad() {
        let resets = Arc::new(AtomicUsize::new(0));
        let mut detector = BoxedSpeechDetector {
            inner: Box::new(ResetCountingDetector {
                resets: Arc::clone(&resets),
            }),
        };

        detector.reset();

        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn arm_discards_precommit_ingress_then_consumes_only_postcommit_audio() {
        let (mut runtime, ingress) = new_runtime();
        write_speech_then_silence(&ingress);
        thread::sleep(Duration::from_millis(15));
        assert_eq!(
            runtime.snapshot().unwrap().status,
            NativeCaptureRuntimeStatus::Parked
        );
        assert_eq!(runtime.metrics().unwrap().ingress_packets_consumed, 0);

        runtime.arm().unwrap();
        thread::sleep(Duration::from_millis(15));
        let after_arm = runtime.snapshot().unwrap();
        assert_eq!(after_arm.status, NativeCaptureRuntimeStatus::Armed);
        assert_eq!(after_arm.metrics.ingress_packets_consumed, 0);
        assert_eq!(after_arm.meter, DispatcherMeter::default());
        assert!(runtime.begin_owned_outcome().unwrap().is_none());

        write_speech_then_silence(&ingress);
        // Re-arming an already armed runtime is idempotent and must not
        // discard audio written after the durable capture boundary.
        runtime.arm().unwrap();
        let lease = wait_for_outcome(&runtime);
        runtime.commit_owned_outcome(lease.token()).unwrap();
        runtime.request_shutdown().unwrap();
        wait_for_drained(&runtime);
        assert!(runtime.join_if_drained().unwrap());
    }

    #[test]
    fn abort_before_arm_discards_prepared_pcm_without_an_outcome() {
        let (mut runtime, ingress) = new_runtime();
        write_speech_then_silence(&ingress);
        thread::sleep(Duration::from_millis(15));
        assert_eq!(runtime.metrics().unwrap().ingress_packets_consumed, 0);

        runtime.abort_before_arm().unwrap();
        wait_for_drained(&runtime);
        assert_eq!(runtime.metrics().unwrap().ingress_packets_consumed, 0);
        assert!(runtime.begin_owned_outcome().unwrap().is_none());
        assert!(runtime.join_if_drained().unwrap());
    }

    #[test]
    fn joins_an_aborted_parked_worker_without_polling_for_drain() {
        let (mut runtime, ingress) = new_runtime();
        write_speech_then_silence(&ingress);

        runtime.abort_before_arm().unwrap();
        runtime.join_after_abort().unwrap();

        assert_eq!(
            runtime.snapshot().unwrap().status,
            NativeCaptureRuntimeStatus::Drained
        );
        assert!(runtime.join_if_drained().unwrap());
    }

    #[test]
    fn reports_missing_engine_as_a_gap() {
        let (mut runtime, ingress) = new_runtime();
        runtime.arm().unwrap();
        write_speech_then_silence(&ingress);

        let lease = wait_for_outcome(&runtime);
        match lease.outcome() {
            AsrOutcome::Gap(gap) => {
                assert_eq!(gap.reason, InferenceGapReason::LocalEngineUnavailable);
            }
            AsrOutcome::Response { .. } => {
                panic!("unconfigured native runtime must not invent an ASR response")
            }
        }
        runtime.commit_owned_outcome(lease.token()).unwrap();
        runtime.request_shutdown().unwrap();
        wait_for_drained(&runtime);
        assert!(runtime.join_if_drained().unwrap());
    }

    #[test]
    fn dispatcher_keeps_consuming_ingress_while_asr_blocks_outside_its_mutex() {
        let gate = Arc::new((Mutex::new(BlockingAsrState::default()), Condvar::new()));
        let (mut runtime, ingress) = new_runtime_with_engine(BlockingAsr::new(Arc::clone(&gate)));
        runtime.arm().unwrap();
        write_speech_then_silence(&ingress);
        wait_for_engine_start(&gate);

        assert_eq!(
            ingress.try_write(320, INFERENCE_SAMPLE_RATE_HZ, 1, &[0.25; 160]),
            CaptureWriteResult::Enqueued
        );
        wait_for_ingress_packets(&runtime, 3);
        assert_eq!(runtime.metrics().unwrap().jobs_completed, 0);
        assert!(runtime.meter().unwrap().peak_dbfs > -20.0);

        release_engine(&gate);
        let first = wait_for_outcome(&runtime);
        assert!(matches!(first.outcome(), AsrOutcome::Response { .. }));
        runtime.commit_owned_outcome(first.token()).unwrap();

        // The post-block packet remains an active utterance until shutdown.
        // Sealing it must still receive the configured one-call ASR budget.
        runtime.request_shutdown().unwrap();
        let final_window = wait_for_outcome(&runtime);
        assert!(matches!(
            final_window.outcome(),
            AsrOutcome::Response { .. }
        ));
        runtime.commit_owned_outcome(final_window.token()).unwrap();
        wait_for_drained(&runtime);
        assert!(runtime.join_if_drained().unwrap());
    }

    #[test]
    fn shutdown_infers_one_sealed_final_window_before_terminalizing_remaining_work() {
        let gate = Arc::new((
            Mutex::new(BlockingAsrState {
                released: true,
                ..BlockingAsrState::default()
            }),
            Condvar::new(),
        ));
        let (mut runtime, ingress) = new_runtime_with_engine(BlockingAsr::new(Arc::clone(&gate)));
        runtime.arm().unwrap();
        assert_eq!(
            ingress.try_write(0, INFERENCE_SAMPLE_RATE_HZ, 1, &[0.5; 160]),
            CaptureWriteResult::Enqueued
        );
        wait_for_ingress_packets(&runtime, 1);

        runtime.request_shutdown().unwrap();
        let final_window = wait_for_outcome(&runtime);
        assert!(matches!(
            final_window.outcome(),
            AsrOutcome::Response { .. }
        ));
        runtime.commit_owned_outcome(final_window.token()).unwrap();
        wait_for_drained(&runtime);
        assert_eq!(runtime.metrics().unwrap().shutdown_outcomes, 0);
        assert_eq!(recover_mutex(&gate.0).calls, 1);
        assert!(runtime.join_if_drained().unwrap());
    }

    #[test]
    fn aborting_an_owned_outcome_makes_the_same_gap_retryable() {
        let (mut runtime, ingress) = new_runtime();
        runtime.arm().unwrap();
        write_speech_then_silence(&ingress);

        let first = wait_for_outcome(&runtime);
        let first_gap_id = match first.outcome() {
            AsrOutcome::Gap(gap) => gap.id,
            AsrOutcome::Response { .. } => panic!("expected an unavailable-engine gap"),
        };
        runtime.abort_owned_outcome(first.token()).unwrap();

        let retry = wait_for_outcome(&runtime);
        let retry_gap_id = match retry.outcome() {
            AsrOutcome::Gap(gap) => gap.id,
            AsrOutcome::Response { .. } => panic!("expected an unavailable-engine retry gap"),
        };
        assert_eq!(retry_gap_id, first_gap_id);
        runtime.commit_owned_outcome(retry.token()).unwrap();
        runtime.request_shutdown().unwrap();
        wait_for_drained(&runtime);
        assert!(runtime.join_if_drained().unwrap());
    }

    #[test]
    fn shutdown_waits_for_outcome_commit_before_draining() {
        let (mut runtime, ingress) = new_runtime();
        runtime.arm().unwrap();
        write_speech_then_silence(&ingress);
        let lease = wait_for_outcome(&runtime);

        runtime.request_shutdown().unwrap();
        thread::sleep(Duration::from_millis(15));
        assert!(!runtime.is_drained().unwrap());
        assert!(!runtime.join_if_drained().unwrap());

        runtime.commit_owned_outcome(lease.token()).unwrap();
        wait_for_drained(&runtime);
        assert!(runtime.join_if_drained().unwrap());
    }
}
