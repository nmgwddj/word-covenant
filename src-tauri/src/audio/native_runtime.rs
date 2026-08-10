//! Native worker ownership for one bounded capture-to-inference bridge.
//!
//! The runtime has no CPAL stream, SQLite connection, Tauri handle, or
//! executable ASR model. It owns the only ingress consumer on a native thread
//! and exposes owned outcome leases so durable projection can happen after
//! releasing the dispatcher mutex.

use super::{
    AsrBridgeConfig, AsrQueueMetrics, CaptureClock, CaptureDispatcher, CaptureIngress,
    DispatcherError, DispatcherMeter, DispatcherRuntime, DispatcherStatus, IngressPumpResult,
    OwnedOutcomeLease, OwnedOutcomeLeaseError, ShutdownDrainResult, WorkerPumpResult,
};
use crate::inference::pipeline::{EnergySpeechDetector, SpeechPipelineConfig, SpeechSegmenter};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_IDLE_WAIT: Duration = Duration::from_millis(2);

type NativeDispatcher = CaptureDispatcher<SpeechSegmenter<EnergySpeechDetector>>;
type RuntimeControl = (Mutex<WorkerControl>, Condvar);

/// Local configuration for a parked native dispatcher worker.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeCaptureRuntimeConfig {
    pub bridge: AsrBridgeConfig,
    pub energy_threshold: f32,
    pub pipeline: SpeechPipelineConfig,
    pub idle_wait: Duration,
}

impl Default for NativeCaptureRuntimeConfig {
    fn default() -> Self {
        Self {
            bridge: AsrBridgeConfig::default(),
            energy_threshold: 0.015,
            pipeline: SpeechPipelineConfig::default(),
            idle_wait: DEFAULT_IDLE_WAIT,
        }
    }
}

impl NativeCaptureRuntimeConfig {
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
    worker_finished: bool,
    signal_generation: u64,
}

enum WorkerWait {
    Immediate,
    Idle,
    ExternalProgress,
    Drained,
}

/// One parked native dispatcher and the thread that drives it.
///
/// Before [`Self::arm`] the thread waits on a condition variable and never
/// calls `CaptureIngress::try_consume`. After arm it pumps at most one ingress
/// packet and one unavailable-engine ASR job per iteration. No PCM crosses a
/// Tauri boundary.
pub struct NativeCaptureRuntime {
    dispatcher: Arc<Mutex<NativeDispatcher>>,
    control: Arc<RuntimeControl>,
    worker: Option<JoinHandle<()>>,
}

impl NativeCaptureRuntime {
    pub fn new(
        ingress: Arc<CaptureIngress>,
        runtime: DispatcherRuntime,
        clock: CaptureClock,
        config: NativeCaptureRuntimeConfig,
    ) -> Result<Self, NativeCaptureRuntimeError> {
        config
            .validate()
            .map_err(NativeCaptureRuntimeError::InvalidConfiguration)?;
        let segmenter = SpeechSegmenter::with_energy_gate(
            runtime.session_id,
            clock.clone(),
            config.energy_threshold,
            config.pipeline,
        )
        .map_err(NativeCaptureRuntimeError::InvalidConfiguration)?;
        let dispatcher = CaptureDispatcher::new(runtime, ingress, clock, segmenter, config.bridge)?;
        let dispatcher = Arc::new(Mutex::new(dispatcher));
        let control = Arc::new((Mutex::new(WorkerControl::default()), Condvar::new()));
        let worker_dispatcher = Arc::clone(&dispatcher);
        let worker_control = Arc::clone(&control);
        let idle_wait = config.idle_wait;
        let worker = thread::Builder::new()
            .name("word-covenant-native-dispatcher".to_owned())
            .spawn(move || run_worker(worker_dispatcher, worker_control, idle_wait))
            .map_err(|error| {
                NativeCaptureRuntimeError::ThreadSpawn(format!(
                    "could not start native capture dispatcher: {error}"
                ))
            })?;

        Ok(Self {
            dispatcher,
            control,
            worker: Some(worker),
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
        let Some(worker) = self.worker.take() else {
            return Ok(true);
        };
        worker
            .join()
            .map_err(|_| NativeCaptureRuntimeError::WorkerPanicked)?;
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
        while !control.worker_finished {
            control = match condition.wait(control) {
                Ok(control) => control,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        drop(control);

        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        worker
            .join()
            .map_err(|_| NativeCaptureRuntimeError::WorkerPanicked)
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
        let (armed, shutdown_requested, _, worker_finished, _) = control_snapshot(&self.control);
        Ok(NativeCaptureRuntimeSnapshot {
            status: runtime_status(dispatcher_status, armed),
            dispatcher_status,
            armed,
            shutdown_requested,
            worker_finished,
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

fn run_worker(
    dispatcher: Arc<Mutex<NativeDispatcher>>,
    control: Arc<RuntimeControl>,
    idle_wait: Duration,
) {
    loop {
        let (armed, shutdown_requested, abort_before_arm, _, signal_generation) =
            control_snapshot(&control);
        if !armed && !shutdown_requested {
            wait_for_signal(&control, signal_generation);
            continue;
        }

        let wait = if abort_before_arm {
            drive_unarmed_abort_once(&dispatcher)
        } else if shutdown_requested {
            drive_shutdown_once(&dispatcher)
        } else {
            drive_running_once(&dispatcher)
        };
        match wait {
            WorkerWait::Immediate => continue,
            WorkerWait::Idle => wait_for_signal_or_timeout(&control, signal_generation, idle_wait),
            WorkerWait::ExternalProgress => wait_for_signal(&control, signal_generation),
            WorkerWait::Drained => {
                mark_worker_finished(&control);
                return;
            }
        }
    }
}

fn drive_running_once(dispatcher: &Arc<Mutex<NativeDispatcher>>) -> WorkerWait {
    let mut dispatcher = recover_mutex(dispatcher);
    let ingress = dispatcher.pump_ingress_once();
    let worker = dispatcher.pump_worker_once(None);

    if matches!(ingress, Ok(IngressPumpResult::BlockedByPendingEvent))
        || matches!(worker, WorkerPumpResult::BlockedByResultQueue)
    {
        return WorkerWait::ExternalProgress;
    }
    if dispatcher.status() == DispatcherStatus::Drained {
        return WorkerWait::Drained;
    }
    if matches!(ingress, Ok(IngressPumpResult::Consumed))
        || matches!(
            worker,
            WorkerPumpResult::Processed | WorkerPumpResult::DeliveredHeldOutcome
        )
    {
        WorkerWait::Immediate
    } else {
        WorkerWait::Idle
    }
}

fn drive_shutdown_once(dispatcher: &Arc<Mutex<NativeDispatcher>>) -> WorkerWait {
    let mut dispatcher = recover_mutex(dispatcher);
    dispatcher.begin_shutdown();
    let _ = dispatcher.pump_ingress_once();
    match dispatcher.drain_shutdown_once() {
        Ok(ShutdownDrainResult::Drained) => WorkerWait::Drained,
        Ok(ShutdownDrainResult::WaitingForIngress) => WorkerWait::Immediate,
        Ok(
            ShutdownDrainResult::WaitingForPendingEvent
            | ShutdownDrainResult::WaitingForOutcomeDelivery
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

fn control_snapshot(control: &Arc<RuntimeControl>) -> (bool, bool, bool, bool, u64) {
    let (mutex, _) = &**control;
    let control = recover_mutex(mutex);
    (
        control.armed,
        control.shutdown_requested,
        control.abort_before_arm_requested,
        control.worker_finished,
        control.signal_generation,
    )
}

fn wake_worker(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    signal_control(&mut control, condition);
}

fn mark_worker_finished(control: &Arc<RuntimeControl>) {
    let (mutex, condition) = &**control;
    let mut control = recover_mutex(mutex);
    control.worker_finished = true;
    signal_control(&mut control, condition);
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
    use crate::inference::{InferenceGapReason, INFERENCE_SAMPLE_RATE_HZ};
    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use std::time::Instant;
    use uuid::Uuid;

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
                language: Some("zh".to_owned()),
                emit_partials: false,
            },
            idle_wait: Duration::from_millis(1),
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
