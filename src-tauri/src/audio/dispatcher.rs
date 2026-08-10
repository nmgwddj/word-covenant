//! Bounded native hand-off from capture ingress to local ASR execution.
//!
//! This module intentionally has no CPAL stream, thread, SQLite handle, or
//! Tauri command. A caller owns one [`CaptureDispatcher`] per active capture
//! runtime and drives its synchronous pump methods from native code. The
//! dispatcher is the sole caller of [`CaptureIngress::try_consume`], so level
//! projection and speech segmentation observe the same PCM sequence.

use super::{CaptureClock, CaptureIngress, CapturePoint, MAX_CAPTURE_SAMPLES_PER_PACKET};
use crate::inference::pipeline::{
    NativePcmPacket, SpeechActivityDetector, SpeechSegmenter, SpeechSegmenterError,
    SpeechWindowEvent, PIPELINE_FRAME_SAMPLES,
};
use crate::inference::{
    AsrEngine, AsrRequest, AsrResponse, InferenceError, InferenceGap, InferenceGapReason,
    InferenceGapStage, ModelProvenance,
};
use chrono::Duration;
use crossbeam_queue::ArrayQueue;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use uuid::Uuid;

pub const DEFAULT_ASR_JOB_QUEUE_CAPACITY: usize = 16;
pub const DEFAULT_ASR_RESULT_QUEUE_CAPACITY: usize = 16;
pub const MAX_ASR_QUEUE_CAPACITY: usize = 256;

// `SpeechSegmenter` emits no more than one completed window per 10 ms input
// frame, plus a discontinuity and a possible pre-discontinuity window. The
// capture packet cap makes this native-only backlog finite. One additional
// slot is reserved for the final window produced while sealing the segmenter.
const MAX_EVENTS_PER_CAPTURE_PACKET: usize =
    2 + MAX_CAPTURE_SAMPLES_PER_PACKET / PIPELINE_FRAME_SAMPLES;
const MAX_PENDING_SEGMENTER_EVENTS: usize = MAX_EVENTS_PER_CAPTURE_PACKET + 1;
const MINIMUM_DBFS: f32 = -96.0;

/// Fixed capacities for the native ASR bridge.
///
/// These queues contain only local Rust values. In particular, neither queue
/// is exposed through a Tauri command or serializable UI projection.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrBridgeConfig {
    pub job_queue_capacity: usize,
    pub result_queue_capacity: usize,
}

impl Default for AsrBridgeConfig {
    fn default() -> Self {
        Self {
            job_queue_capacity: DEFAULT_ASR_JOB_QUEUE_CAPACITY,
            result_queue_capacity: DEFAULT_ASR_RESULT_QUEUE_CAPACITY,
        }
    }
}

impl AsrBridgeConfig {
    pub fn validate(&self) -> Result<(), String> {
        validate_queue_capacity("ASR job queue", self.job_queue_capacity)?;
        validate_queue_capacity("ASR result queue", self.result_queue_capacity)?;
        Ok(())
    }
}

/// A generation fence for one capture dispatcher runtime.
///
/// The underlying UUID is persisted in inference gaps, while the wrapper
/// avoids accidentally passing a session or capture-segment ID as a runtime
/// generation in native code.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct DispatcherRuntimeId(Uuid);

impl DispatcherRuntimeId {
    pub fn new(value: Uuid) -> Result<Self, String> {
        if value.is_nil() {
            return Err("dispatcher runtime ID must not be empty".to_owned());
        }
        Ok(Self(value))
    }

    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl fmt::Display for DispatcherRuntimeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Stable runtime context attached to every job and terminal outcome.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DispatcherRuntime {
    pub id: DispatcherRuntimeId,
    pub session_id: Uuid,
    pub capture_segment_id: Uuid,
    pub capture_anchor: CapturePoint,
}

impl DispatcherRuntime {
    pub fn new(
        id: DispatcherRuntimeId,
        session_id: Uuid,
        capture_segment_id: Uuid,
        capture_anchor: CapturePoint,
    ) -> Result<Self, String> {
        if session_id.is_nil() {
            return Err("dispatcher session ID must not be empty".to_owned());
        }
        if capture_segment_id.is_nil() {
            return Err("dispatcher capture segment ID must not be empty".to_owned());
        }
        Ok(Self {
            id,
            session_id,
            capture_segment_id,
            capture_anchor,
        })
    }
}

/// Compact metadata for an owned native ASR request.
///
/// This deliberately omits the request PCM. It can safely be used by the
/// state layer to fence a result before durable projection.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrJobMetadata {
    pub id: Uuid,
    pub session_id: Uuid,
    pub runtime_id: DispatcherRuntimeId,
    pub capture_segment_id: Uuid,
    pub started_at: CapturePoint,
    pub ended_at: CapturePoint,
}

/// One bounded local ASR request awaiting sequential execution.
///
/// `AsrJob` intentionally does not implement `Serialize`: the request owns
/// PCM and must remain inside the native process.
#[derive(Clone, Debug, PartialEq)]
pub struct AsrJob {
    metadata: AsrJobMetadata,
    request: AsrRequest,
}

impl AsrJob {
    fn new(
        runtime: &DispatcherRuntime,
        request: AsrRequest,
        started_at: CapturePoint,
        ended_at: CapturePoint,
    ) -> Result<Self, String> {
        request.validate()?;
        if request.audio.session_id() != runtime.session_id {
            return Err("ASR request session does not match its dispatcher runtime".to_owned());
        }
        if request.audio.capture_start_ns() != started_at.monotonic_ns
            || request.audio.capture_end_ns() != ended_at.monotonic_ns
        {
            return Err("ASR request capture range does not match dispatcher metadata".to_owned());
        }
        if ended_at.monotonic_ns < started_at.monotonic_ns
            || ended_at.wall_clock < started_at.wall_clock
        {
            return Err("ASR job capture range is inverted".to_owned());
        }

        Ok(Self {
            metadata: AsrJobMetadata {
                id: Uuid::new_v4(),
                session_id: runtime.session_id,
                runtime_id: runtime.id,
                capture_segment_id: runtime.capture_segment_id,
                started_at,
                ended_at,
            },
            request,
        })
    }

    pub fn metadata(&self) -> &AsrJobMetadata {
        &self.metadata
    }

    pub fn request(&self) -> &AsrRequest {
        &self.request
    }
}

/// One native ASR job claimed by the sequential inference worker.
///
/// The lease owns the PCM-bearing request and deliberately cannot be
/// serialized. A worker must return it through
/// [`CaptureDispatcher::complete_asr_job`] after executing inference outside
/// the dispatcher mutex. The dispatcher retains compact metadata while the
/// lease is active so shutdown cannot silently forget an in-flight job.
#[derive(Debug)]
pub struct AsrJobLease {
    token: u64,
    job: AsrJob,
}

impl AsrJobLease {
    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn metadata(&self) -> &AsrJobMetadata {
        self.job.metadata()
    }

    pub fn request(&self) -> &AsrRequest {
        self.job.request()
    }
}

/// One bounded result of asking the dispatcher for work for its ASR worker.
///
/// Only [`Self::Claimed`] transfers PCM-bearing work out of the dispatcher.
/// The remaining variants report bounded delivery/backpressure state without
/// manufacturing a transcript.
#[derive(Debug)]
pub enum AsrJobClaim {
    Claimed(AsrJobLease),
    DeliveredHeldOutcome,
    NoJob,
    BlockedByResultQueue,
    InFlight,
}

/// A local engine result returned by the ASR worker after it has released the
/// dispatcher mutex.
///
/// The engine provenance is copied rather than borrowing a non-`Sync` engine
/// context across threads. A missing engine remains an explicit terminal
/// local-engine-unavailable gap.
#[derive(Debug)]
pub enum AsrJobExecution {
    EngineUnavailable,
    EngineResult {
        model_provenance: ModelProvenance,
        result: Result<AsrResponse, InferenceError>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsrJobLeaseError {
    NoActiveLease,
    TokenMismatch,
    MetadataMismatch,
}

impl fmt::Display for AsrJobLeaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoActiveLease => formatter.write_str("no active ASR job lease exists"),
            Self::TokenMismatch => formatter.write_str("ASR job lease token does not match"),
            Self::MetadataMismatch => {
                formatter.write_str("ASR job lease metadata does not match the active job")
            }
        }
    }
}

impl std::error::Error for AsrJobLeaseError {}

/// A local result available for durable native projection.
///
/// A response can contain transient partial emissions. The state layer keeps
/// partial text transient and persists only validated final emissions. A gap
/// contains no transcript text at all.
#[derive(Clone, Debug, PartialEq)]
pub enum AsrOutcome {
    Response {
        job: AsrJobMetadata,
        response: AsrResponse,
    },
    Gap(InferenceGap),
}

impl AsrOutcome {
    pub fn runtime_id(&self) -> DispatcherRuntimeId {
        match self {
            Self::Response { job, .. } => job.runtime_id,
            Self::Gap(gap) => DispatcherRuntimeId::new(gap.runtime_id)
                .expect("an inference gap validates its non-empty runtime ID"),
        }
    }

    pub fn session_id(&self) -> Uuid {
        match self {
            Self::Response { job, .. } => job.session_id,
            Self::Gap(gap) => gap.session_id,
        }
    }
}

/// Compact input-level telemetry calculated by the dispatcher.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DispatcherMeter {
    pub rms_dbfs: f32,
    pub peak_dbfs: f32,
    pub clipping: bool,
}

impl Default for DispatcherMeter {
    fn default() -> Self {
        Self {
            rms_dbfs: MINIMUM_DBFS,
            peak_dbfs: MINIMUM_DBFS,
            clipping: false,
        }
    }
}

/// Counts and bounded-depth snapshots exposed by the native bridge.
///
/// The structure contains neither raw audio nor transcript text, so it is
/// safe for a compact application projection once the service layer wires it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrQueueMetrics {
    pub ingress_packets_consumed: u64,
    pub ingress_discontinuities: u64,
    pub segmenter_failures: u64,
    pub jobs_admitted: u64,
    pub jobs_completed: u64,
    pub job_queue_saturated: u64,
    pub result_queue_saturated: u64,
    pub unavailable_engine_outcomes: u64,
    pub engine_failure_outcomes: u64,
    pub shutdown_outcomes: u64,
    pub outcome_claims_aborted: u64,
    pub job_queue_high_watermark: usize,
    pub result_queue_high_watermark: usize,
    pub pending_event_high_watermark: usize,
    pub job_queue_depth: usize,
    pub result_queue_depth: usize,
    pub pending_event_depth: usize,
    pub worker_holds_outcome: bool,
    pub owned_outcome_lease_active: bool,
    pub closing: bool,
}

impl Default for AsrQueueMetrics {
    fn default() -> Self {
        Self {
            ingress_packets_consumed: 0,
            ingress_discontinuities: 0,
            segmenter_failures: 0,
            jobs_admitted: 0,
            jobs_completed: 0,
            job_queue_saturated: 0,
            result_queue_saturated: 0,
            unavailable_engine_outcomes: 0,
            engine_failure_outcomes: 0,
            shutdown_outcomes: 0,
            outcome_claims_aborted: 0,
            job_queue_high_watermark: 0,
            result_queue_high_watermark: 0,
            pending_event_high_watermark: 0,
            job_queue_depth: 0,
            result_queue_depth: 0,
            pending_event_depth: 0,
            worker_holds_outcome: false,
            owned_outcome_lease_active: false,
            closing: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatcherStatus {
    Running,
    Closing,
    Drained,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngressPumpResult {
    Consumed,
    NoPacket,
    BlockedByPendingEvent,
    Drained,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerPumpResult {
    Processed,
    DeliveredHeldOutcome,
    NoJob,
    BlockedByResultQueue,
    InFlight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShutdownDrainResult {
    WaitingForIngress,
    WaitingForPendingEvent,
    WaitingForOutcomeDelivery,
    AwaitingInference,
    AwaitingOutcomeCommit,
    Drained,
}

/// Progress made while sealing capture input before shutdown inference.
///
/// This phase deliberately never terminalizes admitted ASR jobs. The native
/// runtime uses it to give a sealed final window its bounded inference budget
/// before [`CaptureDispatcher::drain_shutdown_once`] converts remaining work
/// into terminal gaps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShutdownPreparationResult {
    WaitingForIngress,
    WaitingForPendingEvent,
    ReadyForInference,
    Drained,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DispatcherError {
    InvalidConfiguration(String),
    InvalidRuntime(String),
    ShutdownNotStarted,
    IngressNotDrained,
    UnarmedAbortNotAllowed,
    PreArmIngressDiscardNotAllowed,
    SegmenterEventOverflow,
}

impl fmt::Display for DispatcherError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) | Self::InvalidRuntime(message) => {
                formatter.write_str(message)
            }
            Self::ShutdownNotStarted => formatter.write_str("dispatcher shutdown has not started"),
            Self::IngressNotDrained => {
                formatter.write_str("capture ingress must drain before sealing the segmenter")
            }
            Self::UnarmedAbortNotAllowed => formatter.write_str(
                "unarmed dispatcher abort requires a pristine dispatcher and stopped producer",
            ),
            Self::PreArmIngressDiscardNotAllowed => formatter
                .write_str("pre-arm ingress discard requires a pristine running dispatcher"),
            Self::SegmenterEventOverflow => {
                formatter.write_str("speech segmenter exceeded the bounded dispatcher event limit")
            }
        }
    }
}

impl std::error::Error for DispatcherError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnedOutcomeLeaseError {
    NoActiveLease,
    TokenMismatch,
}

impl fmt::Display for OwnedOutcomeLeaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoActiveLease => {
                formatter.write_str("no owned dispatcher outcome lease is active")
            }
            Self::TokenMismatch => {
                formatter.write_str("dispatcher outcome lease token does not match")
            }
        }
    }
}

impl std::error::Error for OwnedOutcomeLeaseError {}

/// Abstraction retained solely to make the queue contract deterministic in
/// unit tests. Production code uses the implementation for [`SpeechSegmenter`]
/// below; neither variant sends PCM over IPC.
pub trait SpeechWindowSource {
    fn push_packet(
        &mut self,
        packet: NativePcmPacket<'_>,
    ) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError>;

    fn finish(&mut self) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError>;
}

impl<D: SpeechActivityDetector> SpeechWindowSource for SpeechSegmenter<D> {
    fn push_packet(
        &mut self,
        packet: NativePcmPacket<'_>,
    ) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError> {
        Self::push_packet(self, packet)
    }

    fn finish(&mut self) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError> {
        Self::finish(self)
    }
}

enum PendingDispatcherEvent {
    Event(SpeechWindowEvent),
    Outcome(AsrOutcome),
}

struct ActiveOwnedOutcomeLease {
    token: u64,
    outcome: AsrOutcome,
}

struct ActiveAsrJobLease {
    token: u64,
    metadata: AsrJobMetadata,
}

/// The only native consumer of a [`CaptureIngress`] for one active runtime.
///
/// `pump_ingress_once` is deliberately synchronous. The later CPAL bridge can
/// run it on a native worker, but its deterministic form makes backpressure,
/// claim retries, and stop ordering testable without a microphone or thread.
pub struct CaptureDispatcher<S> {
    runtime: DispatcherRuntime,
    ingress: Arc<CaptureIngress>,
    clock: CaptureClock,
    segmenter: S,
    jobs: Arc<ArrayQueue<AsrJob>>,
    results: Arc<ArrayQueue<AsrOutcome>>,
    pending_events: ArrayQueue<SpeechWindowEvent>,
    pending_event: Option<PendingDispatcherEvent>,
    worker_held_outcome: Option<AsrOutcome>,
    active_asr_job_lease: Option<ActiveAsrJobLease>,
    retry_outcome: Option<AsrOutcome>,
    active_owned_outcome_lease: Option<ActiveOwnedOutcomeLease>,
    next_owned_outcome_token: u64,
    next_asr_job_token: u64,
    meter: DispatcherMeter,
    metrics: AsrQueueMetrics,
    status: DispatcherStatus,
    ingress_drained: bool,
    segmenter_sealed: bool,
    last_packet_range: Option<CaptureRange>,
}

impl<S: SpeechWindowSource> CaptureDispatcher<S> {
    pub fn new(
        runtime: DispatcherRuntime,
        ingress: Arc<CaptureIngress>,
        clock: CaptureClock,
        segmenter: S,
        config: AsrBridgeConfig,
    ) -> Result<Self, DispatcherError> {
        config
            .validate()
            .map_err(DispatcherError::InvalidConfiguration)?;
        if clock.point_at_sample_offset(0) != runtime.capture_anchor {
            return Err(DispatcherError::InvalidRuntime(
                "dispatcher capture clock does not match its runtime anchor".to_owned(),
            ));
        }

        Ok(Self {
            runtime,
            ingress,
            clock,
            segmenter,
            jobs: Arc::new(ArrayQueue::new(config.job_queue_capacity)),
            results: Arc::new(ArrayQueue::new(config.result_queue_capacity)),
            pending_events: ArrayQueue::new(MAX_PENDING_SEGMENTER_EVENTS),
            pending_event: None,
            worker_held_outcome: None,
            active_asr_job_lease: None,
            retry_outcome: None,
            active_owned_outcome_lease: None,
            next_owned_outcome_token: 1,
            next_asr_job_token: 1,
            meter: DispatcherMeter::default(),
            metrics: AsrQueueMetrics::default(),
            status: DispatcherStatus::Running,
            ingress_drained: false,
            segmenter_sealed: false,
            last_packet_range: None,
        })
    }

    pub fn runtime(&self) -> &DispatcherRuntime {
        &self.runtime
    }

    pub fn status(&self) -> DispatcherStatus {
        self.status
    }

    pub fn meter(&self) -> &DispatcherMeter {
        &self.meter
    }

    pub fn metrics(&self) -> AsrQueueMetrics {
        let mut metrics = self.metrics.clone();
        metrics.job_queue_depth = self.jobs.len();
        metrics.result_queue_depth = self.results.len();
        metrics.pending_event_depth = self.pending_events.len()
            + usize::from(self.pending_event.is_some())
            + usize::from(self.retry_outcome.is_some());
        metrics.worker_holds_outcome = self.worker_held_outcome.is_some();
        metrics.owned_outcome_lease_active = self.active_owned_outcome_lease.is_some();
        metrics.closing = self.status != DispatcherStatus::Running;
        metrics
    }

    /// Consume at most one capture packet after first making progress on any
    /// earlier segmenter event. A blocked event prevents a later packet from
    /// overtaking it, preserving FIFO order without unbounded buffering.
    pub fn pump_ingress_once(&mut self) -> Result<IngressPumpResult, DispatcherError> {
        if self.status == DispatcherStatus::Drained {
            return Ok(IngressPumpResult::Drained);
        }
        if !self.progress_pending_event() {
            return Ok(IngressPumpResult::BlockedByPendingEvent);
        }

        let ingress = Arc::clone(&self.ingress);
        let mut consumed = None;
        let did_consume = ingress.try_consume(|packet| {
            let range = self.capture_range_for_packet(
                packet.starting_sample_offset,
                packet.sample_rate,
                packet.channels,
                packet.samples.len(),
            );
            self.update_meter(packet.samples);
            let events = self.segmenter.push_packet(NativePcmPacket {
                starting_sample_offset: packet.starting_sample_offset,
                sample_rate_hz: packet.sample_rate,
                channels: packet.channels,
                samples: packet.samples,
            });
            consumed = Some((range, events));
        });

        if !did_consume {
            self.ingress_drained = self.status == DispatcherStatus::Closing;
            return Ok(IngressPumpResult::NoPacket);
        }

        self.metrics.ingress_packets_consumed =
            self.metrics.ingress_packets_consumed.saturating_add(1);
        let (range, events) = consumed.expect("a consumed ingress packet invokes its callback");
        self.last_packet_range = Some(range.clone());
        match events {
            Ok(events) => self.enqueue_segmenter_events(events, range)?,
            Err(error) => {
                self.metrics.segmenter_failures = self.metrics.segmenter_failures.saturating_add(1);
                self.pending_event =
                    Some(PendingDispatcherEvent::Outcome(AsrOutcome::Gap(self.gap(
                        None,
                        CaptureRange {
                            started_at: error.started_at,
                            ended_at: error.ended_at,
                        },
                        InferenceGapStage::Dispatcher,
                        InferenceGapReason::SegmenterFailed,
                    ))));
            }
        }
        let _ = self.progress_pending_event();
        Ok(IngressPumpResult::Consumed)
    }

    /// Claim one bounded ASR job without retaining the dispatcher mutex while
    /// inference executes.
    ///
    /// A single native ASR worker owns this API. It must return a claimed job
    /// through [`Self::complete_asr_job`]. A full result queue terminalizes
    /// the popped job before an engine call, preserving the existing bounded
    /// backpressure and gap accounting behavior.
    pub fn claim_asr_job(&mut self) -> AsrJobClaim {
        if let Some(outcome) = self.worker_held_outcome.take() {
            match self.try_push_result(outcome) {
                Ok(()) => return AsrJobClaim::DeliveredHeldOutcome,
                Err(outcome) => {
                    self.worker_held_outcome = Some(outcome);
                    return AsrJobClaim::BlockedByResultQueue;
                }
            }
        }

        if self.active_asr_job_lease.is_some() {
            return AsrJobClaim::InFlight;
        }

        let Some(job) = self.jobs.pop() else {
            return AsrJobClaim::NoJob;
        };

        // There is a single sequential worker. If delivery is already full,
        // terminalize this job without invoking ASR and retain just one owned
        // outcome until the state layer frees a result slot.
        if self.results.is_full() {
            self.metrics.result_queue_saturated =
                self.metrics.result_queue_saturated.saturating_add(1);
            self.worker_held_outcome = Some(self.gap_for_job(
                &job,
                InferenceGapStage::ResultQueue,
                InferenceGapReason::ResultQueueSaturated,
            ));
            self.metrics.jobs_completed = self.metrics.jobs_completed.saturating_add(1);
            return AsrJobClaim::BlockedByResultQueue;
        }

        let token = self.next_asr_job_token;
        self.next_asr_job_token = self.next_asr_job_token.saturating_add(1).max(1);
        self.active_asr_job_lease = Some(ActiveAsrJobLease {
            token,
            metadata: job.metadata().clone(),
        });
        AsrJobClaim::Claimed(AsrJobLease { token, job })
    }

    /// Complete an ASR job that was previously claimed by the native worker.
    ///
    /// The response has already been produced outside the dispatcher mutex.
    /// This method validates the engine's provenance and converts every
    /// unavailable/failed result into an explicit terminal gap before placing
    /// it on the bounded result queue.
    pub fn complete_asr_job(
        &mut self,
        lease: AsrJobLease,
        execution: AsrJobExecution,
    ) -> Result<WorkerPumpResult, AsrJobLeaseError> {
        self.take_active_asr_job(&lease)?;

        let outcome = match execution {
            AsrJobExecution::EngineUnavailable => {
                self.metrics.unavailable_engine_outcomes =
                    self.metrics.unavailable_engine_outcomes.saturating_add(1);
                self.gap_for_job(
                    &lease.job,
                    InferenceGapStage::Worker,
                    InferenceGapReason::LocalEngineUnavailable,
                )
            }
            AsrJobExecution::EngineResult {
                model_provenance,
                result,
            } => match result {
                Ok(response) => {
                    match response.validate_against(lease.job.request(), &model_provenance) {
                        Ok(()) => AsrOutcome::Response {
                            job: lease.job.metadata().clone(),
                            response,
                        },
                        Err(_) => {
                            self.metrics.engine_failure_outcomes =
                                self.metrics.engine_failure_outcomes.saturating_add(1);
                            self.gap_for_job(
                                &lease.job,
                                InferenceGapStage::Worker,
                                InferenceGapReason::EngineFailed,
                            )
                        }
                    }
                }
                Err(InferenceError::BackendUnavailable(_)) => {
                    self.metrics.unavailable_engine_outcomes =
                        self.metrics.unavailable_engine_outcomes.saturating_add(1);
                    self.gap_for_job(
                        &lease.job,
                        InferenceGapStage::Worker,
                        InferenceGapReason::LocalEngineUnavailable,
                    )
                }
                Err(InferenceError::InvalidInput(_)) | Err(InferenceError::Failed(_)) => {
                    self.metrics.engine_failure_outcomes =
                        self.metrics.engine_failure_outcomes.saturating_add(1);
                    self.gap_for_job(
                        &lease.job,
                        InferenceGapStage::Worker,
                        InferenceGapReason::EngineFailed,
                    )
                }
            },
        };

        self.metrics.jobs_completed = self.metrics.jobs_completed.saturating_add(1);
        match self.try_push_result(outcome) {
            Ok(()) => Ok(WorkerPumpResult::Processed),
            Err(outcome) => {
                self.worker_held_outcome = Some(outcome);
                Ok(WorkerPumpResult::BlockedByResultQueue)
            }
        }
    }

    /// Process one job synchronously for deterministic tests and callers that
    /// intentionally do not need a separate native ASR worker. The native
    /// runtime uses [`Self::claim_asr_job`] and [`Self::complete_asr_job`] so
    /// `AsrEngine::transcribe` never runs while its dispatcher mutex is held.
    /// Passing `None` remains an explicit local-engine-unavailable state.
    pub fn pump_worker_once(&mut self, engine: Option<&mut dyn AsrEngine>) -> WorkerPumpResult {
        let claim = self.claim_asr_job();
        let AsrJobClaim::Claimed(lease) = claim else {
            return match claim {
                AsrJobClaim::DeliveredHeldOutcome => WorkerPumpResult::DeliveredHeldOutcome,
                AsrJobClaim::NoJob => WorkerPumpResult::NoJob,
                AsrJobClaim::BlockedByResultQueue => WorkerPumpResult::BlockedByResultQueue,
                AsrJobClaim::InFlight => WorkerPumpResult::InFlight,
                AsrJobClaim::Claimed(_) => unreachable!("claimed ASR job must match above"),
            };
        };

        let execution = match engine {
            None => AsrJobExecution::EngineUnavailable,
            Some(engine) => AsrJobExecution::EngineResult {
                model_provenance: engine.model_provenance().clone(),
                result: engine.transcribe(lease.request()),
            },
        };
        self.complete_asr_job(lease, execution)
            .expect("a freshly claimed ASR job completes against its active lease")
    }

    /// Begin a durable outcome delivery. The claim restores itself to the head
    /// of the native retry slot on drop or [`OutcomeClaim::abort`], so a failed
    /// SQLite transaction cannot silently lose the result.
    pub fn begin_outcome(&mut self) -> Option<OutcomeClaim<'_, S>> {
        if self.active_owned_outcome_lease.is_some() {
            return None;
        }
        let outcome = self.retry_outcome.take().or_else(|| self.results.pop())?;
        Some(OutcomeClaim {
            dispatcher: self,
            outcome: Some(outcome),
        })
    }

    /// Transfer an outcome into a lease that does not borrow the dispatcher.
    ///
    /// The state layer can release its capture-service mutex, write SQLite,
    /// then acknowledge the token through [`Self::commit_owned_outcome`] or
    /// restore it through [`Self::abort_owned_outcome`]. Only one such lease
    /// can exist at a time; its retained native copy makes an abandoned token
    /// visible as blocked work instead of silently losing an outcome.
    pub fn begin_owned_outcome(&mut self) -> Option<OwnedOutcomeLease> {
        if self.active_owned_outcome_lease.is_some() {
            return None;
        }
        let outcome = self.retry_outcome.take().or_else(|| self.results.pop())?;
        let token = self.next_owned_outcome_token;
        self.next_owned_outcome_token = self.next_owned_outcome_token.saturating_add(1).max(1);
        self.active_owned_outcome_lease = Some(ActiveOwnedOutcomeLease {
            token,
            outcome: outcome.clone(),
        });
        Some(OwnedOutcomeLease { token, outcome })
    }

    /// Mark a previously leased outcome as durably committed.
    pub fn commit_owned_outcome(&mut self, token: u64) -> Result<(), OwnedOutcomeLeaseError> {
        self.take_owned_outcome(token)?;
        Ok(())
    }

    /// Return a previously leased outcome to the retry head after persistence
    /// fails. It will be delivered before any newer result-queue entry.
    pub fn abort_owned_outcome(&mut self, token: u64) -> Result<(), OwnedOutcomeLeaseError> {
        let outcome = self.take_owned_outcome(token)?;
        self.restore_claimed_outcome(outcome);
        Ok(())
    }

    /// Fence producer input before draining. The caller must stop the CPAL
    /// stream before this call, then keep pumping ingress until it reports
    /// `NoPacket` and finally call [`Self::seal_after_ingress_drain`].
    pub fn begin_shutdown(&mut self) {
        if self.status == DispatcherStatus::Running {
            self.status = DispatcherStatus::Closing;
        }
    }

    /// Discard PCM buffered before the caller commits the capture start.
    ///
    /// This is valid only while the dispatcher is still pristine and running:
    /// no packet may have reached the meter or segmenter, and no job, outcome,
    /// or lease may exist. The native runtime calls this immediately before it
    /// arms its worker, defining the first evidence-bearing ingress boundary.
    /// The caller must ensure the dispatcher is not armed while this runs.
    ///
    /// Unlike shutdown, this deliberately leaves the dispatcher running and
    /// does not create a meter update, transcript, or inference gap.
    pub(crate) fn discard_pristine_ingress_before_arm(&mut self) -> Result<usize, DispatcherError> {
        if self.status != DispatcherStatus::Running || !self.is_pristine_unarmed() {
            return Err(DispatcherError::PreArmIngressDiscardNotAllowed);
        }

        // The producer can still be live while the durable start boundary is
        // crossed. Limit this maintenance path to the queue snapshot so an
        // unusually fast callback cannot keep `arm` draining indefinitely.
        let queued_at_boundary = self.ingress.queued_packet_count();
        let mut discarded = 0_usize;
        for _ in 0..queued_at_boundary {
            if self.ingress.try_consume(|_| {}) {
                discarded = discarded.saturating_add(1);
            } else {
                break;
            }
        }
        Ok(discarded)
    }

    /// Discard queued PCM before this dispatcher has ever consumed it.
    ///
    /// This is strictly for a failed staged startup after the producer has
    /// stopped but before the runtime was armed. It bypasses the segmenter so
    /// no request, transcript, or inference gap is produced for an aborted
    /// pre-recording attempt. Active capture shutdown must use
    /// [`Self::begin_shutdown`] and the normal ingress drain instead.
    pub fn abort_unarmed(&mut self) -> Result<usize, DispatcherError> {
        if self.status == DispatcherStatus::Drained || !self.is_pristine_for_unarmed_abort() {
            return Err(DispatcherError::UnarmedAbortNotAllowed);
        }

        let mut discarded = 0_usize;
        while self.ingress.try_consume(|_| {
            discarded = discarded.saturating_add(1);
        }) {}
        self.status = DispatcherStatus::Closing;
        self.ingress_drained = true;
        Ok(discarded)
    }

    /// Flush the segmenter only after the producer has stopped and all queued
    /// ingress PCM has been consumed. This preserves a final bounded window.
    pub fn seal_after_ingress_drain(&mut self) -> Result<(), DispatcherError> {
        if self.status == DispatcherStatus::Running {
            return Err(DispatcherError::ShutdownNotStarted);
        }
        if !self.ingress_drained {
            return Err(DispatcherError::IngressNotDrained);
        }
        if self.segmenter_sealed {
            return Ok(());
        }

        self.segmenter_sealed = true;
        match self.segmenter.finish() {
            Ok(events) => {
                let Some(range) = self.last_packet_range.clone() else {
                    return Ok(());
                };
                self.enqueue_segmenter_events(events, range)?;
            }
            Err(error) => {
                self.metrics.segmenter_failures = self.metrics.segmenter_failures.saturating_add(1);
                self.pending_event =
                    Some(PendingDispatcherEvent::Outcome(AsrOutcome::Gap(self.gap(
                        None,
                        CaptureRange {
                            started_at: error.started_at,
                            ended_at: error.ended_at,
                        },
                        InferenceGapStage::Dispatcher,
                        InferenceGapReason::SegmenterFailed,
                    ))));
            }
        }
        Ok(())
    }

    /// Drain and seal producer input without terminalizing any admitted ASR
    /// jobs. This lets a runtime reserve a finite post-seal inference budget
    /// for the final window before the normal shutdown drain accounts for
    /// work that remains.
    pub fn prepare_shutdown_for_inference_once(
        &mut self,
    ) -> Result<ShutdownPreparationResult, DispatcherError> {
        if self.status == DispatcherStatus::Running {
            return Err(DispatcherError::ShutdownNotStarted);
        }
        if self.status == DispatcherStatus::Drained {
            return Ok(ShutdownPreparationResult::Drained);
        }
        if !self.progress_pending_event() {
            return Ok(ShutdownPreparationResult::WaitingForPendingEvent);
        }
        if !self.ingress_drained {
            return Ok(ShutdownPreparationResult::WaitingForIngress);
        }
        if !self.segmenter_sealed {
            self.seal_after_ingress_drain()?;
            if !self.progress_pending_event() {
                return Ok(ShutdownPreparationResult::WaitingForPendingEvent);
            }
        }

        Ok(ShutdownPreparationResult::ReadyForInference)
    }

    /// One deterministic shutdown step that turns not-yet-executed jobs into
    /// explicit terminal gaps. It is useful when stopping must not wait for a
    /// local engine call. A normal drain can instead use `pump_worker_once`.
    pub fn drain_shutdown_once(&mut self) -> Result<ShutdownDrainResult, DispatcherError> {
        match self.prepare_shutdown_for_inference_once()? {
            ShutdownPreparationResult::WaitingForIngress => {
                return Ok(ShutdownDrainResult::WaitingForIngress)
            }
            ShutdownPreparationResult::WaitingForPendingEvent => {
                return Ok(ShutdownDrainResult::WaitingForPendingEvent)
            }
            ShutdownPreparationResult::Drained => return Ok(ShutdownDrainResult::Drained),
            ShutdownPreparationResult::ReadyForInference => {}
        }

        if self.active_asr_job_lease.is_some() {
            return Ok(ShutdownDrainResult::AwaitingInference);
        }

        if let Some(outcome) = self.worker_held_outcome.take() {
            match self.try_push_result(outcome) {
                Ok(()) => return Ok(ShutdownDrainResult::AwaitingOutcomeCommit),
                Err(outcome) => {
                    self.worker_held_outcome = Some(outcome);
                    return Ok(ShutdownDrainResult::WaitingForOutcomeDelivery);
                }
            }
        }

        if let Some(job) = self.jobs.pop() {
            self.metrics.shutdown_outcomes = self.metrics.shutdown_outcomes.saturating_add(1);
            let outcome = self.gap_for_job(
                &job,
                InferenceGapStage::Shutdown,
                InferenceGapReason::StoppedBeforeInference,
            );
            match self.try_push_result(outcome) {
                Ok(()) => return Ok(ShutdownDrainResult::AwaitingOutcomeCommit),
                Err(outcome) => {
                    self.worker_held_outcome = Some(outcome);
                    return Ok(ShutdownDrainResult::WaitingForOutcomeDelivery);
                }
            }
        }

        if self.has_uncommitted_outcomes() {
            return Ok(ShutdownDrainResult::AwaitingOutcomeCommit);
        }
        self.status = DispatcherStatus::Drained;
        Ok(ShutdownDrainResult::Drained)
    }

    fn restore_claimed_outcome(&mut self, outcome: AsrOutcome) {
        debug_assert!(
            self.retry_outcome.is_none(),
            "a claim holds the only retry outcome"
        );
        self.retry_outcome = Some(outcome);
        self.metrics.outcome_claims_aborted = self.metrics.outcome_claims_aborted.saturating_add(1);
    }

    fn has_uncommitted_outcomes(&self) -> bool {
        self.results.len() > 0
            || self.retry_outcome.is_some()
            || self.active_owned_outcome_lease.is_some()
    }

    fn is_pristine_for_unarmed_abort(&self) -> bool {
        matches!(
            self.status,
            DispatcherStatus::Running | DispatcherStatus::Closing
        ) && self.is_pristine_unarmed()
    }

    fn is_pristine_unarmed(&self) -> bool {
        self.metrics.ingress_packets_consumed == 0
            && self.pending_events.is_empty()
            && self.pending_event.is_none()
            && self.jobs.is_empty()
            && self.results.is_empty()
            && self.worker_held_outcome.is_none()
            && self.active_asr_job_lease.is_none()
            && self.retry_outcome.is_none()
            && self.active_owned_outcome_lease.is_none()
            && self.last_packet_range.is_none()
            && !self.ingress_drained
            && !self.segmenter_sealed
    }

    fn take_owned_outcome(&mut self, token: u64) -> Result<AsrOutcome, OwnedOutcomeLeaseError> {
        let Some(active) = self.active_owned_outcome_lease.take() else {
            return Err(OwnedOutcomeLeaseError::NoActiveLease);
        };
        if active.token != token {
            self.active_owned_outcome_lease = Some(active);
            return Err(OwnedOutcomeLeaseError::TokenMismatch);
        }
        Ok(active.outcome)
    }

    fn take_active_asr_job(&mut self, lease: &AsrJobLease) -> Result<(), AsrJobLeaseError> {
        let Some(active) = self.active_asr_job_lease.take() else {
            return Err(AsrJobLeaseError::NoActiveLease);
        };
        if active.token != lease.token {
            self.active_asr_job_lease = Some(active);
            return Err(AsrJobLeaseError::TokenMismatch);
        }
        if active.metadata != *lease.metadata() {
            self.active_asr_job_lease = Some(active);
            return Err(AsrJobLeaseError::MetadataMismatch);
        }
        Ok(())
    }

    fn progress_pending_event(&mut self) -> bool {
        loop {
            if self.pending_event.is_none() {
                self.pending_event = self.pending_events.pop().map(PendingDispatcherEvent::Event);
            }
            let Some(pending) = self.pending_event.take() else {
                return true;
            };

            match pending {
                PendingDispatcherEvent::Outcome(outcome) => match self.try_push_result(outcome) {
                    Ok(()) => continue,
                    Err(outcome) => {
                        self.pending_event = Some(PendingDispatcherEvent::Outcome(outcome));
                        return false;
                    }
                },
                PendingDispatcherEvent::Event(SpeechWindowEvent::Discontinuity { .. }) => {
                    self.metrics.ingress_discontinuities =
                        self.metrics.ingress_discontinuities.saturating_add(1);
                }
                PendingDispatcherEvent::Event(SpeechWindowEvent::Request {
                    session_id,
                    request,
                }) => {
                    let range = self.capture_range_for_request(&request);
                    if session_id != self.runtime.session_id
                        || request.audio.session_id() != session_id
                    {
                        self.metrics.segmenter_failures =
                            self.metrics.segmenter_failures.saturating_add(1);
                        self.pending_event =
                            Some(PendingDispatcherEvent::Outcome(AsrOutcome::Gap(self.gap(
                                None,
                                range,
                                InferenceGapStage::Dispatcher,
                                InferenceGapReason::SegmenterFailed,
                            ))));
                        continue;
                    }

                    let job = match AsrJob::new(
                        &self.runtime,
                        request,
                        range.started_at.clone(),
                        range.ended_at.clone(),
                    ) {
                        Ok(job) => job,
                        Err(_) => {
                            self.metrics.segmenter_failures =
                                self.metrics.segmenter_failures.saturating_add(1);
                            self.pending_event =
                                Some(PendingDispatcherEvent::Outcome(AsrOutcome::Gap(self.gap(
                                    None,
                                    range,
                                    InferenceGapStage::Dispatcher,
                                    InferenceGapReason::SegmenterFailed,
                                ))));
                            continue;
                        }
                    };

                    match self.jobs.push(job) {
                        Ok(()) => {
                            self.metrics.jobs_admitted =
                                self.metrics.jobs_admitted.saturating_add(1);
                            self.metrics.job_queue_high_watermark =
                                self.metrics.job_queue_high_watermark.max(self.jobs.len());
                        }
                        Err(job) => {
                            self.metrics.job_queue_saturated =
                                self.metrics.job_queue_saturated.saturating_add(1);
                            self.pending_event =
                                Some(PendingDispatcherEvent::Outcome(self.gap_for_job(
                                    &job,
                                    InferenceGapStage::JobQueue,
                                    InferenceGapReason::JobQueueSaturated,
                                )));
                        }
                    }
                }
            }
        }
    }

    fn enqueue_segmenter_events(
        &mut self,
        events: Vec<SpeechWindowEvent>,
        fallback_range: CaptureRange,
    ) -> Result<(), DispatcherError> {
        if events.len() > MAX_EVENTS_PER_CAPTURE_PACKET
            || self.pending_events.len() + events.len() > MAX_PENDING_SEGMENTER_EVENTS
        {
            self.metrics.segmenter_failures = self.metrics.segmenter_failures.saturating_add(1);
            self.pending_event = Some(PendingDispatcherEvent::Outcome(AsrOutcome::Gap(self.gap(
                None,
                fallback_range,
                InferenceGapStage::Dispatcher,
                InferenceGapReason::SegmenterFailed,
            ))));
            return Err(DispatcherError::SegmenterEventOverflow);
        }
        for event in events {
            self.pending_events
                .push(event)
                .expect("the checked bounded segmenter event queue has capacity");
        }
        self.metrics.pending_event_high_watermark = self
            .metrics
            .pending_event_high_watermark
            .max(self.pending_events.len());
        Ok(())
    }

    fn try_push_result(&mut self, outcome: AsrOutcome) -> Result<(), AsrOutcome> {
        match self.results.push(outcome) {
            Ok(()) => {
                self.metrics.result_queue_high_watermark = self
                    .metrics
                    .result_queue_high_watermark
                    .max(self.results.len());
                Ok(())
            }
            Err(outcome) => Err(outcome),
        }
    }

    fn gap_for_job(
        &self,
        job: &AsrJob,
        stage: InferenceGapStage,
        reason: InferenceGapReason,
    ) -> AsrOutcome {
        AsrOutcome::Gap(self.gap(
            Some(job.metadata.id),
            CaptureRange {
                started_at: job.metadata.started_at.clone(),
                ended_at: job.metadata.ended_at.clone(),
            },
            stage,
            reason,
        ))
    }

    fn gap(
        &self,
        job_id: Option<Uuid>,
        range: CaptureRange,
        stage: InferenceGapStage,
        reason: InferenceGapReason,
    ) -> InferenceGap {
        InferenceGap::new(
            Uuid::new_v4(),
            self.runtime.session_id,
            self.runtime.id.as_uuid(),
            self.runtime.capture_segment_id,
            job_id,
            range.started_at,
            range.ended_at,
            stage,
            reason,
        )
        .expect("dispatcher only constructs validated runtime and capture ranges")
    }

    fn capture_range_for_packet(
        &self,
        starting_sample_offset: u64,
        sample_rate: u32,
        channels: u16,
        sample_count: usize,
    ) -> CaptureRange {
        let frames = if channels == 0 {
            0
        } else {
            sample_count / usize::from(channels)
        };
        // CaptureIngress already validates packet layout. Saturation keeps
        // malformed boundary data from wrapping a terminal range backwards.
        let ending_sample_offset = starting_sample_offset.saturating_add(frames as u64);
        debug_assert_eq!(sample_rate, self.clock.sample_rate());
        CaptureRange {
            started_at: self.clock.point_at_sample_offset(starting_sample_offset),
            ended_at: self.clock.point_at_sample_offset(ending_sample_offset),
        }
    }

    fn capture_range_for_request(&self, request: &AsrRequest) -> CaptureRange {
        CaptureRange {
            started_at: self.capture_point_at_monotonic_ns(request.audio.capture_start_ns()),
            ended_at: self.capture_point_at_monotonic_ns(request.audio.capture_end_ns()),
        }
    }

    fn capture_point_at_monotonic_ns(&self, monotonic_ns: u64) -> CapturePoint {
        let anchor = &self.runtime.capture_anchor;
        let wall_clock = if monotonic_ns >= anchor.monotonic_ns {
            anchor.wall_clock
                + Duration::nanoseconds(
                    monotonic_ns
                        .saturating_sub(anchor.monotonic_ns)
                        .min(i64::MAX as u64) as i64,
                )
        } else {
            anchor.wall_clock
                - Duration::nanoseconds(
                    anchor
                        .monotonic_ns
                        .saturating_sub(monotonic_ns)
                        .min(i64::MAX as u64) as i64,
                )
        };
        CapturePoint {
            monotonic_ns,
            wall_clock,
        }
    }

    fn update_meter(&mut self, samples: &[f32]) {
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            self.meter = DispatcherMeter::default();
            return;
        }
        let mut squared_sum = 0.0_f64;
        let mut peak = 0.0_f32;
        let mut clipping = false;
        for sample in samples {
            let magnitude = sample.abs();
            squared_sum += f64::from(*sample) * f64::from(*sample);
            peak = peak.max(magnitude);
            clipping |= magnitude >= 0.999;
        }
        let rms = (squared_sum / samples.len() as f64).sqrt() as f32;
        self.meter = DispatcherMeter {
            rms_dbfs: to_dbfs(rms),
            peak_dbfs: to_dbfs(peak),
            clipping,
        };
    }
}

/// A result removed from the bounded queue but not yet durably acknowledged.
///
/// The dispatcher retains its own copy while this lease is active. Callers
/// must explicitly acknowledge the token with `commit_owned_outcome` after a
/// durable write, or `abort_owned_outcome` after a failed write. Dropping this
/// value alone intentionally leaves the retained outcome blocked, preventing
/// an uncertain persistence result from being silently retried or discarded.
pub struct OwnedOutcomeLease {
    token: u64,
    outcome: AsrOutcome,
}

impl OwnedOutcomeLease {
    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn outcome(&self) -> &AsrOutcome {
        &self.outcome
    }
}

/// A result removed from the bounded queue while the caller still holds a
/// mutable dispatcher borrow. Prefer [`OwnedOutcomeLease`] for persistence
/// that must occur after releasing a capture-service lock.
pub struct OutcomeClaim<'a, S: SpeechWindowSource> {
    dispatcher: &'a mut CaptureDispatcher<S>,
    outcome: Option<AsrOutcome>,
}

impl<S: SpeechWindowSource> OutcomeClaim<'_, S> {
    pub fn outcome(&self) -> &AsrOutcome {
        self.outcome
            .as_ref()
            .expect("an outcome claim exposes its value until commit or abort")
    }

    pub fn commit(mut self) {
        let _ = self.outcome.take();
    }

    pub fn abort(mut self) {
        if let Some(outcome) = self.outcome.take() {
            self.dispatcher.restore_claimed_outcome(outcome);
        }
    }
}

impl<S: SpeechWindowSource> Drop for OutcomeClaim<'_, S> {
    fn drop(&mut self) {
        if let Some(outcome) = self.outcome.take() {
            self.dispatcher.restore_claimed_outcome(outcome);
        }
    }
}

#[derive(Clone, Debug)]
struct CaptureRange {
    started_at: CapturePoint,
    ended_at: CapturePoint,
}

fn validate_queue_capacity(label: &str, capacity: usize) -> Result<(), String> {
    if capacity == 0 {
        return Err(format!("{label} capacity must be greater than zero"));
    }
    if capacity > MAX_ASR_QUEUE_CAPACITY {
        return Err(format!(
            "{label} capacity exceeds the {MAX_ASR_QUEUE_CAPACITY}-item native bound"
        ));
    }
    Ok(())
}

fn to_dbfs(value: f32) -> f32 {
    if value <= 0.0 || !value.is_finite() {
        return MINIMUM_DBFS;
    }
    (20.0 * value.log10()).max(MINIMUM_DBFS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{
        InferenceAudioWindow, InferenceEngine, InferenceExecutionScope, ModelProvenance,
        INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ,
    };
    use chrono::{DateTime, Duration as ChronoDuration, Utc};
    use std::collections::VecDeque;

    struct ScriptedSegmenter {
        events: VecDeque<Vec<SpeechWindowEvent>>,
        finish_events: Vec<SpeechWindowEvent>,
    }

    impl ScriptedSegmenter {
        fn requests(session_id: Uuid, starts: impl IntoIterator<Item = u64>) -> Self {
            Self {
                events: starts
                    .into_iter()
                    .map(|start| {
                        vec![SpeechWindowEvent::Request {
                            session_id,
                            request: request(session_id, start),
                        }]
                    })
                    .collect(),
                finish_events: Vec::new(),
            }
        }
    }

    impl SpeechWindowSource for ScriptedSegmenter {
        fn push_packet(
            &mut self,
            _packet: NativePcmPacket<'_>,
        ) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError> {
            Ok(self.events.pop_front().unwrap_or_default())
        }

        fn finish(&mut self) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError> {
            Ok(std::mem::take(&mut self.finish_events))
        }
    }

    struct EmptyEngine {
        model: ModelProvenance,
        calls: usize,
        failure: Option<InferenceError>,
    }

    impl EmptyEngine {
        fn successful() -> Self {
            Self {
                model: model(),
                calls: 0,
                failure: None,
            }
        }

        fn failing(error: InferenceError) -> Self {
            Self {
                model: model(),
                calls: 0,
                failure: Some(error),
            }
        }
    }

    impl InferenceEngine for EmptyEngine {
        fn model_provenance(&self) -> &ModelProvenance {
            &self.model
        }

        fn execution_scope(&self) -> InferenceExecutionScope {
            InferenceExecutionScope::OnDevice
        }
    }

    impl AsrEngine for EmptyEngine {
        fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError> {
            self.calls += 1;
            if let Some(error) = self.failure.clone() {
                return Err(error);
            }
            AsrResponse::new(request, &self.model, Vec::new()).map_err(InferenceError::invalid)
        }
    }

    fn model() -> ModelProvenance {
        ModelProvenance::new("test", "empty", "v1", "a".repeat(64)).unwrap()
    }

    fn anchor() -> CapturePoint {
        CapturePoint {
            monotonic_ns: 1_000,
            wall_clock: DateTime::<Utc>::UNIX_EPOCH + ChronoDuration::seconds(10),
        }
    }

    fn runtime(session_id: Uuid) -> DispatcherRuntime {
        DispatcherRuntime::new(
            DispatcherRuntimeId::new(Uuid::from_u128(11)).unwrap(),
            session_id,
            Uuid::from_u128(12),
            anchor(),
        )
        .unwrap()
    }

    fn request(session_id: Uuid, capture_start_ns: u64) -> AsrRequest {
        AsrRequest::new(
            InferenceAudioWindow::new(
                session_id,
                capture_start_ns,
                capture_start_ns + 10_000_000,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![0.5; 160],
            )
            .unwrap(),
            Some("zh".to_owned()),
            false,
        )
        .unwrap()
    }

    fn dispatcher(
        segmenter: ScriptedSegmenter,
        job_capacity: usize,
        result_capacity: usize,
    ) -> CaptureDispatcher<ScriptedSegmenter> {
        let session_id = Uuid::from_u128(10);
        CaptureDispatcher::new(
            runtime(session_id),
            CaptureIngress::new(4, 160).unwrap(),
            CaptureClock::new(anchor(), 16_000).unwrap(),
            segmenter,
            AsrBridgeConfig {
                job_queue_capacity: job_capacity,
                result_queue_capacity: result_capacity,
            },
        )
        .unwrap()
    }

    fn write_packet(dispatcher: &CaptureDispatcher<ScriptedSegmenter>, offset: u64) {
        assert_eq!(
            dispatcher.ingress.try_write(offset, 16_000, 1, &[0.5; 160]),
            super::super::CaptureWriteResult::Enqueued
        );
    }

    fn gap(
        dispatcher: &CaptureDispatcher<ScriptedSegmenter>,
        reason: InferenceGapReason,
    ) -> AsrOutcome {
        AsrOutcome::Gap(
            InferenceGap::new(
                Uuid::from_u128(100),
                dispatcher.runtime.session_id,
                dispatcher.runtime.id.as_uuid(),
                dispatcher.runtime.capture_segment_id,
                None,
                anchor(),
                anchor(),
                InferenceGapStage::Dispatcher,
                reason,
            )
            .unwrap(),
        )
    }

    #[test]
    fn rejects_zero_or_excessive_queue_capacities() {
        assert!(AsrBridgeConfig {
            job_queue_capacity: 0,
            result_queue_capacity: 1,
        }
        .validate()
        .is_err());
        assert!(AsrBridgeConfig {
            job_queue_capacity: 1,
            result_queue_capacity: MAX_ASR_QUEUE_CAPACITY + 1,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn one_dispatcher_consumer_updates_meter_and_admits_a_job() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 2);
        write_packet(&dispatcher, 0);

        assert_eq!(
            dispatcher.pump_ingress_once().unwrap(),
            IngressPumpResult::Consumed
        );
        assert_eq!(dispatcher.metrics().ingress_packets_consumed, 1);
        assert_eq!(dispatcher.metrics().jobs_admitted, 1);
        assert_eq!(dispatcher.metrics().job_queue_depth, 1);
        assert!(dispatcher.meter().rms_dbfs > MINIMUM_DBFS);
        assert!(!dispatcher.meter().clipping);
    }

    #[test]
    fn discards_pristine_prearm_pcm_without_exposing_capture_artifacts() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 2);
        for offset in [0, 160, 320, 480] {
            write_packet(&dispatcher, offset);
        }

        assert_eq!(dispatcher.discard_pristine_ingress_before_arm().unwrap(), 4);
        assert_eq!(dispatcher.metrics().ingress_packets_consumed, 0);
        assert_eq!(dispatcher.meter(), &DispatcherMeter::default());
        assert!(dispatcher.begin_outcome().is_none());
        assert_eq!(
            dispatcher.pump_ingress_once().unwrap(),
            IngressPumpResult::NoPacket
        );

        write_packet(&dispatcher, 640);
        assert_eq!(
            dispatcher.pump_ingress_once().unwrap(),
            IngressPumpResult::Consumed
        );
        assert_eq!(dispatcher.metrics().jobs_admitted, 1);
        assert_eq!(
            dispatcher.discard_pristine_ingress_before_arm(),
            Err(DispatcherError::PreArmIngressDiscardNotAllowed)
        );
    }

    #[test]
    fn job_saturation_produces_one_range_bearing_gap_without_invoking_asr() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(
            ScriptedSegmenter::requests(session_id, [1_000, 11_000_000]),
            1,
            2,
        );
        write_packet(&dispatcher, 0);
        write_packet(&dispatcher, 160);
        dispatcher.pump_ingress_once().unwrap();
        dispatcher.pump_ingress_once().unwrap();

        let claim = dispatcher
            .begin_outcome()
            .expect("saturation is terminally reported");
        let AsrOutcome::Gap(gap) = claim.outcome() else {
            panic!("job saturation must not fabricate an ASR response");
        };
        assert_eq!(gap.stage, InferenceGapStage::JobQueue);
        assert_eq!(gap.reason, InferenceGapReason::JobQueueSaturated);
        assert_eq!(gap.started_at.monotonic_ns, 11_000_000);
        assert_eq!(gap.ended_at.monotonic_ns, 21_000_000);
        claim.commit();
        assert_eq!(dispatcher.metrics().job_queue_saturated, 1);
    }

    #[test]
    fn result_saturation_terminalizes_before_calling_asr() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 1);
        dispatcher
            .results
            .push(gap(&dispatcher, InferenceGapReason::SegmenterFailed))
            .unwrap();
        write_packet(&dispatcher, 0);
        dispatcher.pump_ingress_once().unwrap();

        let mut engine = EmptyEngine::successful();
        assert_eq!(
            dispatcher.pump_worker_once(Some(&mut engine)),
            WorkerPumpResult::BlockedByResultQueue
        );
        assert_eq!(engine.calls, 0);
        dispatcher.begin_outcome().unwrap().commit();
        assert_eq!(
            dispatcher.pump_worker_once(Some(&mut engine)),
            WorkerPumpResult::DeliveredHeldOutcome
        );
        let claim = dispatcher.begin_outcome().unwrap();
        let AsrOutcome::Gap(gap) = claim.outcome() else {
            panic!("result saturation must become a gap before ASR execution");
        };
        assert_eq!(gap.reason, InferenceGapReason::ResultQueueSaturated);
        claim.commit();
        assert_eq!(engine.calls, 0);
    }

    #[test]
    fn unavailable_and_failed_engines_have_distinct_terminal_reasons() {
        let session_id = Uuid::from_u128(10);
        let mut unavailable = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 2);
        write_packet(&unavailable, 0);
        unavailable.pump_ingress_once().unwrap();
        unavailable.pump_worker_once(None);
        let claim = unavailable.begin_outcome().unwrap();
        let AsrOutcome::Gap(gap) = claim.outcome() else {
            panic!("unavailable engine must not return text");
        };
        assert_eq!(gap.reason, InferenceGapReason::LocalEngineUnavailable);
        claim.commit();

        let mut failed = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 2);
        write_packet(&failed, 0);
        failed.pump_ingress_once().unwrap();
        let mut engine = EmptyEngine::failing(InferenceError::failed("test failure"));
        failed.pump_worker_once(Some(&mut engine));
        let claim = failed.begin_outcome().unwrap();
        let AsrOutcome::Gap(gap) = claim.outcome() else {
            panic!("failed engine must not return text");
        };
        assert_eq!(gap.reason, InferenceGapReason::EngineFailed);
        claim.commit();
    }

    #[test]
    fn shutdown_waits_for_a_claimed_asr_job_before_terminalizing_remaining_work() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 2);
        write_packet(&dispatcher, 0);
        dispatcher.pump_ingress_once().unwrap();

        let AsrJobClaim::Claimed(lease) = dispatcher.claim_asr_job() else {
            panic!("an admitted job must transfer to the ASR worker");
        };
        dispatcher.begin_shutdown();
        assert_eq!(
            dispatcher.pump_ingress_once().unwrap(),
            IngressPumpResult::NoPacket
        );
        assert_eq!(
            dispatcher.prepare_shutdown_for_inference_once().unwrap(),
            ShutdownPreparationResult::ReadyForInference
        );
        assert_eq!(
            dispatcher.drain_shutdown_once().unwrap(),
            ShutdownDrainResult::AwaitingInference
        );

        let mut engine = EmptyEngine::successful();
        let execution = AsrJobExecution::EngineResult {
            model_provenance: engine.model_provenance().clone(),
            result: engine.transcribe(lease.request()),
        };
        assert_eq!(
            dispatcher.complete_asr_job(lease, execution).unwrap(),
            WorkerPumpResult::Processed
        );
        let claim = dispatcher.begin_outcome().unwrap();
        assert!(matches!(claim.outcome(), AsrOutcome::Response { .. }));
        claim.commit();
        assert_eq!(
            dispatcher.drain_shutdown_once().unwrap(),
            ShutdownDrainResult::Drained
        );
    }

    #[test]
    fn aborted_claim_remains_available_before_later_results() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(
            ScriptedSegmenter::requests(session_id, [1_000, 11_000_000]),
            2,
            2,
        );
        write_packet(&dispatcher, 0);
        write_packet(&dispatcher, 160);
        dispatcher.pump_ingress_once().unwrap();
        dispatcher.pump_ingress_once().unwrap();
        dispatcher.pump_worker_once(None);
        dispatcher.pump_worker_once(None);

        let first = dispatcher.begin_outcome().unwrap();
        let first_start = match first.outcome() {
            AsrOutcome::Gap(gap) => gap.started_at.monotonic_ns,
            AsrOutcome::Response { .. } => panic!("no engine should produce a gap"),
        };
        first.abort();
        let retry = dispatcher.begin_outcome().unwrap();
        let retry_start = match retry.outcome() {
            AsrOutcome::Gap(gap) => gap.started_at.monotonic_ns,
            AsrOutcome::Response { .. } => panic!("no engine should produce a gap"),
        };
        assert_eq!(retry_start, first_start);
        retry.commit();
        dispatcher.begin_outcome().unwrap().commit();
        assert_eq!(dispatcher.metrics().outcome_claims_aborted, 1);
        assert_eq!(dispatcher.metrics().result_queue_high_watermark, 2);
    }

    #[test]
    fn owned_outcome_lease_requires_explicit_acknowledgement_after_drop() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 2);
        write_packet(&dispatcher, 0);
        dispatcher.pump_ingress_once().unwrap();
        dispatcher.pump_worker_once(None);

        let lease = dispatcher
            .begin_owned_outcome()
            .expect("the native outcome is available without borrowing dispatcher");
        let token = lease.token();
        assert!(matches!(lease.outcome(), AsrOutcome::Gap(_)));
        assert!(dispatcher.begin_owned_outcome().is_none());
        drop(lease);

        // The retained dispatcher copy deliberately remains blocked until the
        // persistence owner explicitly decides whether to retry or commit it.
        assert!(dispatcher.begin_owned_outcome().is_none());
        dispatcher.abort_owned_outcome(token).unwrap();
        let retry = dispatcher.begin_owned_outcome().unwrap();
        let retry_token = retry.token();
        assert!(matches!(retry.outcome(), AsrOutcome::Gap(_)));
        drop(retry);
        dispatcher.commit_owned_outcome(retry_token).unwrap();
        assert!(dispatcher.begin_owned_outcome().is_none());
    }

    #[test]
    fn shutdown_drain_terminalizes_pending_work_before_reporting_drained() {
        let session_id = Uuid::from_u128(10);
        let mut dispatcher = dispatcher(ScriptedSegmenter::requests(session_id, [1_000]), 1, 2);
        write_packet(&dispatcher, 0);
        dispatcher.pump_ingress_once().unwrap();
        dispatcher.begin_shutdown();
        assert_eq!(
            dispatcher.pump_ingress_once().unwrap(),
            IngressPumpResult::NoPacket
        );
        dispatcher.seal_after_ingress_drain().unwrap();
        assert_eq!(
            dispatcher.drain_shutdown_once().unwrap(),
            ShutdownDrainResult::AwaitingOutcomeCommit
        );
        let claim = dispatcher.begin_outcome().unwrap();
        let AsrOutcome::Gap(gap) = claim.outcome() else {
            panic!("unprocessed shutdown work must be a gap");
        };
        assert_eq!(gap.stage, InferenceGapStage::Shutdown);
        assert_eq!(gap.reason, InferenceGapReason::StoppedBeforeInference);
        claim.commit();
        assert_eq!(
            dispatcher.drain_shutdown_once().unwrap(),
            ShutdownDrainResult::Drained
        );
        assert_eq!(dispatcher.status(), DispatcherStatus::Drained);
    }
}
