//! Native-only packet normalization and speech-window assembly.
//!
//! This is intentionally a synchronous, bounded contract for fixture input.
//! It does not own a microphone callback, a queue, a database, or a Tauri
//! handle. The CPAL bridge will use it later from its single dispatcher.

use super::{
    AsrEngine, AsrRequest, AsrResponse, InferenceAudioWindow, InferenceError, VadEngine,
    VadRequest, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ, MAX_INFERENCE_WINDOW_SAMPLES,
};
use crate::audio::{CaptureClock, CapturePacket, CapturePoint, MAX_CAPTURE_SAMPLES_PER_PACKET};
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;
use std::fmt;
use uuid::Uuid;

pub const PIPELINE_FRAME_SAMPLES: usize = 160;
const RESAMPLER_INPUT_SAMPLES: usize = 480;
const RESAMPLER_OUTPUT_SAMPLES: usize = PIPELINE_FRAME_SAMPLES;
const MAX_PIPELINE_WINDOW_FRAMES: usize = MAX_INFERENCE_WINDOW_SAMPLES / PIPELINE_FRAME_SAMPLES;
const MAX_PIPELINE_EVENTS_PER_PACKET: usize =
    2 + MAX_CAPTURE_SAMPLES_PER_PACKET / PIPELINE_FRAME_SAMPLES;

/// Borrowed capture PCM. It never crosses a native-process boundary.
#[derive(Clone, Copy, Debug)]
pub struct NativePcmPacket<'a> {
    pub starting_sample_offset: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: &'a [f32],
}

impl<'a> From<&'a CapturePacket> for NativePcmPacket<'a> {
    fn from(packet: &'a CapturePacket) -> Self {
        Self {
            starting_sample_offset: packet.starting_sample_offset,
            sample_rate_hz: packet.sample_rate,
            channels: packet.channels,
            samples: &packet.samples,
        }
    }
}

impl<'a> NativePcmPacket<'a> {
    fn frame_count(&self) -> Result<usize, String> {
        if self.samples.len() > MAX_CAPTURE_SAMPLES_PER_PACKET {
            return Err(format!(
                "native PCM packet exceeds the {MAX_CAPTURE_SAMPLES_PER_PACKET}-sample capture bound"
            ));
        }
        if self.sample_rate_hz == 0 {
            return Err("native PCM sample rate must be greater than zero".to_owned());
        }
        if self.channels == 0 {
            return Err("native PCM channel count must be greater than zero".to_owned());
        }
        let channels = usize::from(self.channels);
        if !self.samples.len().is_multiple_of(channels) {
            return Err("native PCM samples must align to its channel count".to_owned());
        }
        if !self.samples.iter().all(|sample| sample.is_finite()) {
            return Err("native PCM samples must be finite".to_owned());
        }
        Ok(self.samples.len() / channels)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpeechPipelineEvent {
    AsrResponse {
        session_id: Uuid,
        response: AsrResponse,
    },
    Discontinuity {
        session_id: Uuid,
        expected_source_offset: u64,
        received_source_offset: u64,
        at_capture_ns: u64,
    },
}

/// A completed native speech window or a discontinuity in its source PCM.
///
/// [`Self::Request`] owns its [`AsrRequest`] so a native dispatcher can put it
/// on a bounded queue without retaining the capture packet. It remains inside
/// the Rust process and must not be sent through Tauri IPC.
#[derive(Clone, Debug, PartialEq)]
pub enum SpeechWindowEvent {
    Request {
        session_id: Uuid,
        request: AsrRequest,
    },
    Discontinuity {
        session_id: Uuid,
        expected_source_offset: u64,
        received_source_offset: u64,
        at_capture_ns: u64,
    },
}

/// A terminal segmentation failure with the capture range that cannot yield a
/// complete inference outcome.
///
/// The range includes any unfinished audio held before the packet began and
/// the whole packet that failed. A dispatcher can turn it directly into an
/// auditable inference gap without retaining PCM.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeechSegmenterError {
    pub session_id: Uuid,
    pub started_at: CapturePoint,
    pub ended_at: CapturePoint,
    pub message: String,
}

impl fmt::Display for SpeechSegmenterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SpeechSegmenterError {}

/// A frame-level activity decision used to assemble speech windows.
///
/// The default test fixture is an energy gate. A real VAD model can be
/// attached through [`VadSpeechDetector`] without changing the rest of the
/// pipeline.
pub trait SpeechActivityDetector: Send {
    fn is_speech(&mut self, frame: &InferenceAudioWindow) -> Result<bool, InferenceError>;

    /// Drops detector state that must not span a capture discontinuity.
    ///
    /// Stateless fixture detectors can retain the default implementation.
    fn reset(&mut self) {}
}

/// Explicitly temporary energy gate for deterministic local pipeline tests.
/// It is not a production VAD claim.
#[derive(Clone, Debug)]
pub struct EnergySpeechDetector {
    minimum_rms: f32,
}

impl EnergySpeechDetector {
    pub fn new(minimum_rms: f32) -> Result<Self, String> {
        if !minimum_rms.is_finite() || minimum_rms < 0.0 {
            return Err("speech energy threshold must be a finite non-negative value".to_owned());
        }
        Ok(Self { minimum_rms })
    }
}

impl SpeechActivityDetector for EnergySpeechDetector {
    fn is_speech(&mut self, frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
        Ok(root_mean_square(frame.samples())? >= self.minimum_rms)
    }
}

/// Combines a stateful speech detector with a local RMS admission floor.
///
/// The wrapped detector is deliberately evaluated before the RMS result is
/// applied. WebRTC VAD keeps a short signal history, so quiet frames must
/// still reach it in order to clear that native state rather than leaving a
/// stale speech decision alive across room noise.
pub struct EnergyGatedSpeechDetector<D> {
    detector: D,
    gate: SpeechEnergyGate,
}

enum SpeechEnergyGate {
    Manual { minimum_rms: f32 },
    Adaptive { noise_floor_dbfs: Option<f32> },
}

impl<D> EnergyGatedSpeechDetector<D> {
    pub fn new(detector: D, minimum_rms: f32) -> Result<Self, String> {
        if !minimum_rms.is_finite() || minimum_rms < 0.0 {
            return Err("speech energy threshold must be a finite non-negative value".to_owned());
        }
        Ok(Self {
            detector,
            gate: SpeechEnergyGate::Manual { minimum_rms },
        })
    }

    pub fn adaptive(detector: D) -> Self {
        Self {
            detector,
            gate: SpeechEnergyGate::Adaptive {
                noise_floor_dbfs: None,
            },
        }
    }

    pub fn into_inner(self) -> D {
        self.detector
    }

    fn effective_minimum_rms(&self) -> f32 {
        match self.gate {
            SpeechEnergyGate::Manual { minimum_rms } => minimum_rms,
            SpeechEnergyGate::Adaptive { noise_floor_dbfs } => {
                let threshold_dbfs = (noise_floor_dbfs.unwrap_or(-54.0) + 12.0).clamp(-42.0, -24.0);
                10_f32.powf(threshold_dbfs / 20.0)
            }
        }
    }

    fn observe_non_speech_rms(&mut self, rms: f32) {
        let SpeechEnergyGate::Adaptive { noise_floor_dbfs } = &mut self.gate else {
            return;
        };
        let observed_dbfs = if rms > 0.0 {
            (20.0 * rms.log10()).max(-96.0)
        } else {
            -96.0
        };
        *noise_floor_dbfs = Some(match *noise_floor_dbfs {
            Some(previous) => previous * 0.95 + observed_dbfs * 0.05,
            None => observed_dbfs,
        });
    }
}

impl<D: SpeechActivityDetector> SpeechActivityDetector for EnergyGatedSpeechDetector<D> {
    fn is_speech(&mut self, frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
        let detector_reports_speech = self.detector.is_speech(frame)?;
        let rms = root_mean_square(frame.samples())?;
        if !detector_reports_speech {
            self.observe_non_speech_rms(rms);
            return Ok(false);
        }
        Ok(rms >= self.effective_minimum_rms())
    }

    fn reset(&mut self) {
        self.detector.reset();
        if let SpeechEnergyGate::Adaptive { noise_floor_dbfs } = &mut self.gate {
            *noise_floor_dbfs = None;
        }
    }
}

fn root_mean_square(samples: &[f32]) -> Result<f32, InferenceError> {
    if samples.is_empty() {
        return Err(InferenceError::invalid(
            "speech energy evaluation requires at least one sample",
        ));
    }

    let energy = samples.iter().try_fold(0_f64, |total, sample| {
        if !sample.is_finite() {
            return Err(InferenceError::invalid(
                "speech energy samples must be finite",
            ));
        }
        Ok(total + f64::from(*sample) * f64::from(*sample))
    })?;
    Ok((energy / samples.len() as f64).sqrt() as f32)
}

/// Adapts an existing local VAD engine to the frame-level pipeline contract.
pub struct VadSpeechDetector<V> {
    engine: V,
    minimum_probability: f32,
}

impl<V> VadSpeechDetector<V> {
    pub fn new(engine: V, minimum_probability: f32) -> Result<Self, String> {
        if !minimum_probability.is_finite() || !(0.0..=1.0).contains(&minimum_probability) {
            return Err("VAD speech probability threshold must be between zero and one".to_owned());
        }
        Ok(Self {
            engine,
            minimum_probability,
        })
    }

    pub fn into_inner(self) -> V {
        self.engine
    }
}

impl<V: VadEngine> SpeechActivityDetector for VadSpeechDetector<V> {
    fn is_speech(&mut self, frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
        let request = VadRequest::new(frame.clone()).map_err(InferenceError::invalid)?;
        let response = self.engine.detect_voice_activity(&request)?;
        Ok(response.segments.iter().any(|segment| {
            segment.speech_probability >= self.minimum_probability
                && segment.capture_start_ns < frame.capture_end_ns()
                && segment.capture_end_ns > frame.capture_start_ns()
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechPipelineConfig {
    pub pre_roll_frames: usize,
    pub hangover_frames: usize,
    pub maximum_window_frames: usize,
    /// The longest contiguous run of VAD-positive 10 ms frames required
    /// before an utterance can enter ASR. Pre-roll and hangover audio remain
    /// attached but do not count toward this admission threshold.
    pub minimum_speech_frames: usize,
    pub language: Option<String>,
    pub emit_partials: bool,
}

impl Default for SpeechPipelineConfig {
    fn default() -> Self {
        Self {
            pre_roll_frames: 20,
            hangover_frames: 50,
            maximum_window_frames: MAX_PIPELINE_WINDOW_FRAMES,
            minimum_speech_frames: 20,
            language: Some("zh".to_owned()),
            emit_partials: true,
        }
    }
}

impl SpeechPipelineConfig {
    fn validate(&self) -> Result<(), String> {
        if self.maximum_window_frames == 0
            || self.maximum_window_frames > MAX_PIPELINE_WINDOW_FRAMES
        {
            return Err(format!(
                "speech pipeline window must contain 1 through {MAX_PIPELINE_WINDOW_FRAMES} frames"
            ));
        }
        if self.pre_roll_frames >= self.maximum_window_frames {
            return Err(
                "speech pipeline pre-roll must be shorter than its maximum window".to_owned(),
            );
        }
        if self.hangover_frames > self.maximum_window_frames {
            return Err("speech pipeline hangover exceeds its maximum window".to_owned());
        }
        if self.minimum_speech_frames == 0
            || self.minimum_speech_frames > self.maximum_window_frames
        {
            return Err(
                "speech pipeline minimum speech frames must fit inside its maximum window"
                    .to_owned(),
            );
        }
        if self
            .language
            .as_deref()
            .is_some_and(|language| language.trim().is_empty())
        {
            return Err("speech pipeline language must not be empty".to_owned());
        }
        Ok(())
    }
}

struct BufferedFrame {
    started_at: CapturePoint,
    frame: InferenceAudioWindow,
}

struct ActiveUtterance {
    started_at: CapturePoint,
    capture_start_ns: u64,
    capture_end_ns: u64,
    samples: Vec<f32>,
    consecutive_speech_frames: usize,
    longest_speech_run_frames: usize,
}

struct ResampledSamples {
    starting_source_offset: u64,
    samples: Vec<f32>,
}

/// Stateful mono 48 kHz to 16 kHz conversion owned by the dispatcher-side
/// segmenter. The FFT overlap adds half a target frame of delay; the first
/// half-frame is removed and the matching tail is drained at a known boundary.
struct Mono48KhzResampler {
    resampler: FftFixedInOut<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
    stream_start_source_offset: Option<u64>,
    next_output_source_offset: Option<u64>,
    processed_source_samples: usize,
    emitted_output_samples: usize,
    delay_samples_remaining: usize,
}

impl Mono48KhzResampler {
    fn new() -> Result<Self, String> {
        let resampler = FftFixedInOut::<f32>::new(
            48_000,
            INFERENCE_SAMPLE_RATE_HZ as usize,
            RESAMPLER_INPUT_SAMPLES,
            1,
        )
        .map_err(|error| format!("could not initialize local PCM resampler: {error}"))?;
        let delay_samples_remaining = resampler.output_delay();
        Ok(Self {
            resampler,
            input: Vec::with_capacity(RESAMPLER_INPUT_SAMPLES),
            output: vec![0.0; RESAMPLER_OUTPUT_SAMPLES],
            stream_start_source_offset: None,
            next_output_source_offset: None,
            processed_source_samples: 0,
            emitted_output_samples: 0,
            delay_samples_remaining,
        })
    }

    fn push_sample(
        &mut self,
        source_offset: u64,
        sample: f32,
    ) -> Result<Option<ResampledSamples>, String> {
        if self.stream_start_source_offset.is_none() {
            self.stream_start_source_offset = Some(source_offset);
            self.next_output_source_offset = Some(source_offset);
        }
        self.input.push(sample);
        if self.input.len() < RESAMPLER_INPUT_SAMPLES {
            return Ok(None);
        }

        self.processed_source_samples = self
            .processed_source_samples
            .checked_add(RESAMPLER_INPUT_SAMPLES)
            .ok_or_else(|| "resampler source sample count overflowed".to_owned())?;
        let written = self.process_current_input()?;
        let skipped = self.delay_samples_remaining.min(written);
        self.delay_samples_remaining -= skipped;
        self.take_output(skipped, written)
    }

    fn drain(&mut self) -> Result<Option<ResampledSamples>, String> {
        let target_output_samples = self.processed_source_samples / 3;
        let remaining = target_output_samples
            .checked_sub(self.emitted_output_samples)
            .ok_or_else(|| "resampler emitted more PCM than its source duration".to_owned())?;
        if remaining == 0 {
            return Ok(None);
        }

        self.input.clear();
        self.input.resize(RESAMPLER_INPUT_SAMPLES, 0.0);
        let written = self.process_current_input()?;
        self.take_output(0, remaining.min(written))
    }

    fn process_current_input(&mut self) -> Result<usize, String> {
        let (consumed, written) = self
            .resampler
            .process_into_buffer(
                std::slice::from_ref(&self.input),
                std::slice::from_mut(&mut self.output),
                None,
            )
            .map_err(|error| format!("local PCM resampling failed: {error}"))?;
        if consumed != RESAMPLER_INPUT_SAMPLES || written != RESAMPLER_OUTPUT_SAMPLES {
            return Err(format!(
                "local PCM resampler returned an unexpected {consumed}:{written} frame ratio"
            ));
        }
        self.input.clear();
        Ok(written)
    }

    fn take_output(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<Option<ResampledSamples>, String> {
        if start >= end {
            return Ok(None);
        }
        let starting_source_offset = self
            .next_output_source_offset
            .ok_or_else(|| "resampler output is missing its capture offset".to_owned())?;
        let samples = self.output[start..end].to_vec();
        let source_duration = u64::try_from(samples.len())
            .map_err(|_| "resampler output length cannot fit the capture clock".to_owned())?
            .checked_mul(3)
            .ok_or_else(|| "resampler output capture duration overflowed".to_owned())?;
        self.next_output_source_offset = Some(
            starting_source_offset
                .checked_add(source_duration)
                .ok_or_else(|| "resampler output capture offset overflowed".to_owned())?,
        );
        self.emitted_output_samples = self
            .emitted_output_samples
            .checked_add(samples.len())
            .ok_or_else(|| "resampler output sample count overflowed".to_owned())?;
        Ok(Some(ResampledSamples {
            starting_source_offset,
            samples,
        }))
    }

    fn reset(&mut self) {
        self.resampler.reset();
        self.input.clear();
        self.stream_start_source_offset = None;
        self.next_output_source_offset = None;
        self.processed_source_samples = 0;
        self.emitted_output_samples = 0;
        self.delay_samples_remaining = self.resampler.output_delay();
    }
}

impl ActiveUtterance {
    fn from_frame(frame: BufferedFrame, is_speech: bool) -> Self {
        let BufferedFrame { started_at, frame } = frame;
        Self {
            started_at,
            capture_start_ns: frame.capture_start_ns(),
            capture_end_ns: frame.capture_end_ns(),
            samples: frame.samples().to_vec(),
            consecutive_speech_frames: usize::from(is_speech),
            longest_speech_run_frames: usize::from(is_speech),
        }
    }

    fn append(&mut self, frame: &InferenceAudioWindow, is_speech: bool) -> Result<(), String> {
        if frame.capture_start_ns() != self.capture_end_ns {
            return Err("speech pipeline frames must remain contiguous".to_owned());
        }
        let new_length = self
            .samples
            .len()
            .checked_add(frame.samples().len())
            .ok_or_else(|| "speech pipeline sample count overflowed".to_owned())?;
        if new_length > MAX_INFERENCE_WINDOW_SAMPLES {
            return Err("speech pipeline window exceeds the inference limit".to_owned());
        }
        self.samples.extend_from_slice(frame.samples());
        self.capture_end_ns = frame.capture_end_ns();
        if is_speech {
            self.consecutive_speech_frames = self
                .consecutive_speech_frames
                .checked_add(1)
                .ok_or_else(|| "speech pipeline speech frame count overflowed".to_owned())?;
            self.longest_speech_run_frames = self
                .longest_speech_run_frames
                .max(self.consecutive_speech_frames);
        } else {
            self.consecutive_speech_frames = 0;
        }
        Ok(())
    }

    fn frame_count(&self) -> usize {
        self.samples.len() / PIPELINE_FRAME_SAMPLES
    }
}

/// Bounded native PCM segmentation with no ASR engine or persistent state.
///
/// Callers receive owned [`AsrRequest`] values and decide how to execute,
/// project, and audit them. This is the handoff point used by the native
/// dispatcher; it does not own a microphone callback, queue, database, or
/// Tauri handle.
pub struct SpeechSegmenter<D> {
    session_id: Uuid,
    clock: CaptureClock,
    detector: D,
    config: SpeechPipelineConfig,
    source_channels: Option<u16>,
    resampler: Option<Mono48KhzResampler>,
    expected_source_offset: Option<u64>,
    last_normalized_source_offset: Option<u64>,
    pending_frame_start_offset: Option<u64>,
    pending_samples: Vec<f32>,
    pre_roll: VecDeque<BufferedFrame>,
    active: Option<ActiveUtterance>,
    trailing_silence_frames: usize,
}

impl<D: SpeechActivityDetector> SpeechSegmenter<D> {
    pub fn new(
        session_id: Uuid,
        clock: CaptureClock,
        detector: D,
        config: SpeechPipelineConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        match clock.sample_rate() {
            INFERENCE_SAMPLE_RATE_HZ | 48_000 => {}
            sample_rate => {
                return Err(format!(
                    "speech pipeline supports only 16000 Hz or 48000 Hz input, received {sample_rate} Hz"
                ));
            }
        }

        let resampler = if clock.sample_rate() == 48_000 {
            Some(Mono48KhzResampler::new()?)
        } else {
            None
        };

        Ok(Self {
            session_id,
            clock,
            detector,
            config,
            source_channels: None,
            resampler,
            expected_source_offset: None,
            last_normalized_source_offset: None,
            pending_frame_start_offset: None,
            pending_samples: Vec::with_capacity(PIPELINE_FRAME_SAMPLES),
            pre_roll: VecDeque::new(),
            active: None,
            trailing_silence_frames: 0,
        })
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub fn push_packet(
        &mut self,
        packet: NativePcmPacket<'_>,
    ) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError> {
        let unfinished_started_at = self.unfinished_started_at();
        let fallback_packet_end_offset = self.fallback_packet_end_offset(packet);
        let source_frames = match packet.frame_count() {
            Ok(source_frames) => source_frames,
            Err(message) => {
                return Err(self.error_for_packet(
                    unfinished_started_at,
                    packet.starting_sample_offset,
                    fallback_packet_end_offset,
                    message,
                ));
            }
        };
        let source_frames_u64 = match u64::try_from(source_frames) {
            Ok(source_frames) => source_frames,
            Err(_) => {
                return Err(self.error_for_packet(
                    unfinished_started_at,
                    packet.starting_sample_offset,
                    packet.starting_sample_offset,
                    "native PCM frame count cannot be represented by the capture clock",
                ));
            }
        };
        let packet_end_offset = match packet.starting_sample_offset.checked_add(source_frames_u64) {
            Some(packet_end_offset) => packet_end_offset,
            None => {
                return Err(self.error_for_packet(
                    unfinished_started_at,
                    packet.starting_sample_offset,
                    packet.starting_sample_offset,
                    "native PCM packet end offset overflowed",
                ));
            }
        };
        if packet.sample_rate_hz != self.clock.sample_rate() {
            return Err(self.error_for_packet(
                unfinished_started_at,
                packet.starting_sample_offset,
                packet_end_offset,
                "native PCM packet sample rate does not match its capture clock",
            ));
        }
        if !matches!(packet.sample_rate_hz, INFERENCE_SAMPLE_RATE_HZ | 48_000) {
            return Err(self.error_for_packet(
                unfinished_started_at,
                packet.starting_sample_offset,
                packet_end_offset,
                format!(
                    "speech pipeline supports only 16000 Hz or 48000 Hz input, received {} Hz",
                    packet.sample_rate_hz
                ),
            ));
        }
        let expected_source_offset = self.expected_source_offset;
        if let Some(expected_source_offset) = expected_source_offset {
            if packet.starting_sample_offset < expected_source_offset {
                return Err(self.error_for_packet(
                    unfinished_started_at,
                    packet.starting_sample_offset,
                    packet_end_offset,
                    format!(
                        "native PCM packet source offset moved backwards or repeated: expected at least {expected_source_offset}, received {}",
                        packet.starting_sample_offset
                    ),
                ));
            }
        }

        match self.source_channels {
            Some(channels) if channels != packet.channels => {
                return Err(self.error_for_packet(
                    unfinished_started_at,
                    packet.starting_sample_offset,
                    packet_end_offset,
                    "native PCM channel count changed during a speech pipeline session",
                ));
            }
            Some(_) => {}
            None => self.source_channels = Some(packet.channels),
        }

        let mut events = Vec::new();
        if let Some(expected_source_offset) =
            expected_source_offset.filter(|expected| packet.starting_sample_offset > *expected)
        {
            if let Err(message) = self.drain_resampler(&mut events) {
                return Err(self.abort_packet(
                    unfinished_started_at,
                    packet.starting_sample_offset,
                    packet_end_offset,
                    message,
                ));
            }
            let request = match self.finalize_active() {
                Ok(request) => request,
                Err(message) => {
                    return Err(self.abort_packet(
                        unfinished_started_at,
                        packet.starting_sample_offset,
                        packet_end_offset,
                        message,
                    ));
                }
            };
            if let Some(request) = request {
                events.push(self.request_event(request));
            }
            self.clear_unfinished_audio();
            events.push(SpeechWindowEvent::Discontinuity {
                session_id: self.session_id,
                expected_source_offset,
                received_source_offset: packet.starting_sample_offset,
                at_capture_ns: self
                    .clock
                    .point_at_sample_offset(packet.starting_sample_offset)
                    .monotonic_ns,
            });
        }

        let channels = usize::from(packet.channels);
        for frame_index in 0..source_frames {
            let frame_offset = match u64::try_from(frame_index) {
                Ok(frame_offset) => frame_offset,
                Err(_) => {
                    return Err(self.abort_packet(
                        unfinished_started_at,
                        packet.starting_sample_offset,
                        packet_end_offset,
                        "native PCM frame index cannot be represented by the capture clock",
                    ));
                }
            };
            let source_offset = match packet.starting_sample_offset.checked_add(frame_offset) {
                Some(source_offset) => source_offset,
                None => {
                    return Err(self.abort_packet(
                        unfinished_started_at,
                        packet.starting_sample_offset,
                        packet_end_offset,
                        "native PCM source offset overflowed",
                    ));
                }
            };
            let start = match frame_index.checked_mul(channels) {
                Some(start) => start,
                None => {
                    return Err(self.abort_packet(
                        unfinished_started_at,
                        packet.starting_sample_offset,
                        packet_end_offset,
                        "native PCM frame offset overflowed",
                    ));
                }
            };
            let mono = packet.samples[start..start + channels]
                .iter()
                .map(|sample| f64::from(*sample))
                .sum::<f64>()
                / channels as f64;

            if packet.sample_rate_hz == INFERENCE_SAMPLE_RATE_HZ {
                if let Err(message) =
                    self.push_normalized_sample(source_offset, mono as f32, &mut events)
                {
                    return Err(self.abort_packet(
                        unfinished_started_at,
                        packet.starting_sample_offset,
                        packet_end_offset,
                        message,
                    ));
                }
                continue;
            }

            let normalized = {
                match self
                    .resampler
                    .as_mut()
                    .expect("48 kHz capture initializes its resampler")
                    .push_sample(source_offset, mono as f32)
                {
                    Ok(normalized) => normalized,
                    Err(message) => {
                        return Err(self.abort_packet(
                            unfinished_started_at,
                            packet.starting_sample_offset,
                            packet_end_offset,
                            message,
                        ));
                    }
                }
            };
            if let Some(normalized) = normalized {
                if let Err(message) = self.push_resampled_samples(normalized, &mut events) {
                    return Err(self.abort_packet(
                        unfinished_started_at,
                        packet.starting_sample_offset,
                        packet_end_offset,
                        message,
                    ));
                }
            }
        }

        self.expected_source_offset = Some(packet_end_offset);
        debug_assert!(
            events.len() <= MAX_PIPELINE_EVENTS_PER_PACKET,
            "a bounded native PCM packet cannot create unbounded pipeline events"
        );
        Ok(events)
    }

    /// Flushes a final active utterance at a known capture stop. A partial 10
    /// ms frame is discarded because it cannot carry a valid inference clock.
    pub fn finish(&mut self) -> Result<Vec<SpeechWindowEvent>, SpeechSegmenterError> {
        let unfinished_started_at = self.unfinished_started_at();
        let mut events = Vec::new();
        if let Err(message) = self.drain_resampler(&mut events) {
            return Err(self.abort_finish(unfinished_started_at, message));
        }
        let request = match self.finalize_active() {
            Ok(request) => request,
            Err(message) => return Err(self.abort_finish(unfinished_started_at, message)),
        };
        if let Some(request) = request {
            events.push(self.request_event(request));
        }
        self.clear_unfinished_audio();
        Ok(events)
    }

    fn fallback_packet_end_offset(&self, packet: NativePcmPacket<'_>) -> u64 {
        let frame_count = if packet.channels == 0 {
            0
        } else {
            packet.samples.len() / usize::from(packet.channels)
        };
        packet
            .starting_sample_offset
            .saturating_add(u64::try_from(frame_count).unwrap_or(u64::MAX))
    }

    fn unfinished_started_at(&self) -> Option<CapturePoint> {
        let mut started_at = self.active.as_ref().map(|active| active.started_at.clone());
        if let Some(frame) = self.pre_roll.front() {
            Self::keep_earliest_capture_point(&mut started_at, frame.started_at.clone());
        }
        if let Some(source_offset) = self.pending_frame_start_offset {
            Self::keep_earliest_capture_point(
                &mut started_at,
                self.clock.point_at_sample_offset(source_offset),
            );
        }
        started_at
    }

    fn keep_earliest_capture_point(current: &mut Option<CapturePoint>, candidate: CapturePoint) {
        if current
            .as_ref()
            .is_none_or(|current| candidate.monotonic_ns < current.monotonic_ns)
        {
            *current = Some(candidate);
        }
    }

    fn error_for_packet(
        &self,
        unfinished_started_at: Option<CapturePoint>,
        packet_start_offset: u64,
        packet_end_offset: u64,
        message: impl Into<String>,
    ) -> SpeechSegmenterError {
        let packet_started_at = self.clock.point_at_sample_offset(packet_start_offset);
        let started_at = unfinished_started_at
            .filter(|unfinished| unfinished.monotonic_ns < packet_started_at.monotonic_ns)
            .unwrap_or(packet_started_at);
        let packet_ended_at = self.clock.point_at_sample_offset(packet_end_offset);
        let ended_at = if packet_ended_at.monotonic_ns >= started_at.monotonic_ns {
            packet_ended_at
        } else {
            started_at.clone()
        };
        SpeechSegmenterError {
            session_id: self.session_id,
            started_at,
            ended_at,
            message: message.into(),
        }
    }

    fn abort_packet(
        &mut self,
        unfinished_started_at: Option<CapturePoint>,
        packet_start_offset: u64,
        packet_end_offset: u64,
        message: impl Into<String>,
    ) -> SpeechSegmenterError {
        let error = self.error_for_packet(
            unfinished_started_at,
            packet_start_offset,
            packet_end_offset,
            message,
        );
        self.clear_unfinished_audio();
        self.expected_source_offset = Some(packet_end_offset);
        error
    }

    fn abort_finish(
        &mut self,
        unfinished_started_at: Option<CapturePoint>,
        message: impl Into<String>,
    ) -> SpeechSegmenterError {
        let ending_source_offset = self.expected_source_offset.unwrap_or_default();
        let error = self.error_for_packet(
            unfinished_started_at,
            ending_source_offset,
            ending_source_offset,
            message,
        );
        self.clear_unfinished_audio();
        error
    }

    fn push_normalized_sample(
        &mut self,
        source_offset: u64,
        sample: f32,
        events: &mut Vec<SpeechWindowEvent>,
    ) -> Result<(), String> {
        let source_stride = self.source_stride();
        if let Some(previous) = self.last_normalized_source_offset {
            let expected = previous
                .checked_add(source_stride)
                .ok_or_else(|| "normalized PCM source offset overflowed".to_owned())?;
            if source_offset != expected {
                return Err("normalized PCM samples are not contiguous".to_owned());
            }
        }
        self.last_normalized_source_offset = Some(source_offset);

        if self.pending_samples.is_empty() {
            self.pending_frame_start_offset = Some(source_offset);
        }
        self.pending_samples.push(sample);
        if self.pending_samples.len() != PIPELINE_FRAME_SAMPLES {
            return Ok(());
        }

        let start_source_offset = self
            .pending_frame_start_offset
            .take()
            .expect("the first normalized sample records a frame start");
        let frame_source_duration = u64::try_from(PIPELINE_FRAME_SAMPLES)
            .map_err(|_| "pipeline frame length cannot be represented by the capture clock")?
            .checked_mul(source_stride)
            .ok_or_else(|| "pipeline frame source duration overflowed".to_owned())?;
        let end_source_offset = start_source_offset
            .checked_add(frame_source_duration)
            .ok_or_else(|| "pipeline frame end offset overflowed".to_owned())?;
        let samples = std::mem::replace(
            &mut self.pending_samples,
            Vec::with_capacity(PIPELINE_FRAME_SAMPLES),
        );
        let started_at = self.clock.point_at_sample_offset(start_source_offset);
        let ended_at = self.clock.point_at_sample_offset(end_source_offset);
        let frame = InferenceAudioWindow::new(
            self.session_id,
            started_at.monotonic_ns,
            ended_at.monotonic_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            samples,
        )?;
        if let Some(request) = self.process_frame(BufferedFrame { started_at, frame })? {
            events.push(self.request_event(request));
        }
        Ok(())
    }

    fn push_resampled_samples(
        &mut self,
        resampled: ResampledSamples,
        events: &mut Vec<SpeechWindowEvent>,
    ) -> Result<(), String> {
        for (index, sample) in resampled.samples.into_iter().enumerate() {
            let source_offset = u64::try_from(index)
                .map_err(|_| "resampler output index cannot fit the capture clock".to_owned())?
                .checked_mul(self.source_stride())
                .and_then(|offset| resampled.starting_source_offset.checked_add(offset))
                .ok_or_else(|| "resampler output source offset overflowed".to_owned())?;
            self.push_normalized_sample(source_offset, sample, events)?;
        }
        Ok(())
    }

    fn drain_resampler(&mut self, events: &mut Vec<SpeechWindowEvent>) -> Result<(), String> {
        let drained = self
            .resampler
            .as_mut()
            .map(Mono48KhzResampler::drain)
            .transpose()?;
        if let Some(Some(drained)) = drained {
            self.push_resampled_samples(drained, events)?;
        }
        Ok(())
    }

    fn process_frame(&mut self, frame: BufferedFrame) -> Result<Option<AsrRequest>, String> {
        let speech = self
            .detector
            .is_speech(&frame.frame)
            .map_err(|error| format!("local speech detector failed: {error}"))?;

        if self.active.is_some() && !speech && self.config.hangover_frames == 0 {
            let response = self.finalize_active()?;
            self.push_pre_roll(frame);
            return Ok(response);
        }

        if let Some(active) = self.active.as_mut() {
            active.append(&frame.frame, speech)?;
            self.trailing_silence_frames = if speech {
                0
            } else {
                self.trailing_silence_frames.saturating_add(1)
            };
            let should_finalize = active.frame_count() >= self.config.maximum_window_frames
                || (!speech && self.trailing_silence_frames >= self.config.hangover_frames);
            if should_finalize {
                return self.finalize_active();
            }
            return Ok(None);
        }

        if !speech {
            self.push_pre_roll(frame);
            return Ok(None);
        }

        let mut pre_roll = std::mem::take(&mut self.pre_roll);
        let active = match pre_roll.pop_front() {
            Some(first) => {
                let mut active = ActiveUtterance::from_frame(first, false);
                for previous in pre_roll {
                    active.append(&previous.frame, false)?;
                }
                active.append(&frame.frame, true)?;
                active
            }
            None => ActiveUtterance::from_frame(frame, true),
        };
        self.trailing_silence_frames = 0;
        let should_finalize = active.frame_count() >= self.config.maximum_window_frames;
        self.active = Some(active);
        if should_finalize {
            self.finalize_active()
        } else {
            Ok(None)
        }
    }

    fn push_pre_roll(&mut self, frame: BufferedFrame) {
        if self.config.pre_roll_frames == 0 {
            return;
        }
        while self.pre_roll.len() >= self.config.pre_roll_frames {
            self.pre_roll.pop_front();
        }
        self.pre_roll.push_back(frame);
    }

    fn finalize_active(&mut self) -> Result<Option<AsrRequest>, String> {
        let Some(active) = self.active.take() else {
            return Ok(None);
        };
        self.trailing_silence_frames = 0;
        if active.longest_speech_run_frames < self.config.minimum_speech_frames {
            return Ok(None);
        }
        let window = InferenceAudioWindow::new(
            self.session_id,
            active.capture_start_ns,
            active.capture_end_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            active.samples,
        )?;
        let request = AsrRequest::new(
            window,
            self.config.language.clone(),
            self.config.emit_partials,
        )?;
        Ok(Some(request))
    }

    fn clear_unfinished_audio(&mut self) {
        self.pending_samples.clear();
        self.pending_frame_start_offset = None;
        self.last_normalized_source_offset = None;
        self.pre_roll.clear();
        self.active = None;
        self.trailing_silence_frames = 0;
        self.detector.reset();
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
        }
    }

    fn request_event(&self, request: AsrRequest) -> SpeechWindowEvent {
        SpeechWindowEvent::Request {
            session_id: self.session_id,
            request,
        }
    }

    fn source_stride(&self) -> u64 {
        u64::from(self.clock.sample_rate() / INFERENCE_SAMPLE_RATE_HZ)
    }
}

impl SpeechSegmenter<EnergySpeechDetector> {
    pub fn with_energy_gate(
        session_id: Uuid,
        clock: CaptureClock,
        minimum_rms: f32,
        config: SpeechPipelineConfig,
    ) -> Result<Self, String> {
        Self::new(
            session_id,
            clock,
            EnergySpeechDetector::new(minimum_rms)?,
            config,
        )
    }
}

/// M2.1-compatible synchronous ASR wrapper around [`SpeechSegmenter`].
///
/// New native code should use [`SpeechSegmenter`] and enqueue its owned
/// [`AsrRequest`] values. Existing fixture consumers can retain this type and
/// receive the same validated response and discontinuity events as before.
pub struct SpeechPipeline<D, A> {
    segmenter: SpeechSegmenter<D>,
    asr: A,
}

impl<D: SpeechActivityDetector, A: AsrEngine> SpeechPipeline<D, A> {
    pub fn new(
        session_id: Uuid,
        clock: CaptureClock,
        detector: D,
        asr: A,
        config: SpeechPipelineConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            segmenter: SpeechSegmenter::new(session_id, clock, detector, config)?,
            asr,
        })
    }

    pub fn session_id(&self) -> Uuid {
        self.segmenter.session_id()
    }

    pub fn push_packet(
        &mut self,
        packet: NativePcmPacket<'_>,
    ) -> Result<Vec<SpeechPipelineEvent>, String> {
        let events = self
            .segmenter
            .push_packet(packet)
            .map_err(|error| error.to_string())?;
        self.transcribe_window_events(events)
    }

    /// Flushes a final active utterance at a known capture stop. A partial 10
    /// ms frame is discarded because it cannot carry a valid inference clock.
    pub fn finish(&mut self) -> Result<Vec<SpeechPipelineEvent>, String> {
        let events = self.segmenter.finish().map_err(|error| error.to_string())?;
        self.transcribe_window_events(events)
    }

    fn transcribe_window_events(
        &mut self,
        events: Vec<SpeechWindowEvent>,
    ) -> Result<Vec<SpeechPipelineEvent>, String> {
        let mut pipeline_events = Vec::with_capacity(events.len());
        for event in events {
            match event {
                SpeechWindowEvent::Request {
                    session_id,
                    request,
                } => {
                    let response = self
                        .asr
                        .transcribe(&request)
                        .map_err(|error| format!("local ASR engine failed: {error}"))?;
                    response
                        .validate_against(&request, self.asr.model_provenance())
                        .map_err(|error| {
                            format!("local ASR engine returned an invalid response: {error}")
                        })?;
                    pipeline_events.push(SpeechPipelineEvent::AsrResponse {
                        session_id,
                        response,
                    });
                }
                SpeechWindowEvent::Discontinuity {
                    session_id,
                    expected_source_offset,
                    received_source_offset,
                    at_capture_ns,
                } => pipeline_events.push(SpeechPipelineEvent::Discontinuity {
                    session_id,
                    expected_source_offset,
                    received_source_offset,
                    at_capture_ns,
                }),
            }
        }
        Ok(pipeline_events)
    }
}

impl<A: AsrEngine> SpeechPipeline<EnergySpeechDetector, A> {
    pub fn with_energy_gate(
        session_id: Uuid,
        clock: CaptureClock,
        minimum_rms: f32,
        asr: A,
        config: SpeechPipelineConfig,
    ) -> Result<Self, String> {
        Self::new(
            session_id,
            clock,
            EnergySpeechDetector::new(minimum_rms)?,
            asr,
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::CapturePoint;
    use crate::inference::{
        AsrResponseDisposition, InferenceEngine, ModelProvenance, TranscriptEmission,
        TranscriptEmissionKind,
    };
    use chrono::{DateTime, Duration, Utc};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    #[derive(Clone)]
    struct RecordingAsr {
        model: ModelProvenance,
        windows: Arc<Mutex<Vec<InferenceAudioWindow>>>,
    }

    impl RecordingAsr {
        fn new(windows: Arc<Mutex<Vec<InferenceAudioWindow>>>) -> Self {
            Self {
                model: model(),
                windows,
            }
        }
    }

    impl InferenceEngine for RecordingAsr {
        fn model_provenance(&self) -> &ModelProvenance {
            &self.model
        }
    }

    impl AsrEngine for RecordingAsr {
        fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError> {
            let window = request.audio.clone();
            self.windows.lock().unwrap().push(window.clone());
            AsrResponse::new(
                request,
                &self.model,
                vec![TranscriptEmission {
                    utterance_key: format!("test-{}", window.capture_start_ns()),
                    capture_start_ns: window.capture_start_ns(),
                    capture_end_ns: window.capture_end_ns(),
                    text: "本机测试".to_owned(),
                    kind: TranscriptEmissionKind::Final,
                    revision: 1,
                    word_timings: Vec::new(),
                    model_provenance: self.model.clone(),
                }],
            )
            .map_err(InferenceError::invalid)
        }
    }

    struct UnvalidatedResponseAsr {
        model: ModelProvenance,
    }

    impl InferenceEngine for UnvalidatedResponseAsr {
        fn model_provenance(&self) -> &ModelProvenance {
            &self.model
        }
    }

    impl AsrEngine for UnvalidatedResponseAsr {
        fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError> {
            Ok(AsrResponse {
                emissions: vec![TranscriptEmission {
                    utterance_key: "unvalidated-response".to_owned(),
                    capture_start_ns: request.audio.capture_start_ns(),
                    capture_end_ns: request.audio.capture_end_ns(),
                    text: "错误模型溯源".to_owned(),
                    kind: TranscriptEmissionKind::Final,
                    revision: 1,
                    word_timings: Vec::new(),
                    model_provenance: ModelProvenance::new(
                        "fixture",
                        "unexpected-pipeline-asr",
                        "v1",
                        "d".repeat(64),
                    )
                    .expect("fixture model provenance is valid"),
                }],
                disposition: AsrResponseDisposition::Transcript,
            })
        }
    }

    #[derive(Default)]
    struct AlwaysSpeech;

    impl SpeechActivityDetector for AlwaysSpeech {
        fn is_speech(&mut self, _frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
            Ok(true)
        }
    }

    struct ScriptedSpeechDetector {
        decisions: VecDeque<Result<bool, InferenceError>>,
    }

    impl ScriptedSpeechDetector {
        fn new(decisions: impl IntoIterator<Item = Result<bool, InferenceError>>) -> Self {
            Self {
                decisions: decisions.into_iter().collect(),
            }
        }
    }

    impl SpeechActivityDetector for ScriptedSpeechDetector {
        fn is_speech(&mut self, _frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
            self.decisions.pop_front().unwrap_or(Ok(true))
        }
    }

    struct CountingSpeechDetector {
        calls: Arc<AtomicUsize>,
    }

    impl SpeechActivityDetector for CountingSpeechDetector {
        fn is_speech(&mut self, _frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(true)
        }
    }

    struct ResetRequiredDetector {
        saw_frame: bool,
        resets: Arc<AtomicUsize>,
    }

    impl ResetRequiredDetector {
        fn new(resets: Arc<AtomicUsize>) -> Self {
            Self {
                saw_frame: false,
                resets,
            }
        }
    }

    impl SpeechActivityDetector for ResetRequiredDetector {
        fn is_speech(&mut self, _frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
            if self.saw_frame {
                return Err(InferenceError::failed(
                    "detector state crossed a capture discontinuity",
                ));
            }
            self.saw_frame = true;
            Ok(true)
        }

        fn reset(&mut self) {
            self.saw_frame = false;
            self.resets.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn model() -> ModelProvenance {
        ModelProvenance::new("fixture", "pipeline-asr", "v1", "c".repeat(64)).unwrap()
    }

    fn clock(sample_rate: u32) -> CaptureClock {
        CaptureClock::new(
            CapturePoint {
                monotonic_ns: 1_000,
                wall_clock: DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(1),
            },
            sample_rate,
        )
        .unwrap()
    }

    fn config() -> SpeechPipelineConfig {
        SpeechPipelineConfig {
            pre_roll_frames: 0,
            hangover_frames: 1,
            maximum_window_frames: 16,
            minimum_speech_frames: 1,
            language: Some("zh".to_owned()),
            emit_partials: true,
        }
    }

    fn packet(
        starting_sample_offset: u64,
        sample_rate_hz: u32,
        channels: u16,
        samples: Vec<f32>,
    ) -> CapturePacket {
        CapturePacket {
            starting_sample_offset,
            sample_rate: sample_rate_hz,
            channels,
            samples,
        }
    }

    fn response_count(events: &[SpeechPipelineEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, SpeechPipelineEvent::AsrResponse { .. }))
            .count()
    }

    fn requests(events: Vec<SpeechWindowEvent>) -> Vec<AsrRequest> {
        events
            .into_iter()
            .filter_map(|event| match event {
                SpeechWindowEvent::Request { request, .. } => Some(request),
                SpeechWindowEvent::Discontinuity { .. } => None,
            })
            .collect()
    }

    #[test]
    fn energy_gate_observes_quiet_frames_before_rejecting_them() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut detector = EnergyGatedSpeechDetector::new(
            CountingSpeechDetector {
                calls: Arc::clone(&calls),
            },
            0.05,
        )
        .unwrap();
        let quiet = InferenceAudioWindow::new(
            Uuid::nil(),
            0,
            10_000_000,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.01; PIPELINE_FRAME_SAMPLES],
        )
        .unwrap();
        let audible = InferenceAudioWindow::new(
            Uuid::nil(),
            10_000_000,
            20_000_000,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            vec![0.2; PIPELINE_FRAME_SAMPLES],
        )
        .unwrap();

        assert!(!detector.is_speech(&quiet).unwrap());
        assert!(detector.is_speech(&audible).unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn adaptive_energy_gate_learns_only_from_vad_negative_frames() {
        let mut detector =
            EnergyGatedSpeechDetector::adaptive(ScriptedSpeechDetector::new([Ok(true), Ok(true)]));
        let frame = |amplitude: f32| {
            InferenceAudioWindow::new(
                Uuid::nil(),
                0,
                10_000_000,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![amplitude; PIPELINE_FRAME_SAMPLES],
            )
            .unwrap()
        };

        assert!(detector.is_speech(&frame(0.2)).unwrap());
        assert!(
            detector.is_speech(&frame(0.01)).unwrap(),
            "VAD-positive audio must not raise the adaptive noise floor"
        );
    }

    #[test]
    fn adaptive_energy_gate_uses_noise_margin_and_clamped_thresholds() {
        let frame = |amplitude: f32| {
            InferenceAudioWindow::new(
                Uuid::nil(),
                0,
                10_000_000,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![amplitude; PIPELINE_FRAME_SAMPLES],
            )
            .unwrap()
        };
        let mut quiet_room =
            EnergyGatedSpeechDetector::adaptive(ScriptedSpeechDetector::new([Ok(false), Ok(true)]));
        assert!(!quiet_room.is_speech(&frame(0.001)).unwrap());
        assert!(quiet_room.is_speech(&frame(0.01)).unwrap());

        let mut noisy_room =
            EnergyGatedSpeechDetector::adaptive(ScriptedSpeechDetector::new([Ok(false), Ok(true)]));
        assert!(!noisy_room.is_speech(&frame(0.1)).unwrap());
        assert!(
            !noisy_room.is_speech(&frame(0.05)).unwrap(),
            "the upper -24 dBFS clamp must still reject sub-threshold VAD positives"
        );
    }

    #[test]
    fn adaptive_energy_gate_resets_its_noise_estimate_after_a_discontinuity() {
        let mut detector =
            EnergyGatedSpeechDetector::adaptive(ScriptedSpeechDetector::new([Ok(false), Ok(true)]));
        let frame = |amplitude: f32| {
            InferenceAudioWindow::new(
                Uuid::nil(),
                0,
                10_000_000,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![amplitude; PIPELINE_FRAME_SAMPLES],
            )
            .unwrap()
        };

        assert!(!detector.is_speech(&frame(0.1)).unwrap());
        detector.reset();
        assert!(detector.is_speech(&frame(0.01)).unwrap());
    }

    #[test]
    fn segmenter_discards_a_brief_vad_positive_run_at_finish() {
        let decisions = (0..19).map(|_| Ok(true));
        let mut segmenter = SpeechSegmenter::new(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            ScriptedSpeechDetector::new(decisions),
            SpeechPipelineConfig {
                minimum_speech_frames: 20,
                maximum_window_frames: 24,
                ..config()
            },
        )
        .unwrap();

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&packet(
                0,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![0.2; PIPELINE_FRAME_SAMPLES * 19],
            )))
            .unwrap()
            .is_empty());
        assert!(segmenter.finish().unwrap().is_empty());
    }

    #[test]
    fn segmenter_does_not_accumulate_intermittent_vad_noise_into_speech() {
        let decisions = (0..20).flat_map(|_| [Ok(true), Ok(false)]);
        let mut segmenter = SpeechSegmenter::new(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            ScriptedSpeechDetector::new(decisions),
            SpeechPipelineConfig {
                hangover_frames: 50,
                minimum_speech_frames: 20,
                maximum_window_frames: 80,
                ..config()
            },
        )
        .unwrap();

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&packet(
                0,
                INFERENCE_SAMPLE_RATE_HZ,
                INFERENCE_CHANNELS,
                vec![0.2; PIPELINE_FRAME_SAMPLES * 40],
            )))
            .unwrap()
            .is_empty());
        assert!(segmenter.finish().unwrap().is_empty());
    }

    #[test]
    fn segmenter_admits_qualified_speech_without_counting_pre_roll_or_hangover() {
        let decisions = std::iter::repeat_n(Ok(true), 20).chain(std::iter::once(Ok(false)));
        let mut segmenter = SpeechSegmenter::new(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            ScriptedSpeechDetector::new(std::iter::repeat_n(Ok(false), 2).chain(decisions)),
            SpeechPipelineConfig {
                pre_roll_frames: 2,
                hangover_frames: 1,
                minimum_speech_frames: 20,
                maximum_window_frames: 24,
                ..config()
            },
        )
        .unwrap();
        let mut samples = vec![0.0; PIPELINE_FRAME_SAMPLES * 2];
        samples.extend(std::iter::repeat_n(0.2, PIPELINE_FRAME_SAMPLES * 20));
        samples.extend(std::iter::repeat_n(0.0, PIPELINE_FRAME_SAMPLES));

        let requests = requests(
            segmenter
                .push_packet(NativePcmPacket::from(&packet(
                    0,
                    INFERENCE_SAMPLE_RATE_HZ,
                    INFERENCE_CHANNELS,
                    samples,
                )))
                .unwrap(),
        );

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].audio.frame_count(), PIPELINE_FRAME_SAMPLES * 23);
    }

    #[test]
    fn segmenter_reports_the_lost_range_and_recovers_after_a_mid_packet_vad_failure() {
        let session_id = Uuid::new_v4();
        let capture_clock = clock(INFERENCE_SAMPLE_RATE_HZ);
        let mut segmenter = SpeechSegmenter::new(
            session_id,
            capture_clock.clone(),
            ScriptedSpeechDetector::new([
                Ok(true),
                Ok(true),
                Err(InferenceError::failed("fixture VAD failure")),
                Ok(true),
            ]),
            SpeechPipelineConfig {
                maximum_window_frames: 2,
                ..config()
            },
        )
        .unwrap();

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&packet(
                0,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                vec![0.2; PIPELINE_FRAME_SAMPLES],
            )))
            .unwrap()
            .is_empty());
        let error = segmenter
            .push_packet(NativePcmPacket::from(&packet(
                PIPELINE_FRAME_SAMPLES as u64,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                vec![0.2; PIPELINE_FRAME_SAMPLES * 2],
            )))
            .unwrap_err();

        assert_eq!(error.session_id, session_id);
        assert_eq!(
            error.started_at,
            capture_clock.point_at_sample_offset(0),
            "the error range includes the unfinished request before this packet"
        );
        assert_eq!(
            error.ended_at,
            capture_clock.point_at_sample_offset((PIPELINE_FRAME_SAMPLES * 3) as u64),
            "the error range includes the complete packet, including frames after the failure"
        );
        assert_eq!(
            error.message,
            "local speech detector failed: fixture VAD failure"
        );
        assert_eq!(error.to_string(), error.message);

        let resumed = segmenter
            .push_packet(NativePcmPacket::from(&packet(
                (PIPELINE_FRAME_SAMPLES * 3) as u64,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                vec![0.2; PIPELINE_FRAME_SAMPLES],
            )))
            .unwrap();
        assert!(
            resumed.is_empty(),
            "the packet immediately after a failure must not see a synthetic discontinuity"
        );
        let requests = requests(segmenter.finish().unwrap());

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].audio.capture_start_ns(), 30_001_000);
        assert_eq!(requests[0].audio.capture_end_ns(), 40_001_000);
    }

    #[test]
    fn segmenter_yields_an_owned_request_at_finish_without_an_asr_engine() {
        let session_id = Uuid::new_v4();
        let mut segmenter = SpeechSegmenter::new(
            session_id,
            clock(INFERENCE_SAMPLE_RATE_HZ),
            AlwaysSpeech,
            config(),
        )
        .unwrap();
        let samples = (0..PIPELINE_FRAME_SAMPLES)
            .map(|index| index as f32 / PIPELINE_FRAME_SAMPLES as f32)
            .collect::<Vec<_>>();

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&packet(
                0,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                samples.clone(),
            )))
            .unwrap()
            .is_empty());
        let mut events = segmenter.finish().unwrap();

        assert_eq!(events.len(), 1);
        let SpeechWindowEvent::Request {
            session_id: event_session_id,
            request,
        } = events.pop().unwrap()
        else {
            panic!("segmenter finish must yield an ASR request");
        };
        assert_eq!(event_session_id, session_id);
        assert_eq!(request.audio.session_id(), session_id);
        assert_eq!(request.audio.samples(), samples);
        assert_eq!(request.audio.sample_rate_hz(), INFERENCE_SAMPLE_RATE_HZ);
        assert_eq!(request.audio.channels(), INFERENCE_CHANNELS);
        assert_eq!(request.audio.capture_start_ns(), 1_000);
        assert_eq!(request.audio.capture_end_ns(), 10_001_000);
        assert_eq!(request.language.as_deref(), Some("zh"));
        assert!(request.emit_partials);
    }

    #[test]
    fn segmenter_downmixes_and_resamples_48khz_pcm_across_packet_boundaries() {
        let session_id = Uuid::new_v4();
        let mut segmenter =
            SpeechSegmenter::new(session_id, clock(48_000), AlwaysSpeech, config()).unwrap();
        let stereo_tone = |start: usize, length: usize| {
            (start..start + length)
                .flat_map(|frame| {
                    let phase = 2.0 * std::f32::consts::PI * 1_000.0 * frame as f32 / 48_000.0;
                    let sample = phase.sin() * 0.8;
                    [sample, sample]
                })
                .collect::<Vec<_>>()
        };
        let first = packet(0, 48_000, 2, stereo_tone(0, 1_997));
        let second = packet(1_997, 48_000, 2, stereo_tone(1_997, 2_803));

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&first))
            .unwrap()
            .is_empty());
        assert!(segmenter
            .push_packet(NativePcmPacket::from(&second))
            .unwrap()
            .is_empty());
        let requests = requests(segmenter.finish().unwrap());

        assert_eq!(requests.len(), 1);
        let audio = &requests[0].audio;
        assert_eq!(audio.session_id(), session_id);
        assert_eq!(audio.frame_count(), PIPELINE_FRAME_SAMPLES * 10);
        let retained_rms = root_mean_square(&audio.samples()[PIPELINE_FRAME_SAMPLES..]).unwrap();
        assert!(
            (0.45..=0.65).contains(&retained_rms),
            "speech-band tone RMS must survive resampling, observed {retained_rms}"
        );
        assert_eq!(audio.capture_start_ns(), 1_000);
        assert_eq!(audio.capture_end_ns(), 100_001_000);
    }

    #[test]
    fn segmenter_attenuates_above_nyquist_audio_when_resampling_48khz_pcm() {
        let mut segmenter =
            SpeechSegmenter::new(Uuid::new_v4(), clock(48_000), AlwaysSpeech, config()).unwrap();
        let samples = (0..4_800)
            .map(|frame| {
                let phase = 2.0 * std::f32::consts::PI * 12_000.0 * frame as f32 / 48_000.0;
                phase.sin() * 0.8
            })
            .collect::<Vec<_>>();

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&packet(0, 48_000, 1, samples)))
            .unwrap()
            .is_empty());
        let requests = requests(segmenter.finish().unwrap());

        assert_eq!(requests.len(), 1);
        let audio = &requests[0].audio;
        assert_eq!(audio.frame_count(), PIPELINE_FRAME_SAMPLES * 10);
        let aliased_rms = root_mean_square(&audio.samples()[PIPELINE_FRAME_SAMPLES..]).unwrap();
        assert!(
            aliased_rms < 0.03,
            "out-of-band tone must be filtered before downsampling, observed RMS {aliased_rms}"
        );
        assert_eq!(audio.capture_start_ns(), 1_000);
        assert_eq!(audio.capture_end_ns(), 100_001_000);
    }

    #[test]
    fn segmenter_adds_bounded_pre_roll_and_hangover_to_its_request() {
        let mut segmenter = SpeechSegmenter::new(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            EnergySpeechDetector::new(0.05).unwrap(),
            SpeechPipelineConfig {
                pre_roll_frames: 2,
                hangover_frames: 2,
                maximum_window_frames: 16,
                ..config()
            },
        )
        .unwrap();
        let mut samples = Vec::new();
        for amplitude in [0.0_f32, 0.0, 0.2, 0.0, 0.0] {
            samples.extend(std::iter::repeat_n(amplitude, PIPELINE_FRAME_SAMPLES));
        }

        let requests = requests(
            segmenter
                .push_packet(NativePcmPacket::from(&packet(0, 16_000, 1, samples)))
                .unwrap(),
        );

        assert_eq!(requests.len(), 1);
        let audio = &requests[0].audio;
        assert_eq!(audio.frame_count(), PIPELINE_FRAME_SAMPLES * 5);
        assert_eq!(audio.capture_start_ns(), 1_000);
        assert_eq!(audio.capture_end_ns(), 50_001_000);
    }

    #[test]
    fn segmenter_seals_a_request_before_emitting_a_discontinuity() {
        let session_id = Uuid::new_v4();
        let mut segmenter =
            SpeechSegmenter::new(session_id, clock(16_000), AlwaysSpeech, config()).unwrap();
        let first = packet(0, 16_000, 1, vec![0.2; PIPELINE_FRAME_SAMPLES]);
        let second = packet(320, 16_000, 1, vec![0.2; PIPELINE_FRAME_SAMPLES]);

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&first))
            .unwrap()
            .is_empty());
        let mut events = segmenter
            .push_packet(NativePcmPacket::from(&second))
            .unwrap();

        assert_eq!(events.len(), 2);
        let SpeechWindowEvent::Request { request, .. } = events.remove(0) else {
            panic!("a discontinuity must first seal the preceding speech request");
        };
        assert_eq!(request.audio.capture_start_ns(), 1_000);
        assert_eq!(request.audio.capture_end_ns(), 10_001_000);
        assert!(matches!(
            events.as_slice(),
            [SpeechWindowEvent::Discontinuity {
                session_id: event_session_id,
                expected_source_offset: 160,
                received_source_offset: 320,
                at_capture_ns: 20_001_000,
            }] if *event_session_id == session_id
        ));

        let requests = requests(segmenter.finish().unwrap());
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].audio.capture_start_ns(), 20_001_000);
        assert_eq!(requests[0].audio.capture_end_ns(), 30_001_000);
    }

    #[test]
    fn resets_stateful_vad_before_processing_audio_after_a_discontinuity() {
        let resets = Arc::new(AtomicUsize::new(0));
        let mut segmenter = SpeechSegmenter::new(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            ResetRequiredDetector::new(Arc::clone(&resets)),
            config(),
        )
        .unwrap();

        assert!(segmenter
            .push_packet(NativePcmPacket::from(&packet(
                0,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                vec![0.2; PIPELINE_FRAME_SAMPLES],
            )))
            .unwrap()
            .is_empty());
        let events = segmenter
            .push_packet(NativePcmPacket::from(&packet(
                (PIPELINE_FRAME_SAMPLES * 2) as u64,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                vec![0.2; PIPELINE_FRAME_SAMPLES],
            )))
            .unwrap();

        assert!(matches!(
            events.first(),
            Some(SpeechWindowEvent::Request { .. })
        ));
        assert!(matches!(
            events.get(1),
            Some(SpeechWindowEvent::Discontinuity { .. })
        ));
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn segmenter_forces_a_request_at_the_thirty_second_limit() {
        let mut segmenter = SpeechSegmenter::new(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            AlwaysSpeech,
            SpeechPipelineConfig {
                maximum_window_frames: MAX_PIPELINE_WINDOW_FRAMES,
                ..config()
            },
        )
        .unwrap();
        let mut emitted_requests = Vec::new();
        let frames_per_packet = 8_000_u64;
        let packets = u64::try_from(MAX_INFERENCE_WINDOW_SAMPLES)
            .unwrap()
            .checked_div(frames_per_packet)
            .unwrap();

        for packet_index in 0..packets {
            let events = segmenter
                .push_packet(NativePcmPacket::from(&packet(
                    packet_index * frames_per_packet,
                    INFERENCE_SAMPLE_RATE_HZ,
                    1,
                    vec![0.2; frames_per_packet as usize],
                )))
                .unwrap();
            emitted_requests.extend(requests(events));
        }

        assert_eq!(emitted_requests.len(), 1);
        let audio = &emitted_requests[0].audio;
        assert_eq!(audio.samples().len(), MAX_INFERENCE_WINDOW_SAMPLES);
        assert_eq!(audio.duration_ns(), 30_000_000_000);
        assert!(segmenter.finish().unwrap().is_empty());
    }

    #[test]
    fn converts_identity_pcm_to_a_native_inference_window() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let session_id = Uuid::new_v4();
        let mut pipeline = SpeechPipeline::new(
            session_id,
            clock(INFERENCE_SAMPLE_RATE_HZ),
            AlwaysSpeech,
            RecordingAsr::new(Arc::clone(&windows)),
            config(),
        )
        .unwrap();

        let samples = (0..PIPELINE_FRAME_SAMPLES)
            .map(|index| index as f32 / PIPELINE_FRAME_SAMPLES as f32)
            .collect::<Vec<_>>();
        assert!(pipeline
            .push_packet(NativePcmPacket::from(&packet(
                0,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                samples.clone(),
            )))
            .unwrap()
            .is_empty());
        let events = pipeline.finish().unwrap();

        assert_eq!(response_count(&events), 1);
        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].session_id(), session_id);
        assert_eq!(windows[0].samples(), samples);
        assert_eq!(windows[0].capture_start_ns(), 1_000);
        assert_eq!(windows[0].capture_end_ns(), 10_001_000);
    }

    #[test]
    fn downmixes_and_resamples_48khz_pcm_across_packet_boundaries() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let session_id = Uuid::new_v4();
        let mut pipeline = SpeechPipeline::new(
            session_id,
            clock(48_000),
            AlwaysSpeech,
            RecordingAsr::new(Arc::clone(&windows)),
            config(),
        )
        .unwrap();
        let stereo_tone = |start: usize, length: usize| {
            (start..start + length)
                .flat_map(|frame| {
                    let phase = 2.0 * std::f32::consts::PI * 1_000.0 * frame as f32 / 48_000.0;
                    let sample = phase.sin() * 0.8;
                    [sample, sample]
                })
                .collect::<Vec<_>>()
        };
        let first = packet(0, 48_000, 2, stereo_tone(0, 200));
        let second = packet(200, 48_000, 2, stereo_tone(200, 280));

        assert!(pipeline
            .push_packet(NativePcmPacket::from(&first))
            .unwrap()
            .is_empty());
        assert!(pipeline
            .push_packet(NativePcmPacket::from(&second))
            .unwrap()
            .is_empty());
        pipeline.finish().unwrap();

        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].frame_count(), PIPELINE_FRAME_SAMPLES);
        let retained_rms = root_mean_square(windows[0].samples()).unwrap();
        assert!((0.45..=0.65).contains(&retained_rms));
        assert_eq!(windows[0].capture_start_ns(), 1_000);
        assert_eq!(windows[0].capture_end_ns(), 10_001_000);
    }

    #[test]
    fn resamples_a_48khz_packet_from_its_actual_capture_offset() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let capture_clock = clock(48_000);
        let mut pipeline = SpeechPipeline::new(
            Uuid::new_v4(),
            capture_clock.clone(),
            AlwaysSpeech,
            RecordingAsr::new(Arc::clone(&windows)),
            config(),
        )
        .unwrap();
        let input = packet(1, 48_000, 1, vec![0.25; RESAMPLER_INPUT_SAMPLES]);

        assert!(pipeline
            .push_packet(NativePcmPacket::from(&input))
            .unwrap()
            .is_empty());
        assert_eq!(response_count(&pipeline.finish().unwrap()), 1);

        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].frame_count(), PIPELINE_FRAME_SAMPLES);
        assert!(windows[0].samples().iter().all(|sample| sample.is_finite()));
        assert_eq!(
            windows[0].capture_start_ns(),
            capture_clock.point_at_sample_offset(1).monotonic_ns
        );
        assert_eq!(
            windows[0].capture_end_ns(),
            capture_clock.point_at_sample_offset(481).monotonic_ns
        );
    }

    #[test]
    fn adds_bounded_pre_roll_and_hangover_to_a_speech_window() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SpeechPipeline::with_energy_gate(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            0.05,
            RecordingAsr::new(Arc::clone(&windows)),
            SpeechPipelineConfig {
                pre_roll_frames: 2,
                hangover_frames: 2,
                maximum_window_frames: 16,
                ..config()
            },
        )
        .unwrap();
        let mut samples = Vec::new();
        for amplitude in [0.0_f32, 0.0, 0.2, 0.0, 0.0] {
            samples.extend(std::iter::repeat_n(amplitude, PIPELINE_FRAME_SAMPLES));
        }

        let events = pipeline
            .push_packet(NativePcmPacket::from(&packet(0, 16_000, 1, samples)))
            .unwrap();

        assert_eq!(response_count(&events), 1);
        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].frame_count(), PIPELINE_FRAME_SAMPLES * 5);
        assert_eq!(windows[0].capture_start_ns(), 1_000);
        assert_eq!(windows[0].capture_end_ns(), 50_001_000);
    }

    #[test]
    fn zero_hangover_finalizes_before_the_first_silent_frame() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SpeechPipeline::with_energy_gate(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            0.05,
            RecordingAsr::new(Arc::clone(&windows)),
            SpeechPipelineConfig {
                hangover_frames: 0,
                ..config()
            },
        )
        .unwrap();
        let mut samples = vec![0.2; PIPELINE_FRAME_SAMPLES];
        samples.extend(std::iter::repeat_n(0.0, PIPELINE_FRAME_SAMPLES));

        let events = pipeline
            .push_packet(NativePcmPacket::from(&packet(0, 16_000, 1, samples)))
            .unwrap();

        assert_eq!(response_count(&events), 1);
        assert!(pipeline.finish().unwrap().is_empty());
        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].frame_count(), PIPELINE_FRAME_SAMPLES);
        assert_eq!(windows[0].capture_start_ns(), 1_000);
        assert_eq!(windows[0].capture_end_ns(), 10_001_000);
    }

    #[test]
    fn rejects_an_asr_response_that_bypasses_request_and_model_validation() {
        let mut pipeline = SpeechPipeline::new(
            Uuid::new_v4(),
            clock(INFERENCE_SAMPLE_RATE_HZ),
            AlwaysSpeech,
            UnvalidatedResponseAsr { model: model() },
            config(),
        )
        .unwrap();

        assert!(pipeline
            .push_packet(NativePcmPacket::from(&packet(
                0,
                INFERENCE_SAMPLE_RATE_HZ,
                1,
                vec![0.2; PIPELINE_FRAME_SAMPLES],
            )))
            .unwrap()
            .is_empty());
        let error = pipeline.finish().unwrap_err();

        assert!(error.contains("model provenance"));
    }

    #[test]
    fn emits_a_discontinuity_without_joining_speech_across_the_gap() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let session_id = Uuid::new_v4();
        let mut pipeline = SpeechPipeline::new(
            session_id,
            clock(16_000),
            AlwaysSpeech,
            RecordingAsr::new(Arc::clone(&windows)),
            config(),
        )
        .unwrap();
        let first = packet(0, 16_000, 1, vec![0.2; PIPELINE_FRAME_SAMPLES]);
        let second = packet(320, 16_000, 1, vec![0.2; PIPELINE_FRAME_SAMPLES]);

        assert!(pipeline
            .push_packet(NativePcmPacket::from(&first))
            .unwrap()
            .is_empty());
        let events = pipeline
            .push_packet(NativePcmPacket::from(&second))
            .unwrap();

        assert_eq!(response_count(&events), 1);
        assert!(matches!(
            events.as_slice(),
            [
                SpeechPipelineEvent::AsrResponse { .. },
                SpeechPipelineEvent::Discontinuity {
                    session_id: event_session_id,
                    expected_source_offset: 160,
                    received_source_offset: 320,
                    at_capture_ns: 20_001_000,
                },
            ] if *event_session_id == session_id
        ));
        pipeline.finish().unwrap();
        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].frame_count(), PIPELINE_FRAME_SAMPLES);
        assert_eq!(windows[1].frame_count(), PIPELINE_FRAME_SAMPLES);
    }

    #[test]
    fn rejects_backwards_or_repeated_source_packet_offsets_without_resetting_audio() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SpeechPipeline::new(
            Uuid::new_v4(),
            clock(16_000),
            AlwaysSpeech,
            RecordingAsr::new(Arc::clone(&windows)),
            config(),
        )
        .unwrap();
        let first = packet(0, 16_000, 1, vec![0.2; PIPELINE_FRAME_SAMPLES]);

        assert!(pipeline
            .push_packet(NativePcmPacket::from(&first))
            .unwrap()
            .is_empty());
        for invalid_offset in [80, 0] {
            let error = pipeline
                .push_packet(NativePcmPacket::from(&packet(
                    invalid_offset,
                    16_000,
                    1,
                    vec![0.2; PIPELINE_FRAME_SAMPLES],
                )))
                .unwrap_err();
            assert!(error.contains("backwards or repeated"));
        }
        assert!(pipeline
            .push_packet(NativePcmPacket::from(&packet(
                PIPELINE_FRAME_SAMPLES as u64,
                16_000,
                1,
                vec![0.2; PIPELINE_FRAME_SAMPLES],
            )))
            .unwrap()
            .is_empty());
        assert_eq!(response_count(&pipeline.finish().unwrap()), 1);

        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].frame_count(), PIPELINE_FRAME_SAMPLES * 2);
    }

    #[test]
    fn forces_bounded_windows_before_the_inference_limit() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SpeechPipeline::new(
            Uuid::new_v4(),
            clock(16_000),
            AlwaysSpeech,
            RecordingAsr::new(Arc::clone(&windows)),
            SpeechPipelineConfig {
                maximum_window_frames: 2,
                ..config()
            },
        )
        .unwrap();
        let input = packet(0, 16_000, 1, vec![0.2; PIPELINE_FRAME_SAMPLES * 3]);

        let events = pipeline.push_packet(NativePcmPacket::from(&input)).unwrap();
        assert_eq!(response_count(&events), 1);
        pipeline.finish().unwrap();

        let windows = windows.lock().unwrap();
        assert_eq!(
            windows
                .iter()
                .map(InferenceAudioWindow::frame_count)
                .collect::<Vec<_>>(),
            vec![PIPELINE_FRAME_SAMPLES * 2, PIPELINE_FRAME_SAMPLES]
        );
    }

    #[test]
    fn rejects_unsupported_or_malformed_native_pcm() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SpeechPipeline::new(
            Uuid::new_v4(),
            clock(16_000),
            AlwaysSpeech,
            RecordingAsr::new(windows),
            config(),
        )
        .unwrap();

        let unsupported = packet(0, 44_100, 1, vec![0.0; PIPELINE_FRAME_SAMPLES]);
        assert!(pipeline
            .push_packet(NativePcmPacket::from(&unsupported))
            .unwrap_err()
            .contains("sample rate"));
        let malformed = packet(0, 16_000, 2, vec![0.0; PIPELINE_FRAME_SAMPLES - 1]);
        assert!(pipeline
            .push_packet(NativePcmPacket::from(&malformed))
            .unwrap_err()
            .contains("align"));
        let non_finite = packet(0, 16_000, 1, vec![f32::NAN; PIPELINE_FRAME_SAMPLES]);
        assert!(pipeline
            .push_packet(NativePcmPacket::from(&non_finite))
            .unwrap_err()
            .contains("finite"));
        let oversized = packet(0, 16_000, 1, vec![0.0; MAX_CAPTURE_SAMPLES_PER_PACKET + 1]);
        assert!(pipeline
            .push_packet(NativePcmPacket::from(&oversized))
            .unwrap_err()
            .contains("capture bound"));
    }
}
