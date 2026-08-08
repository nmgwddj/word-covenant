//! Native-only packet normalization and speech-window assembly.
//!
//! This is intentionally a synchronous, bounded contract for fixture input.
//! It does not own a microphone callback, a queue, a database, or a Tauri
//! handle. The CPAL bridge will use it later from its single dispatcher.

use super::{
    AsrEngine, AsrRequest, AsrResponse, InferenceAudioWindow, InferenceError, VadEngine,
    VadRequest, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ, MAX_INFERENCE_WINDOW_SAMPLES,
};
use crate::audio::{CaptureClock, CapturePacket, MAX_CAPTURE_SAMPLES_PER_PACKET};
use std::collections::VecDeque;
use uuid::Uuid;

pub const PIPELINE_FRAME_SAMPLES: usize = 160;
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

/// A frame-level activity decision used to assemble speech windows.
///
/// The default test fixture is an energy gate. A real VAD model can be
/// attached through [`VadSpeechDetector`] without changing the rest of the
/// pipeline.
pub trait SpeechActivityDetector: Send {
    fn is_speech(&mut self, frame: &InferenceAudioWindow) -> Result<bool, InferenceError>;
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
        let samples = frame.samples();
        let energy = samples
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>();
        let rms = (energy / samples.len() as f64).sqrt() as f32;
        Ok(rms >= self.minimum_rms)
    }
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
    pub language: Option<String>,
    pub emit_partials: bool,
}

impl Default for SpeechPipelineConfig {
    fn default() -> Self {
        Self {
            pre_roll_frames: 20,
            hangover_frames: 50,
            maximum_window_frames: MAX_PIPELINE_WINDOW_FRAMES,
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

struct ActiveUtterance {
    capture_start_ns: u64,
    capture_end_ns: u64,
    samples: Vec<f32>,
}

impl ActiveUtterance {
    fn from_frame(frame: &InferenceAudioWindow) -> Self {
        Self {
            capture_start_ns: frame.capture_start_ns(),
            capture_end_ns: frame.capture_end_ns(),
            samples: frame.samples().to_vec(),
        }
    }

    fn append(&mut self, frame: &InferenceAudioWindow) -> Result<(), String> {
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
        Ok(())
    }

    fn frame_count(&self) -> usize {
        self.samples.len() / PIPELINE_FRAME_SAMPLES
    }
}

/// Bounded native PCM pipeline. It has no direct access to persistent state;
/// callers decide how returned local ASR responses are projected and audited.
pub struct SpeechPipeline<D, A> {
    session_id: Uuid,
    clock: CaptureClock,
    detector: D,
    asr: A,
    config: SpeechPipelineConfig,
    source_channels: Option<u16>,
    expected_source_offset: Option<u64>,
    last_normalized_source_offset: Option<u64>,
    pending_frame_start_offset: Option<u64>,
    pending_samples: Vec<f32>,
    pre_roll: VecDeque<InferenceAudioWindow>,
    active: Option<ActiveUtterance>,
    trailing_silence_frames: usize,
}

impl<D: SpeechActivityDetector, A: AsrEngine> SpeechPipeline<D, A> {
    pub fn new(
        session_id: Uuid,
        clock: CaptureClock,
        detector: D,
        asr: A,
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

        Ok(Self {
            session_id,
            clock,
            detector,
            asr,
            config,
            source_channels: None,
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
    ) -> Result<Vec<SpeechPipelineEvent>, String> {
        let source_frames = packet.frame_count()?;
        if packet.sample_rate_hz != self.clock.sample_rate() {
            return Err(
                "native PCM packet sample rate does not match its capture clock".to_owned(),
            );
        }
        if !matches!(packet.sample_rate_hz, INFERENCE_SAMPLE_RATE_HZ | 48_000) {
            return Err(format!(
                "speech pipeline supports only 16000 Hz or 48000 Hz input, received {} Hz",
                packet.sample_rate_hz
            ));
        }
        let expected_source_offset = self.expected_source_offset;
        if let Some(expected_source_offset) = expected_source_offset {
            if packet.starting_sample_offset < expected_source_offset {
                return Err(format!(
                    "native PCM packet source offset moved backwards or repeated: expected at least {expected_source_offset}, received {}",
                    packet.starting_sample_offset
                ));
            }
        }

        match self.source_channels {
            Some(channels) if channels != packet.channels => {
                return Err(
                    "native PCM channel count changed during a speech pipeline session".to_owned(),
                );
            }
            Some(_) => {}
            None => self.source_channels = Some(packet.channels),
        }

        let mut events = Vec::new();
        if let Some(expected_source_offset) =
            expected_source_offset.filter(|expected| packet.starting_sample_offset > *expected)
        {
            if let Some(response) = self.finalize_active()? {
                events.push(self.response_event(response));
            }
            self.clear_unfinished_audio();
            events.push(SpeechPipelineEvent::Discontinuity {
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
            let source_offset = packet
                .starting_sample_offset
                .checked_add(u64::try_from(frame_index).map_err(|_| {
                    "native PCM frame index cannot be represented by the capture clock"
                })?)
                .ok_or_else(|| "native PCM source offset overflowed".to_owned())?;
            let start = frame_index
                .checked_mul(channels)
                .ok_or_else(|| "native PCM frame offset overflowed".to_owned())?;
            let mono = packet.samples[start..start + channels]
                .iter()
                .map(|sample| f64::from(*sample))
                .sum::<f64>()
                / channels as f64;

            // 48 kHz is an exact integer multiple of the 16 kHz inference
            // rate. This deterministic decimator is deliberately limited to
            // fixture work; a production CPAL bridge will use a vetted filter.
            if packet.sample_rate_hz == INFERENCE_SAMPLE_RATE_HZ || source_offset % 3 == 0 {
                self.push_normalized_sample(source_offset, mono as f32, &mut events)?;
            }
        }

        self.expected_source_offset = Some(
            packet
                .starting_sample_offset
                .checked_add(u64::try_from(source_frames).map_err(|_| {
                    "native PCM frame count cannot be represented by the capture clock"
                })?)
                .ok_or_else(|| "native PCM packet end offset overflowed".to_owned())?,
        );
        debug_assert!(
            events.len() <= MAX_PIPELINE_EVENTS_PER_PACKET,
            "a bounded native PCM packet cannot create unbounded pipeline events"
        );
        Ok(events)
    }

    /// Flushes a final active utterance at a known capture stop. A partial 10
    /// ms frame is discarded because it cannot carry a valid inference clock.
    pub fn finish(&mut self) -> Result<Vec<SpeechPipelineEvent>, String> {
        let mut events = Vec::new();
        if let Some(response) = self.finalize_active()? {
            events.push(self.response_event(response));
        }
        self.clear_unfinished_audio();
        Ok(events)
    }

    fn push_normalized_sample(
        &mut self,
        source_offset: u64,
        sample: f32,
        events: &mut Vec<SpeechPipelineEvent>,
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
        let frame = InferenceAudioWindow::new(
            self.session_id,
            self.clock
                .point_at_sample_offset(start_source_offset)
                .monotonic_ns,
            self.clock
                .point_at_sample_offset(end_source_offset)
                .monotonic_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            samples,
        )?;
        if let Some(response) = self.process_frame(frame)? {
            events.push(self.response_event(response));
        }
        Ok(())
    }

    fn process_frame(
        &mut self,
        frame: InferenceAudioWindow,
    ) -> Result<Option<AsrResponse>, String> {
        let speech = self
            .detector
            .is_speech(&frame)
            .map_err(|error| format!("local speech detector failed: {error}"))?;

        if self.active.is_some() && !speech && self.config.hangover_frames == 0 {
            let response = self.finalize_active()?;
            self.push_pre_roll(frame);
            return Ok(response);
        }

        if let Some(active) = self.active.as_mut() {
            active.append(&frame)?;
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

        let mut frames = std::mem::take(&mut self.pre_roll);
        frames.push_back(frame);
        let mut frames = frames.into_iter();
        let first = frames
            .next()
            .expect("a speech start always includes its current frame");
        let mut active = ActiveUtterance::from_frame(&first);
        for previous in frames {
            active.append(&previous)?;
        }
        self.trailing_silence_frames = 0;
        let should_finalize = active.frame_count() >= self.config.maximum_window_frames;
        self.active = Some(active);
        if should_finalize {
            self.finalize_active()
        } else {
            Ok(None)
        }
    }

    fn push_pre_roll(&mut self, frame: InferenceAudioWindow) {
        if self.config.pre_roll_frames == 0 {
            return;
        }
        while self.pre_roll.len() >= self.config.pre_roll_frames {
            self.pre_roll.pop_front();
        }
        self.pre_roll.push_back(frame);
    }

    fn finalize_active(&mut self) -> Result<Option<AsrResponse>, String> {
        let Some(active) = self.active.take() else {
            return Ok(None);
        };
        self.trailing_silence_frames = 0;
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
        let response = self
            .asr
            .transcribe(&request)
            .map_err(|error| format!("local ASR engine failed: {error}"))?;
        response
            .validate_against(&request, self.asr.model_provenance())
            .map_err(|error| format!("local ASR engine returned an invalid response: {error}"))?;
        Ok(Some(response))
    }

    fn clear_unfinished_audio(&mut self) {
        self.pending_samples.clear();
        self.pending_frame_start_offset = None;
        self.last_normalized_source_offset = None;
        self.pre_roll.clear();
        self.active = None;
        self.trailing_silence_frames = 0;
    }

    fn response_event(&self, response: AsrResponse) -> SpeechPipelineEvent {
        SpeechPipelineEvent::AsrResponse {
            session_id: self.session_id,
            response,
        }
    }

    fn source_stride(&self) -> u64 {
        u64::from(self.clock.sample_rate() / INFERENCE_SAMPLE_RATE_HZ)
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
        InferenceEngine, ModelProvenance, TranscriptEmission, TranscriptEmissionKind,
    };
    use chrono::{DateTime, Duration, Utc};
    use std::sync::{Arc, Mutex};

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
    fn downmixes_and_decimates_48khz_pcm_across_packet_boundaries() {
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
        let stereo = |start: usize, length: usize| {
            (start..start + length)
                .flat_map(|frame| [frame as f32 / 480.0, 0.2])
                .collect::<Vec<_>>()
        };
        let first = packet(0, 48_000, 2, stereo(0, 200));
        let second = packet(200, 48_000, 2, stereo(200, 280));

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
        assert!((windows[0].samples()[0] - 0.1).abs() < f32::EPSILON);
        assert!((windows[0].samples()[1] - 0.103_125).abs() < f32::EPSILON);
        assert_eq!(windows[0].capture_start_ns(), 1_000);
        assert_eq!(windows[0].capture_end_ns(), 10_001_000);
    }

    #[test]
    fn decimates_a_48khz_packet_that_starts_between_decimation_ticks() {
        let windows = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = SpeechPipeline::new(
            Uuid::new_v4(),
            clock(48_000),
            AlwaysSpeech,
            RecordingAsr::new(Arc::clone(&windows)),
            config(),
        )
        .unwrap();
        let input = packet(
            1,
            48_000,
            1,
            (1..=480).map(|offset| offset as f32).collect(),
        );

        assert!(pipeline
            .push_packet(NativePcmPacket::from(&input))
            .unwrap()
            .is_empty());
        assert_eq!(response_count(&pipeline.finish().unwrap()), 1);

        let windows = windows.lock().unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].frame_count(), PIPELINE_FRAME_SAMPLES);
        assert_eq!(windows[0].samples()[0], 3.0);
        assert_eq!(windows[0].samples()[PIPELINE_FRAME_SAMPLES - 1], 480.0);
        assert_eq!(windows[0].capture_start_ns(), 63_500);
        assert_eq!(windows[0].capture_end_ns(), 10_063_500);
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
