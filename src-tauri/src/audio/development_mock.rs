use super::{CaptureClock, CapturePacket, CapturePoint, CaptureSource, TestCaptureSource};
use crate::domain::{CaptureSession, TranscriptSpan};
use crate::inference::pipeline::{
    EnergySpeechDetector, NativePcmPacket, SpeechPipeline, SpeechPipelineConfig,
    SpeechPipelineEvent,
};
use crate::inference::{
    AsrEngine, AsrRequest, AsrResponse, InferenceEngine, InferenceError, ModelProvenance,
    TranscriptEmission, TranscriptEmissionKind,
};
use serde::Serialize;
use std::collections::VecDeque;
use uuid::Uuid;

pub const DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE: usize = 10;
pub(crate) const DEVELOPMENT_MOCK_MAX_PENDING_ASR_RESPONSES: usize = 16;

const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u16 = 1;
const FRAMES_PER_PACKET: usize = 320;
const FRAMES_PER_PACKET_U64: u64 = 320;
const TOTAL_PACKETS: usize = 600;
const WAVE_FREQUENCY_HZ: f32 = 220.0;
const WAVE_AMPLITUDE: f32 = 0.08;
const ENERGY_GATE_MINIMUM_RMS: f32 = 0.03;
const PIPELINE_HANGOVER_FRAMES: usize = 16;
const SCRIPTED_ASR_ARTIFACT_SHA256: &str =
    "98b983076c7d53180585574129340b2174852783bc8e65fb3dbddb58d43d872d";

#[derive(Clone, Copy, Debug)]
struct DevelopmentMockCue {
    start_sample_offset: u64,
    end_sample_offset: u64,
    partial_text: &'static str,
    text: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DevelopmentMockProgress {
    pub session_id: Uuid,
    pub packets_advanced: usize,
    pub spans: Vec<TranscriptSpan>,
    pub exhausted: bool,
}

/// Deterministic, local-only capture input for debug builds and Rust tests.
///
/// The source emits deterministic local PCM packets to exercise the capture
/// clock and a native fixture pipeline. It deliberately does not represent a
/// microphone, production speech recognizer, or speaker-identification
/// implementation.
pub struct DevelopmentMockRunner {
    session_id: Uuid,
    source: TestCaptureSource,
    pipeline: SpeechPipeline<EnergySpeechDetector, ScriptedMockAsr>,
    remaining_packets: usize,
    pending_asr_responses: VecDeque<PendingAsrResponse>,
    next_delivery_id: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct DevelopmentMockAsrDelivery {
    pub(crate) id: u64,
    pub(crate) response: AsrResponse,
}

#[derive(Clone, Debug)]
struct PendingAsrResponse {
    id: u64,
    response: AsrResponse,
    in_flight: bool,
}

impl DevelopmentMockRunner {
    pub fn new(session: &CaptureSession) -> Result<Self, String> {
        let packets = (0..TOTAL_PACKETS).map(|index| {
            let starting_sample_offset = u64::try_from(index)
                .expect("packet index fits")
                .saturating_mul(FRAMES_PER_PACKET_U64);
            CapturePacket {
                starting_sample_offset,
                sample_rate: SAMPLE_RATE,
                channels: CHANNELS,
                samples: scripted_pcm_for_packet(starting_sample_offset),
            }
        });
        let mut source = TestCaptureSource::new(packets);
        source.start()?;

        let clock = CaptureClock::new(
            CapturePoint {
                monotonic_ns: session.started_monotonic_ns,
                wall_clock: session.started_at,
            },
            SAMPLE_RATE,
        )?;
        let pipeline = SpeechPipeline::with_energy_gate(
            session.id,
            clock.clone(),
            ENERGY_GATE_MINIMUM_RMS,
            ScriptedMockAsr::new(&clock)?,
            scripted_pipeline_config(),
        )?;

        Ok(Self {
            session_id: session.id,
            source,
            pipeline,
            remaining_packets: TOTAL_PACKETS,
            pending_asr_responses: VecDeque::new(),
            next_delivery_id: 1,
        })
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub fn stop(&mut self) -> Result<(), String> {
        self.source.stop()?;
        let events = self.pipeline.finish()?;
        self.enqueue_asr_responses(events)
    }

    pub fn advance(&mut self, packet_count: usize) -> Result<DevelopmentMockProgress, String> {
        if packet_count == 0 || packet_count > DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE {
            return Err(format!(
                "development mock packet count must be between 1 and {DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE}"
            ));
        }

        let mut packets_advanced = 0;
        for _ in 0..packet_count {
            if self.pending_asr_responses.len() >= DEVELOPMENT_MOCK_MAX_PENDING_ASR_RESPONSES {
                return Err(format!(
                    "development mock has reached its {DEVELOPMENT_MOCK_MAX_PENDING_ASR_RESPONSES}-response delivery limit"
                ));
            }
            let Some(packet) = self.source.try_next_packet() else {
                break;
            };
            packets_advanced += 1;
            self.remaining_packets = self.remaining_packets.saturating_sub(1);
            let events = self.pipeline.push_packet(NativePcmPacket::from(&packet))?;
            self.enqueue_asr_responses(events)?;
        }

        if self.remaining_packets == 0 {
            let events = self.pipeline.finish()?;
            self.enqueue_asr_responses(events)?;
        }

        Ok(DevelopmentMockProgress {
            session_id: self.session_id,
            packets_advanced,
            spans: Vec::new(),
            exhausted: self.remaining_packets == 0,
        })
    }

    /// Claims the next native-only response for delivery. The response stays
    /// queued until the caller explicitly commits its successful persistence.
    pub(crate) fn begin_pending_asr_delivery(
        &mut self,
    ) -> Result<Option<DevelopmentMockAsrDelivery>, String> {
        let Some(pending) = self.pending_asr_responses.front_mut() else {
            return Ok(None);
        };
        if pending.in_flight {
            return Err("development mock ASR response delivery is already in flight".to_owned());
        }
        pending.in_flight = true;
        Ok(Some(DevelopmentMockAsrDelivery {
            id: pending.id,
            response: pending.response.clone(),
        }))
    }

    /// Acknowledges exactly the response that committed to durable storage.
    pub(crate) fn commit_pending_asr_delivery(&mut self, id: u64) -> Result<(), String> {
        let Some(pending) = self.pending_asr_responses.front() else {
            return Err("development mock ASR response delivery is no longer pending".to_owned());
        };
        if pending.id != id || !pending.in_flight {
            return Err(
                "development mock ASR response delivery acknowledgement is out of order".to_owned(),
            );
        }
        self.pending_asr_responses.pop_front();
        Ok(())
    }

    /// Makes a failed delivery available to the next local retry without
    /// discarding its partial/final response payload.
    pub(crate) fn abort_pending_asr_delivery(&mut self, id: u64) -> Result<(), String> {
        let Some(pending) = self.pending_asr_responses.front_mut() else {
            return Err("development mock ASR response delivery is no longer pending".to_owned());
        };
        if pending.id != id || !pending.in_flight {
            return Err(
                "development mock ASR response delivery release is out of order".to_owned(),
            );
        }
        pending.in_flight = false;
        Ok(())
    }

    pub(crate) fn has_pending_asr_responses(&self) -> bool {
        !self.pending_asr_responses.is_empty()
    }

    fn enqueue_asr_responses(&mut self, events: Vec<SpeechPipelineEvent>) -> Result<(), String> {
        let mut responses = Vec::new();
        for event in events {
            match event {
                SpeechPipelineEvent::AsrResponse {
                    session_id,
                    response,
                } if session_id == self.session_id => responses.push(response),
                SpeechPipelineEvent::AsrResponse { session_id, .. } => {
                    return Err(format!(
                        "development mock pipeline returned a response for a different session: {session_id}"
                    ));
                }
                SpeechPipelineEvent::Discontinuity {
                    expected_source_offset,
                    received_source_offset,
                    ..
                } => {
                    return Err(format!(
                        "development mock source is unexpectedly discontinuous: expected offset {expected_source_offset}, received {received_source_offset}"
                    ));
                }
            }
        }
        let available = DEVELOPMENT_MOCK_MAX_PENDING_ASR_RESPONSES
            .checked_sub(self.pending_asr_responses.len())
            .expect("a bounded response queue never exceeds its maximum");
        if responses.len() > available {
            return Err(format!(
                "development mock response delivery would exceed its {DEVELOPMENT_MOCK_MAX_PENDING_ASR_RESPONSES}-response limit"
            ));
        }
        for response in responses {
            let id = self.next_delivery_id;
            self.next_delivery_id = self
                .next_delivery_id
                .checked_add(1)
                .ok_or_else(|| "development mock ASR response delivery ID overflowed".to_owned())?;
            self.pending_asr_responses.push_back(PendingAsrResponse {
                id,
                response,
                in_flight: false,
            });
        }
        Ok(())
    }
}

struct ScriptedMockAsr {
    model_provenance: ModelProvenance,
    pending_cues: VecDeque<ScriptedMockAsrCue>,
}

#[derive(Clone, Debug)]
struct ScriptedMockAsrCue {
    utterance_key: String,
    capture_start_ns: u64,
    capture_end_ns: u64,
    partial_text: &'static str,
    text: &'static str,
}

impl ScriptedMockAsr {
    fn new(clock: &CaptureClock) -> Result<Self, String> {
        let model_provenance = ModelProvenance::new(
            "word-covenant-development-mock",
            "scripted-local-asr",
            "debug-v1",
            SCRIPTED_ASR_ARTIFACT_SHA256,
        )?;
        let pending_cues = SCRIPTED_CUES
            .iter()
            .map(|cue| {
                let start = clock
                    .point_at_sample_offset(cue.start_sample_offset)
                    .monotonic_ns;
                let end = clock
                    .point_at_sample_offset(cue.end_sample_offset)
                    .monotonic_ns;
                ScriptedMockAsrCue {
                    utterance_key: format!(
                        "development-mock-{}-{}",
                        cue.start_sample_offset, cue.end_sample_offset
                    ),
                    capture_start_ns: start,
                    capture_end_ns: end,
                    partial_text: cue.partial_text,
                    text: cue.text,
                }
            })
            .collect();
        Ok(Self {
            model_provenance,
            pending_cues,
        })
    }
}

impl InferenceEngine for ScriptedMockAsr {
    fn model_provenance(&self) -> &ModelProvenance {
        &self.model_provenance
    }
}

impl AsrEngine for ScriptedMockAsr {
    fn transcribe(&mut self, request: &AsrRequest) -> Result<AsrResponse, InferenceError> {
        request.validate().map_err(InferenceError::invalid)?;
        let Some(cue) = self.pending_cues.front() else {
            return AsrResponse::new(request, &self.model_provenance, Vec::new())
                .map_err(InferenceError::invalid);
        };

        // Stopping inside a scripted sentence produces no invented final
        // transcript. The next full fixture run begins with a fresh adapter.
        if request.audio.capture_end_ns() < cue.capture_end_ns {
            return AsrResponse::new(request, &self.model_provenance, Vec::new())
                .map_err(InferenceError::invalid);
        }
        if request.audio.capture_start_ns() > cue.capture_start_ns {
            return Err(InferenceError::failed(
                "development mock ASR window starts after its scripted sentence",
            ));
        }

        let cue = self
            .pending_cues
            .pop_front()
            .expect("a checked scripted cue remains queued");
        let partial_end_ns = cue
            .capture_start_ns
            .saturating_add(cue.capture_end_ns.saturating_sub(cue.capture_start_ns) / 2);
        AsrResponse::new(
            request,
            &self.model_provenance,
            vec![
                TranscriptEmission {
                    utterance_key: cue.utterance_key.clone(),
                    capture_start_ns: cue.capture_start_ns,
                    capture_end_ns: partial_end_ns,
                    text: cue.partial_text.to_owned(),
                    kind: TranscriptEmissionKind::Partial,
                    revision: 1,
                    word_timings: Vec::new(),
                    model_provenance: self.model_provenance.clone(),
                },
                TranscriptEmission {
                    utterance_key: cue.utterance_key,
                    capture_start_ns: cue.capture_start_ns,
                    capture_end_ns: cue.capture_end_ns,
                    text: cue.text.to_owned(),
                    kind: TranscriptEmissionKind::Final,
                    revision: 2,
                    word_timings: Vec::new(),
                    model_provenance: self.model_provenance.clone(),
                },
            ],
        )
        .map_err(InferenceError::invalid)
    }
}

const SCRIPTED_CUES: [DevelopmentMockCue; 3] = [
    DevelopmentMockCue {
        start_sample_offset: 0,
        end_sample_offset: 44_800,
        partial_text: "partialonlyone",
        text: "本次记录仅保存在本机。",
    },
    DevelopmentMockCue {
        start_sample_offset: 49_600,
        end_sample_offset: 115_200,
        partial_text: "partialonlytwo",
        text: "出网行为需要在行动前单独授权。",
    },
    DevelopmentMockCue {
        start_sample_offset: 121_600,
        end_sample_offset: 182_400,
        partial_text: "partialonlythree",
        text: "先生成一份待确认的行动草案。",
    },
];

fn scripted_pipeline_config() -> SpeechPipelineConfig {
    // The shortest scripted silence is 300 ms, so 160 ms gives each sentence
    // a deterministic finalization boundary without joining neighbors.
    SpeechPipelineConfig {
        pre_roll_frames: 0,
        hangover_frames: PIPELINE_HANGOVER_FRAMES,
        emit_partials: true,
        ..SpeechPipelineConfig::default()
    }
}

fn scripted_pcm_for_packet(starting_sample_offset: u64) -> Vec<f32> {
    (0..FRAMES_PER_PACKET)
        .map(|frame| {
            let sample_offset = starting_sample_offset
                .saturating_add(u64::try_from(frame).expect("packet frame index fits in a u64"));
            if !SCRIPTED_CUES.iter().any(|cue| {
                cue.start_sample_offset <= sample_offset && sample_offset < cue.end_sample_offset
            }) {
                return 0.0;
            }

            let phase = 2.0 * std::f32::consts::PI * WAVE_FREQUENCY_HZ * sample_offset as f32
                / SAMPLE_RATE as f32;
            WAVE_AMPLITUDE * phase.sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::TranscriptEmissionKind;
    use chrono::{DateTime, Duration, Utc};

    fn session() -> CaptureSession {
        CaptureSession::begin(1_000, DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(10))
    }

    #[test]
    fn emits_scripted_asr_responses_only_after_pipeline_finalization() {
        let session = session();
        let mut runner = DevelopmentMockRunner::new(&session).unwrap();

        // The first cue ends after 140 packets, but the energy-gated
        // pipeline must observe its hangover before calling the local ASR
        // adapter. No partial result is a public transcript projection.
        for _ in 0..14 {
            let progress = runner.advance(10).unwrap();
            assert!(progress.spans.is_empty());
            assert!(!runner.has_pending_asr_responses());
        }

        let progress = runner.advance(10).unwrap();
        assert_eq!(progress.packets_advanced, 10);
        assert!(progress.spans.is_empty());
        let delivery = runner
            .begin_pending_asr_delivery()
            .unwrap()
            .expect("pipeline queued the scripted response for native delivery");
        let response = &delivery.response;
        assert_eq!(response.emissions.len(), 2);
        assert_eq!(
            response
                .emissions
                .iter()
                .map(|emission| (emission.kind, emission.revision))
                .collect::<Vec<_>>(),
            vec![
                (TranscriptEmissionKind::Partial, 1),
                (TranscriptEmissionKind::Final, 2),
            ]
        );
        let final_emission = response
            .emissions
            .iter()
            .find(|emission| emission.kind == TranscriptEmissionKind::Final)
            .expect("scripted response includes a final emission");
        assert_eq!(
            final_emission.capture_start_ns,
            session.started_monotonic_ns
        );
        assert_eq!(
            final_emission.capture_end_ns - final_emission.capture_start_ns,
            2_800_000_000
        );
        assert_eq!(final_emission.text, "本次记录仅保存在本机。");
        runner.abort_pending_asr_delivery(delivery.id).unwrap();
    }

    #[test]
    fn bounds_manual_advances() {
        let mut runner = DevelopmentMockRunner::new(&session()).unwrap();

        assert!(runner.advance(0).is_err());
        assert!(runner
            .advance(DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE + 1)
            .is_err());
    }

    #[test]
    fn uses_a_low_amplitude_wave_only_during_scripted_cues() {
        let speaking_packet = scripted_pcm_for_packet(0);
        let silent_packet = scripted_pcm_for_packet(44_800);

        assert!(speaking_packet.iter().any(|sample| *sample != 0.0));
        assert!(speaking_packet
            .iter()
            .all(|sample| sample.abs() <= WAVE_AMPLITUDE));
        assert!(silent_packet.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn remains_active_for_the_full_twelve_second_script_cycle() {
        let mut runner = DevelopmentMockRunner::new(&session()).unwrap();

        for _ in 0..59 {
            assert!(!runner.advance(10).unwrap().exhausted);
        }
        assert!(runner.advance(10).unwrap().exhausted);
    }
}
