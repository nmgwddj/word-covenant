use super::{CaptureClock, CapturePacket, CapturePoint, CaptureSource, TestCaptureSource};
use crate::domain::{CaptureSession, TranscriptSource, TranscriptSpan};
use serde::Serialize;
use std::collections::VecDeque;
use uuid::Uuid;

pub const DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE: usize = 10;

const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u16 = 1;
const FRAMES_PER_PACKET: usize = 320;
const FRAMES_PER_PACKET_U64: u64 = 320;
const TOTAL_PACKETS: usize = 600;
const WAVE_FREQUENCY_HZ: f32 = 220.0;
const WAVE_AMPLITUDE: f32 = 0.08;

#[derive(Clone, Copy, Debug)]
struct DevelopmentMockCue {
    start_sample_offset: u64,
    end_sample_offset: u64,
    speaker_cluster_id: &'static str,
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
/// clock and then projects a fixed transcript script. It deliberately does not represent a
/// microphone, speech recognizer, or speaker-identification implementation.
pub struct DevelopmentMockRunner {
    session_id: Uuid,
    source: TestCaptureSource,
    clock: CaptureClock,
    pending_cues: VecDeque<DevelopmentMockCue>,
    remaining_packets: usize,
}

impl DevelopmentMockRunner {
    pub fn new(session: &CaptureSession) -> Result<Self, String> {
        let pending_cues = VecDeque::from(scripted_cues());
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

        Ok(Self {
            session_id: session.id,
            source,
            clock,
            pending_cues,
            remaining_packets: TOTAL_PACKETS,
        })
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub fn stop(&mut self) -> Result<(), String> {
        self.source.stop()
    }

    pub fn advance(&mut self, packet_count: usize) -> Result<DevelopmentMockProgress, String> {
        if packet_count == 0 || packet_count > DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE {
            return Err(format!(
                "development mock packet count must be between 1 and {DEVELOPMENT_MOCK_MAX_PACKETS_PER_ADVANCE}"
            ));
        }

        let mut packets_advanced = 0;
        let mut spans = Vec::new();

        for _ in 0..packet_count {
            let Some(packet) = self.source.try_next_packet() else {
                break;
            };
            packets_advanced += 1;
            self.remaining_packets = self.remaining_packets.saturating_sub(1);

            let packet_end_sample_offset = packet
                .starting_sample_offset
                .saturating_add(u64::try_from(packet.frame_count()).expect("frame count fits"));
            while self
                .pending_cues
                .front()
                .is_some_and(|cue| cue.end_sample_offset <= packet_end_sample_offset)
            {
                let cue = self
                    .pending_cues
                    .pop_front()
                    .expect("cue exists after front check");
                let start = self.clock.point_at_sample_offset(cue.start_sample_offset);
                let end = self.clock.point_at_sample_offset(cue.end_sample_offset);
                spans.push(TranscriptSpan::new(
                    self.session_id,
                    start.monotonic_ns,
                    end.monotonic_ns,
                    Some(cue.speaker_cluster_id.to_owned()),
                    cue.text,
                    true,
                    1,
                    TranscriptSource::Synthetic,
                )?);
            }
        }

        Ok(DevelopmentMockProgress {
            session_id: self.session_id,
            packets_advanced,
            spans,
            exhausted: self.remaining_packets == 0,
        })
    }
}

const SCRIPTED_CUES: [DevelopmentMockCue; 3] = [
    DevelopmentMockCue {
        start_sample_offset: 0,
        end_sample_offset: 44_800,
        speaker_cluster_id: "speaker-1",
        text: "本次记录仅保存在本机。",
    },
    DevelopmentMockCue {
        start_sample_offset: 49_600,
        end_sample_offset: 115_200,
        speaker_cluster_id: "speaker-2",
        text: "出网行为需要在行动前单独授权。",
    },
    DevelopmentMockCue {
        start_sample_offset: 121_600,
        end_sample_offset: 182_400,
        speaker_cluster_id: "speaker-1",
        text: "先生成一份待确认的行动草案。",
    },
];

fn scripted_cues() -> Vec<DevelopmentMockCue> {
    SCRIPTED_CUES.to_vec()
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
    use chrono::{DateTime, Duration, Utc};

    fn session() -> CaptureSession {
        CaptureSession::begin(1_000, DateTime::<Utc>::UNIX_EPOCH + Duration::seconds(10))
    }

    #[test]
    fn emits_scripted_spans_at_the_sample_clock_offsets() {
        let session = session();
        let mut runner = DevelopmentMockRunner::new(&session).unwrap();

        for _ in 0..13 {
            let progress = runner.advance(10).unwrap();
            assert!(progress.spans.is_empty());
        }

        let progress = runner.advance(10).unwrap();
        assert_eq!(progress.packets_advanced, 10);
        assert_eq!(progress.spans.len(), 1);
        let span = &progress.spans[0];
        assert_eq!(span.session_id, session.id);
        assert_eq!(span.capture_start_ns, session.started_monotonic_ns);
        assert_eq!(span.capture_end_ns - span.capture_start_ns, 2_800_000_000);
        assert_eq!(span.speaker_cluster_id.as_deref(), Some("speaker-1"));
        assert_eq!(span.text, "本次记录仅保存在本机。");
        assert_eq!(span.source, TranscriptSource::Synthetic);
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
