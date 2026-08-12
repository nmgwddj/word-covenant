//! Native WebRTC VAD adapter for the fixed 10 ms inference frame boundary.
//!
//! The adapter accepts only in-process, 16 kHz mono audio frames. It keeps the
//! libfvad state and temporary signed PCM inside native memory and does not
//! implement Serde, persistence, logging, or network access.

use super::{
    pipeline::SpeechActivityDetector, InferenceAudioWindow, InferenceError, INFERENCE_CHANNELS,
    INFERENCE_SAMPLE_RATE_HZ,
};
use webrtc_vad::{SampleRate, Vad, VadMode};

pub const WEBRTC_VAD_FRAME_DURATION_MS: u32 = 10;
pub const WEBRTC_VAD_FRAME_SAMPLES: usize =
    (INFERENCE_SAMPLE_RATE_HZ as usize * WEBRTC_VAD_FRAME_DURATION_MS as usize) / 1_000;

/// libfvad aggressiveness setting used by [`WebRtcVad`].
///
/// `Aggressive` is the production default: it is less likely to turn room
/// noise into an ASR request, while the speech pipeline keeps pre-roll and
/// hangover frames to reduce clipped utterances.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebRtcVadMode {
    Quality,
    LowBitrate,
    Aggressive,
    VeryAggressive,
}

impl WebRtcVadMode {
    fn into_native(self) -> VadMode {
        match self {
            Self::Quality => VadMode::Quality,
            Self::LowBitrate => VadMode::LowBitrate,
            Self::Aggressive => VadMode::Aggressive,
            Self::VeryAggressive => VadMode::VeryAggressive,
        }
    }
}

/// Stateful WebRTC voice activity detector for exact 10 ms inference frames.
///
/// The underlying `webrtc-vad::Vad` contains a native pointer and is neither
/// cloneable nor shareable. It is moved only into the single native capture
/// dispatcher that owns its detector state.
pub struct WebRtcVad {
    inner: Vad,
    mode: WebRtcVadMode,
}

// `WebRtcVad` has exclusive mutable access through `&mut self`; libfvad has
// no thread-affine state, so moving this owned detector to its worker is safe.
unsafe impl Send for WebRtcVad {}

impl WebRtcVad {
    pub fn new() -> Self {
        Self::with_mode(WebRtcVadMode::Aggressive)
    }

    pub fn with_mode(mode: WebRtcVadMode) -> Self {
        Self {
            inner: Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, mode.into_native()),
            mode,
        }
    }

    pub fn mode(&self) -> WebRtcVadMode {
        self.mode
    }

    /// Clears libfvad's short-lived signal history at a capture discontinuity.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Evaluates one exact 10 ms, 16 kHz mono frame.
    pub fn detect_frame(&mut self, frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
        let pcm = pcm_frame(frame)?;
        self.inner
            .is_voice_segment(&pcm)
            .map_err(|_| InferenceError::invalid("WebRTC VAD requires an exact 10 ms PCM frame"))
    }
}

impl Default for WebRtcVad {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechActivityDetector for WebRtcVad {
    fn is_speech(&mut self, frame: &InferenceAudioWindow) -> Result<bool, InferenceError> {
        self.detect_frame(frame)
    }

    fn reset(&mut self) {
        Self::reset(self);
    }
}

fn pcm_frame(
    frame: &InferenceAudioWindow,
) -> Result<[i16; WEBRTC_VAD_FRAME_SAMPLES], InferenceError> {
    frame.validate().map_err(InferenceError::invalid)?;
    if frame.sample_rate_hz() != INFERENCE_SAMPLE_RATE_HZ || frame.channels() != INFERENCE_CHANNELS
    {
        return Err(InferenceError::invalid(
            "WebRTC VAD requires 16 kHz mono inference audio",
        ));
    }
    if frame.frame_count() != WEBRTC_VAD_FRAME_SAMPLES {
        return Err(InferenceError::invalid(format!(
            "WebRTC VAD requires exactly {WEBRTC_VAD_FRAME_SAMPLES} samples per frame"
        )));
    }

    let mut pcm = [0_i16; WEBRTC_VAD_FRAME_SAMPLES];
    for (output, sample) in pcm.iter_mut().zip(frame.samples()) {
        *output = normalized_sample_to_pcm16(*sample)?;
    }
    Ok(pcm)
}

fn normalized_sample_to_pcm16(sample: f32) -> Result<i16, InferenceError> {
    if !sample.is_finite() {
        return Err(InferenceError::invalid("WebRTC VAD samples must be finite"));
    }
    if sample <= -1.0 {
        return Ok(i16::MIN);
    }
    if sample >= 1.0 {
        return Ok(i16::MAX);
    }

    Ok((sample * f32::from(i16::MAX)).round() as i16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{InferenceAudioWindow, INFERENCE_CHANNELS, INFERENCE_SAMPLE_RATE_HZ};
    use uuid::Uuid;

    fn frame(samples: Vec<f32>) -> InferenceAudioWindow {
        let end_ns = (samples.len() as u64 * 1_000_000_000) / u64::from(INFERENCE_SAMPLE_RATE_HZ);
        InferenceAudioWindow::new(
            Uuid::nil(),
            0,
            end_ns,
            INFERENCE_SAMPLE_RATE_HZ,
            INFERENCE_CHANNELS,
            samples,
        )
        .unwrap()
    }

    fn speech_like_fixture() -> Vec<f32> {
        // A deterministic, voiced-band fixture. It is only a VAD test signal,
        // never a substitute for actual transcription input or speaker logic.
        (0..WEBRTC_VAD_FRAME_SAMPLES)
            .map(|index| {
                let phase = index as f32 / INFERENCE_SAMPLE_RATE_HZ as f32;
                let carrier = (std::f32::consts::TAU * 180.0 * phase).sin();
                let harmonic = (std::f32::consts::TAU * 540.0 * phase).sin() * 0.22;
                (carrier * 0.68) + harmonic
            })
            .collect()
    }

    #[test]
    fn keeps_silence_out_of_the_speech_path() {
        let mut detector = WebRtcVad::new();

        assert!(!detector
            .detect_frame(&frame(vec![0.0; WEBRTC_VAD_FRAME_SAMPLES]))
            .unwrap());
    }

    #[test]
    fn evaluates_a_deterministic_voiced_fixture() {
        let mut detector = WebRtcVad::with_mode(WebRtcVadMode::Quality);
        let voiced = frame(speech_like_fixture());

        let decisions = (0..4)
            .map(|_| detector.detect_frame(&voiced).unwrap())
            .collect::<Vec<_>>();

        assert!(
            decisions.iter().any(|decision| *decision),
            "the WebRTC VAD fixture must exercise its voiced path"
        );
    }

    #[test]
    fn rejects_non_10ms_frames_before_calling_libfvad() {
        let mut detector = WebRtcVad::new();
        let error = detector
            .detect_frame(&frame(vec![0.0; WEBRTC_VAD_FRAME_SAMPLES * 2]))
            .unwrap_err();

        assert!(error.to_string().contains("exactly"));
    }

    #[test]
    fn saturates_normalized_samples_and_rejects_non_finite_values() {
        assert_eq!(normalized_sample_to_pcm16(-3.0).unwrap(), i16::MIN);
        assert_eq!(normalized_sample_to_pcm16(3.0).unwrap(), i16::MAX);
        assert_eq!(normalized_sample_to_pcm16(0.5).unwrap(), 16_384);
        assert!(normalized_sample_to_pcm16(f32::NAN).is_err());
    }

    #[test]
    fn reset_preserves_the_configured_mode() {
        let mut detector = WebRtcVad::with_mode(WebRtcVadMode::VeryAggressive);
        let _ = detector.detect_frame(&frame(speech_like_fixture()));

        detector.reset();

        assert_eq!(detector.mode(), WebRtcVadMode::VeryAggressive);
        assert!(!detector
            .detect_frame(&frame(vec![0.0; WEBRTC_VAD_FRAME_SAMPLES]))
            .unwrap());
    }
}
