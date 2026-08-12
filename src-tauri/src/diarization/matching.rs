use super::{SpeakerEmbedding, SpeakerSampleQuality};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerMatchPolicy {
    minimum_similarity: f32,
    minimum_runner_up_margin: f32,
    minimum_voiced_duration_ns: u64,
    minimum_voiced_ratio: f32,
    minimum_signal_quality: f32,
    maximum_overlap_probability: f32,
}

impl SpeakerMatchPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        minimum_similarity: f32,
        minimum_runner_up_margin: f32,
        minimum_voiced_duration_ns: u64,
        minimum_voiced_ratio: f32,
        minimum_signal_quality: f32,
        maximum_overlap_probability: f32,
    ) -> Result<Self, String> {
        for (label, value) in [
            ("minimum similarity", minimum_similarity),
            ("minimum runner-up margin", minimum_runner_up_margin),
            ("minimum voiced ratio", minimum_voiced_ratio),
            ("minimum signal quality", minimum_signal_quality),
            ("maximum overlap probability", maximum_overlap_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "speaker match {label} must be between zero and one"
                ));
            }
        }
        Ok(Self {
            minimum_similarity,
            minimum_runner_up_margin,
            minimum_voiced_duration_ns,
            minimum_voiced_ratio,
            minimum_signal_quality,
            maximum_overlap_probability,
        })
    }

    pub fn sample_rejection(self, quality: SpeakerSampleQuality) -> Option<SpeakerSampleRejection> {
        if quality.voiced_duration_ns() < self.minimum_voiced_duration_ns {
            Some(SpeakerSampleRejection::TooShort)
        } else if quality.voiced_ratio() < self.minimum_voiced_ratio {
            Some(SpeakerSampleRejection::InsufficientVoice)
        } else if quality.signal_quality() < self.minimum_signal_quality {
            Some(SpeakerSampleRejection::LowSignalQuality)
        } else if quality.overlap_probability() > self.maximum_overlap_probability {
            Some(SpeakerSampleRejection::PossibleOverlap)
        } else {
            None
        }
    }

    pub(crate) fn minimum_similarity(self) -> f32 {
        self.minimum_similarity
    }

    pub(crate) fn minimum_runner_up_margin(self) -> f32 {
        self.minimum_runner_up_margin
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeakerSampleRejection {
    TooShort,
    InsufficientVoice,
    LowSignalQuality,
    PossibleOverlap,
}

#[derive(Clone, Copy, Debug)]
pub struct SpeakerMatchCandidate<'a> {
    pub profile_id: Uuid,
    pub prototype: &'a SpeakerEmbedding,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpeakerMatchDecision {
    Matched {
        profile_id: Uuid,
        similarity: f32,
        runner_up_margin: f32,
        runner_up_similarity: Option<f32>,
    },
    Unknown {
        best_similarity: Option<f32>,
    },
    Ambiguous {
        best_profile_id: Uuid,
        best_similarity: f32,
        runner_up_similarity: f32,
    },
    Ineligible(SpeakerSampleRejection),
}

pub fn cosine_similarity(left: &SpeakerEmbedding, right: &SpeakerEmbedding) -> Result<f32, String> {
    if !left.is_compatible_with(right) {
        return Err("speaker embeddings belong to incompatible model spaces".to_owned());
    }
    Ok(left
        .values()
        .iter()
        .zip(right.values())
        .map(|(left, right)| left * right)
        .sum::<f32>()
        .clamp(-1.0, 1.0))
}

pub fn match_speaker_profile(
    observation: &SpeakerEmbedding,
    quality: SpeakerSampleQuality,
    candidates: &[SpeakerMatchCandidate<'_>],
    policy: SpeakerMatchPolicy,
) -> SpeakerMatchDecision {
    if let Some(rejection) = policy.sample_rejection(quality) {
        return SpeakerMatchDecision::Ineligible(rejection);
    }

    // A profile may own several confirmed prototypes. Collapse those to the
    // best score per profile before applying the runner-up margin; otherwise
    // two prototypes from the same person can make that person look ambiguous.
    let mut best_by_profile = BTreeMap::<Uuid, f32>::new();
    for candidate in candidates {
        let Ok(similarity) = cosine_similarity(observation, candidate.prototype) else {
            continue;
        };
        best_by_profile
            .entry(candidate.profile_id)
            .and_modify(|current| *current = current.max(similarity))
            .or_insert(similarity);
    }
    let mut compatible = best_by_profile.into_iter().collect::<Vec<_>>();
    compatible.sort_by(|left, right| right.1.total_cmp(&left.1));
    let Some((best_profile_id, best_similarity)) = compatible.first().copied() else {
        return SpeakerMatchDecision::Unknown {
            best_similarity: None,
        };
    };
    if best_similarity < policy.minimum_similarity() {
        return SpeakerMatchDecision::Unknown {
            best_similarity: Some(best_similarity),
        };
    }

    let runner_up_similarity = compatible.get(1).map(|candidate| candidate.1);
    let runner_up_margin =
        runner_up_similarity.map_or(1.0, |runner_up| best_similarity - runner_up);
    if let Some(runner_up_similarity) = runner_up_similarity {
        if runner_up_margin < policy.minimum_runner_up_margin() {
            return SpeakerMatchDecision::Ambiguous {
                best_profile_id,
                best_similarity,
                runner_up_similarity,
            };
        }
    }
    SpeakerMatchDecision::Matched {
        profile_id: best_profile_id,
        similarity: best_similarity,
        runner_up_margin,
        runner_up_similarity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ModelProvenance;

    fn model(version: &str) -> ModelProvenance {
        ModelProvenance::new("fixture", "speaker-embedding", version, "a".repeat(64)).unwrap()
    }

    fn embedding(values: &[f32]) -> SpeakerEmbedding {
        SpeakerEmbedding::new(model("v1"), values.to_vec()).unwrap()
    }

    fn quality() -> SpeakerSampleQuality {
        SpeakerSampleQuality::new(2_000_000_000, 0.9, 0.8, 0.05).unwrap()
    }

    fn policy() -> SpeakerMatchPolicy {
        SpeakerMatchPolicy::new(0.80, 0.08, 1_000_000_000, 0.7, 0.6, 0.2).unwrap()
    }

    #[test]
    fn accepts_only_a_high_similarity_candidate_with_a_clear_margin() {
        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        let first = embedding(&[1.0, 0.0, 0.0]);
        let second = embedding(&[0.0, 1.0, 0.0]);
        let observation = embedding(&[0.98, 0.05, 0.0]);

        assert!(matches!(
            match_speaker_profile(
                &observation,
                quality(),
                &[
                    SpeakerMatchCandidate {
                        profile_id: first_id,
                        prototype: &first,
                    },
                    SpeakerMatchCandidate {
                        profile_id: second_id,
                        prototype: &second,
                    },
                ],
                policy(),
            ),
            SpeakerMatchDecision::Matched { profile_id, .. } if profile_id == first_id
        ));
    }

    #[test]
    fn leaves_low_similarity_and_close_candidates_unassigned() {
        let first_id = Uuid::new_v4();
        let first = embedding(&[1.0, 0.0]);
        let second = embedding(&[0.98, 0.2]);
        let unknown = embedding(&[0.0, 1.0]);

        assert!(matches!(
            match_speaker_profile(
                &unknown,
                quality(),
                &[SpeakerMatchCandidate {
                    profile_id: first_id,
                    prototype: &first,
                }],
                policy(),
            ),
            SpeakerMatchDecision::Unknown { .. }
        ));
        assert!(matches!(
            match_speaker_profile(
                &first,
                quality(),
                &[
                    SpeakerMatchCandidate {
                        profile_id: first_id,
                        prototype: &first,
                    },
                    SpeakerMatchCandidate {
                        profile_id: Uuid::new_v4(),
                        prototype: &second,
                    },
                ],
                policy(),
            ),
            SpeakerMatchDecision::Ambiguous { .. }
        ));
    }

    #[test]
    fn rejects_bad_samples_and_ignores_incompatible_model_spaces() {
        let profile_id = Uuid::new_v4();
        let observation = embedding(&[1.0, 0.0]);
        let incompatible = SpeakerEmbedding::new(model("v2"), vec![1.0, 0.0]).unwrap();
        let short = SpeakerSampleQuality::new(100_000_000, 0.9, 0.9, 0.0).unwrap();

        assert_eq!(
            match_speaker_profile(&observation, short, &[], policy()),
            SpeakerMatchDecision::Ineligible(SpeakerSampleRejection::TooShort)
        );
        assert_eq!(
            match_speaker_profile(
                &observation,
                quality(),
                &[SpeakerMatchCandidate {
                    profile_id,
                    prototype: &incompatible,
                }],
                policy(),
            ),
            SpeakerMatchDecision::Unknown {
                best_similarity: None
            }
        );
        assert!(cosine_similarity(&observation, &incompatible).is_err());
    }

    #[test]
    fn multiple_prototypes_for_one_profile_do_not_compete_as_runner_up() {
        let profile_id = Uuid::new_v4();
        let first = embedding(&[1.0, 0.0]);
        let second = embedding(&[0.99, 0.04]);
        let observation = embedding(&[0.995, 0.02]);

        assert!(matches!(
            match_speaker_profile(
                &observation,
                quality(),
                &[
                    SpeakerMatchCandidate {
                        profile_id,
                        prototype: &first,
                    },
                    SpeakerMatchCandidate {
                        profile_id,
                        prototype: &second,
                    },
                ],
                policy(),
            ),
            SpeakerMatchDecision::Matched {
                profile_id: matched,
                ..
            } if matched == profile_id
        ));
    }
}
