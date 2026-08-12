use super::{
    cosine_similarity, SpeakerEmbedding, SpeakerMatchPolicy, SpeakerSampleQuality,
    SpeakerSampleRejection,
};
use uuid::Uuid;

pub const MAX_ANONYMOUS_SPEAKER_CLUSTERS: usize = 32;
const MAX_CENTROID_OBSERVATION_WEIGHT: u32 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct AnonymousSpeakerCluster {
    id: String,
    centroid: SpeakerEmbedding,
    observation_count: u32,
}

impl AnonymousSpeakerCluster {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn centroid(&self) -> &SpeakerEmbedding {
        &self.centroid
    }

    pub fn observation_count(&self) -> u32 {
        self.observation_count
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnonymousSpeakerAssignment {
    Assigned { cluster_id: String, similarity: f32 },
    Created { cluster_id: String },
    Ambiguous,
    Ineligible(SpeakerSampleRejection),
    IncompatibleModelSpace,
    CapacityReached,
}

/// Bounded anonymous clustering for exactly one recording session. It never
/// owns names or persistent profile state.
#[derive(Clone, Debug)]
pub struct SessionSpeakerClusterer {
    session_id: Uuid,
    policy: SpeakerMatchPolicy,
    clusters: Vec<AnonymousSpeakerCluster>,
}

impl SessionSpeakerClusterer {
    pub fn new(session_id: Uuid, policy: SpeakerMatchPolicy) -> Result<Self, String> {
        if session_id.is_nil() {
            return Err("speaker cluster session ID must not be empty".to_owned());
        }
        Ok(Self {
            session_id,
            policy,
            clusters: Vec::new(),
        })
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub fn clusters(&self) -> &[AnonymousSpeakerCluster] {
        &self.clusters
    }

    pub fn assign(
        &mut self,
        observation: SpeakerEmbedding,
        quality: SpeakerSampleQuality,
    ) -> AnonymousSpeakerAssignment {
        if let Some(rejection) = self.policy.sample_rejection(quality) {
            return AnonymousSpeakerAssignment::Ineligible(rejection);
        }
        if self
            .clusters
            .first()
            .is_some_and(|cluster| !observation.is_compatible_with(&cluster.centroid))
        {
            return AnonymousSpeakerAssignment::IncompatibleModelSpace;
        }

        let mut candidates = self
            .clusters
            .iter()
            .enumerate()
            .filter_map(|(index, cluster)| {
                cosine_similarity(&observation, &cluster.centroid)
                    .ok()
                    .map(|similarity| (index, similarity))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
        if let Some((best_index, best_similarity)) = candidates.first().copied() {
            if best_similarity >= self.policy.minimum_similarity() {
                if candidates.get(1).is_some_and(|runner_up| {
                    best_similarity - runner_up.1 < self.policy.minimum_runner_up_margin()
                }) {
                    return AnonymousSpeakerAssignment::Ambiguous;
                }
                let cluster = &mut self.clusters[best_index];
                if let Ok(centroid) =
                    updated_centroid(&cluster.centroid, &observation, cluster.observation_count)
                {
                    cluster.centroid = centroid;
                }
                cluster.observation_count = cluster.observation_count.saturating_add(1);
                return AnonymousSpeakerAssignment::Assigned {
                    cluster_id: cluster.id.clone(),
                    similarity: best_similarity,
                };
            }
        }

        if self.clusters.len() >= MAX_ANONYMOUS_SPEAKER_CLUSTERS {
            return AnonymousSpeakerAssignment::CapacityReached;
        }
        let cluster_id = format!("speaker-{}", Uuid::new_v4());
        self.clusters.push(AnonymousSpeakerCluster {
            id: cluster_id.clone(),
            centroid: observation,
            observation_count: 1,
        });
        AnonymousSpeakerAssignment::Created { cluster_id }
    }
}

fn updated_centroid(
    current: &SpeakerEmbedding,
    observation: &SpeakerEmbedding,
    observation_count: u32,
) -> Result<SpeakerEmbedding, String> {
    if !current.is_compatible_with(observation) {
        return Err("cannot update a centroid across speaker model spaces".to_owned());
    }
    let weight = observation_count.min(MAX_CENTROID_OBSERVATION_WEIGHT) as f32;
    let values = current
        .values()
        .iter()
        .zip(observation.values())
        .map(|(current, observation)| (current * weight + observation) / (weight + 1.0))
        .collect::<Vec<_>>();
    SpeakerEmbedding::new(current.model().clone(), values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ModelProvenance;

    fn model(version: &str) -> ModelProvenance {
        ModelProvenance::new("fixture", "speaker-embedding", version, "b".repeat(64)).unwrap()
    }

    fn embedding(values: &[f32]) -> SpeakerEmbedding {
        SpeakerEmbedding::new(model("v1"), values.to_vec()).unwrap()
    }

    fn quality() -> SpeakerSampleQuality {
        SpeakerSampleQuality::new(2_000_000_000, 0.9, 0.9, 0.0).unwrap()
    }

    fn policy() -> SpeakerMatchPolicy {
        SpeakerMatchPolicy::new(0.80, 0.05, 1_000_000_000, 0.7, 0.6, 0.2).unwrap()
    }

    #[test]
    fn creates_then_reuses_a_nearby_anonymous_cluster() {
        let session_id = Uuid::new_v4();
        let mut clusterer = SessionSpeakerClusterer::new(session_id, policy()).unwrap();
        let created = clusterer.assign(embedding(&[1.0, 0.0, 0.0]), quality());
        let AnonymousSpeakerAssignment::Created { cluster_id } = created else {
            panic!("first eligible voice must create an anonymous cluster");
        };

        assert!(matches!(
            clusterer.assign(embedding(&[0.99, 0.03, 0.0]), quality()),
            AnonymousSpeakerAssignment::Assigned { cluster_id: assigned, .. } if assigned == cluster_id
        ));
        assert_eq!(clusterer.clusters()[0].observation_count(), 2);
        assert_eq!(clusterer.session_id(), session_id);
    }

    #[test]
    fn creates_distinct_clusters_and_leaves_ambiguous_or_bad_audio_unassigned() {
        let mut clusterer = SessionSpeakerClusterer::new(Uuid::new_v4(), policy()).unwrap();
        clusterer.assign(embedding(&[1.0, 0.0]), quality());
        assert!(matches!(
            clusterer.assign(embedding(&[0.0, 1.0]), quality()),
            AnonymousSpeakerAssignment::Created { .. }
        ));
        assert!(matches!(
            clusterer.assign(
                embedding(&[1.0, 0.0]),
                SpeakerSampleQuality::new(50_000_000, 0.9, 0.9, 0.0).unwrap()
            ),
            AnonymousSpeakerAssignment::Ineligible(SpeakerSampleRejection::TooShort)
        ));
    }

    #[test]
    fn leaves_two_above_threshold_candidates_ambiguous() {
        let ambiguity_policy =
            SpeakerMatchPolicy::new(0.65, 0.05, 1_000_000_000, 0.7, 0.6, 0.2).unwrap();
        let mut clusterer = SessionSpeakerClusterer::new(Uuid::new_v4(), ambiguity_policy).unwrap();
        clusterer.assign(embedding(&[1.0, 0.0]), quality());
        clusterer.assign(embedding(&[0.0, 1.0]), quality());

        assert!(matches!(
            clusterer.assign(embedding(&[0.71, 0.71]), quality()),
            AnonymousSpeakerAssignment::Ambiguous
        ));
        assert_eq!(clusterer.clusters()[0].observation_count(), 1);
        assert_eq!(clusterer.clusters()[1].observation_count(), 1);
    }

    #[test]
    fn does_not_compare_incompatible_model_spaces() {
        let mut clusterer = SessionSpeakerClusterer::new(Uuid::new_v4(), policy()).unwrap();
        clusterer.assign(embedding(&[1.0, 0.0]), quality());
        let incompatible = SpeakerEmbedding::new(model("v2"), vec![1.0, 0.0]).unwrap();

        assert_eq!(
            clusterer.assign(incompatible, quality()),
            AnonymousSpeakerAssignment::IncompatibleModelSpace
        );
        assert_eq!(clusterer.clusters().len(), 1);
    }
}
