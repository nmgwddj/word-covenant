mod clustering;
mod embedding;
mod matching;
#[cfg(target_os = "macos")]
mod onnx_embedding;

pub use clustering::{
    AnonymousSpeakerAssignment, AnonymousSpeakerCluster, SessionSpeakerClusterer,
};
pub use embedding::{
    SpeakerEmbedding, SpeakerEmbeddingEngine, SpeakerSampleQuality,
    MAX_SPEAKER_EMBEDDING_DIMENSIONS,
};
pub use matching::{
    cosine_similarity, match_speaker_profile, SpeakerMatchCandidate, SpeakerMatchDecision,
    SpeakerMatchPolicy, SpeakerSampleRejection,
};
#[cfg(target_os = "macos")]
pub use onnx_embedding::{
    bundled_speaker_model, BundledSpeakerModel, OnnxSpeakerEmbeddingEngine, SpeakerModelManifest,
};
