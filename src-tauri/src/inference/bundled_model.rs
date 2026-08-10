//! Verified metadata and resource loading for the model shipped with macOS.
//!
//! This module owns no network client. The model lock is compiled into the
//! native binary and must match the manifest packaged beside the model before
//! the artifact can be loaded by Whisper.

use super::{
    model_registry::{LocalModelKind, RegisteredModel, VerifiedModelArtifact},
    WHISPER_CPP_GGML_INPUT_FORMAT,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

const BUNDLED_MANIFEST_RELATIVE_PATH: &str = "models/manifest.json";
const BUNDLED_RESOURCE_DIRECTORY: &str = "models";
const COPY_BUFFER_BYTES: usize = 64 * 1024;
const SHA256_HEX_LENGTH: usize = 64;

/// Immutable identity of the audited, byte-identical default model.
///
/// A changed model must receive a different identity in its reviewed lock
/// file rather than replacing this artifact in place.
pub const BUNDLED_ASR_MODEL_ID: Uuid = Uuid::from_u128(0x32ce_7670_d303_4566_9cc3_123a_380b_efe9);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct BundledModelSource {
    repository: String,
    revision: String,
    url: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
struct BundledModelDefinition {
    schema_version: u32,
    model_id: Uuid,
    model_kind: LocalModelKind,
    input_format: String,
    variant: String,
    multilingual: bool,
    artifact_file_name: String,
    size_bytes: u64,
    sha256: String,
    version: String,
    model_card_id: String,
    license_id: String,
    license_confirmed_at: DateTime<Utc>,
    source: BundledModelSource,
}

/// Deliberately path-free state that can cross the WebView boundary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BundledAsrStatus {
    pub available: bool,
    pub model_id: Option<Uuid>,
    pub message: Option<String>,
}

impl BundledAsrStatus {
    pub(crate) fn ready() -> Self {
        Self {
            available: true,
            model_id: Some(BUNDLED_ASR_MODEL_ID),
            message: None,
        }
    }

    pub(crate) fn unavailable() -> Self {
        Self {
            available: false,
            model_id: None,
            message: Some("内置离线转写模型不可用，请重新安装应用".to_owned()),
        }
    }
}

#[derive(Debug)]
pub enum BundledModelError {
    EmbeddedManifest,
    ResourceRoot,
    ResourceManifest,
    ResourceManifestDoesNotMatch,
    InvalidDefinition,
    ArtifactMissing,
    ArtifactIsSymlink,
    ArtifactIsNotRegularFile,
    ArtifactSizeMismatch,
    ArtifactHashMismatch,
    Io,
}

impl fmt::Display for BundledModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmbeddedManifest => "the bundled model lock is invalid",
            Self::ResourceRoot => "the bundled model resource directory is invalid",
            Self::ResourceManifest => "the bundled model manifest is unavailable or invalid",
            Self::ResourceManifestDoesNotMatch => {
                "the bundled model manifest does not match the application lock"
            }
            Self::InvalidDefinition => "the bundled model definition is invalid",
            Self::ArtifactMissing => "the bundled model artifact is missing",
            Self::ArtifactIsSymlink => "the bundled model artifact must not be a symbolic link",
            Self::ArtifactIsNotRegularFile => "the bundled model artifact is not a regular file",
            Self::ArtifactSizeMismatch => "the bundled model artifact has an unexpected size",
            Self::ArtifactHashMismatch => {
                "the bundled model artifact failed integrity verification"
            }
            Self::Io => "the bundled model artifact could not be read",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for BundledModelError {}

/// Validates the app-packaged manifest and model bytes without retaining the
/// model buffer. Startup uses this to project only compact model metadata.
/// This never performs a network request or exposes a resource path over IPC.
pub(crate) fn verified_bundled_model_metadata(
    resource_dir: impl AsRef<Path>,
) -> Result<RegisteredModel, BundledModelError> {
    let (resource_root, definition) = packaged_model_definition(resource_dir.as_ref())?;
    verified_model_metadata_from_definition(&resource_root, &definition)
}

/// Reads the bundled model from one no-follow file descriptor, verifies the
/// bytes that were read, and returns those exact owned bytes to the local
/// Whisper adapter. The adapter must use its buffer API; it must not reopen a
/// resource path after this function returns.
#[cfg_attr(not(all(target_os = "macos", not(test))), allow(dead_code))]
pub(crate) fn verified_bundled_model(
    resource_dir: impl AsRef<Path>,
) -> Result<VerifiedModelArtifact, BundledModelError> {
    let (resource_root, definition) = packaged_model_definition(resource_dir.as_ref())?;
    verified_model_from_definition(&resource_root, &definition)
}

fn packaged_model_definition(
    resource_dir: &Path,
) -> Result<(PathBuf, BundledModelDefinition), BundledModelError> {
    let expected = embedded_definition()?;
    let resource_root = canonical_resource_root(resource_dir)?;
    let manifest_path =
        resolve_regular_resource_file(&resource_root, Path::new(BUNDLED_MANIFEST_RELATIVE_PATH))?;
    let manifest =
        fs::read_to_string(&manifest_path).map_err(|_| BundledModelError::ResourceManifest)?;
    let packaged: BundledModelDefinition =
        serde_json::from_str(&manifest).map_err(|_| BundledModelError::ResourceManifest)?;

    if packaged != expected {
        return Err(BundledModelError::ResourceManifestDoesNotMatch);
    }

    Ok((resource_root, expected))
}

fn embedded_definition() -> Result<BundledModelDefinition, BundledModelError> {
    let definition = serde_json::from_str(include_str!("../../../models/whisper-base.lock.json"))
        .map_err(|_| BundledModelError::EmbeddedManifest)?;
    validate_definition(&definition)?;
    Ok(definition)
}

fn verified_model_from_definition(
    resource_root: &Path,
    definition: &BundledModelDefinition,
) -> Result<VerifiedModelArtifact, BundledModelError> {
    let mut file = open_bundled_model_file(resource_root, definition)?;
    let (bytes, actual_size, actual_sha256) =
        read_model_bytes_with_sha256(&mut file, definition.size_bytes)?;
    verify_model_integrity(definition, actual_size, &actual_sha256)?;

    Ok(VerifiedModelArtifact::from_verified_native_bytes(
        registered_model_from_definition(definition)?,
        bytes,
    ))
}

fn verified_model_metadata_from_definition(
    resource_root: &Path,
    definition: &BundledModelDefinition,
) -> Result<RegisteredModel, BundledModelError> {
    let mut file = open_bundled_model_file(resource_root, definition)?;
    let (actual_size, actual_sha256) = read_with_sha256(&mut file)?;
    verify_model_integrity(definition, actual_size, &actual_sha256)?;

    registered_model_from_definition(definition)
}

fn verify_model_integrity(
    definition: &BundledModelDefinition,
    actual_size: u64,
    actual_sha256: &str,
) -> Result<(), BundledModelError> {
    if actual_size != definition.size_bytes {
        return Err(BundledModelError::ArtifactSizeMismatch);
    }
    if actual_sha256 != definition.sha256 {
        return Err(BundledModelError::ArtifactHashMismatch);
    }
    Ok(())
}

fn open_bundled_model_file(
    resource_root: &Path,
    definition: &BundledModelDefinition,
) -> Result<File, BundledModelError> {
    validate_definition(definition)?;
    let resource_root = canonical_resource_root(resource_root)?;
    let relative_path = Path::new(BUNDLED_RESOURCE_DIRECTORY).join(&definition.artifact_file_name);
    let artifact_path = resolve_regular_resource_file(&resource_root, &relative_path)?;
    let file = open_no_follow_regular_file(&artifact_path)?;
    let metadata = file.metadata().map_err(|_| BundledModelError::Io)?;

    // The descriptor, rather than a path preflight, is authoritative. A
    // replacement between the preflight and `open` still cannot reach Whisper:
    // this handle is rechecked and its bytes are hashed below.
    if !metadata.file_type().is_file() {
        return Err(BundledModelError::ArtifactIsNotRegularFile);
    }
    if metadata.len() != definition.size_bytes {
        return Err(BundledModelError::ArtifactSizeMismatch);
    }

    Ok(file)
}

fn registered_model_from_definition(
    definition: &BundledModelDefinition,
) -> Result<RegisteredModel, BundledModelError> {
    validate_definition(definition)?;
    Ok(RegisteredModel {
        id: definition.model_id,
        model_kind: definition.model_kind,
        // This marker is native-only and never resolved through ModelRegistry.
        // It remains relative so RegisteredModel cannot accidentally expose a
        // packaged resource path through serialization or audit payloads.
        file_path: PathBuf::from(format!("bundled-{}.model", definition.model_id)),
        file_size_bytes: definition.size_bytes,
        sha256: definition.sha256.clone(),
        version: definition.version.clone(),
        input_format: definition.input_format.clone(),
        model_card_id: definition.model_card_id.clone(),
        license_id: definition.license_id.clone(),
        license_confirmed_at: definition.license_confirmed_at,
        imported_at: definition.license_confirmed_at,
    })
}

fn validate_definition(definition: &BundledModelDefinition) -> Result<(), BundledModelError> {
    if definition.schema_version != 1
        || definition.model_id != BUNDLED_ASR_MODEL_ID
        || definition.model_kind != LocalModelKind::SpeechRecognition
        || definition.input_format != WHISPER_CPP_GGML_INPUT_FORMAT
        || definition.variant != "base"
        || !definition.multilingual
        || definition.artifact_file_name != "ggml-base.bin"
        || definition.size_bytes == 0
        || definition.sha256.len() != SHA256_HEX_LENGTH
        || !definition
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || definition.sha256 != definition.sha256.to_ascii_lowercase()
        || definition.version.trim().is_empty()
        || definition.model_card_id.trim().is_empty()
        || definition.license_id.trim().is_empty()
        || definition.source.repository.trim().is_empty()
        || definition.source.revision.trim().is_empty()
        || !definition.source.url.starts_with("https://")
    {
        return Err(BundledModelError::InvalidDefinition);
    }
    Ok(())
}

fn canonical_resource_root(root: &Path) -> Result<PathBuf, BundledModelError> {
    if !root.is_absolute() {
        return Err(BundledModelError::ResourceRoot);
    }
    let metadata = fs::symlink_metadata(root).map_err(|_| BundledModelError::ResourceRoot)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(BundledModelError::ResourceRoot);
    }
    fs::canonicalize(root).map_err(|_| BundledModelError::ResourceRoot)
}

fn resolve_regular_resource_file(
    resource_root: &Path,
    relative_path: &Path,
) -> Result<PathBuf, BundledModelError> {
    if relative_path.components().next().is_none()
        || relative_path.is_absolute()
        || !relative_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(BundledModelError::InvalidDefinition);
    }

    let candidate = resource_root.join(relative_path);
    let mut component_path = resource_root.to_path_buf();
    for component in relative_path.components() {
        let Component::Normal(component) = component else {
            unreachable!("resource path was checked above");
        };
        component_path.push(component);
        let metadata = match fs::symlink_metadata(&component_path) {
            Ok(metadata) => metadata,
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                return Err(BundledModelError::ArtifactMissing);
            }
            Err(_) => return Err(BundledModelError::Io),
        };
        if metadata.file_type().is_symlink() {
            return Err(BundledModelError::ArtifactIsSymlink);
        }
    }

    let metadata = fs::symlink_metadata(&candidate).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            BundledModelError::ArtifactMissing
        } else {
            BundledModelError::Io
        }
    })?;
    if !metadata.file_type().is_file() {
        return Err(BundledModelError::ArtifactIsNotRegularFile);
    }

    let canonical_candidate = fs::canonicalize(&candidate).map_err(|_| BundledModelError::Io)?;
    if !canonical_candidate.starts_with(resource_root) {
        return Err(BundledModelError::ArtifactIsSymlink);
    }
    Ok(canonical_candidate)
}

fn open_no_follow_regular_file(path: &Path) -> Result<File, BundledModelError> {
    #[cfg(unix)]
    {
        let file = OpenOptions::new()
            .read(true)
            // O_NONBLOCK prevents a swapped special file from stalling before
            // the descriptor's regular-file metadata check can fail closed.
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
            .map_err(|error| match error.raw_os_error() {
                Some(libc::ELOOP) => BundledModelError::ArtifactIsSymlink,
                _ if error.kind() == io::ErrorKind::NotFound => BundledModelError::ArtifactMissing,
                _ => BundledModelError::Io,
            })?;
        let metadata = file.metadata().map_err(|_| BundledModelError::Io)?;
        if !metadata.file_type().is_file() {
            return Err(BundledModelError::ArtifactIsNotRegularFile);
        }
        Ok(file)
    }

    #[cfg(not(unix))]
    {
        let _ = path;
        // This product ships the bundled runtime on macOS. Other targets fail
        // closed instead of silently opening a resource without no-follow
        // semantics.
        Err(BundledModelError::Io)
    }
}

fn read_with_sha256(file: &mut File) -> Result<(u64, String), BundledModelError> {
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut size = 0_u64;

    loop {
        let read = file.read(&mut buffer).map_err(|_| BundledModelError::Io)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size = size
            .checked_add(u64::try_from(read).expect("read size fits in u64"))
            .ok_or(BundledModelError::Io)?;
    }

    Ok((size, hex_digest(&hasher.finalize())))
}

fn read_model_bytes_with_sha256(
    file: &mut File,
    expected_size: u64,
) -> Result<(Box<[u8]>, u64, String), BundledModelError> {
    let capacity =
        usize::try_from(expected_size).map_err(|_| BundledModelError::ArtifactSizeMismatch)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| BundledModelError::Io)?;

    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut size = 0_u64;
    loop {
        let read = file.read(&mut buffer).map_err(|_| BundledModelError::Io)?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(u64::try_from(read).expect("read size fits in u64"))
            .ok_or(BundledModelError::Io)?;
        if size > expected_size {
            return Err(BundledModelError::ArtifactSizeMismatch);
        }
        hasher.update(&buffer[..read]);
        bytes.extend_from_slice(&buffer[..read]);
    }

    if size != expected_size {
        return Err(BundledModelError::ArtifactSizeMismatch);
    }

    Ok((
        bytes.into_boxed_slice(),
        size,
        hex_digest(&hasher.finalize()),
    ))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to a string cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("word-covenant-bundled-model-{}", Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn definition(bytes: &[u8]) -> BundledModelDefinition {
        BundledModelDefinition {
            schema_version: 1,
            model_id: BUNDLED_ASR_MODEL_ID,
            model_kind: LocalModelKind::SpeechRecognition,
            input_format: WHISPER_CPP_GGML_INPUT_FORMAT.to_owned(),
            variant: "base".to_owned(),
            multilingual: true,
            artifact_file_name: "ggml-base.bin".to_owned(),
            size_bytes: bytes.len() as u64,
            sha256: hex_digest(&Sha256::digest(bytes)),
            version: "fixture-base".to_owned(),
            model_card_id: "ggerganov/whisper.cpp".to_owned(),
            license_id: "MIT".to_owned(),
            license_confirmed_at: DateTime::<Utc>::UNIX_EPOCH,
            source: BundledModelSource {
                repository: "ggerganov/whisper.cpp".to_owned(),
                revision: "fixture".to_owned(),
                url: "https://example.invalid/model.bin".to_owned(),
            },
        }
    }

    #[test]
    fn verifies_a_regular_manifest_defined_resource() {
        let directory = TestDirectory::new();
        let bytes = b"bundled fixture model";
        let definition = definition(bytes);
        let models = directory.path.join(BUNDLED_RESOURCE_DIRECTORY);
        fs::create_dir_all(&models).unwrap();
        fs::write(models.join(&definition.artifact_file_name), bytes).unwrap();

        let artifact = verified_model_from_definition(&directory.path, &definition).unwrap();

        assert_eq!(artifact.model().id, BUNDLED_ASR_MODEL_ID);
        assert_eq!(artifact.model().sha256, definition.sha256);
        assert_eq!(artifact.model().file_size_bytes, bytes.len() as u64);
        assert_eq!(&*artifact.into_verified_bytes(), bytes);
    }

    #[test]
    fn retains_verified_bytes_after_the_resource_path_is_replaced() {
        let directory = TestDirectory::new();
        let bytes = b"bundled fixture model";
        let definition = definition(bytes);
        let models = directory.path.join(BUNDLED_RESOURCE_DIRECTORY);
        let model_path = models.join(&definition.artifact_file_name);
        fs::create_dir_all(&models).unwrap();
        fs::write(&model_path, bytes).unwrap();

        let artifact = verified_model_from_definition(&directory.path, &definition).unwrap();
        fs::write(&model_path, vec![b'x'; bytes.len()]).unwrap();

        assert_eq!(&*artifact.into_verified_bytes(), bytes);
    }

    #[test]
    fn rejects_tampered_bytes_before_issuing_an_artifact() {
        let directory = TestDirectory::new();
        let definition = definition(b"expected bundled bytes");
        let models = directory.path.join(BUNDLED_RESOURCE_DIRECTORY);
        fs::create_dir_all(&models).unwrap();
        fs::write(
            models.join(&definition.artifact_file_name),
            b"substituted bytes",
        )
        .unwrap();

        assert!(matches!(
            verified_model_from_definition(&directory.path, &definition),
            Err(BundledModelError::ArtifactSizeMismatch | BundledModelError::ArtifactHashMismatch)
        ));
    }

    #[test]
    fn rejects_unsafe_or_incompatible_manifest_metadata() {
        let bytes = b"fixture";
        let mut bundled_definition = definition(bytes);
        bundled_definition.multilingual = false;
        assert!(matches!(
            registered_model_from_definition(&bundled_definition),
            Err(BundledModelError::InvalidDefinition)
        ));

        bundled_definition = definition(bytes);
        bundled_definition.artifact_file_name = "../ggml-base.bin".to_owned();
        assert!(matches!(
            registered_model_from_definition(&bundled_definition),
            Err(BundledModelError::InvalidDefinition)
        ));
    }

    #[test]
    fn compiles_the_reviewed_default_model_lock() {
        let definition = embedded_definition().unwrap();

        assert_eq!(definition.model_id, BUNDLED_ASR_MODEL_ID);
        assert_eq!(definition.artifact_file_name, "ggml-base.bin");
        assert_eq!(definition.size_bytes, 147_951_465);
        assert_eq!(
            definition.sha256,
            "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_symlinked_resource_artifact() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let bytes = b"fixture";
        let definition = definition(bytes);
        let models = directory.path.join(BUNDLED_RESOURCE_DIRECTORY);
        let outside = directory.path.join("outside.bin");
        fs::create_dir_all(&models).unwrap();
        fs::write(&outside, bytes).unwrap();
        symlink(&outside, models.join(&definition.artifact_file_name)).unwrap();

        assert!(matches!(
            verified_model_from_definition(&directory.path, &definition),
            Err(BundledModelError::ArtifactIsSymlink)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn no_follow_open_rejects_a_symlink_even_without_a_path_preflight() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let outside = directory.path.join("outside.bin");
        let link = directory.path.join("model.bin");
        fs::write(&outside, b"fixture").unwrap();
        symlink(&outside, &link).unwrap();

        assert!(matches!(
            open_no_follow_regular_file(&link),
            Err(BundledModelError::ArtifactIsSymlink)
        ));
    }
}
