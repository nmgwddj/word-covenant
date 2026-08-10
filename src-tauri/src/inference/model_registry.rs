//! Local-only registration of user-imported inference model files.
//!
//! This module deliberately has no network client, URL handling, or download
//! path. A caller must supply a local regular file, an expected SHA-256 digest,
//! and an explicit acknowledgement of the model card and license.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

const COPY_BUFFER_BYTES: usize = 64 * 1024;
const SHA256_HEX_LENGTH: usize = 64;
const MANAGED_MODEL_EXTENSION: &str = "model";

/// The local inference role served by a model file.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalModelKind {
    SpeechRecognition,
    VoiceActivityDetection,
    SpeakerEmbedding,
}

/// Evidence that the user reviewed a model card and explicitly accepted its
/// license before the model can be registered.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LicenseAcknowledgement {
    pub model_card_id: String,
    pub license_id: String,
    pub confirmed_at: DateTime<Utc>,
}

impl LicenseAcknowledgement {
    pub fn new(
        model_card_id: impl Into<String>,
        license_id: impl Into<String>,
        confirmed_at: DateTime<Utc>,
    ) -> Result<Self, ModelRegistryError> {
        let acknowledgement = Self {
            model_card_id: model_card_id.into(),
            license_id: license_id.into(),
            confirmed_at,
        };
        acknowledgement.validate()?;
        Ok(acknowledgement)
    }

    fn validate(&self) -> Result<(), ModelRegistryError> {
        require_non_empty("model card identifier", &self.model_card_id)?;
        require_non_empty("license identifier", &self.license_id)
    }
}

/// A user-initiated request to copy a local model file into application-owned
/// storage. `expected_sha256` is intentionally required so import verifies the
/// copied bytes against a digest obtained from the model's trusted metadata.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelImportRequest {
    pub id: Uuid,
    pub source_path: PathBuf,
    pub model_kind: LocalModelKind,
    pub version: String,
    pub input_format: String,
    pub expected_sha256: String,
    pub license_acknowledgement: Option<LicenseAcknowledgement>,
}

/// Immutable metadata retained after a successful local import.
///
/// `file_path` is a safe relative path below the application-managed model
/// root. It is deliberately omitted from every Serde representation so a
/// native filesystem path can never cross the WebView boundary or enter an
/// audit payload. The source file remains untouched.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisteredModel {
    pub id: Uuid,
    pub model_kind: LocalModelKind,
    #[serde(skip)]
    pub file_path: PathBuf,
    pub file_size_bytes: u64,
    pub sha256: String,
    pub version: String,
    pub input_format: String,
    pub model_card_id: String,
    pub license_id: String,
    pub license_confirmed_at: DateTime<Utc>,
    pub imported_at: DateTime<Utc>,
}

/// A model artifact that the native registry has just re-opened and verified.
///
/// This type intentionally has no Serde implementation and keeps its absolute
/// managed path private. Native inference adapters consume it immediately;
/// filesystem paths must never reach Tauri IPC, audit payloads, or the WebView.
#[derive(Debug)]
pub struct VerifiedModelArtifact {
    model: RegisteredModel,
    path: PathBuf,
}

impl VerifiedModelArtifact {
    pub fn model(&self) -> &RegisteredModel {
        &self.model
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// An in-memory index of locally imported models.
///
/// Persistence is intentionally owned by a higher layer. This keeps the file
/// import boundary testable and lets persistence restore only registrations
/// that have already passed the same policy checks.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    /// Canonical application-managed root. This is configured only after it
    /// has passed the native filesystem safety checks below.
    managed_root: Option<PathBuf>,
    by_id: BTreeMap<Uuid, RegisteredModel>,
    id_by_sha256: BTreeMap<String, Uuid>,
    id_by_path: BTreeMap<PathBuf, Uuid>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild a registry from model metadata already persisted by the local
    /// application store.
    ///
    /// Every artifact is reopened and verified from the supplied managed root
    /// before its metadata can be indexed. This performs no network activity.
    /// A malformed, replaced, or duplicate record leaves no partially
    /// restored registry behind.
    pub fn from_persisted(
        app_managed_root: impl AsRef<Path>,
        models: impl IntoIterator<Item = RegisteredModel>,
    ) -> Result<Self, ModelRegistryError> {
        let models = models.into_iter().collect::<Vec<_>>();
        if models.is_empty() {
            return Ok(Self::new());
        }

        let mut registry = Self::new();
        let managed_root = resolve_existing_managed_root(app_managed_root.as_ref())?;
        registry.configure_managed_root(managed_root)?;
        for model in models {
            registry.register_persisted(model)?;
        }
        Ok(registry)
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn get(&self, id: Uuid) -> Option<&RegisteredModel> {
        self.by_id.get(&id)
    }

    /// Resolves one registered artifact for a native inference caller only.
    /// The byte size and SHA-256 are checked immediately before this path is
    /// returned, so a file replaced after application startup is detected.
    /// A future worker must retain and hash a no-follow file handle through
    /// loading rather than reopen this path, closing the remaining TOCTOU
    /// window. This absolute path is never serializable or WebView-facing.
    pub fn verified_artifact_path(&self, id: Uuid) -> Result<PathBuf, ModelRegistryError> {
        Ok(self.verified_artifact(id)?.path)
    }

    /// Resolves a registered artifact into a native-only capability after
    /// verifying its managed location, size, and SHA-256 immediately before
    /// loading. The capability deliberately cannot be serialized or created
    /// from an arbitrary path.
    pub fn verified_artifact(&self, id: Uuid) -> Result<VerifiedModelArtifact, ModelRegistryError> {
        let model = self
            .get(id)
            .ok_or(ModelRegistryError::UnknownModelId { id })?;
        let managed_root = self
            .managed_root
            .as_deref()
            .ok_or(ModelRegistryError::ManagedRootNotConfigured)?;
        verify_persisted_artifact(managed_root, model)?;
        let path = resolve_persisted_artifact(managed_root, &model.file_path)?;
        Ok(VerifiedModelArtifact {
            model: model.clone(),
            path,
        })
    }

    pub fn models(&self) -> impl Iterator<Item = &RegisteredModel> {
        self.by_id.values()
    }

    /// Register metadata that was already persisted after a prior local
    /// import.
    ///
    /// The managed model root must have been configured by a successful import
    /// or [`Self::from_persisted`]. The artifact's existence, type, size, and
    /// digest are verified before metadata is indexed. The digest is
    /// normalized to lowercase so duplicate detection remains
    /// case-insensitive for hexadecimal SHA-256 values.
    pub fn register_persisted(
        &mut self,
        model: RegisteredModel,
    ) -> Result<RegisteredModel, ModelRegistryError> {
        let model = validate_persisted_model(model)?;
        let managed_root = self
            .managed_root
            .as_deref()
            .ok_or(ModelRegistryError::ManagedRootNotConfigured)?;
        verify_persisted_artifact(managed_root, &model)?;
        self.insert_model(model.clone())?;
        Ok(model)
    }

    /// Undo a successful in-memory registration when its SQLite/audit write
    /// cannot be committed.
    ///
    /// This only removes entries from the registry's maps. In particular, it
    /// never deletes the application-managed model artifact; an explicit,
    /// separately-audited removal flow owns that responsibility.
    pub fn rollback_registration(&mut self, id: Uuid) -> Option<RegisteredModel> {
        let model = self.by_id.remove(&id)?;

        if self.id_by_sha256.get(&model.sha256).copied() == Some(id) {
            self.id_by_sha256.remove(&model.sha256);
        }
        if self.id_by_path.get(&model.file_path).copied() == Some(id) {
            self.id_by_path.remove(&model.file_path);
        }

        Some(model)
    }

    /// Removes a model artifact after its in-memory registration was rolled
    /// back. The relative path is resolved and checked only inside this native
    /// module; callers never receive an absolute managed path.
    pub fn remove_managed_artifact(
        &self,
        model: &RegisteredModel,
    ) -> Result<(), ModelRegistryError> {
        let managed_root = self
            .managed_root
            .as_deref()
            .ok_or(ModelRegistryError::ManagedRootNotConfigured)?;
        validate_registered_path(&model.file_path)?;
        let artifact_path = resolve_persisted_artifact(managed_root, &model.file_path)?;
        fs::remove_file(&artifact_path).map_err(|source| ModelRegistryError::Io {
            action: "remove managed model artifact",
            path: artifact_path,
            source,
        })
    }

    /// Copy and register a local model using the current wall clock for the
    /// import timestamp. This function performs no network activity.
    pub fn import(
        &mut self,
        app_managed_root: impl AsRef<Path>,
        request: ModelImportRequest,
    ) -> Result<RegisteredModel, ModelRegistryError> {
        self.import_at(app_managed_root, request, Utc::now())
    }

    /// Copy and register a local model at a caller-supplied timestamp.
    ///
    /// The explicit timestamp makes import policy deterministic under test;
    /// production callers should normally use [`Self::import`].
    pub fn import_at(
        &mut self,
        app_managed_root: impl AsRef<Path>,
        request: ModelImportRequest,
        imported_at: DateTime<Utc>,
    ) -> Result<RegisteredModel, ModelRegistryError> {
        validate_request(&request)?;

        if self.by_id.contains_key(&request.id) {
            return Err(ModelRegistryError::DuplicateModelId { id: request.id });
        }

        let expected_sha256 = normalize_sha256(&request.expected_sha256)?;
        ensure_regular_nonempty_source(&request.source_path)?;
        let managed_root = prepare_managed_root(app_managed_root.as_ref())?;
        self.configure_managed_root(managed_root.clone())?;
        let managed_relative_path = managed_model_relative_path(request.id);
        let target_path = managed_model_path(&managed_root, &managed_relative_path)?;

        if let Some(existing_model_id) = self.id_by_path.get(&managed_relative_path) {
            return Err(ModelRegistryError::DuplicateManagedPath {
                path: managed_relative_path,
                existing_model_id: *existing_model_id,
            });
        }
        ensure_target_is_new(&target_path)?;

        let (file_size_bytes, actual_sha256) =
            copy_with_sha256(&request.source_path, &target_path)?;

        if actual_sha256 != expected_sha256 {
            remove_failed_import(&target_path);
            return Err(ModelRegistryError::HashMismatch {
                expected: expected_sha256,
                actual: actual_sha256,
            });
        }

        if let Some(existing_model_id) = self.id_by_sha256.get(&actual_sha256) {
            remove_failed_import(&target_path);
            return Err(ModelRegistryError::DuplicateSha256 {
                sha256: actual_sha256,
                existing_model_id: *existing_model_id,
            });
        }

        let Some(acknowledgement) = request.license_acknowledgement else {
            return Err(ModelRegistryError::LicenseAcknowledgementRequired);
        };
        let registration = RegisteredModel {
            id: request.id,
            model_kind: request.model_kind,
            file_path: managed_relative_path,
            file_size_bytes,
            sha256: actual_sha256.clone(),
            version: request.version,
            input_format: request.input_format,
            model_card_id: acknowledgement.model_card_id,
            license_id: acknowledgement.license_id,
            license_confirmed_at: acknowledgement.confirmed_at,
            imported_at,
        };

        match self.register_persisted(registration) {
            Ok(registration) => Ok(registration),
            Err(error) => {
                remove_failed_import(&target_path);
                Err(error)
            }
        }
    }

    fn insert_model(&mut self, model: RegisteredModel) -> Result<(), ModelRegistryError> {
        if self.by_id.contains_key(&model.id) {
            return Err(ModelRegistryError::DuplicateModelId { id: model.id });
        }
        if let Some(existing_model_id) = self.id_by_path.get(&model.file_path) {
            return Err(ModelRegistryError::DuplicateManagedPath {
                path: model.file_path,
                existing_model_id: *existing_model_id,
            });
        }
        if let Some(existing_model_id) = self.id_by_sha256.get(&model.sha256) {
            return Err(ModelRegistryError::DuplicateSha256 {
                sha256: model.sha256,
                existing_model_id: *existing_model_id,
            });
        }

        self.id_by_sha256.insert(model.sha256.clone(), model.id);
        self.id_by_path.insert(model.file_path.clone(), model.id);
        self.by_id.insert(model.id, model);
        Ok(())
    }

    fn configure_managed_root(&mut self, managed_root: PathBuf) -> Result<(), ModelRegistryError> {
        if let Some(existing_root) = &self.managed_root {
            if existing_root != &managed_root {
                return Err(ModelRegistryError::ManagedRootChanged {
                    existing: existing_root.clone(),
                    requested: managed_root,
                });
            }
            return Ok(());
        }

        self.managed_root = Some(managed_root);
        Ok(())
    }
}

#[derive(Debug)]
pub enum ModelRegistryError {
    LicenseAcknowledgementRequired,
    EmptyMetadata {
        field: &'static str,
    },
    InvalidExpectedSha256 {
        value: String,
    },
    InvalidRegisteredSha256 {
        value: String,
    },
    RegisteredFileSizeMustBeNonZero {
        path: PathBuf,
    },
    RegisteredPathMustBeRelative {
        path: PathBuf,
    },
    UnsafeRegisteredPath {
        path: PathBuf,
    },
    SourcePathMustBeAbsolute {
        path: PathBuf,
    },
    SourceNotFound {
        path: PathBuf,
    },
    SourceNotRegularFile {
        path: PathBuf,
    },
    SourceEmpty {
        path: PathBuf,
    },
    ManagedRootMustBeAbsolute {
        path: PathBuf,
    },
    UnsafeManagedRoot {
        path: PathBuf,
    },
    ManagedRootIsSymlink {
        path: PathBuf,
    },
    ManagedRootIsNotDirectory {
        path: PathBuf,
    },
    ManagedRootNotFound {
        path: PathBuf,
    },
    ManagedRootNotConfigured,
    ManagedRootChanged {
        existing: PathBuf,
        requested: PathBuf,
    },
    TargetEscapesManagedRoot {
        root: PathBuf,
        target: PathBuf,
    },
    HashMismatch {
        expected: String,
        actual: String,
    },
    TargetPathAlreadyExists {
        path: PathBuf,
    },
    DuplicateModelId {
        id: Uuid,
    },
    UnknownModelId {
        id: Uuid,
    },
    DuplicateManagedPath {
        path: PathBuf,
        existing_model_id: Uuid,
    },
    DuplicateSha256 {
        sha256: String,
        existing_model_id: Uuid,
    },
    FileSizeOverflow {
        path: PathBuf,
    },
    RegisteredArtifactNotFound {
        path: PathBuf,
    },
    RegisteredArtifactIsSymlink {
        path: PathBuf,
    },
    RegisteredArtifactNotRegularFile {
        path: PathBuf,
    },
    RegisteredArtifactEscapesManagedRoot {
        root: PathBuf,
        path: PathBuf,
    },
    RegisteredArtifactSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    RegisteredArtifactHashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
}

impl fmt::Display for ModelRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LicenseAcknowledgementRequired => {
                formatter.write_str("a model card and license acknowledgement is required")
            }
            Self::EmptyMetadata { field } => write!(formatter, "{field} must not be empty"),
            Self::InvalidExpectedSha256 { value } => {
                write!(
                    formatter,
                    "expected SHA-256 must be 64 hexadecimal characters: {value:?}"
                )
            }
            Self::InvalidRegisteredSha256 { value } => {
                write!(
                    formatter,
                    "registered SHA-256 must be 64 hexadecimal characters: {value:?}"
                )
            }
            Self::RegisteredFileSizeMustBeNonZero { path } => write!(
                formatter,
                "registered model file size must be non-zero: {}",
                path.display()
            ),
            Self::RegisteredPathMustBeRelative { path } => write!(
                formatter,
                "registered model file path must be relative to the managed model root: {}",
                path.display()
            ),
            Self::UnsafeRegisteredPath { path } => write!(
                formatter,
                "registered model file path must not contain traversal or current-directory components: {}",
                path.display()
            ),
            Self::SourcePathMustBeAbsolute { path } => write!(
                formatter,
                "model source file path must be absolute: {}",
                path.display()
            ),
            Self::SourceNotFound { path } => {
                write!(
                    formatter,
                    "model source file does not exist: {}",
                    path.display()
                )
            }
            Self::SourceNotRegularFile { path } => {
                write!(
                    formatter,
                    "model source must be a regular file: {}",
                    path.display()
                )
            }
            Self::SourceEmpty { path } => {
                write!(
                    formatter,
                    "model source file must not be empty: {}",
                    path.display()
                )
            }
            Self::ManagedRootMustBeAbsolute { path } => write!(
                formatter,
                "application-managed model root must be an absolute path: {}",
                path.display()
            ),
            Self::UnsafeManagedRoot { path } => write!(
                formatter,
                "application-managed model root must not contain parent traversal: {}",
                path.display()
            ),
            Self::ManagedRootIsSymlink { path } => write!(
                formatter,
                "application-managed model root must not be a symbolic link: {}",
                path.display()
            ),
            Self::ManagedRootIsNotDirectory { path } => write!(
                formatter,
                "application-managed model root is not a directory: {}",
                path.display()
            ),
            Self::ManagedRootNotFound { path } => write!(
                formatter,
                "application-managed model root does not exist: {}",
                path.display()
            ),
            Self::ManagedRootNotConfigured => {
                formatter.write_str("application-managed model root is not configured")
            }
            Self::ManagedRootChanged {
                existing,
                requested,
            } => write!(
                formatter,
                "application-managed model root cannot change from {} to {}",
                existing.display(),
                requested.display()
            ),
            Self::TargetEscapesManagedRoot { root, target } => write!(
                formatter,
                "managed model target {} escapes root {}",
                target.display(),
                root.display()
            ),
            Self::HashMismatch { expected, actual } => write!(
                formatter,
                "model SHA-256 does not match expected digest: expected {expected}, got {actual}"
            ),
            Self::TargetPathAlreadyExists { path } => write!(
                formatter,
                "managed model target already exists: {}",
                path.display()
            ),
            Self::DuplicateModelId { id } => {
                write!(formatter, "model identifier is already registered: {id}")
            }
            Self::UnknownModelId { id } => {
                write!(formatter, "model identifier is not registered: {id}")
            }
            Self::DuplicateManagedPath {
                path,
                existing_model_id,
            } => write!(
                formatter,
                "managed model path {} is already registered by {existing_model_id}",
                path.display()
            ),
            Self::DuplicateSha256 {
                sha256,
                existing_model_id,
            } => write!(
                formatter,
                "SHA-256 {sha256} is already registered by {existing_model_id}"
            ),
            Self::FileSizeOverflow { path } => {
                write!(
                    formatter,
                    "model file is too large: {}",
                    path.display()
                )
            }
            Self::RegisteredArtifactNotFound { path } => write!(
                formatter,
                "registered managed model artifact does not exist: {}",
                path.display()
            ),
            Self::RegisteredArtifactIsSymlink { path } => write!(
                formatter,
                "registered managed model artifact must not traverse a symbolic link: {}",
                path.display()
            ),
            Self::RegisteredArtifactNotRegularFile { path } => write!(
                formatter,
                "registered managed model artifact is not a regular file: {}",
                path.display()
            ),
            Self::RegisteredArtifactEscapesManagedRoot { root, path } => write!(
                formatter,
                "registered managed model artifact {} resolves outside root {}",
                path.display(),
                root.display()
            ),
            Self::RegisteredArtifactSizeMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "registered managed model artifact size does not match for {}: expected {expected}, got {actual}",
                path.display()
            ),
            Self::RegisteredArtifactHashMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "registered managed model artifact SHA-256 does not match for {}: expected {expected}, got {actual}",
                path.display()
            ),
            Self::Io {
                action,
                path,
                source,
            } => write!(formatter, "failed to {action} {}: {source}", path.display()),
        }
    }
}

impl std::error::Error for ModelRegistryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn validate_request(request: &ModelImportRequest) -> Result<(), ModelRegistryError> {
    require_non_empty("model version", &request.version)?;
    require_non_empty("model input format", &request.input_format)?;
    request
        .license_acknowledgement
        .as_ref()
        .ok_or(ModelRegistryError::LicenseAcknowledgementRequired)?
        .validate()
}

fn validate_persisted_model(
    mut model: RegisteredModel,
) -> Result<RegisteredModel, ModelRegistryError> {
    require_non_empty("model version", &model.version)?;
    require_non_empty("model input format", &model.input_format)?;
    require_non_empty("model card identifier", &model.model_card_id)?;
    require_non_empty("license identifier", &model.license_id)?;

    if model.file_size_bytes == 0 {
        return Err(ModelRegistryError::RegisteredFileSizeMustBeNonZero {
            path: model.file_path,
        });
    }
    validate_registered_path(&model.file_path)?;
    model.sha256 = normalize_registered_sha256(&model.sha256)?;
    Ok(model)
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), ModelRegistryError> {
    if value.trim().is_empty() {
        return Err(ModelRegistryError::EmptyMetadata { field });
    }
    Ok(())
}

fn normalize_sha256(value: &str) -> Result<String, ModelRegistryError> {
    if value.len() != SHA256_HEX_LENGTH || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ModelRegistryError::InvalidExpectedSha256 {
            value: value.to_owned(),
        });
    }
    Ok(value.to_ascii_lowercase())
}

fn normalize_registered_sha256(value: &str) -> Result<String, ModelRegistryError> {
    if value.len() != SHA256_HEX_LENGTH || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ModelRegistryError::InvalidRegisteredSha256 {
            value: value.to_owned(),
        });
    }
    Ok(value.to_ascii_lowercase())
}

fn validate_registered_path(path: &Path) -> Result<(), ModelRegistryError> {
    if path.is_absolute() {
        return Err(ModelRegistryError::RegisteredPathMustBeRelative {
            path: path.to_path_buf(),
        });
    }
    let mut components = path.components();
    if components.next().is_none()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(ModelRegistryError::UnsafeRegisteredPath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn prepare_managed_root(root: &Path) -> Result<PathBuf, ModelRegistryError> {
    validate_managed_root(root)?;

    match fs::symlink_metadata(root) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(ModelRegistryError::ManagedRootIsSymlink {
                path: root.to_path_buf(),
            });
        }
        Ok(metadata) if !metadata.file_type().is_dir() => {
            return Err(ModelRegistryError::ManagedRootIsNotDirectory {
                path: root.to_path_buf(),
            });
        }
        Ok(_) => {}
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(root).map_err(|source| ModelRegistryError::Io {
                action: "create application-managed model directory",
                path: root.to_path_buf(),
                source,
            })?;
        }
        Err(source) => {
            return Err(ModelRegistryError::Io {
                action: "inspect application-managed model directory",
                path: root.to_path_buf(),
                source,
            });
        }
    }

    canonical_managed_root(root)
}

fn resolve_existing_managed_root(root: &Path) -> Result<PathBuf, ModelRegistryError> {
    validate_managed_root(root)?;

    match fs::symlink_metadata(root) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(ModelRegistryError::ManagedRootIsSymlink {
                path: root.to_path_buf(),
            });
        }
        Ok(metadata) if !metadata.file_type().is_dir() => {
            return Err(ModelRegistryError::ManagedRootIsNotDirectory {
                path: root.to_path_buf(),
            });
        }
        Ok(_) => canonical_managed_root(root),
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            Err(ModelRegistryError::ManagedRootNotFound {
                path: root.to_path_buf(),
            })
        }
        Err(source) => Err(ModelRegistryError::Io {
            action: "inspect application-managed model directory",
            path: root.to_path_buf(),
            source,
        }),
    }
}

fn validate_managed_root(root: &Path) -> Result<(), ModelRegistryError> {
    if !root.is_absolute() {
        return Err(ModelRegistryError::ManagedRootMustBeAbsolute {
            path: root.to_path_buf(),
        });
    }
    if root
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(ModelRegistryError::UnsafeManagedRoot {
            path: root.to_path_buf(),
        });
    }
    Ok(())
}

fn canonical_managed_root(root: &Path) -> Result<PathBuf, ModelRegistryError> {
    let metadata = fs::symlink_metadata(root).map_err(|source| ModelRegistryError::Io {
        action: "inspect application-managed model directory",
        path: root.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() {
        return Err(ModelRegistryError::ManagedRootIsSymlink {
            path: root.to_path_buf(),
        });
    }
    if !metadata.file_type().is_dir() {
        return Err(ModelRegistryError::ManagedRootIsNotDirectory {
            path: root.to_path_buf(),
        });
    }

    fs::canonicalize(root).map_err(|source| ModelRegistryError::Io {
        action: "resolve application-managed model directory",
        path: root.to_path_buf(),
        source,
    })
}

fn managed_model_relative_path(model_id: Uuid) -> PathBuf {
    PathBuf::from(format!("{model_id}.{MANAGED_MODEL_EXTENSION}"))
}

fn managed_model_path(root: &Path, relative_path: &Path) -> Result<PathBuf, ModelRegistryError> {
    validate_registered_path(relative_path)?;
    let target = root.join(relative_path);
    if target.parent() != Some(root) || !target.starts_with(root) {
        return Err(ModelRegistryError::TargetEscapesManagedRoot {
            root: root.to_path_buf(),
            target,
        });
    }
    Ok(target)
}

fn resolve_persisted_artifact(
    managed_root: &Path,
    relative_path: &Path,
) -> Result<PathBuf, ModelRegistryError> {
    validate_registered_path(relative_path)?;
    let candidate = managed_root.join(relative_path);
    let canonical_path = match fs::canonicalize(&candidate) {
        Ok(path) => path,
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            return Err(ModelRegistryError::RegisteredArtifactNotFound {
                path: relative_path.to_path_buf(),
            });
        }
        Err(source) => {
            return Err(ModelRegistryError::Io {
                action: "resolve registered managed model artifact",
                path: candidate,
                source,
            });
        }
    };

    if !canonical_path.starts_with(managed_root) {
        return Err(ModelRegistryError::RegisteredArtifactEscapesManagedRoot {
            root: managed_root.to_path_buf(),
            path: relative_path.to_path_buf(),
        });
    }

    let mut component_path = managed_root.to_path_buf();
    for component in relative_path.components() {
        let Component::Normal(component) = component else {
            unreachable!("registered path was validated as normal components");
        };
        component_path.push(component);
        let metadata =
            fs::symlink_metadata(&component_path).map_err(|source| ModelRegistryError::Io {
                action: "inspect registered managed model artifact",
                path: component_path.clone(),
                source,
            })?;
        if metadata.file_type().is_symlink() {
            return Err(ModelRegistryError::RegisteredArtifactIsSymlink {
                path: relative_path.to_path_buf(),
            });
        }
    }

    let metadata = fs::symlink_metadata(&candidate).map_err(|source| ModelRegistryError::Io {
        action: "inspect registered managed model artifact",
        path: candidate,
        source,
    })?;
    if !metadata.file_type().is_file() {
        return Err(ModelRegistryError::RegisteredArtifactNotRegularFile {
            path: relative_path.to_path_buf(),
        });
    }

    Ok(canonical_path)
}

fn verify_persisted_artifact(
    managed_root: &Path,
    model: &RegisteredModel,
) -> Result<(), ModelRegistryError> {
    let artifact_path = resolve_persisted_artifact(managed_root, &model.file_path)?;
    let (actual_size, actual_sha256) = read_with_sha256(&artifact_path)?;

    if actual_size != model.file_size_bytes {
        return Err(ModelRegistryError::RegisteredArtifactSizeMismatch {
            path: model.file_path.clone(),
            expected: model.file_size_bytes,
            actual: actual_size,
        });
    }
    if actual_sha256 != model.sha256 {
        return Err(ModelRegistryError::RegisteredArtifactHashMismatch {
            path: model.file_path.clone(),
            expected: model.sha256.clone(),
            actual: actual_sha256,
        });
    }

    Ok(())
}

fn ensure_target_is_new(target: &Path) -> Result<(), ModelRegistryError> {
    match fs::symlink_metadata(target) {
        Ok(_) => Err(ModelRegistryError::TargetPathAlreadyExists {
            path: target.to_path_buf(),
        }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(ModelRegistryError::Io {
            action: "inspect managed model target",
            path: target.to_path_buf(),
            source,
        }),
    }
}

fn ensure_regular_nonempty_source(source_path: &Path) -> Result<(), ModelRegistryError> {
    if !source_path.is_absolute() {
        return Err(ModelRegistryError::SourcePathMustBeAbsolute {
            path: source_path.to_path_buf(),
        });
    }

    let metadata = match fs::symlink_metadata(source_path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            return Err(ModelRegistryError::SourceNotFound {
                path: source_path.to_path_buf(),
            });
        }
        Err(source) => {
            return Err(ModelRegistryError::Io {
                action: "inspect model source file",
                path: source_path.to_path_buf(),
                source,
            });
        }
    };

    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(ModelRegistryError::SourceNotRegularFile {
            path: source_path.to_path_buf(),
        });
    }
    if metadata.len() == 0 {
        return Err(ModelRegistryError::SourceEmpty {
            path: source_path.to_path_buf(),
        });
    }
    Ok(())
}

fn copy_with_sha256(
    source_path: &Path,
    target_path: &Path,
) -> Result<(u64, String), ModelRegistryError> {
    let mut source = File::open(source_path).map_err(|source| ModelRegistryError::Io {
        action: "open model source file",
        path: source_path.to_path_buf(),
        source,
    })?;
    let mut target = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target_path)
    {
        Ok(target) => target,
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            return Err(ModelRegistryError::TargetPathAlreadyExists {
                path: target_path.to_path_buf(),
            });
        }
        Err(source) => {
            return Err(ModelRegistryError::Io {
                action: "create managed model target",
                path: target_path.to_path_buf(),
                source,
            });
        }
    };

    let (file_size_bytes, sha256) =
        match stream_copy_with_sha256(&mut source, &mut target, source_path, target_path) {
            Ok(result) => result,
            Err(error) => {
                drop(target);
                remove_failed_import(target_path);
                return Err(error);
            }
        };

    if file_size_bytes == 0 {
        drop(target);
        remove_failed_import(target_path);
        return Err(ModelRegistryError::SourceEmpty {
            path: source_path.to_path_buf(),
        });
    }

    if let Err(source) = target.sync_all() {
        drop(target);
        remove_failed_import(target_path);
        return Err(ModelRegistryError::Io {
            action: "sync managed model target",
            path: target_path.to_path_buf(),
            source,
        });
    }

    Ok((file_size_bytes, sha256))
}

fn read_with_sha256(path: &Path) -> Result<(u64, String), ModelRegistryError> {
    let mut file = File::open(path).map_err(|source| ModelRegistryError::Io {
        action: "open registered managed model artifact",
        path: path.to_path_buf(),
        source,
    })?;
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut file_size_bytes = 0_u64;

    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| ModelRegistryError::Io {
                action: "read registered managed model artifact",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }

        hasher.update(&buffer[..read]);
        file_size_bytes = file_size_bytes
            .checked_add(u64::try_from(read).expect("read size fits in u64"))
            .ok_or_else(|| ModelRegistryError::FileSizeOverflow {
                path: path.to_path_buf(),
            })?;
    }

    Ok((file_size_bytes, hex_digest(&hasher.finalize())))
}

fn stream_copy_with_sha256(
    source: &mut File,
    target: &mut File,
    source_path: &Path,
    target_path: &Path,
) -> Result<(u64, String), ModelRegistryError> {
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut file_size_bytes = 0_u64;

    loop {
        let read = source
            .read(&mut buffer)
            .map_err(|source| ModelRegistryError::Io {
                action: "read model source file",
                path: source_path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }

        target
            .write_all(&buffer[..read])
            .map_err(|source| ModelRegistryError::Io {
                action: "write managed model target",
                path: target_path.to_path_buf(),
                source,
            })?;
        hasher.update(&buffer[..read]);
        file_size_bytes = file_size_bytes
            .checked_add(u64::try_from(read).expect("read size fits in u64"))
            .ok_or_else(|| ModelRegistryError::FileSizeOverflow {
                path: source_path.to_path_buf(),
            })?;
    }

    let digest = hasher.finalize();
    Ok((file_size_bytes, hex_digest(&digest)))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to a string cannot fail");
    }
    output
}

fn remove_failed_import(target_path: &Path) {
    let _ = fs::remove_file(target_path);
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
                .join(format!("word-covenant-model-registry-{}", Uuid::new_v4()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn child(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn acknowledgement() -> LicenseAcknowledgement {
        LicenseAcknowledgement::new("openai/whisper-tiny", "MIT", DateTime::<Utc>::UNIX_EPOCH)
            .unwrap()
    }

    fn digest(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        hex_digest(&digest)
    }

    fn request(source_path: PathBuf, expected_sha256: String) -> ModelImportRequest {
        ModelImportRequest {
            id: Uuid::new_v4(),
            source_path,
            model_kind: LocalModelKind::SpeechRecognition,
            version: "1.0.0".to_owned(),
            input_format: "16 kHz mono PCM".to_owned(),
            expected_sha256,
            license_acknowledgement: Some(acknowledgement()),
        }
    }

    fn persisted_model(file_path: impl Into<PathBuf>, bytes: &[u8]) -> RegisteredModel {
        RegisteredModel {
            id: Uuid::new_v4(),
            model_kind: LocalModelKind::SpeechRecognition,
            file_path: file_path.into(),
            file_size_bytes: bytes.len() as u64,
            sha256: digest(bytes),
            version: "1.0.0".to_owned(),
            input_format: "16 kHz mono PCM".to_owned(),
            model_card_id: "openai/whisper-tiny".to_owned(),
            license_id: "MIT".to_owned(),
            license_confirmed_at: DateTime::<Utc>::UNIX_EPOCH,
            imported_at: DateTime::<Utc>::UNIX_EPOCH,
        }
    }

    fn write_persisted_model(
        managed_root: &Path,
        relative_path: impl AsRef<Path>,
        bytes: &[u8],
    ) -> RegisteredModel {
        let relative_path = relative_path.as_ref();
        let artifact_path = managed_root.join(relative_path);
        fs::create_dir_all(artifact_path.parent().unwrap()).unwrap();
        fs::write(&artifact_path, bytes).unwrap();
        persisted_model(relative_path, bytes)
    }

    #[test]
    fn imports_a_local_model_and_records_its_provenance() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let managed_root = directory.child("managed-models");
        let bytes = vec![0x5a; COPY_BUFFER_BYTES + 17];
        fs::write(&source_path, &bytes).unwrap();

        let mut registry = ModelRegistry::new();
        let imported_at = DateTime::<Utc>::UNIX_EPOCH;
        let registration = registry
            .import_at(
                &managed_root,
                request(source_path.clone(), digest(&bytes)),
                imported_at,
            )
            .unwrap();

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get(registration.id), Some(&registration));
        assert_eq!(registration.model_kind, LocalModelKind::SpeechRecognition);
        assert_eq!(registration.file_size_bytes, bytes.len() as u64);
        assert_eq!(registration.sha256, digest(&bytes));
        assert_eq!(registration.version, "1.0.0");
        assert_eq!(registration.input_format, "16 kHz mono PCM");
        assert_eq!(registration.model_card_id, "openai/whisper-tiny");
        assert_eq!(registration.license_id, "MIT");
        assert_eq!(
            registration.license_confirmed_at,
            DateTime::<Utc>::UNIX_EPOCH
        );
        assert_eq!(registration.imported_at, imported_at);
        assert!(!registration.file_path.is_absolute());
        assert!(registration
            .file_path
            .components()
            .all(|component| matches!(component, Component::Normal(_))));
        assert_eq!(
            fs::read(managed_root.join(&registration.file_path)).unwrap(),
            bytes
        );
        assert_eq!(
            fs::read(source_path).unwrap(),
            vec![0x5a; COPY_BUFFER_BYTES + 17]
        );
        let serialized = serde_json::to_value(&registration).unwrap();
        assert!(serialized.get("filePath").is_none());
    }

    #[test]
    fn restores_only_a_verified_managed_artifact() {
        let directory = TestDirectory::new();
        let managed_root = directory.child("managed-models");
        let bytes = b"existing managed model";
        let model = write_persisted_model(&managed_root, "existing.model", bytes);
        let id = model.id;

        let registry = ModelRegistry::from_persisted(&managed_root, [model.clone()]).unwrap();

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get(id), Some(&model));
        assert_eq!(
            fs::read(managed_root.join(&model.file_path)).unwrap(),
            bytes
        );
    }

    #[test]
    fn rechecks_a_model_artifact_when_a_native_worker_requests_it() {
        let directory = TestDirectory::new();
        let managed_root = directory.child("managed-models");
        let model = write_persisted_model(&managed_root, "existing.model", b"original model");
        let registry = ModelRegistry::from_persisted(&managed_root, [model.clone()]).unwrap();
        let artifact = registry.verified_artifact(model.id).unwrap();

        assert_eq!(
            artifact.path(),
            fs::canonicalize(managed_root.join(&model.file_path))
                .unwrap()
                .as_path()
        );
        assert_eq!(artifact.model(), &model);
        assert_eq!(
            registry.verified_artifact_path(model.id).unwrap(),
            fs::canonicalize(managed_root.join(&model.file_path)).unwrap()
        );

        fs::write(managed_root.join(&model.file_path), b"replaced model").unwrap();
        assert!(matches!(
            registry.verified_artifact(model.id),
            Err(ModelRegistryError::RegisteredArtifactSizeMismatch { .. })
                | Err(ModelRegistryError::RegisteredArtifactHashMismatch { .. })
        ));
        assert!(matches!(
            registry.verified_artifact_path(Uuid::new_v4()),
            Err(ModelRegistryError::UnknownModelId { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_persisted_identifiers_hashes_and_paths() {
        let directory = TestDirectory::new();
        let managed_root = directory.child("managed-models");
        let first = write_persisted_model(&managed_root, "first.model", b"first model");
        let duplicate_id = RegisteredModel {
            id: first.id,
            ..write_persisted_model(&managed_root, "different-id.model", b"different id")
        };
        let mut registry = ModelRegistry::from_persisted(&managed_root, [first.clone()]).unwrap();
        let id_error = registry.register_persisted(duplicate_id).unwrap_err();
        assert!(matches!(
            id_error,
            ModelRegistryError::DuplicateModelId { id } if id == first.id
        ));
        assert_eq!(registry.len(), 1);

        let duplicate_sha256 = RegisteredModel {
            sha256: first.sha256.to_ascii_uppercase(),
            ..write_persisted_model(&managed_root, "different-hash.model", b"first model")
        };
        let sha256_error = registry.register_persisted(duplicate_sha256).unwrap_err();
        assert!(matches!(
            sha256_error,
            ModelRegistryError::DuplicateSha256 {
                existing_model_id,
                ..
            } if existing_model_id == first.id
        ));
        assert_eq!(registry.len(), 1);

        let duplicate_path = RegisteredModel {
            id: Uuid::new_v4(),
            file_path: first.file_path.clone(),
            ..first.clone()
        };
        let path_error = registry.register_persisted(duplicate_path).unwrap_err();
        assert!(matches!(
            path_error,
            ModelRegistryError::DuplicateManagedPath {
                existing_model_id,
                ..
            } if existing_model_id == first.id
        ));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn revalidates_persisted_model_metadata_before_indexing() {
        let directory = TestDirectory::new();
        let managed_root = directory.child("managed-models");
        let bytes = b"valid model metadata";
        fs::create_dir(&managed_root).unwrap();

        let empty_version = RegisteredModel {
            version: " \t".to_owned(),
            ..persisted_model("empty-version.model", bytes)
        };
        let error = ModelRegistry::from_persisted(&managed_root, [empty_version]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::EmptyMetadata {
                field: "model version"
            }
        ));

        let invalid_sha256 = RegisteredModel {
            sha256: "not-a-sha256".to_owned(),
            ..persisted_model("invalid-digest.model", bytes)
        };
        let error = ModelRegistry::from_persisted(&managed_root, [invalid_sha256]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::InvalidRegisteredSha256 { .. }
        ));

        let zero_sized = RegisteredModel {
            file_size_bytes: 0,
            ..persisted_model("zero-sized.model", bytes)
        };
        let error = ModelRegistry::from_persisted(&managed_root, [zero_sized]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredFileSizeMustBeNonZero { .. }
        ));

        let absolute_path = RegisteredModel {
            file_path: directory.child("absolute.model"),
            ..persisted_model("unused.model", bytes)
        };
        let error = ModelRegistry::from_persisted(&managed_root, [absolute_path]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredPathMustBeRelative { .. }
        ));
    }

    #[test]
    fn rejects_unsafe_relative_paths_before_resolving_them() {
        let directory = TestDirectory::new();
        let managed_root = directory.child("managed-models");
        fs::create_dir(&managed_root).unwrap();

        for path in [
            PathBuf::new(),
            PathBuf::from("."),
            PathBuf::from("./current.model"),
            PathBuf::from("../outside.model"),
            PathBuf::from("nested/../outside.model"),
        ] {
            let model = persisted_model(path, b"model bytes");
            let error = ModelRegistry::from_persisted(&managed_root, [model]).unwrap_err();
            assert!(matches!(
                error,
                ModelRegistryError::UnsafeRegisteredPath { .. }
            ));
        }
    }

    #[test]
    fn rejects_missing_replaced_or_non_regular_restored_artifacts() {
        let directory = TestDirectory::new();
        let managed_root = directory.child("managed-models");
        fs::create_dir(&managed_root).unwrap();

        let missing = persisted_model("missing.model", b"missing model");
        let error = ModelRegistry::from_persisted(&managed_root, [missing]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredArtifactNotFound { .. }
        ));

        let size_mismatch = write_persisted_model(&managed_root, "size.model", b"model bytes");
        let size_mismatch = RegisteredModel {
            file_size_bytes: size_mismatch.file_size_bytes + 1,
            ..size_mismatch
        };
        let error = ModelRegistry::from_persisted(&managed_root, [size_mismatch]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredArtifactSizeMismatch { .. }
        ));

        let hash_mismatch = write_persisted_model(&managed_root, "hash.model", b"model bytes");
        let hash_mismatch = RegisteredModel {
            sha256: "0".repeat(SHA256_HEX_LENGTH),
            ..hash_mismatch
        };
        let error = ModelRegistry::from_persisted(&managed_root, [hash_mismatch]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredArtifactHashMismatch { .. }
        ));

        fs::create_dir(managed_root.join("directory.model")).unwrap();
        let directory_model = persisted_model("directory.model", b"model bytes");
        let error = ModelRegistry::from_persisted(&managed_root, [directory_model]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredArtifactNotRegularFile { .. }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_or_outside_restored_artifacts() {
        let directory = TestDirectory::new();
        let managed_root = directory.child("managed-models");
        fs::create_dir(&managed_root).unwrap();
        let inside_target = managed_root.join("inside.model");
        fs::write(&inside_target, b"inside model").unwrap();

        let linked_path = managed_root.join("linked.model");
        std::os::unix::fs::symlink(&inside_target, &linked_path).unwrap();
        let linked_model = persisted_model("linked.model", b"inside model");
        let error = ModelRegistry::from_persisted(&managed_root, [linked_model]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredArtifactIsSymlink { .. }
        ));

        let outside = directory.child("outside.model");
        fs::write(&outside, b"outside model").unwrap();
        let escaped_path = managed_root.join("escaped.model");
        std::os::unix::fs::symlink(&outside, &escaped_path).unwrap();
        let escaped_model = persisted_model("escaped.model", b"outside model");
        let error = ModelRegistry::from_persisted(&managed_root, [escaped_model]).unwrap_err();
        assert!(matches!(
            error,
            ModelRegistryError::RegisteredArtifactEscapesManagedRoot { .. }
        ));
    }

    #[test]
    fn rolls_back_only_the_in_memory_registration() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let managed_root = directory.child("managed-models");
        let bytes = b"local model";
        fs::write(&source_path, bytes).unwrap();
        let mut registry = ModelRegistry::new();
        let registration = registry
            .import_at(
                &managed_root,
                request(source_path, digest(bytes)),
                Utc::now(),
            )
            .unwrap();
        let artifact_path = managed_root.join(&registration.file_path);

        let rolled_back = registry.rollback_registration(registration.id).unwrap();

        assert_eq!(rolled_back, registration);
        assert!(registry.is_empty());
        assert_eq!(fs::read(&artifact_path).unwrap(), bytes);
        assert!(registry.rollback_registration(registration.id).is_none());

        registry.register_persisted(rolled_back).unwrap();
        assert_eq!(registry.get(registration.id), Some(&registration));

        let removed = registry.rollback_registration(registration.id).unwrap();
        registry.remove_managed_artifact(&removed).unwrap();
        assert!(!artifact_path.exists());
    }

    #[test]
    fn rejects_a_missing_license_acknowledgement_before_creating_a_target() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let bytes = b"local model";
        fs::write(&source_path, bytes).unwrap();
        let managed_root = directory.child("managed-models");
        let mut import = request(source_path, digest(bytes));
        import.license_acknowledgement = None;

        let error = ModelRegistry::new()
            .import_at(&managed_root, import, Utc::now())
            .unwrap_err();

        assert!(matches!(
            error,
            ModelRegistryError::LicenseAcknowledgementRequired
        ));
        assert!(!managed_root.exists());
    }

    #[test]
    fn removes_the_copy_when_the_expected_hash_does_not_match() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let bytes = b"local model";
        fs::write(&source_path, bytes).unwrap();
        let mut registry = ModelRegistry::new();
        let mut import = request(source_path, digest(bytes));
        import.expected_sha256 = "0".repeat(SHA256_HEX_LENGTH);
        let managed_root = directory.child("managed-models");

        let error = registry
            .import_at(&managed_root, import, Utc::now())
            .unwrap_err();

        assert!(matches!(error, ModelRegistryError::HashMismatch { .. }));
        assert!(registry.is_empty());
        assert!(fs::read_dir(managed_root).unwrap().next().is_none());
    }

    #[test]
    fn rejects_duplicate_content_and_removes_the_second_copy() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let bytes = b"local model";
        fs::write(&source_path, bytes).unwrap();
        let expected_sha256 = digest(bytes);
        let managed_root = directory.child("managed-models");
        let mut registry = ModelRegistry::new();

        registry
            .import_at(
                &managed_root,
                request(source_path.clone(), expected_sha256.clone()),
                Utc::now(),
            )
            .unwrap();
        let error = registry
            .import_at(
                &managed_root,
                request(source_path, expected_sha256),
                Utc::now(),
            )
            .unwrap_err();

        assert!(matches!(error, ModelRegistryError::DuplicateSha256 { .. }));
        assert_eq!(registry.len(), 1);
        assert_eq!(fs::read_dir(managed_root).unwrap().count(), 1);
    }

    #[test]
    fn rejects_a_duplicate_identifier_before_copying_again() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let bytes = b"local model";
        fs::write(&source_path, bytes).unwrap();
        let managed_root = directory.child("managed-models");
        let mut registry = ModelRegistry::new();
        let first_request = request(source_path.clone(), digest(bytes));
        let duplicate_id = first_request.id;

        registry
            .import_at(&managed_root, first_request, Utc::now())
            .unwrap();

        let mut duplicate_request = request(source_path, digest(bytes));
        duplicate_request.id = duplicate_id;
        let error = registry
            .import_at(&managed_root, duplicate_request, Utc::now())
            .unwrap_err();

        assert!(matches!(error, ModelRegistryError::DuplicateModelId { .. }));
        assert_eq!(fs::read_dir(managed_root).unwrap().count(), 1);
    }

    #[test]
    fn rejects_parent_traversal_in_the_managed_root() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let bytes = b"local model";
        fs::write(&source_path, bytes).unwrap();
        let unsafe_root = directory.child("models").join("..").join("outside");

        let error = ModelRegistry::new()
            .import_at(
                &unsafe_root,
                request(source_path, digest(bytes)),
                Utc::now(),
            )
            .unwrap_err();

        assert!(matches!(
            error,
            ModelRegistryError::UnsafeManagedRoot { .. }
        ));
        assert!(!directory.child("outside").exists());
    }

    #[test]
    fn rejects_missing_and_empty_sources() {
        let directory = TestDirectory::new();
        let missing_source = directory.child("missing.gguf");
        let managed_root = directory.child("managed-models");

        let missing_error = ModelRegistry::new()
            .import_at(
                &managed_root,
                request(missing_source, "0".repeat(SHA256_HEX_LENGTH)),
                Utc::now(),
            )
            .unwrap_err();
        assert!(matches!(
            missing_error,
            ModelRegistryError::SourceNotFound { .. }
        ));

        let empty_source = directory.child("empty.gguf");
        fs::write(&empty_source, []).unwrap();
        let empty_error = ModelRegistry::new()
            .import_at(
                &managed_root,
                request(empty_source, "0".repeat(SHA256_HEX_LENGTH)),
                Utc::now(),
            )
            .unwrap_err();
        assert!(matches!(
            empty_error,
            ModelRegistryError::SourceEmpty { .. }
        ));

        let directory_source = directory.child("not-a-model");
        fs::create_dir(&directory_source).unwrap();
        let directory_error = ModelRegistry::new()
            .import_at(
                &managed_root,
                request(directory_source, "0".repeat(SHA256_HEX_LENGTH)),
                Utc::now(),
            )
            .unwrap_err();
        assert!(matches!(
            directory_error,
            ModelRegistryError::SourceNotRegularFile { .. }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_symbolic_link_as_the_managed_root() {
        let directory = TestDirectory::new();
        let source_path = directory.child("selected-model.gguf");
        let bytes = b"local model";
        fs::write(&source_path, bytes).unwrap();
        let outside = directory.child("outside");
        fs::create_dir(&outside).unwrap();
        let symlink_root = directory.child("managed-models-link");
        std::os::unix::fs::symlink(&outside, &symlink_root).unwrap();

        let error = ModelRegistry::new()
            .import_at(
                symlink_root,
                request(source_path, digest(bytes)),
                Utc::now(),
            )
            .unwrap_err();

        assert!(matches!(
            error,
            ModelRegistryError::ManagedRootIsSymlink { .. }
        ));
        assert!(fs::read_dir(outside).unwrap().next().is_none());
    }
}
