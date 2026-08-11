use crate::audio::SpeechDetectionSettings;
use crate::domain::{DataCategory, SpeakerCluster, TranscriptSpan};
use crate::inference::bundled_model::BundledAsrStatus;
use crate::inference::model_registry::{
    LicenseAcknowledgement, LocalModelKind, ModelImportRequest, RegisteredModel,
};
use crate::policy::{EgressApproval, PolicyDecision};
use crate::state::{
    ActiveLocalAsrProfile, AgentAction, AppState, PrivacyStatus, SpeakerOperationResult,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::PathBuf;
use tauri::{State, WebviewWindow};
use tauri_plugin_dialog::DialogExt;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateEgressApprovalInput {
    pub tool_id: String,
    pub origin: String,
    pub data_categories: BTreeSet<DataCategory>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HttpProfileAttemptInput {
    pub tool_id: String,
    pub origin: String,
    pub data_categories: BTreeSet<DataCategory>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectInputDeviceInput {
    pub device_uid: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportLocalModelInput {
    pub source_path: String,
    pub model_kind: LocalModelKind,
    pub version: String,
    pub input_format: String,
    pub expected_sha256: String,
    pub model_card_id: String,
    pub license_id: String,
    pub license_acknowledged: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectActiveLocalAsrModelInput {
    pub model_id: Uuid,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSpeakerClusterInput {
    pub session_id: Uuid,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenameSpeakerClusterInput {
    pub session_id: Uuid,
    pub cluster_id: String,
    pub expected_label_revision: u32,
    pub label: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReassignTranscriptSpeakerInput {
    pub session_id: Uuid,
    pub logical_span_id: Uuid,
    pub expected_revision: u32,
    pub target_cluster_id: Option<String>,
}

#[cfg(any(test, debug_assertions))]
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdvanceDevelopmentMockInput {
    pub packet_count: usize,
}

#[tauri::command]
pub fn get_privacy_status(state: State<'_, AppState>) -> Result<PrivacyStatus, String> {
    state.privacy_status()
}

#[tauri::command]
pub fn set_egress_enabled(
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<PrivacyStatus, String> {
    state.set_egress_enabled(enabled)
}

#[tauri::command]
pub fn get_speech_detection_settings(
    state: State<'_, AppState>,
) -> Result<SpeechDetectionSettings, String> {
    state.speech_detection_settings()
}

#[tauri::command]
pub fn set_speech_detection_settings(
    state: State<'_, AppState>,
    input: SpeechDetectionSettings,
) -> Result<SpeechDetectionSettings, String> {
    state.set_speech_detection_settings(input)
}

#[tauri::command]
pub fn start_session(state: State<'_, AppState>) -> Result<crate::domain::CaptureSession, String> {
    state.start_session()
}

#[tauri::command]
pub fn get_capture_projection(
    state: State<'_, AppState>,
) -> Result<crate::audio::CaptureProjection, String> {
    state.capture_projection()
}

#[tauri::command]
pub fn select_input_device(
    state: State<'_, AppState>,
    input: SelectInputDeviceInput,
) -> Result<crate::audio::CaptureProjection, String> {
    state.select_input_device(input.device_uid)
}

#[cfg(any(test, debug_assertions))]
#[tauri::command]
pub fn start_development_mock_session(
    state: State<'_, AppState>,
) -> Result<crate::domain::CaptureSession, String> {
    state.start_development_mock_session()
}

#[cfg(any(test, debug_assertions))]
#[tauri::command]
pub fn advance_development_mock(
    state: State<'_, AppState>,
    input: AdvanceDevelopmentMockInput,
) -> Result<crate::audio::DevelopmentMockProgress, String> {
    state.advance_development_mock(input.packet_count)
}

#[tauri::command]
pub fn stop_session(
    state: State<'_, AppState>,
) -> Result<Option<crate::domain::CaptureSession>, String> {
    state.stop_session()
}

#[tauri::command]
pub fn list_timeline(
    state: State<'_, AppState>,
    session_id: Option<Uuid>,
) -> Result<Vec<TranscriptSpan>, String> {
    state.list_timeline(session_id)
}

#[tauri::command]
pub fn list_speaker_clusters(
    state: State<'_, AppState>,
    session_id: Option<Uuid>,
) -> Result<Vec<SpeakerCluster>, String> {
    state.list_speaker_clusters(session_id)
}

#[tauri::command]
pub fn create_speaker_cluster(
    state: State<'_, AppState>,
    input: CreateSpeakerClusterInput,
) -> Result<SpeakerOperationResult, String> {
    state.create_speaker_cluster(input.session_id)
}

#[tauri::command]
pub fn rename_speaker_cluster(
    state: State<'_, AppState>,
    input: RenameSpeakerClusterInput,
) -> Result<SpeakerOperationResult, String> {
    state.rename_speaker_cluster(
        input.session_id,
        input.cluster_id,
        input.expected_label_revision,
        input.label,
    )
}

#[tauri::command]
pub fn reassign_transcript_speaker(
    state: State<'_, AppState>,
    input: ReassignTranscriptSpeakerInput,
) -> Result<SpeakerOperationResult, String> {
    state.reassign_transcript_speaker(
        input.session_id,
        input.logical_span_id,
        input.expected_revision,
        input.target_cluster_id,
    )
}

#[tauri::command]
pub fn list_local_models(state: State<'_, AppState>) -> Result<Vec<RegisteredModel>, String> {
    state.list_local_models()
}

#[tauri::command]
pub fn get_bundled_asr_status(state: State<'_, AppState>) -> Result<BundledAsrStatus, String> {
    state.bundled_asr_status()
}

#[tauri::command]
pub fn get_active_local_asr_profile(
    state: State<'_, AppState>,
) -> Result<Option<ActiveLocalAsrProfile>, String> {
    state.active_local_asr_profile()
}

#[tauri::command]
pub fn select_active_local_asr_model(
    state: State<'_, AppState>,
    input: SelectActiveLocalAsrModelInput,
) -> Result<ActiveLocalAsrProfile, String> {
    state.select_active_local_asr_model(input.model_id)
}

#[tauri::command]
pub async fn select_local_model_file(window: WebviewWindow) -> Result<Option<String>, String> {
    let dialog = window
        .dialog()
        .file()
        .set_parent(&window)
        .set_title("选择本地模型文件");
    let selected_file = tauri::async_runtime::spawn_blocking(move || dialog.blocking_pick_file())
        .await
        .map_err(|error| format!("local model file picker did not complete: {error}"))?;
    let selected_path = selected_file
        .map(|file| {
            file.into_path()
                .map_err(|_| "selected model file is not a local filesystem path".to_owned())
        })
        .transpose()?;

    selected_local_model_path(selected_path)
}

#[tauri::command]
pub fn import_local_model(
    state: State<'_, AppState>,
    input: ImportLocalModelInput,
) -> Result<RegisteredModel, String> {
    if !input.license_acknowledged {
        return Err("confirm the model card and license before importing a local model".to_owned());
    }
    let source_path = PathBuf::from(input.source_path);
    if !source_path.is_absolute() {
        return Err("select an absolute local model file path".to_owned());
    }
    let acknowledgement =
        LicenseAcknowledgement::new(input.model_card_id, input.license_id, Utc::now())
            .map_err(|error| format!("invalid model license acknowledgement: {error}"))?;
    state.import_local_model(ModelImportRequest {
        id: uuid::Uuid::new_v4(),
        source_path,
        model_kind: input.model_kind,
        version: input.version,
        input_format: input.input_format,
        expected_sha256: input.expected_sha256,
        license_acknowledgement: Some(acknowledgement),
    })
}

fn selected_local_model_path(selected_path: Option<PathBuf>) -> Result<Option<String>, String> {
    selected_path
        .map(|path| {
            if !path.is_absolute() {
                return Err("selected model file must use an absolute local path".to_owned());
            }

            path.into_os_string()
                .into_string()
                .map_err(|_| "selected model file path must be valid Unicode".to_owned())
        })
        .transpose()
}

#[tauri::command]
pub fn create_egress_approval(
    state: State<'_, AppState>,
    input: CreateEgressApprovalInput,
) -> Result<EgressApproval, String> {
    state.create_egress_approval(
        input.tool_id,
        input.origin,
        input.data_categories,
        input.expires_at,
    )
}

#[tauri::command]
pub fn revoke_egress_approval(
    state: State<'_, AppState>,
    approval_id: Uuid,
) -> Result<bool, String> {
    state.revoke_egress_approval(approval_id)
}

#[tauri::command]
pub fn propose_local_speech(state: State<'_, AppState>) -> Result<AgentAction, String> {
    state.propose_local_speech()
}

#[tauri::command]
pub fn list_actions(state: State<'_, AppState>) -> Result<Vec<AgentAction>, String> {
    state.list_actions()
}

#[tauri::command]
pub fn attempt_http_profile(
    state: State<'_, AppState>,
    input: HttpProfileAttemptInput,
) -> Result<PolicyDecision, String> {
    state.evaluate_http_profile(input.tool_id, input.origin, input.data_categories)
}

#[cfg(test)]
mod tests {
    use super::{
        selected_local_model_path, CreateSpeakerClusterInput, ReassignTranscriptSpeakerInput,
        RenameSpeakerClusterInput, SelectActiveLocalAsrModelInput,
    };
    use crate::audio::SpeechDetectionSettings;
    use serde_json::json;
    use std::path::PathBuf;
    use uuid::Uuid;

    #[test]
    fn local_model_file_selection_returns_none_when_the_picker_is_cancelled() {
        assert_eq!(selected_local_model_path(None).unwrap(), None);
    }

    #[test]
    fn local_model_file_selection_only_returns_absolute_paths() {
        assert_eq!(
            selected_local_model_path(Some(PathBuf::from("/tmp/model.gguf"))).unwrap(),
            Some("/tmp/model.gguf".to_owned())
        );
        assert_eq!(
            selected_local_model_path(Some(PathBuf::from("model.gguf"))).unwrap_err(),
            "selected model file must use an absolute local path"
        );
    }

    #[test]
    fn speaker_command_inputs_use_the_camel_case_bridge_contract() {
        let session_id = Uuid::new_v4();
        let logical_span_id = Uuid::new_v4();

        let create: CreateSpeakerClusterInput = serde_json::from_value(json!({
            "sessionId": session_id,
        }))
        .unwrap();
        assert_eq!(create.session_id, session_id);

        let rename: RenameSpeakerClusterInput = serde_json::from_value(json!({
            "sessionId": session_id,
            "clusterId": "speaker-01234567-89ab-cdef-0123-456789abcdef",
            "expectedLabelRevision": 4,
            "label": "会议主持人",
        }))
        .unwrap();
        assert_eq!(rename.session_id, session_id);
        assert_eq!(
            rename.cluster_id,
            "speaker-01234567-89ab-cdef-0123-456789abcdef"
        );
        assert_eq!(rename.expected_label_revision, 4);
        assert_eq!(rename.label, "会议主持人");

        let reassign: ReassignTranscriptSpeakerInput = serde_json::from_value(json!({
            "sessionId": session_id,
            "logicalSpanId": logical_span_id,
            "expectedRevision": 7,
            "targetClusterId": null,
        }))
        .unwrap();
        assert_eq!(reassign.session_id, session_id);
        assert_eq!(reassign.logical_span_id, logical_span_id);
        assert_eq!(reassign.expected_revision, 7);
        assert_eq!(reassign.target_cluster_id, None);

        let select: SelectActiveLocalAsrModelInput = serde_json::from_value(json!({
            "modelId": session_id,
        }))
        .unwrap();
        assert_eq!(select.model_id, session_id);
    }

    #[test]
    fn speech_detection_settings_use_the_camel_case_bridge_contract() {
        let settings: SpeechDetectionSettings = serde_json::from_value(json!({
            "rmsThresholdDbfs": -24,
        }))
        .unwrap();

        assert_eq!(settings.rms_threshold_dbfs, -24);
        assert_eq!(
            serde_json::to_value(settings).unwrap(),
            json!({ "rmsThresholdDbfs": -24 })
        );
    }
}
