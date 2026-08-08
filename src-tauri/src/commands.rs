use crate::domain::{DataCategory, TranscriptSpan};
use crate::policy::{EgressApproval, PolicyDecision};
use crate::state::{AgentAction, AppState, PrivacyStatus};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::collections::BTreeSet;
use tauri::State;
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
