use std::sync::atomic::{AtomicU8, Ordering};
use tauri::{Emitter, Manager};

pub mod audio;
pub mod audit;
pub mod commands;
pub mod domain;
pub mod inference;
pub mod policy;
pub mod state;

const EXIT_DRAIN_IDLE: u8 = 0;
const EXIT_DRAINING: u8 = 1;
const EXIT_READY_TO_EXIT: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExitRequestAction {
    StartDrain,
    KeepOpen,
    AllowExit,
}

/// Claim an exit request without allowing a later request to bypass an
/// in-progress durable capture drain.
fn exit_request_action(exit_state: &AtomicU8) -> ExitRequestAction {
    loop {
        match exit_state.load(Ordering::Acquire) {
            EXIT_DRAIN_IDLE => {
                if exit_state
                    .compare_exchange(
                        EXIT_DRAIN_IDLE,
                        EXIT_DRAINING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return ExitRequestAction::StartDrain;
                }
            }
            EXIT_DRAINING => return ExitRequestAction::KeepOpen,
            EXIT_READY_TO_EXIT => return ExitRequestAction::AllowExit,
            _ => exit_state.store(EXIT_DRAIN_IDLE, Ordering::Release),
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .setup(|app| {
            let data_dir = app.path().app_local_data_dir()?;
            std::fs::create_dir_all(&data_dir)?;
            app.manage(state::AppState::open(
                data_dir.join("word-covenant.sqlite3"),
            )?);

            #[cfg(target_os = "macos")]
            {
                let app_handle = app.handle().clone();
                std::thread::Builder::new()
                    .name("word-covenant-capture-projection".to_owned())
                    .spawn(move || {
                        let mut emitted_revision = None;
                        loop {
                            let projection =
                                app_handle.state::<state::AppState>().capture_projection();
                            if let Ok(projection) = projection {
                                let revision = projection.revision;
                                if emitted_revision != Some(revision) {
                                    let _ = app_handle.emit("capture-projection", &projection);
                                    emitted_revision = Some(revision);
                                }
                            }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                    })?;
            }

            #[cfg(debug_assertions)] // only include this code on debug builds
            {
                if let Some(window) = tauri::Manager::get_webview_window(app, "main") {
                    window.open_devtools();
                }
            }
            Ok(())
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_prevent_default::init());

    #[cfg(debug_assertions)]
    let builder = builder.invoke_handler(tauri::generate_handler![
        commands::get_privacy_status,
        commands::set_egress_enabled,
        commands::start_session,
        commands::get_capture_projection,
        commands::select_input_device,
        commands::start_development_mock_session,
        commands::advance_development_mock,
        commands::stop_session,
        commands::list_timeline,
        commands::list_speaker_clusters,
        commands::create_speaker_cluster,
        commands::rename_speaker_cluster,
        commands::reassign_transcript_speaker,
        commands::list_local_models,
        commands::select_local_model_file,
        commands::import_local_model,
        commands::create_egress_approval,
        commands::revoke_egress_approval,
        commands::propose_local_speech,
        commands::list_actions,
        commands::attempt_http_profile,
    ]);

    #[cfg(not(debug_assertions))]
    let builder = builder.invoke_handler(tauri::generate_handler![
        commands::get_privacy_status,
        commands::set_egress_enabled,
        commands::start_session,
        commands::get_capture_projection,
        commands::select_input_device,
        commands::stop_session,
        commands::list_timeline,
        commands::list_speaker_clusters,
        commands::create_speaker_cluster,
        commands::rename_speaker_cluster,
        commands::reassign_transcript_speaker,
        commands::list_local_models,
        commands::select_local_model_file,
        commands::import_local_model,
        commands::create_egress_approval,
        commands::revoke_egress_approval,
        commands::propose_local_speech,
        commands::list_actions,
        commands::attempt_http_profile,
    ]);

    // A normal application exit must use the same drain path as an explicit
    // Stop command. Otherwise the immutable SessionStarted bundle could be
    // left without terminal capture/inference evidence.
    let exit_state = AtomicU8::new(EXIT_DRAIN_IDLE);
    let app = builder
        .build(tauri::generate_context!())
        .expect("error while building Tauri application");
    app.run(move |app_handle, event| {
        if let tauri::RunEvent::ExitRequested { api, code, .. } = event {
            match exit_request_action(&exit_state) {
                ExitRequestAction::StartDrain => {
                    api.prevent_exit();
                    match app_handle.state::<state::AppState>().stop_session() {
                        Ok(_) => {
                            exit_state.store(EXIT_READY_TO_EXIT, Ordering::Release);
                            app_handle.exit(code.unwrap_or_default());
                        }
                        Err(error) => {
                            exit_state.store(EXIT_DRAIN_IDLE, Ordering::Release);
                            eprintln!("could not durably stop native capture before exit: {error}");
                        }
                    }
                }
                ExitRequestAction::KeepOpen => api.prevent_exit(),
                ExitRequestAction::AllowExit => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_repeated_exit_requests_while_capture_drain_is_running() {
        let state = AtomicU8::new(EXIT_DRAIN_IDLE);

        assert_eq!(exit_request_action(&state), ExitRequestAction::StartDrain);
        assert_eq!(exit_request_action(&state), ExitRequestAction::KeepOpen);

        state.store(EXIT_READY_TO_EXIT, Ordering::Release);
        assert_eq!(exit_request_action(&state), ExitRequestAction::AllowExit);
    }

    #[test]
    fn failed_capture_drain_returns_exit_state_to_idle() {
        let state = AtomicU8::new(EXIT_DRAIN_IDLE);

        assert_eq!(exit_request_action(&state), ExitRequestAction::StartDrain);
        state.store(EXIT_DRAIN_IDLE, Ordering::Release);

        assert_eq!(exit_request_action(&state), ExitRequestAction::StartDrain);
    }
}
