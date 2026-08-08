use tauri::{Emitter, Manager};

pub mod audio;
pub mod audit;
pub mod commands;
pub mod domain;
pub mod inference;
pub mod policy;
pub mod state;

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
        commands::list_local_models,
        commands::select_local_model_file,
        commands::import_local_model,
        commands::create_egress_approval,
        commands::revoke_egress_approval,
        commands::propose_local_speech,
        commands::list_actions,
        commands::attempt_http_profile,
    ]);

    builder
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
