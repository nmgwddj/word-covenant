use tauri::Manager;

pub mod audio;
pub mod audit;
pub mod commands;
pub mod domain;
pub mod policy;
pub mod state;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app.path().app_local_data_dir()?;
            std::fs::create_dir_all(&data_dir)?;
            app.manage(state::AppState::open(
                data_dir.join("word-covenant.sqlite3"),
            )?);

            #[cfg(debug_assertions)] // only include this code on debug builds
            {
                if let Some(window) = tauri::Manager::get_webview_window(app, "main") {
                    window.open_devtools();
                }
            }
            Ok(())
        })
        .plugin(tauri_plugin_prevent_default::init())
        .invoke_handler(tauri::generate_handler![
            commands::get_privacy_status,
            commands::set_egress_enabled,
            commands::start_session,
            commands::stop_session,
            commands::list_timeline,
            commands::create_egress_approval,
            commands::revoke_egress_approval,
            commands::propose_local_speech,
            commands::list_actions,
            commands::attempt_http_profile,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
