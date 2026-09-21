//! In-place updates from the GitHub release's `latest.json` (see the
//! `plugins.updater` block in tauri.conf.json and the release workflow).

use serde::Serialize;
use tauri::{AppHandle, Manager};
use tauri_plugin_updater::UpdaterExt;

#[derive(Serialize)]
pub struct UpdateInfo {
    pub version: String,
}

/// Whether a newer release exists. Works on every build, including ones that
/// can't install it themselves (see `can_install_update`), so those can still
/// point the user at the release page.
#[tauri::command]
pub async fn check_for_update(app: AppHandle) -> Result<Option<UpdateInfo>, String> {
    let updater = app.updater().map_err(|e| e.to_string())?;
    let update = updater.check().await.map_err(|e| e.to_string())?;
    Ok(update.map(|u| UpdateInfo { version: u.version }))
}

/// Re-checks rather than holding the `Update` from `check_for_update`, which
/// isn't storable between command calls. Resolves only on failure: a
/// successful install restarts the app.
#[tauri::command]
pub async fn install_update(app: AppHandle) -> Result<(), String> {
    if !can_install(&app) {
        return Err("This build can't update itself".into());
    }
    let updater = app.updater().map_err(|e| e.to_string())?;
    let Some(update) = updater.check().await.map_err(|e| e.to_string())? else {
        return Err("No update available".into());
    };
    update
        .download_and_install(|_chunk, _total| {}, || {})
        .await
        .map_err(|e| e.to_string())?;
    app.restart();
}

#[tauri::command]
pub fn can_install_update(app: AppHandle) -> bool {
    can_install(&app)
}

fn can_install(app: &AppHandle) -> bool {
    // The bundled-CUDA build is a portable zip with no installer, and its
    // update entry is the plain installer, which would replace it with the
    // non-CUDA app.
    let portable_cuda = app
        .path()
        .resource_dir()
        .map(|r| r.join("engine-payload").join("nvidia").is_dir())
        .unwrap_or(false);
    if portable_cuda {
        return false;
    }
    // Only an AppImage can be swapped in place; AppImage sets $APPIMAGE.
    #[cfg(target_os = "linux")]
    {
        return std::env::var_os("APPIMAGE").is_some();
    }
    #[cfg(not(target_os = "linux"))]
    true
}
