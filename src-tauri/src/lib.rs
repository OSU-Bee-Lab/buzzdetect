use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

// Every structured progress line the Python engine prints on stdout starts
// with this marker (see engine/src/pipeline/progress_json.py) so it can be
// told apart from ordinary log/print output sharing the same stream.
const PROGRESS_MARKER: &str = "BDPROGRESS ";

#[derive(Default)]
struct AnalysisState(Mutex<Option<Child>>);

// engine/ is bundled with buzzdetect2 and ships its own venv (see
// engine/requirements.txt, set up with `uv venv` / `uv pip install`), so the
// interpreter path is fixed rather than a user setting.
fn resolve_engine_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let bundled = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("engine");
    if bundled.join("buzzdetect_cli.py").exists() {
        return Ok(bundled);
    }
    // Dev mode: nothing's bundled yet, engine/ lives next to the project root.
    std::env::current_dir()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or_else(|| "could not resolve project root".to_string())
        .map(|p| p.join("engine"))
}

fn resolve_python_bin(engine_dir: &PathBuf) -> PathBuf {
    engine_dir.join(".venv").join("bin").join("python3")
}

#[tauri::command]
fn list_models(app: AppHandle) -> Result<Vec<String>, String> {
    let models_dir = resolve_engine_dir(&app)?.join("models");
    let mut out = vec![];
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("model.py").exists() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    out.push(name.to_string());
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

#[tauri::command]
fn get_model_classes(app: AppHandle, modelname: String) -> Result<Vec<String>, String> {
    let config_path = resolve_engine_dir(&app)?
        .join("models")
        .join(&modelname)
        .join("config_model.json");
    let text = std::fs::read_to_string(&config_path).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let classes = value
        .get("classes")
        .and_then(|c| c.as_array())
        .ok_or("config_model.json has no 'classes' field")?;
    let mut out: Vec<String> = classes
        .iter()
        .filter_map(|c| c.as_str().map(|s| s.to_string()))
        .collect();
    out.sort();
    Ok(out)
}

#[derive(Serialize, Clone)]
struct Manifest {
    modelname: String,
    classes_out: Option<Vec<String>>,
    framehop_prop: Option<f64>,
}

// buzzdetect writes this into dir_out to record the settings that determine
// result schema/resumability (engine/src/pipeline/manifest.py). If present,
// a run into that folder is locked to match it, so the frontend needs to
// read it before letting the user pick incompatible settings.
#[tauri::command]
fn read_manifest(dir_out: String) -> Result<Option<Manifest>, String> {
    let path = PathBuf::from(&dir_out).join("buzzdetect_manifest.json");
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let modelname = value
        .get("modelname")
        .and_then(|v| v.as_str())
        .ok_or("manifest has no modelname")?
        .to_string();
    let classes_out = value.get("classes_out").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|c| c.as_str().map(|s| s.to_string()))
            .collect()
    });
    let framehop_prop = value.get("framehop_prop").and_then(|v| v.as_f64());
    Ok(Some(Manifest {
        modelname,
        classes_out,
        framehop_prop,
    }))
}

#[derive(Debug, Deserialize)]
pub struct AnalysisSettings {
    modelname: String,
    dir_audio: String,
    dir_out: String,
    // Empty/omitted means "all classes" (buzzdetect_cli's 'all' sentinel).
    #[serde(default)]
    classes_out: Vec<String>,
    #[serde(default = "default_chunklength")]
    chunklength: f64,
    #[serde(default = "default_analyzers_cpu")]
    analyzers_cpu: u32,
    #[serde(default)]
    analyzers_gpu: u32,
    #[serde(default = "default_framehop_prop")]
    framehop_prop: f64,
    #[serde(default)]
    n_streamers: Option<u32>,
    #[serde(default)]
    stream_buffer_depth: Option<u32>,
    #[serde(default = "default_verbosity_print")]
    verbosity_print: String,
    #[serde(default = "default_verbosity_log")]
    verbosity_log: String,
    #[serde(default)]
    log_progress: bool,
}

fn default_chunklength() -> f64 {
    200.0
}
fn default_analyzers_cpu() -> u32 {
    2
}
fn default_framehop_prop() -> f64 {
    1.0
}
fn default_verbosity_print() -> String {
    "PROGRESS".into()
}
fn default_verbosity_log() -> String {
    "DEBUG".into()
}

#[derive(Serialize, Clone)]
struct EngineExit {
    code: Option<i32>,
}

#[tauri::command]
fn start_analysis(
    app: AppHandle,
    state: State<AnalysisState>,
    settings: AnalysisSettings,
) -> Result<(), String> {
    if settings.classes_out.is_empty() {
        return Err("Select at least one class to output".into());
    }

    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if guard.is_some() {
        return Err("An analysis is already running".into());
    }

    let engine_dir = resolve_engine_dir(&app)?;
    let python_bin = resolve_python_bin(&engine_dir);
    if !python_bin.exists() {
        return Err(format!(
            "Engine's Python environment not found at {}. Run `uv venv --python 3.13 .venv && uv pip install -r requirements.txt` in engine/.",
            python_bin.display()
        ));
    }

    let mut cmd = Command::new(&python_bin);
    cmd.current_dir(&engine_dir)
        .arg("buzzdetect_cli.py")
        .arg("--modelname")
        .arg(&settings.modelname)
        .arg("--dir_audio")
        .arg(&settings.dir_audio)
        .arg("--dir_out")
        .arg(&settings.dir_out)
        .arg("--chunklength")
        .arg(settings.chunklength.to_string())
        .arg("--analyzers_cpu")
        .arg(settings.analyzers_cpu.to_string())
        .arg("--analyzers_gpu")
        .arg(settings.analyzers_gpu.to_string())
        .arg("--framehop_prop")
        .arg(settings.framehop_prop.to_string())
        .arg("--verbosity_print")
        .arg(&settings.verbosity_print)
        .arg("--verbosity_log")
        .arg(&settings.verbosity_log)
        .arg("--log_progress")
        .arg(settings.log_progress.to_string())
        .arg("--classes_out")
        .args(&settings.classes_out)
        // Unbuffer Python's stdout so BDPROGRESS lines arrive as they're
        // printed rather than sitting in a pipe buffer until it fills.
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(n) = settings.n_streamers {
        cmd.arg("--n_streamers").arg(n.to_string());
    }
    if let Some(n) = settings.stream_buffer_depth {
        cmd.arg("--stream_buffer_depth").arg(n.to_string());
    }

    let mut child = cmd.spawn().map_err(|e| format!("failed to launch engine: {e}"))?;

    // reconcile_with_manifest (buzzdetect_cli.py) can prompt y/N on stdin if
    // the output folder already holds results from different settings.
    // There's no non-interactive flag for that yet, so we pre-empt it here:
    // always answer yes (adopt the existing settings) rather than let the
    // prompt hang forever with no attached terminal.
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(b"y\n");
    }

    let stdout = child.stdout.take().ok_or("failed to capture engine stdout")?;
    let stderr = child.stderr.take().ok_or("failed to capture engine stderr")?;

    spawn_line_reader(app.clone(), stdout, false);
    spawn_line_reader(app.clone(), stderr, true);

    let app_for_wait = app.clone();
    // Child itself is stored so cancel_analysis can kill it; a second thread
    // polls it to know when to emit engine-exit.
    *guard = Some(child);
    drop(guard);

    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let state_handle = app_for_wait.state::<AnalysisState>();
        let mut guard = match state_handle.0.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if let Some(child) = guard.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let _ = app_for_wait.emit(
                        "engine-exit",
                        EngineExit {
                            code: status.code(),
                        },
                    );
                    *guard = None;
                    return;
                }
                Ok(None) => continue,
                Err(_) => return,
            }
        } else {
            // Cancelled from elsewhere.
            return;
        }
    });

    Ok(())
}

fn spawn_line_reader<R: std::io::Read + Send + 'static>(app: AppHandle, reader: R, is_stderr: bool) {
    std::thread::spawn(move || {
        let buf = BufReader::new(reader);
        for line in buf.lines() {
            let Ok(line) = line else { break };
            if let Some(json_str) = line.strip_prefix(PROGRESS_MARKER) {
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(value) => {
                        let _ = app.emit("engine-progress", value);
                        continue;
                    }
                    Err(_) => { /* fall through to plain log line */ }
                }
            }
            let _ = app.emit("engine-log", serde_json::json!({ "line": line, "stderr": is_stderr }));
        }
    });
}

#[tauri::command]
fn cancel_analysis(state: State<AnalysisState>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = guard.take() {
        child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AnalysisState::default())
        .invoke_handler(tauri::generate_handler![
            start_analysis,
            cancel_analysis,
            list_models,
            get_model_classes,
            read_manifest
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
