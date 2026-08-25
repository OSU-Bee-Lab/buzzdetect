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

// How to invoke the Python engine. Two shapes, because the app has to work
// both as a shipped bundle and out of a checkout:
//
// - Bundled: a PyInstaller sidecar (see engine/buzzdetect.spec) sits next to
//   the app executable, and the parts buzzdetect loads off disk at runtime --
//   models, the ONNX embedder, the stream drivers -- ship as the
//   engine-payload resource directory. Nothing on the user's machine is
//   needed: no Python, no venv.
// - Dev: no sidecar has been built, so run engine/buzzdetect_cli.py out of
//   engine/.venv the way `buzzdetect_cli.py` is run by hand.
//
// Either way the process runs with a working directory containing models/,
// embedders/ and src/stream/drivers/, which is what makes the relative paths
// in engine/src/config.py resolve.
struct Engine {
    program: PathBuf,
    // Dev only: the CLI script to hand the interpreter. Empty when the
    // sidecar, which is the CLI, is what's being run.
    prefix_args: Vec<PathBuf>,
    workdir: PathBuf,
}

#[cfg(target_os = "windows")]
const SIDECAR_NAME: &str = "buzzdetect.exe";
#[cfg(not(target_os = "windows"))]
const SIDECAR_NAME: &str = "buzzdetect";

fn resolve_engine(app: &AppHandle) -> Result<Engine, String> {
    // Tauri strips the target triple when it copies an externalBin into the
    // bundle, so the sidecar lands beside the app executable under its plain
    // name.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let sidecar = exe_dir.join(SIDECAR_NAME);
            if sidecar.exists() {
                if let Ok(resources) = app.path().resource_dir() {
                    let payload = resources.join("engine-payload");
                    if payload.join("models").exists() {
                        return Ok(Engine {
                            program: sidecar,
                            prefix_args: vec![],
                            workdir: payload,
                        });
                    }
                }
            }
        }
    }

    let engine_dir = std::env::current_dir()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or_else(|| "could not resolve project root".to_string())?
        .join("engine");
    let venv_bin = if cfg!(target_os = "windows") {
        engine_dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        engine_dir.join(".venv").join("bin").join("python3")
    };
    Ok(Engine {
        program: venv_bin,
        prefix_args: vec![engine_dir.join("buzzdetect_cli.py")],
        workdir: engine_dir,
    })
}

#[tauri::command]
fn list_models(app: AppHandle) -> Result<Vec<String>, String> {
    let models_dir = resolve_engine(&app)?.workdir.join("models");
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
    let config_path = resolve_engine(&app)?
        .workdir
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

// Whether this build's engine can run a worker on a GPU. Written into the
// payload at build time by scripts/build-engine.mjs, which asks the frozen
// runtime itself -- the CPU installers carry a CPU-only onnxruntime, the CUDA
// ones carry onnxruntime-gpu, and macOS gets CoreML either way. The frontend
// uses it to hide the GPU analyzer control rather than offer a setting that
// would quietly run on the CPU.
#[tauri::command]
fn gpu_available(app: AppHandle) -> Result<bool, String> {
    let path = resolve_engine(&app)?.workdir.join("gpu-providers.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        // No payload means a checkout, where whatever is in engine/.venv
        // decides. Offer the option and let the engine's own warning speak.
        return Ok(true);
    };
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    Ok(value
        .get("gpu_providers")
        .and_then(|p| p.as_array())
        .map(|providers| !providers.is_empty())
        .unwrap_or(false))
}

#[derive(Serialize, Clone)]
struct Manifest {
    modelname: String,
    classes_out: Option<Vec<String>>,
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
    Ok(Some(Manifest {
        modelname,
        classes_out,
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
    #[serde(default)]
    gpu_fp16: bool,
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

    let engine = resolve_engine(&app)?;
    if !engine.program.exists() {
        return Err(format!(
            "Engine not found at {}. In a checkout, build the sidecar with `node scripts/build-engine.mjs`, or set up engine/.venv with `uv venv --python 3.13 .venv && uv pip install -r requirements.txt`.",
            engine.program.display()
        ));
    }

    let mut cmd = Command::new(&engine.program);
    cmd.current_dir(&engine.workdir)
        .args(&engine.prefix_args)
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
        // Reduced precision is a runtime property of the GPU session, not part
        // of the result schema, so it travels as an environment variable rather
        // than a CLI argument (see engine/src/inference/onnx.py).
        .env("BUZZDETECT_GPU_FP16", if settings.gpu_fp16 { "1" } else { "0" })
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(n) = settings.n_streamers {
        cmd.arg("--n_streamers").arg(n.to_string());
    }
    if let Some(n) = settings.stream_buffer_depth {
        cmd.arg("--stream_buffer_depth").arg(n.to_string());
    }

    // The CUDA build's NVIDIA runtime ships as loose libraries in the payload
    // rather than frozen into the sidecar -- 2.5GB in one file is more than
    // makensis will bundle (see engine/buzzdetect.spec's strip_nvidia).
    // onnxruntime dlopen()s them by bare soname, so the child needs that
    // directory on its loader search path, and it has to be set here, in the
    // environment the child is spawned with: both loaders read the variable
    // once, at process start. Absent on the CPU builds, where there's no such
    // directory to find.
    let nvidia = engine.workdir.join("nvidia");
    if nvidia.is_dir() {
        let var_name = if cfg!(target_os = "windows") {
            "PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        let inherited = std::env::var_os(var_name).unwrap_or_default();
        let mut search = vec![nvidia];
        search.extend(std::env::split_paths(&inherited));
        if let Ok(joined) = std::env::join_paths(search) {
            cmd.env(var_name, joined);
        }
    }

    // Its own process group, so cancel_analysis can signal the whole tree.
    // PyInstaller's onefile bootloader forks the real engine as a child of
    // itself, so the pid we get back here is a wrapper: killing just that pid
    // leaves the analysis running, still writing results and still emitting
    // progress on the stdout pipe we're reading.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
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

// How long a cancelled engine gets to wind down before it's killed outright.
const CANCEL_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

#[cfg(unix)]
fn signal_engine(pid: u32, signal: i32) {
    // Negative pid = the whole process group, which is the point: see the
    // process_group call in start_analysis.
    unsafe {
        libc::kill(-(pid as i32), signal);
    }
}

#[cfg(windows)]
fn signal_engine(pid: u32, _signal: i32) {
    // No process groups to signal; /T walks the child tree instead, which is
    // what actually gets PyInstaller's forked worker.
    let _ = Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid.to_string()])
        .status();
}

#[tauri::command]
fn cancel_analysis(app: AppHandle, state: State<AnalysisState>) -> Result<(), String> {
    let pid = {
        let guard = state.0.lock().map_err(|e| e.to_string())?;
        match guard.as_ref() {
            Some(child) => child.id(),
            None => return Ok(()),
        }
    };

    // Deliberately does NOT clear AnalysisState. The waiter thread spawned by
    // start_analysis is what reaps the child and emits engine-exit, and the
    // frontend keeps the run locked until that lands -- so taking the child
    // out here would strand the UI in a stopping state forever.
    #[cfg(unix)]
    signal_engine(pid, libc::SIGTERM);
    #[cfg(windows)]
    signal_engine(pid, 0);

    // Escalate if it hasn't gone by itself. Results are written per chunk, so
    // the worst a hard kill costs is the chunk in flight, which the next run
    // picks up again.
    std::thread::spawn(move || {
        std::thread::sleep(CANCEL_GRACE);
        let state = app.state::<AnalysisState>();
        let still_running = state
            .0
            .lock()
            .map(|guard| guard.as_ref().map(|child| child.id()) == Some(pid))
            .unwrap_or(false);
        if still_running {
            #[cfg(unix)]
            signal_engine(pid, libc::SIGKILL);
            #[cfg(windows)]
            signal_engine(pid, 0);
        }
    });

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
            gpu_available,
            read_manifest
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
