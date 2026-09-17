use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, State};

// Every structured progress line the Python engine prints on stdout starts
// with this marker (see engine/src/pipeline/progress_json.py) so it can be
// told apart from ordinary log/print output sharing the same stream.
const PROGRESS_MARKER: &str = "BDPROGRESS ";

#[derive(Default)]
struct AnalysisState(Mutex<Option<Child>>);

// Everything the current run has told the frontend, kept so a page that
// reloads mid-run can rebuild its view (see attach_analysis). The run outlives
// the page: the child and its reader threads belong to the Rust side, but the
// progress store is page state, and the webview can be reloaded out from under
// it -- by the OS reclaiming its content process, or a plain Cmd+R.
//
// Every emitted event and log line carries a `seq` from one counter, so a page
// can tell which live events its snapshot already covered.
#[derive(Default)]
struct RunRecord {
    started_at_ms: u64,
    next_seq: u64,
    // None where a chunk_done was superseded by its file's final one: the
    // store sets a finished file's doneSeconds outright, so only in-flight
    // files need their chunks replayed. Keeps this O(files), not O(chunks).
    events: Vec<Option<serde_json::Value>>,
    open_chunks: HashMap<String, Vec<usize>>,
    logs: VecDeque<serde_json::Value>,
}

// Matches the frontend store's own cap.
const RECORDED_LOG_LINES: usize = 500;

static RUN_RECORD: Mutex<Option<RunRecord>> = Mutex::new(None);

impl RunRecord {
    fn take_seq(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }

    fn push_event(&mut self, mut value: serde_json::Value) -> serde_json::Value {
        let seq = self.take_seq();
        if let Some(obj) = value.as_object_mut() {
            obj.insert("seq".into(), seq.into());
        }
        let chunk_path = (value["event"] == "chunk_done")
            .then(|| value["path"].as_str().map(str::to_owned))
            .flatten();
        if let Some(path) = chunk_path {
            if value["done"] == true {
                for idx in self.open_chunks.remove(&path).unwrap_or_default() {
                    self.events[idx] = None;
                }
            } else {
                self.open_chunks.entry(path).or_default().push(self.events.len());
            }
        }
        self.events.push(Some(value.clone()));
        value
    }

    fn push_log(&mut self, line: String, is_stderr: bool) -> serde_json::Value {
        let seq = self.take_seq();
        let value = serde_json::json!({ "line": line, "stderr": is_stderr, "seq": seq });
        self.logs.push_back(value.clone());
        while self.logs.len() > RECORDED_LOG_LINES {
            self.logs.pop_front();
        }
        value
    }
}

// How to invoke the Python engine. Two shapes, because the app has to work
// both as a shipped bundle and out of a checkout:
//
// - Bundled: the PyInstaller onedir engine (see engine/buzzdetect.spec) ships
//   inside the engine-payload resource directory at engine-bin/, alongside the
//   parts buzzdetect loads off disk at runtime -- models, the ONNX embedder,
//   the stream drivers. onedir rather than a single-file externalBin because a
//   onefile binary re-extracts itself on every launch (~25s of frozen window,
//   measured). Nothing on the user's machine is needed: no Python, no venv.
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
const SIDECAR_NAME: &str = "buzzdetect-engine.exe";
#[cfg(not(target_os = "windows"))]
const SIDECAR_NAME: &str = "buzzdetect-engine";

fn resolve_engine(app: &AppHandle) -> Result<Engine, String> {
    // Bundled: the onedir engine sits at engine-payload/engine-bin/, and the
    // launcher inside it carries the same name the onefile sidecar used to.
    // The payload directory itself is the working directory, so models/ and
    // src/stream/drivers/ resolve.
    if let Ok(resources) = app.path().resource_dir() {
        let payload = resources.join("engine-payload");
        let launcher = payload.join("engine-bin").join(SIDECAR_NAME);
        if launcher.exists() {
            return Ok(Engine {
                program: launcher,
                prefix_args: vec![],
                workdir: payload,
            });
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

// A model is just model.onnx + config_model.json in a folder. buzzdetect ships
// some in the bundle (engine-payload/models, read-only); users import their own
// into app-local data, which is where import_model writes and where the engine
// is pointed via BUZZDETECT_MODELS_PATH. Kept out of the bundle deliberately:
// writing into a signed .app or Program Files needs privileges and breaks the
// signature (engine/src/inference/models.py has the engine end of this).
const MODEL_MARKER: &str = "config_model.json";

// The framing parameters export_onnx.py writes alongside the class list. A dir
// missing any of these isn't a model this engine can run -- reject it at import
// rather than mid-analysis.
const REQUIRED_CONFIG_KEYS: [&str; 7] = [
    "classes",
    "samplerate",
    "framelength_s",
    "digits_time",
    "digits_results",
    "samples_hop",
    "samples_min",
];

// Files copied out of an imported model directory. Everything else there
// (TensorFlow weights, training history, analysis output) the engine can't use.
const IMPORT_FILES: [&str; 6] = [
    "model.onnx",
    "model.fp16.onnx",
    "config_model.json",
    "translation.csv",
    "weights.csv",
    "README.md",
];

const README: &str = "README.md";

/// Per-user model store, outside the app bundle. `None` if the platform data
/// dir can't be resolved (shouldn't happen in practice).
fn user_models_dir(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_local_data_dir().ok().map(|d| d.join("models"))
}

/// Every model root the engine will search, highest priority first: the
/// bundled dir, then the user store.
fn model_roots(app: &AppHandle) -> Vec<PathBuf> {
    let mut roots = vec![resolve_engine(app)
        .map(|e| e.workdir.join("models"))
        .unwrap_or_default()];
    if let Some(user) = user_models_dir(app) {
        roots.push(user);
    }
    roots
}

fn model_dir(app: &AppHandle, modelname: &str) -> Option<PathBuf> {
    model_roots(app)
        .into_iter()
        .map(|r| r.join(modelname))
        .find(|p| p.join(MODEL_MARKER).is_file())
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    /// A user-imported model, so remove_model can delete it. Bundled models are
    /// part of the install and can't be removed here.
    removable: bool,
    /// config_model.json's optional one-line `description`, written by hand.
    description: Option<String>,
    has_readme: bool,
}

impl ModelInfo {
    fn read(dir: &std::path::Path, name: &str, removable: bool) -> ModelInfo {
        let config = read_config(dir).unwrap_or_default();
        ModelInfo {
            name: name.to_string(),
            removable,
            description: description_of(&config),
            has_readme: dir.join(README).is_file(),
        }
    }
}

fn read_config(dir: &std::path::Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join(MODEL_MARKER)).ok()?;
    serde_json::from_str(&text).ok()
}

/// A blank or non-string description is no description.
fn description_of(config: &serde_json::Value) -> Option<String> {
    config
        .get("description")
        .and_then(|d| d.as_str())
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(str::to_string)
}

#[tauri::command]
fn list_models(app: AppHandle) -> Result<Vec<ModelInfo>, String> {
    let roots = model_roots(&app);
    let mut out: Vec<ModelInfo> = vec![];
    for (i, root) in roots.iter().enumerate() {
        let removable = i > 0; // root 0 is the bundled dir
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.join(MODEL_MARKER).is_file() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                // Earlier root wins a name collision: a bundled model shadows
                // an imported one, matching the engine's resolution order.
                if out.iter().any(|m| m.name == name) {
                    continue;
                }
                out.push(ModelInfo::read(&path, name, removable));
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[tauri::command]
fn get_model_classes(app: AppHandle, modelname: String) -> Result<Vec<String>, String> {
    let config_path = model_dir(&app, &modelname)
        .ok_or_else(|| format!("model '{modelname}' not found"))?
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

/// Validate that `dir` holds a model this engine can run, and return its
/// config_model.json parsed. Mirrors engine/src/inference/models.py's
/// _validate_config so the failure shows up at import, not mid-analysis.
fn validate_model_dir(dir: &std::path::Path) -> Result<serde_json::Value, String> {
    if !dir.join("model.onnx").is_file() {
        return Err("that folder has no model.onnx".into());
    }
    let config_path = dir.join("config_model.json");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|_| "that folder has no config_model.json".to_string())?;
    let config: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("config_model.json is not valid JSON: {e}"))?;

    let missing: Vec<&str> = REQUIRED_CONFIG_KEYS
        .iter()
        .copied()
        .filter(|k| config.get(k).is_none())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "config_model.json is missing required key(s): {}. Re-export the \
             model with buzzdetect-training's tools/export_onnx.py.",
            missing.join(", ")
        ));
    }
    let classes_ok = config
        .get("classes")
        .and_then(|c| c.as_array())
        .map(|a| !a.is_empty() && a.iter().all(|c| c.is_string()))
        .unwrap_or(false);
    if !classes_ok {
        return Err("config_model.json: \"classes\" must be a non-empty list of strings".into());
    }
    Ok(config)
}

/// Extract a .zip to a fresh scratch directory the caller is responsible for
/// removing. zip's `extract` uses enclosed_name(), so entries can't escape.
fn unzip_to_scratch(zip_path: &std::path::Path) -> Result<PathBuf, String> {
    let file = std::fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("not a readable .zip: {e}"))?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let scratch = std::env::temp_dir().join(format!("buzzdetect-import-{nanos}"));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    if let Err(e) = archive.extract(&scratch) {
        let _ = std::fs::remove_dir_all(&scratch);
        return Err(format!("couldn't unpack the .zip: {e}"));
    }
    Ok(scratch)
}

/// The directory holding config_model.json within an extracted zip: either the
/// root itself or a single wrapper folder one level down.
fn find_model_root(extracted: &std::path::Path) -> Option<PathBuf> {
    if extracted.join(MODEL_MARKER).is_file() {
        return Some(extracted.to_path_buf());
    }
    for entry in std::fs::read_dir(extracted).ok()?.flatten() {
        let path = entry.path();
        if path.join(MODEL_MARKER).is_file() {
            return Some(path);
        }
    }
    None
}

/// Copy an imported model into the per-user store. `src` is either a folder the
/// user picked or a .zip of one; the folder name (or the .zip's filename)
/// becomes the model name. Refuses a name that collides with an existing model.
#[tauri::command]
fn import_model(app: AppHandle, src: String) -> Result<ModelInfo, String> {
    let src = PathBuf::from(&src);
    let is_zip = src
        .extension()
        .map(|e| e.eq_ignore_ascii_case("zip"))
        .unwrap_or(false);

    // For a zip, the name is its filename stem; for a folder, the folder name.
    let name = src
        .file_stem()
        .filter(|_| is_zip || src.is_dir())
        .and_then(|n| n.to_str())
        .ok_or("pick a model folder or a .zip file")?
        .to_string();
    if name.starts_with('.') || name.contains(['/', '\\']) {
        return Err(format!("'{name}' is not a usable model name"));
    }

    // Resolve the folder to copy from, unpacking a zip into scratch first.
    let scratch = if is_zip {
        Some(unzip_to_scratch(&src)?)
    } else if src.is_dir() {
        None
    } else {
        return Err("pick a model folder or a .zip file".into());
    };
    let result = (|| {
        let model_src = match &scratch {
            Some(dir) => find_model_root(dir)
                .ok_or_else(|| "that .zip has no config_model.json in it".to_string())?,
            None => src.clone(),
        };

        validate_model_dir(&model_src)?;

        if model_dir(&app, &name).is_some() {
            return Err(format!(
                "a model named '{name}' already exists. Rename it and try again."
            ));
        }

        let dest = user_models_dir(&app)
            .ok_or("couldn't resolve the app data directory")?
            .join(&name);
        std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;

        let copied = (|| -> std::io::Result<()> {
            for file in IMPORT_FILES {
                let from = model_src.join(file);
                if from.is_file() {
                    std::fs::copy(&from, dest.join(file))?;
                }
            }
            Ok(())
        })();
        if let Err(e) = copied {
            let _ = std::fs::remove_dir_all(&dest);
            return Err(format!("failed to copy the model in: {e}"));
        }

        Ok(ModelInfo::read(&dest, &name, true))
    })();

    if let Some(dir) = scratch {
        let _ = std::fs::remove_dir_all(dir);
    }
    result
}

/// Everything the model info window shows: the README, and the thresholds that
/// buzzdetect-training wrote into config_model.json. `thresholds` is a plain
/// {class: number} map; `threshold_stats` is how much each one rests on. Both
/// are passed through as-is, and either may be absent.
#[derive(Serialize)]
struct ModelDetails {
    name: String,
    description: Option<String>,
    readme: Option<String>,
    thresholds: Option<serde_json::Value>,
    threshold_stats: Option<serde_json::Value>,
}

fn model_details_in(dir: &std::path::Path, name: &str) -> ModelDetails {
    let config = read_config(dir).unwrap_or_default();
    let object = |key: &str| config.get(key).filter(|v| v.is_object()).cloned();
    ModelDetails {
        name: name.to_string(),
        description: description_of(&config),
        readme: std::fs::read_to_string(dir.join(README)).ok(),
        thresholds: object("thresholds"),
        threshold_stats: object("threshold_stats"),
    }
}

#[tauri::command]
fn model_details(app: AppHandle, modelname: String) -> Result<ModelDetails, String> {
    let dir = model_dir(&app, &modelname).ok_or_else(|| format!("model '{modelname}' not found"))?;
    Ok(model_details_in(&dir, &modelname))
}

const MODEL_INFO_WINDOW: &str = "model-info";

fn percent_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Open the model info window on `modelname`, or point the open one at it.
/// async because building a window from a synchronous command deadlocks on
/// Windows.
#[tauri::command]
async fn open_model_info(app: AppHandle, modelname: String) -> Result<(), String> {
    if let Some(window) = app.get_webview_window(MODEL_INFO_WINDOW) {
        app.emit_to(MODEL_INFO_WINDOW, "model-info-select", &modelname)
            .map_err(|e| e.to_string())?;
        let _ = window.unminimize();
        return window.set_focus().map_err(|e| e.to_string());
    }
    let url = format!("model-info?model={}", percent_encode(&modelname));
    tauri::WebviewWindowBuilder::new(&app, MODEL_INFO_WINDOW, tauri::WebviewUrl::App(url.into()))
        .title("Model info")
        .inner_size(860.0, 640.0)
        .min_inner_size(480.0, 320.0)
        .build()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// `rel` resolved inside the model directory, or None if it names nothing
/// there -- including anything that climbs out of it.
fn file_in_model_dir(dir: &std::path::Path, rel: &str) -> Option<PathBuf> {
    let root = dir.canonicalize().ok()?;
    let path = root.join(rel).canonicalize().ok()?;
    (path.starts_with(&root) && path.is_file()).then_some(path)
}

/// Open a file a README links to (e.g. tests/metrics.svg) with the system's
/// default app.
#[tauri::command]
fn open_model_file(app: AppHandle, modelname: String, rel: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let dir = model_dir(&app, &modelname).ok_or_else(|| format!("model '{modelname}' not found"))?;
    let path = file_in_model_dir(&dir, &rel)
        .ok_or_else(|| format!("{rel} isn't in this model's folder"))?;
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| e.to_string())
}

/// Delete a user-imported model. Bundled models aren't in the user store and
/// can't be removed here.
#[tauri::command]
fn remove_model(app: AppHandle, name: String) -> Result<(), String> {
    let dir = user_models_dir(&app)
        .ok_or("couldn't resolve the app data directory")?
        .join(&name);
    // Guard against `..` and absolute names slipping through.
    let user_root = user_models_dir(&app).unwrap();
    if dir.parent() != Some(user_root.as_path()) || !dir.join(MODEL_MARKER).is_file() {
        return Err(format!("'{name}' is not an imported model"));
    }
    std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())
}

/// What the frontend needs to decide whether to offer the GPU controls.
///
/// Two separate questions, because they have different answers. `supported` is
/// about this build -- the CPU installers carry a CPU-only onnxruntime and can
/// never use a GPU, so there's nothing to offer and nothing to check. `usable`
/// is about this machine, and is the one that needs an actual look.
#[derive(Serialize)]
struct GpuStatus {
    supported: bool,
    usable: bool,
    providers: Vec<String>,
    detail: Option<String>,
}

// How long to let the probe run before giving up on it. Generous: it pays for
// the engine's first frozen import and for CUDA initialising a context on a
// cold driver. The point is only that a wedged driver can't leave the UI waiting.
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);

/// Whether a GPU worker would actually reach a GPU on this machine.
///
/// `gpu-providers.json` is written at build time by scripts/build-engine.mjs
/// and answers the build question only -- onnxruntime reports the providers it
/// was compiled with, which is the same answer on a workstation with a full
/// CUDA install and on a laptop with no NVIDIA hardware at all. So a build that
/// could use a GPU has to ask the engine to try one; see probe_gpu in
/// engine/src/inference/onnx.py.
///
/// The bundled-CUDA build is the exception: it ships the NVIDIA runtime itself
/// (engine-payload/nvidia), so there's nothing about the machine left to
/// discover and the probe would only cost a second at startup.
#[tauri::command]
async fn gpu_status(app: AppHandle) -> Result<GpuStatus, String> {
    let engine = resolve_engine(&app)?;
    let workdir = engine.workdir.clone();

    let built_in = std::fs::read_to_string(workdir.join("gpu-providers.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .map(|value| {
            value
                .get("gpu_providers")
                .and_then(|p| p.as_array())
                .map(|providers| {
                    providers
                        .iter()
                        .filter_map(|p| p.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        });

    // No file at all means a checkout, where engine/.venv decides and there's
    // no build-time answer to consult. Probe it like any other GPU build.
    if let Some(providers) = &built_in {
        if providers.is_empty() {
            return Ok(GpuStatus {
                supported: false,
                usable: false,
                providers: vec![],
                detail: None,
            });
        }
    }

    // Two builds have nothing to discover, and shouldn't spend several seconds
    // of startup discovering it:
    //
    // - the bundled-CUDA build, which ships the NVIDIA runtime itself
    //   (engine-payload/nvidia) rather than looking for the machine's;
    // - any build whose only GPU provider is CoreML, which is part of macOS and
    //   so can't be absent the way a CUDA install can.
    let self_contained = workdir.join("nvidia").is_dir()
        || built_in.as_deref().is_some_and(|providers| {
            !providers.is_empty()
                && providers
                    .iter()
                    .all(|p| p == "CoreMLExecutionProvider")
        });
    if self_contained {
        return Ok(GpuStatus {
            supported: true,
            usable: true,
            providers: built_in.unwrap_or_default(),
            detail: None,
        });
    }

    match probe_gpu(&engine) {
        Ok(providers) if !providers.is_empty() => Ok(GpuStatus {
            supported: true,
            usable: true,
            providers,
            detail: None,
        }),
        Ok(_) => Ok(GpuStatus {
            supported: true,
            usable: false,
            providers: vec![],
            detail: Some(
                "No usable GPU runtime was found on this machine. For NVIDIA GPUs, \
                 install CUDA 12 and cuDNN 9; this build can't use CUDA 11 or cuDNN 8."
                    .into(),
            ),
        }),
        Err(e) => Ok(GpuStatus {
            supported: true,
            usable: false,
            providers: vec![],
            detail: Some(format!("Couldn't check this machine for a GPU: {e}")),
        }),
    }
}

/// Run the engine's own GPU probe and read back the providers it managed to load.
fn probe_gpu(engine: &Engine) -> Result<Vec<String>, String> {
    let mut cmd = Command::new(&engine.program);
    cmd.current_dir(&engine.workdir)
        .args(&engine.prefix_args)
        .arg("--probe_gpu")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let child = cmd.spawn().map_err(|e| e.to_string())?;
    let pid = child.id();

    // A driver in a bad state can hang session creation rather than failing it,
    // and the frontend is sitting on a spinner until this returns.
    let finished = Arc::new(AtomicBool::new(false));
    {
        let finished = finished.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + PROBE_TIMEOUT;
            while Instant::now() < deadline {
                if finished.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            kill_pid(pid);
        });
    }

    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    finished.store(true, Ordering::Relaxed);

    if !output.status.success() {
        return Err("the engine's GPU probe exited without an answer".into());
    }

    // onnxruntime and CoreML both narrate on stdout as well as stderr, so take
    // the line that parses rather than assuming the last one is ours.
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            continue;
        };
        if let Some(providers) = value.get("gpu_providers").and_then(|p| p.as_array()) {
            return Ok(providers
                .iter()
                .filter_map(|p| p.as_str().map(str::to_string))
                .collect());
        }
    }
    Err("the engine's GPU probe printed no result".into())
}

/// Kill one process, and only that one -- unlike signal_engine, which signals a
/// whole process group and would take this app down with it.
#[cfg(unix)]
fn kill_pid(pid: u32) {
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[cfg(windows)]
fn kill_pid(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .status();
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

// The command line for one analysis, in the order buzzdetect_cli.py's argparse
// wants it: --classes_out takes any number of values, so nothing bare may
// follow it -- only another flag.
fn engine_args(settings: &AnalysisSettings) -> Vec<String> {
    let mut args = vec![
        "--modelname".into(),
        settings.modelname.clone(),
        "--dir_audio".into(),
        settings.dir_audio.clone(),
        "--dir_out".into(),
        settings.dir_out.clone(),
        "--chunklength".into(),
        settings.chunklength.to_string(),
        "--analyzers_cpu".into(),
        settings.analyzers_cpu.to_string(),
        "--analyzers_gpu".into(),
        settings.analyzers_gpu.to_string(),
        "--verbosity_print".into(),
        settings.verbosity_print.clone(),
        "--verbosity_log".into(),
        settings.verbosity_log.clone(),
        "--log_progress".into(),
        settings.log_progress.to_string(),
        "--classes_out".into(),
    ];
    args.extend(settings.classes_out.iter().cloned());

    if let Some(n) = settings.n_streamers {
        args.push("--n_streamers".into());
        args.push(n.to_string());
    }
    if let Some(n) = settings.stream_buffer_depth {
        args.push("--stream_buffer_depth".into());
        args.push(n.to_string());
    }
    args
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
        .args(engine_args(&settings))
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

    // Point the engine at the per-user model store (outside the app bundle) so
    // an imported model resolves by name just like a bundled one. The bundled
    // dir is found via the child's cwd; this is the extra root.
    if let Some(user_models) = user_models_dir(&app) {
        if user_models.is_dir() {
            cmd.env("BUZZDETECT_MODELS_PATH", &user_models);
        }
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
    //
    // The pipe is deliberately left open afterwards rather than dropped: it's
    // also how cancel_analysis asks for a tidy stop (STOP_COMMAND in
    // engine/src/pipeline/interrupt.py), which is the only way to ask on
    // Windows, where there is no SIGTERM to send.
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(b"y\n");
        let _ = stdin.flush();
    }

    if let Ok(mut record) = RUN_RECORD.lock() {
        *record = Some(RunRecord {
            started_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or_default(),
            ..Default::default()
        });
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

#[derive(Serialize)]
struct RunSnapshot {
    running: bool,
    started_at_ms: u64,
    events: Vec<serde_json::Value>,
    logs: Vec<serde_json::Value>,
}

/// What a freshly loaded page needs to pick up a run already in progress.
/// The page subscribes to the live events first and asks for this second,
/// then drops any live event whose seq the snapshot already covered.
#[tauri::command]
fn attach_analysis(state: State<AnalysisState>) -> Result<RunSnapshot, String> {
    let running = state.0.lock().map_err(|e| e.to_string())?.is_some();
    let record = RUN_RECORD.lock().map_err(|e| e.to_string())?;
    Ok(match (running, record.as_ref()) {
        (true, Some(r)) => RunSnapshot {
            running,
            started_at_ms: r.started_at_ms,
            events: r.events.iter().flatten().cloned().collect(),
            logs: r.logs.iter().cloned().collect(),
        },
        _ => RunSnapshot { running: false, started_at_ms: 0, events: vec![], logs: vec![] },
    })
}

/// When the engine last said anything. A cancelled engine is given room to
/// wind down for as long as it's still reporting; see cancel_analysis.
static LAST_ENGINE_OUTPUT: Mutex<Option<Instant>> = Mutex::new(None);

fn note_engine_output() {
    if let Ok(mut last) = LAST_ENGINE_OUTPUT.lock() {
        *last = Some(Instant::now());
    }
}

fn engine_quiet_for() -> Duration {
    LAST_ENGINE_OUTPUT
        .lock()
        .ok()
        .and_then(|last| *last)
        .map(|t| t.elapsed())
        .unwrap_or_default()
}

// What one line of the engine's output turns into. A marked line that doesn't
// parse is not an error: it becomes an ordinary log line, so a stray
// BDPROGRESS in someone's print output can't take the run down.
#[derive(Debug, PartialEq)]
enum EngineLine {
    Progress(serde_json::Value),
    Log(String),
}

fn classify_line(line: String) -> EngineLine {
    if let Some(json_str) = line.strip_prefix(PROGRESS_MARKER) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
            return EngineLine::Progress(value);
        }
    }
    EngineLine::Log(line)
}

fn spawn_line_reader<R: std::io::Read + Send + 'static>(app: AppHandle, reader: R, is_stderr: bool) {
    std::thread::spawn(move || {
        let buf = BufReader::new(reader);
        for line in buf.lines() {
            let Ok(line) = line else { break };
            note_engine_output();
            // Recorded and emitted under one lock, so a snapshot taken by
            // attach_analysis falls cleanly between two events.
            let Ok(mut record) = RUN_RECORD.lock() else { break };
            let Some(record) = record.as_mut() else { continue };
            match classify_line(line) {
                EngineLine::Progress(value) => {
                    let _ = app.emit("engine-progress", record.push_event(value));
                }
                EngineLine::Log(line) => {
                    let _ = app.emit("engine-log", record.push_log(line, is_stderr));
                }
            }
        }
    });
}

// A cancelled engine winds itself down -- workers finish the chunk in flight,
// then the streamers, analyzers and writer report themselves out in turn --
// and the point of stopping it politely is that the user gets to watch that
// happen. So the clock that ends its life is a silence, not a stopwatch: it
// gets killed once it has stopped saying anything for this long...
const CANCEL_QUIET_GRACE: Duration = Duration::from_secs(15);
// ...with an outer bound for an engine that's chatty but wedged.
const CANCEL_MAX_GRACE: Duration = Duration::from_secs(120);
// How often the escalation thread rechecks those two.
const CANCEL_POLL: Duration = Duration::from_millis(250);

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

    // Ask, rather than signal. The engine takes this as a request to run its
    // early-exit path, which unwinds the workers in order and logs each one
    // out -- all of which the user is still watching, since the pipes stay
    // open until the process actually goes.
    {
        let mut guard = state.0.lock().map_err(|e| e.to_string())?;
        if let Some(stdin) = guard.as_mut().and_then(|child| child.stdin.as_mut()) {
            let _ = stdin.write_all(b"STOP\n");
            let _ = stdin.flush();
        }
    }
    note_engine_output(); // start the silence clock from the request itself

    // Deliberately does NOT clear AnalysisState. The waiter thread spawned by
    // start_analysis is what reaps the child and emits engine-exit, and the
    // frontend keeps the run locked until that lands -- so taking the child
    // out here would strand the UI in a stopping state forever.

    // Escalate only once it's gone quiet (or taken far too long). Results are
    // written per chunk, so the worst a hard kill costs is the chunk in
    // flight, which the next run picks up again.
    std::thread::spawn(move || {
        let deadline = Instant::now() + CANCEL_MAX_GRACE;
        loop {
            std::thread::sleep(CANCEL_POLL);
            let state = app.state::<AnalysisState>();
            let still_running = state
                .0
                .lock()
                .map(|guard| guard.as_ref().map(|child| child.id()) == Some(pid))
                .unwrap_or(false);
            if !still_running {
                return;
            }
            if engine_quiet_for() < CANCEL_QUIET_GRACE && Instant::now() < deadline {
                continue;
            }
            #[cfg(unix)]
            {
                signal_engine(pid, libc::SIGTERM);
                std::thread::sleep(EXIT_GRACE);
                signal_engine(pid, libc::SIGKILL);
            }
            #[cfg(windows)]
            signal_engine(pid, 0);
            return;
        }
    });

    Ok(())
}

/// The stop button's second click: skip the wind-down and kill the whole
/// engine tree now. Like cancel_analysis, it leaves AnalysisState alone for the
/// waiter thread to reap and report as engine-exit.
#[tauri::command]
fn kill_analysis(state: State<AnalysisState>) -> Result<(), String> {
    let pid = {
        let guard = state.0.lock().map_err(|e| e.to_string())?;
        match guard.as_ref() {
            Some(child) => child.id(),
            None => return Ok(()),
        }
    };
    #[cfg(unix)]
    signal_engine(pid, libc::SIGKILL);
    #[cfg(windows)]
    signal_engine(pid, 0);
    Ok(())
}

// How long an engine gets to wind down when the app itself is on the way out.
// Shorter than CANCEL_GRACE: the user is closing the window, not waiting on a
// tidy stop, and a chunk in flight is re-analysed by the next run anyway.
const EXIT_GRACE: Duration = Duration::from_secs(2);

/// Stop the engine when the app exits, instead of leaving it orphaned.
///
/// The engine runs in its own process group (see start_analysis), so nothing
/// takes it down with the app: it survives, but not usefully -- the app's end
/// of the stdout pipe closes with it, and the engine wedges on its next
/// progress write. So it has to be killed explicitly.
///
/// Not a complete guarantee, and can't be: a SIGKILL or a force-quit gives the
/// app no chance to run this, and the engine is orphaned again. It covers
/// closing the window and quitting, which is how the app is actually exited.
fn kill_engine_on_exit(app: &AppHandle) {
    let state = app.state::<AnalysisState>();
    // Taken out of the state rather than borrowed: nothing else is going to
    // reap this child, and the waiter thread's engine-exit event has no
    // frontend left to reach.
    let Some(mut child) = state.0.lock().ok().and_then(|mut guard| guard.take()) else {
        return;
    };
    let pid = child.id();

    #[cfg(unix)]
    {
        signal_engine(pid, libc::SIGTERM);
        let deadline = Instant::now() + EXIT_GRACE;
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(_) => return,
            }
        }
        signal_engine(pid, libc::SIGKILL);
        let _ = child.wait();
    }

    // taskkill /F /T is already a hard kill of the whole tree, so there's
    // nothing to escalate to and nothing to wait for.
    #[cfg(windows)]
    {
        signal_engine(pid, 0);
        let _ = child.wait();
    }
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
            kill_analysis,
            attach_analysis,
            list_models,
            get_model_classes,
            model_details,
            open_model_info,
            open_model_file,
            import_model,
            remove_model,
            gpu_status,
            read_manifest
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                kill_engine_on_exit(app);
            }
        });
}

#[cfg(test)]
mod tests {
    // The engine is a subprocess with a text protocol, so what's testable on
    // this side is the two edges of that: the command line the child is given,
    // and what its output turns into. See engine/tests/test_cli.py for the
    // same protocol asserted from the engine's end.
    use super::*;

    fn settings(json: serde_json::Value) -> AnalysisSettings {
        serde_json::from_value(json).expect("settings")
    }

    fn minimal() -> AnalysisSettings {
        settings(serde_json::json!({
            "modelname": "model_general_v3",
            "dir_audio": "/data/audio",
            "dir_out": "/data/out",
            "classes_out": ["ins_buzz"],
        }))
    }

    fn value_after(args: &[String], flag: &str) -> Option<String> {
        args.iter().position(|a| a == flag).map(|i| args[i + 1].clone())
    }

    #[test]
    fn settings_fall_back_to_the_defaults_the_ui_shows() {
        let s = minimal();
        assert_eq!(s.chunklength, 200.0);
        assert_eq!(s.analyzers_cpu, 2);
        assert_eq!(s.analyzers_gpu, 0);
        assert_eq!(s.verbosity_print, "PROGRESS");
        assert_eq!(s.verbosity_log, "DEBUG");
        assert!(!s.gpu_fp16);
        assert!(!s.log_progress);
        assert!(s.n_streamers.is_none());
        assert!(s.stream_buffer_depth.is_none());
    }

    #[test]
    fn every_setting_reaches_the_engine() {
        let args = engine_args(&settings(serde_json::json!({
            "modelname": "m",
            "dir_audio": "/a",
            "dir_out": "/o",
            "classes_out": ["ins_buzz", "frog"],
            "chunklength": 50.5,
            "analyzers_cpu": 4,
            "analyzers_gpu": 1,
            "verbosity_print": "DEBUG",
            "verbosity_log": "INFO",
            "log_progress": true,
            "n_streamers": 3,
            "stream_buffer_depth": 7,
        })));
        assert_eq!(value_after(&args, "--modelname").as_deref(), Some("m"));
        assert_eq!(value_after(&args, "--dir_audio").as_deref(), Some("/a"));
        assert_eq!(value_after(&args, "--dir_out").as_deref(), Some("/o"));
        assert_eq!(value_after(&args, "--chunklength").as_deref(), Some("50.5"));
        assert_eq!(value_after(&args, "--analyzers_cpu").as_deref(), Some("4"));
        assert_eq!(value_after(&args, "--analyzers_gpu").as_deref(), Some("1"));
        assert_eq!(value_after(&args, "--verbosity_print").as_deref(), Some("DEBUG"));
        assert_eq!(value_after(&args, "--verbosity_log").as_deref(), Some("INFO"));
        assert_eq!(value_after(&args, "--log_progress").as_deref(), Some("true"));
        assert_eq!(value_after(&args, "--n_streamers").as_deref(), Some("3"));
        assert_eq!(value_after(&args, "--stream_buffer_depth").as_deref(), Some("7"));
    }

    #[test]
    fn unset_streamer_settings_are_left_to_the_engine_to_work_out() {
        let args = engine_args(&minimal());
        assert!(!args.iter().any(|a| a == "--n_streamers"));
        assert!(!args.iter().any(|a| a == "--stream_buffer_depth"));
    }

    #[test]
    fn every_selected_class_is_passed_and_nothing_bare_follows_them() {
        // --classes_out takes any number of values, so a bare argument after
        // it would be swallowed as another class.
        let args = engine_args(&settings(serde_json::json!({
            "modelname": "m",
            "dir_audio": "/a",
            "dir_out": "/o",
            "classes_out": ["ins_buzz", "frog", "human"],
            "n_streamers": 3,
        })));
        let at = args.iter().position(|a| a == "--classes_out").unwrap();
        assert_eq!(&args[at + 1..at + 4], ["ins_buzz", "frog", "human"]);
        assert!(args[at + 4..].chunks(2).all(|pair| pair[0].starts_with("--")));
    }

    #[test]
    fn a_path_with_spaces_stays_one_argument() {
        let args = engine_args(&settings(serde_json::json!({
            "modelname": "m",
            "dir_audio": "/data/my recordings",
            "dir_out": "/o",
            "classes_out": ["ins_buzz"],
        })));
        assert_eq!(value_after(&args, "--dir_audio").as_deref(), Some("/data/my recordings"));
    }

    #[test]
    fn a_finished_file_replays_without_its_intermediate_chunks() {
        let mut r = RunRecord::default();
        let chunk = |path: &str, done: bool| {
            serde_json::json!({ "event": "chunk_done", "path": path, "chunk_start": 0, "chunk_end": 1, "done": done })
        };
        r.push_event(serde_json::json!({ "event": "file_start", "path": "a.wav" }));
        r.push_event(chunk("a.wav", false));
        r.push_event(chunk("b.wav", false));
        r.push_log("hello".into(), false);
        r.push_event(chunk("a.wav", false));
        let last = r.push_event(chunk("a.wav", true));

        assert_eq!(last["seq"], 5);
        let kept: Vec<_> = r.events.iter().flatten().collect();
        assert_eq!(kept.len(), 3);
        assert_eq!(kept[1]["path"], "b.wav");
        assert_eq!(kept[2]["done"], true);
        assert_eq!(r.logs[0]["seq"], 3);
    }

    #[test]
    fn a_marked_line_becomes_a_progress_event() {
        let line = format!("{PROGRESS_MARKER}{{\"event\": \"chunk_done\", \"done\": true}}");
        match classify_line(line) {
            EngineLine::Progress(value) => {
                assert_eq!(value["event"], "chunk_done");
                assert_eq!(value["done"], true);
            }
            other => panic!("expected a progress event, got {other:?}"),
        }
    }

    #[test]
    fn an_ordinary_log_line_is_left_alone() {
        let line = "2026-05-18 10:00:00 [INFO] streamer 0: launching".to_string();
        assert_eq!(classify_line(line.clone()), EngineLine::Log(line));
    }

    #[test]
    fn a_marked_line_that_does_not_parse_is_logged_rather_than_dropped() {
        let line = format!("{PROGRESS_MARKER}not json");
        assert_eq!(classify_line(line.clone()), EngineLine::Log(line));
    }

    #[test]
    fn the_marker_has_to_start_the_line() {
        let line = format!("some prefix {PROGRESS_MARKER}{{}}");
        assert!(matches!(classify_line(line), EngineLine::Log(_)));
    }

    #[test]
    fn the_marker_is_the_one_the_engine_writes() {
        // engine/src/pipeline/progress_json.py's MARKER, trailing space included.
        assert_eq!(PROGRESS_MARKER, "BDPROGRESS ");
    }

    #[test]
    fn an_output_folder_with_no_manifest_reads_as_nothing_to_match() {
        let dir = std::env::temp_dir().join("buzzdetect-test-no-manifest");
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(dir.join("buzzdetect_manifest.json"));
        assert!(read_manifest(dir.to_string_lossy().into()).unwrap().is_none());
    }

    #[test]
    fn a_manifest_is_read_for_the_two_fields_the_ui_locks_on() {
        let dir = std::env::temp_dir().join("buzzdetect-test-manifest");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("buzzdetect_manifest.json"),
            r#"{"modelname": "model_general_v3", "output_mode": "activations",
                "classes_out": ["frog", "ins_buzz"], "precision": null, "framehop_prop": 1}"#,
        )
        .unwrap();
        let manifest = read_manifest(dir.to_string_lossy().into()).unwrap().unwrap();
        assert_eq!(manifest.modelname, "model_general_v3");
        assert_eq!(manifest.classes_out.unwrap(), ["frog", "ins_buzz"]);
    }

    #[test]
    fn a_detections_manifest_locks_no_classes() {
        let dir = std::env::temp_dir().join("buzzdetect-test-manifest-detections");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("buzzdetect_manifest.json"),
            r#"{"modelname": "m", "output_mode": "detections", "classes_out": null,
                "precision": 0.95, "framehop_prop": 1}"#,
        )
        .unwrap();
        let manifest = read_manifest(dir.to_string_lossy().into()).unwrap().unwrap();
        assert!(manifest.classes_out.is_none());
    }

    #[test]
    fn an_unreadable_manifest_is_an_error_rather_than_a_wrong_answer() {
        let dir = std::env::temp_dir().join("buzzdetect-test-manifest-bad");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("buzzdetect_manifest.json"), "{ not json").unwrap();
        assert!(read_manifest(dir.to_string_lossy().into()).is_err());

        std::fs::write(dir.join("buzzdetect_manifest.json"), r#"{"precision": null}"#).unwrap();
        assert!(read_manifest(dir.to_string_lossy().into()).is_err());
    }

    fn good_config() -> &'static str {
        r#"{"classes": ["a", "b"], "samplerate": 16000, "framelength_s": 0.96,
            "digits_time": 2, "digits_results": 2, "samples_hop": 15360,
            "samples_min": 15600}"#
    }

    #[test]
    fn a_model_dir_needs_the_onnx_and_a_complete_config() {
        let dir = std::env::temp_dir().join("buzzdetect-test-import-ok");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // No model.onnx yet.
        std::fs::write(dir.join("config_model.json"), good_config()).unwrap();
        assert!(validate_model_dir(&dir).is_err());

        std::fs::write(dir.join("model.onnx"), b"not really a graph").unwrap();
        assert!(validate_model_dir(&dir).is_ok());
    }

    #[test]
    fn a_config_missing_a_framing_key_is_rejected_by_name() {
        let dir = std::env::temp_dir().join("buzzdetect-test-import-missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.onnx"), b"x").unwrap();
        let stripped = good_config().replace(r#""samples_hop": 15360,"#, "");
        std::fs::write(dir.join("config_model.json"), stripped).unwrap();

        let err = validate_model_dir(&dir).unwrap_err();
        assert!(err.contains("samples_hop"), "{err}");
    }

    #[test]
    fn a_zip_wrapping_the_model_in_a_folder_is_found_and_unpacked() {
        use std::io::Write;
        let zip_path = std::env::temp_dir().join("buzzdetect-test-bundle.zip");
        let file = std::fs::File::create(&zip_path).unwrap();
        let mut w = zip::ZipWriter::new(file);
        let opts: zip::write::SimpleFileOptions = Default::default();
        w.start_file("redwood/config_model.json", opts).unwrap();
        w.write_all(good_config().as_bytes()).unwrap();
        w.start_file("redwood/model.onnx", opts).unwrap();
        w.write_all(b"graph").unwrap();
        w.finish().unwrap();

        let scratch = unzip_to_scratch(&zip_path).unwrap();
        let root = find_model_root(&scratch).expect("model root");
        assert_eq!(root.file_name().unwrap(), "redwood");
        assert!(validate_model_dir(&root).is_ok());
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    fn a_model_without_the_optional_keys_has_no_details() {
        let dir = std::env::temp_dir().join("buzzdetect-test-details-bare");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config_model.json"), good_config()).unwrap();

        let info = ModelInfo::read(&dir, "bare", false);
        assert_eq!(info.description, None);
        assert!(!info.has_readme);
        let details = model_details_in(&dir, "bare");
        assert!(details.readme.is_none() && details.thresholds.is_none());
    }

    #[test]
    fn description_thresholds_and_readme_are_read() {
        let dir = std::env::temp_dir().join("buzzdetect-test-details-full");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = good_config().replacen(
            '{',
            r#"{"description": "  Buzz.  ", "thresholds": {"a": -1.2},
                "threshold_stats": {"a": {"folds": 8}},"#,
            1,
        );
        std::fs::write(dir.join("config_model.json"), config).unwrap();
        std::fs::write(dir.join("README.md"), "# hi").unwrap();

        let info = ModelInfo::read(&dir, "full", true);
        assert_eq!(info.description.as_deref(), Some("Buzz."));
        assert!(info.has_readme);
        let details = model_details_in(&dir, "full");
        assert_eq!(details.readme.as_deref(), Some("# hi"));
        assert_eq!(details.thresholds.unwrap()["a"], -1.2);
        assert_eq!(details.threshold_stats.unwrap()["a"]["folds"], 8);
    }

    #[test]
    fn a_blank_description_is_none() {
        let config: serde_json::Value = serde_json::json!({"description": "   "});
        assert_eq!(description_of(&config), None);
        assert_eq!(description_of(&serde_json::json!({"description": 3})), None);
    }

    #[test]
    fn a_readme_link_cannot_leave_the_model_folder() {
        let dir = std::env::temp_dir().join("buzzdetect-test-details-links");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("tests")).unwrap();
        std::fs::write(dir.join("tests").join("metrics.svg"), "<svg/>").unwrap();
        std::fs::write(dir.parent().unwrap().join("buzzdetect-outside.txt"), "x").unwrap();

        assert!(file_in_model_dir(&dir, "tests/metrics.svg").is_some());
        assert!(file_in_model_dir(&dir, "../buzzdetect-outside.txt").is_none());
        assert!(file_in_model_dir(&dir, "tests").is_none());
        assert!(file_in_model_dir(&dir, "missing.svg").is_none());
    }

    #[test]
    fn a_model_name_is_encoded_for_the_window_url() {
        assert_eq!(percent_encode("model_general_v3"), "model_general_v3");
        assert_eq!(percent_encode("a b&c"), "a%20b%26c");
    }

    #[test]
    fn empty_class_list_is_rejected() {
        let dir = std::env::temp_dir().join("buzzdetect-test-import-noclasses");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.onnx"), b"x").unwrap();
        std::fs::write(
            dir.join("config_model.json"),
            good_config().replace(r#"["a", "b"]"#, "[]"),
        )
        .unwrap();
        assert!(validate_model_dir(&dir).is_err());
    }
}
