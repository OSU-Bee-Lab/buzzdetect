//! The model catalog: models published as GitHub releases, which the Models
//! window lists beside the installed ones and can download.
//!
//! `models.json` is the only asset of a release tagged `models`. Each entry
//! names a model, the base URL its files sit under (its own `model-<name>`
//! release), and each file's sha256 -- except the README's, so a README can be
//! edited in place without anyone being offered an update.
//! scripts/publish-model.mjs writes both ends.
//!
//! A name is a model's identity. When an imported model's files stop matching
//! the catalog's hashes, it is offered as an update: that's how a bad binary
//! gets replaced. Bundled models are listed in the catalog too, for their
//! READMEs, but can't be updated from here -- the bundled copy shadows anything
//! in the user store with the same name.

use super::{
    list_models, model_dir, user_models_dir, validate_model_dir, AnalysisState, ModelDetails,
    IMPORT_FILES, MODEL_MARKER, README,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Emitter, Manager};

const CATALOG_URL: &str =
    "https://github.com/OSU-Bee-Lab/buzzdetect/releases/download/models/models.json";
/// Overrides CATALOG_URL. An http(s) URL, or a local path -- in which case the
/// entries' `url`s may be local directories too, for trying a catalog out
/// without publishing it.
const CATALOG_URL_ENV: &str = "BUZZDETECT_CATALOG_URL";
const CATALOG_TTL: Duration = Duration::from_secs(600);
const CATALOG_TIMEOUT: Duration = Duration::from_secs(20);
const FILE_TIMEOUT: Duration = Duration::from_secs(600);
const PREFS_FILE: &str = "model_prefs.json";

/// Emitted to every window after anything that changes the Models list:
/// a download, import, removal, or (un)ignore.
pub const MODELS_CHANGED: &str = "models-changed";

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
struct Catalog {
    #[serde(default)]
    models: Vec<CatalogEntry>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
struct CatalogEntry {
    name: String,
    #[serde(default)]
    description: Option<String>,
    /// The oldest app that can run this model, e.g. one released after the
    /// engine started requiring a new config key. Absent means any.
    #[serde(default)]
    min_app_version: Option<String>,
    /// The directory the files sit under, usually the model's release.
    url: String,
    files: BTreeMap<String, CatalogFile>,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
struct CatalogFile {
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    size: Option<u64>,
}

/// A directory name the app will create: nothing hidden, nothing that climbs
/// out of the models dir or trips over a platform's reserved characters.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Parse a catalog, dropping any entry the app couldn't safely install: an
/// unusable name, no model.onnx or config, or a file the app doesn't copy
/// (which is also what keeps a file name from being a path).
fn parse_catalog(bytes: &[u8]) -> Result<Catalog, String> {
    let mut catalog: Catalog = serde_json::from_slice(bytes)
        .map_err(|e| format!("the model catalog isn't valid: {e}"))?;
    catalog.models.retain(|m| {
        valid_name(&m.name)
            && m.files.contains_key("model.onnx")
            && m.files.contains_key(MODEL_MARKER)
            && m.files.keys().all(|f| IMPORT_FILES.contains(&f.as_str()))
    });
    Ok(catalog)
}

fn compatible(entry: &CatalogEntry, app_version: &semver::Version) -> bool {
    match &entry.min_app_version {
        None => true,
        Some(min) => semver::Version::parse(min)
            .map(|min| *app_version >= min)
            .unwrap_or(false),
    }
}

fn asset_url(base: &str, file: &str) -> String {
    format!("{}/{}", base.trim_end_matches('/'), file)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    if let Some(c) = CLIENT.get() {
        return Ok(c);
    }
    // reqwest is built without a bundled provider, as the updater has it.
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let client = reqwest::Client::builder()
        .user_agent(concat!("buzzdetect/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    Ok(CLIENT.get_or_init(|| client))
}

/// The bytes at `url`: over http(s), or read off disk for anything else.
async fn fetch(url: &str, timeout: Duration) -> Result<Vec<u8>, String> {
    if !(url.starts_with("https://") || url.starts_with("http://")) {
        let path = url.strip_prefix("file://").unwrap_or(url);
        return std::fs::read(path).map_err(|e| format!("{path}: {e}"));
    }
    let resp = client()?
        .get(url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{url}: HTTP {}", resp.status()));
    }
    resp.bytes().await.map(|b| b.to_vec()).map_err(|e| e.to_string())
}

static CATALOG: Mutex<Option<(Instant, Catalog)>> = Mutex::new(None);

fn cached_catalog() -> Option<Catalog> {
    let guard = CATALOG.lock().ok()?;
    let (at, catalog) = guard.as_ref()?;
    (at.elapsed() < CATALOG_TTL).then(|| catalog.clone())
}

async fn catalog(refresh: bool) -> Result<Catalog, String> {
    if !refresh {
        if let Some(c) = cached_catalog() {
            return Ok(c);
        }
    }
    let url = std::env::var(CATALOG_URL_ENV).unwrap_or_else(|_| CATALOG_URL.to_string());
    let catalog = parse_catalog(&fetch(&url, CATALOG_TIMEOUT).await?)?;
    if let Ok(mut guard) = CATALOG.lock() {
        *guard = Some((Instant::now(), catalog.clone()));
    }
    Ok(catalog)
}

async fn catalog_entry(name: &str) -> Result<CatalogEntry, String> {
    catalog(false)
        .await?
        .models
        .into_iter()
        .find(|m| m.name == name)
        .ok_or_else(|| format!("'{name}' isn't in the model catalog"))
}

#[derive(Serialize, Deserialize, Default)]
struct Prefs {
    #[serde(default)]
    ignored: Vec<String>,
}

fn prefs_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_local_data_dir().ok().map(|d| d.join(PREFS_FILE))
}

fn read_prefs(app: &AppHandle) -> Prefs {
    prefs_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// sha256 of a file, remembered by (size, mtime) so listing the models doesn't
/// rehash every onnx file each time the window opens.
fn file_sha256(path: &Path) -> Option<String> {
    static CACHE: Mutex<Option<HashMap<PathBuf, (u64, SystemTime, String)>>> = Mutex::new(None);
    let meta = std::fs::metadata(path).ok()?;
    let stamp = (meta.len(), meta.modified().ok()?);
    if let Some((len, mtime, hash)) = CACHE
        .lock()
        .ok()?
        .get_or_insert_with(HashMap::new)
        .get(path)
    {
        if (*len, *mtime) == stamp {
            return Some(hash.clone());
        }
    }
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path).ok()?, &mut hasher).ok()?;
    let hash = hex(&hasher.finalize());
    CACHE
        .lock()
        .ok()?
        .get_or_insert_with(HashMap::new)
        .insert(path.to_path_buf(), (stamp.0, stamp.1, hash.clone()));
    Some(hash)
}

/// Whether any hashed file in the entry is missing from `dir` or differs.
fn differs(dir: &Path, entry: &CatalogEntry) -> bool {
    entry.files.iter().any(|(file, meta)| match &meta.sha256 {
        None => false,
        Some(want) => file_sha256(&dir.join(file))
            .map(|got| !got.eq_ignore_ascii_case(want))
            .unwrap_or(true),
    })
}

/// An installed model, as far as the overview needs to know.
struct Installed {
    name: String,
    bundled: bool,
    description: Option<String>,
    /// Its files don't match the catalog's. Only computed for imported models.
    differs: bool,
}

/// One row of the Models window's list.
#[derive(Serialize, Debug, PartialEq)]
struct ModelRow {
    name: String,
    description: Option<String>,
    installed: bool,
    bundled: bool,
    in_catalog: bool,
    /// False when the catalog says it needs a newer app than this one.
    compatible: bool,
    min_app_version: Option<String>,
    /// An imported model whose files differ from the catalog's.
    update: bool,
    ignored: bool,
    /// Worth a badge: a compatible model not installed and not ignored, or
    /// an update. Updates can't be ignored -- they're how fixes arrive.
    notify: bool,
    /// Sum of the catalog's file sizes, for a model that isn't installed.
    download_size: Option<u64>,
}

/// Installed models first, in list_models' order, then the catalog's other
/// entries in catalog order.
fn rows(
    installed: &[Installed],
    catalog: Option<&Catalog>,
    ignored: &HashSet<String>,
    app_version: &semver::Version,
) -> Vec<ModelRow> {
    let entries: &[CatalogEntry] = catalog.map(|c| c.models.as_slice()).unwrap_or(&[]);
    let find = |name: &str| entries.iter().find(|e| e.name == name);
    let mut out: Vec<ModelRow> = installed
        .iter()
        .map(|m| {
            let entry = find(&m.name);
            let compatible = entry.map(|e| compatible(e, app_version)).unwrap_or(true);
            let update = !m.bundled && entry.is_some() && compatible && m.differs;
            ModelRow {
                name: m.name.clone(),
                description: m
                    .description
                    .clone()
                    .or_else(|| entry.and_then(|e| e.description.clone())),
                installed: true,
                bundled: m.bundled,
                in_catalog: entry.is_some(),
                compatible,
                min_app_version: entry.and_then(|e| e.min_app_version.clone()),
                update,
                ignored: false,
                notify: update,
                download_size: None,
            }
        })
        .collect();
    for e in entries {
        if installed.iter().any(|m| m.name == e.name) {
            continue;
        }
        let compatible = compatible(e, app_version);
        let is_ignored = ignored.contains(&e.name);
        out.push(ModelRow {
            name: e.name.clone(),
            description: e.description.clone(),
            installed: false,
            bundled: false,
            in_catalog: true,
            compatible,
            min_app_version: e.min_app_version.clone(),
            update: false,
            ignored: is_ignored,
            notify: compatible && !is_ignored,
            download_size: Some(e.files.values().filter_map(|f| f.size).sum()),
        });
    }
    out
}

#[derive(Serialize)]
pub struct ModelsOverview {
    models: Vec<ModelRow>,
    /// Why the catalog couldn't be read, if it couldn't. The installed models
    /// are still listed; there's just nothing to offer.
    catalog_error: Option<String>,
}

#[tauri::command]
pub async fn models_overview(app: AppHandle, refresh: bool) -> Result<ModelsOverview, String> {
    let fetched = catalog(refresh).await;
    let entries = fetched.as_ref().map(|c| c.models.as_slice()).unwrap_or(&[]);
    let installed: Vec<Installed> = list_models(app.clone())?
        .into_iter()
        .map(|m| {
            let bundled = !m.removable;
            let differs = !bundled
                && entries
                    .iter()
                    .find(|e| e.name == m.name)
                    .zip(model_dir(&app, &m.name))
                    .map(|(e, dir)| differs(&dir, e))
                    .unwrap_or(false);
            Installed {
                name: m.name,
                bundled,
                description: m.description,
                differs,
            }
        })
        .collect();
    let ignored: HashSet<String> = read_prefs(&app).ignored.into_iter().collect();
    Ok(ModelsOverview {
        models: rows(
            &installed,
            fetched.as_ref().ok(),
            &ignored,
            &app.package_info().version,
        ),
        catalog_error: fetched.err(),
    })
}

/// Fetch every file in the entry into `dir`, checking each hash as it lands.
async fn download_into(entry: &CatalogEntry, dir: &Path) -> Result<(), String> {
    for (file, meta) in &entry.files {
        let bytes = fetch(&asset_url(&entry.url, file), FILE_TIMEOUT)
            .await
            .map_err(|e| format!("couldn't download {file}: {e}"))?;
        if let Some(want) = &meta.sha256 {
            if !hex(&Sha256::digest(&bytes)).eq_ignore_ascii_case(want) {
                return Err(format!(
                    "{file} didn't match the catalog's checksum. Try again in a \
                     minute; the model may be in the middle of being republished."
                ));
            }
        }
        std::fs::write(dir.join(file), &bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Move a verified download into place, swapping out an older copy. Both
/// sit in the same directory, so these are renames rather than copies.
fn install(staging: &Path, dest: &Path, old: &Path) -> Result<(), String> {
    if !dest.exists() {
        return std::fs::rename(staging, dest).map_err(|e| e.to_string());
    }
    std::fs::rename(dest, old).map_err(|e| format!("couldn't replace the old copy: {e}"))?;
    if let Err(e) = std::fs::rename(staging, dest) {
        let _ = std::fs::rename(old, dest);
        return Err(format!("couldn't replace the old copy: {e}"));
    }
    let _ = std::fs::remove_dir_all(old);
    Ok(())
}

/// Download a catalog model into the user store, or replace an imported copy
/// whose files have changed. Files land in a hidden sibling directory first
/// (list_models skips dot-dirs), so a failure never leaves a half-written
/// model where the engine would find it.
#[tauri::command]
pub async fn download_model(app: AppHandle, name: String) -> Result<(), String> {
    static IN_FLIGHT: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let entry = catalog_entry(&name).await?;
    if !compatible(&entry, &app.package_info().version) {
        return Err(format!(
            "{name} needs buzzdetect {} or newer",
            entry.min_app_version.as_deref().unwrap_or("?")
        ));
    }
    if list_models(app.clone())?
        .iter()
        .any(|m| m.name == name && !m.removable)
    {
        return Err(format!("{name} comes with buzzdetect and can't be replaced here"));
    }
    let user = user_models_dir(&app).ok_or("couldn't resolve the app data directory")?;
    let dest = user.join(&name);
    if dest.exists() {
        let running = app
            .state::<AnalysisState>()
            .0
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false);
        if running {
            return Err("can't update a model while an analysis is running".into());
        }
    }
    {
        let mut guard = IN_FLIGHT.lock().map_err(|e| e.to_string())?;
        if !guard.get_or_insert_with(HashSet::new).insert(name.clone()) {
            return Err(format!("{name} is already downloading"));
        }
    }

    let staging = user.join(format!(".{name}.download"));
    let old = user.join(format!(".{name}.old"));
    let result = async {
        // Leftovers from a download the app quit in the middle of.
        let _ = std::fs::remove_dir_all(&staging);
        let _ = std::fs::remove_dir_all(&old);
        std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
        download_into(&entry, &staging).await?;
        validate_model_dir(&staging)?;
        install(&staging, &dest, &old)
    }
    .await;
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    if let Ok(mut guard) = IN_FLIGHT.lock() {
        guard.get_or_insert_with(HashSet::new).remove(&name);
    }
    result?;
    let _ = app.emit(MODELS_CHANGED, ());
    Ok(())
}

/// Stop (or resume) badging a catalog model the user doesn't want.
#[tauri::command]
pub fn set_model_ignored(app: AppHandle, name: String, ignored: bool) -> Result<(), String> {
    let path = prefs_path(&app).ok_or("couldn't resolve the app data directory")?;
    let mut prefs = read_prefs(&app);
    prefs.ignored.retain(|n| *n != name);
    if ignored {
        prefs.ignored.push(name);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&prefs).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| e.to_string())?;
    let _ = app.emit(MODELS_CHANGED, ());
    Ok(())
}

/// A catalog model's details, read from its release: the config (for the
/// thresholds and description) and the README. For an installed model the
/// window shows its local config but this README, which may be newer.
#[tauri::command]
pub async fn catalog_model_details(name: String) -> Result<ModelDetails, String> {
    let entry = catalog_entry(&name).await?;
    let config: serde_json::Value = serde_json::from_slice(
        &fetch(&asset_url(&entry.url, MODEL_MARKER), CATALOG_TIMEOUT).await?,
    )
    .map_err(|e| format!("{name}'s config_model.json isn't valid: {e}"))?;
    let readme = if entry.files.contains_key(README) {
        fetch(&asset_url(&entry.url, README), CATALOG_TIMEOUT)
            .await
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
    } else {
        None
    };
    Ok(ModelDetails::from_parts(&name, &config, readme))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str) -> CatalogEntry {
        let mut files = BTreeMap::new();
        files.insert(
            "model.onnx".to_string(),
            CatalogFile { sha256: Some("aa".into()), size: Some(10) },
        );
        files.insert(
            "config_model.json".to_string(),
            CatalogFile { sha256: Some("bb".into()), size: Some(2) },
        );
        files.insert("README.md".to_string(), CatalogFile::default());
        CatalogEntry {
            name: name.into(),
            description: Some(format!("{name} from the catalog")),
            min_app_version: None,
            url: "https://example.org/x/".into(),
            files,
        }
    }

    fn installed(name: &str, bundled: bool, differs: bool) -> Installed {
        Installed {
            name: name.into(),
            bundled,
            description: None,
            differs,
        }
    }

    fn v(s: &str) -> semver::Version {
        semver::Version::parse(s).unwrap()
    }

    #[test]
    fn parse_drops_entries_it_could_not_install() {
        let mut bad_file = entry("bad_file");
        bad_file
            .files
            .insert("../../evil".into(), CatalogFile::default());
        let mut no_onnx = entry("no_onnx");
        no_onnx.files.remove("model.onnx");
        let catalog = Catalog {
            models: vec![entry("ok"), entry("../escape"), entry(".hidden"), bad_file, no_onnx],
        };
        let parsed = parse_catalog(&serde_json::to_vec(&catalog).unwrap()).unwrap();
        let names: Vec<_> = parsed.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["ok"]);
    }

    #[test]
    fn parse_accepts_what_the_publish_script_writes() {
        let text = r#"{"models": [{"name": "m", "url": "https://x/", "files": {
            "model.onnx": {"sha256": "ab", "size": 1},
            "config_model.json": {"sha256": "cd", "size": 1},
            "README.md": {}}}]}"#;
        let parsed = parse_catalog(text.as_bytes()).unwrap();
        assert_eq!(parsed.models.len(), 1);
        assert!(parsed.models[0].files["README.md"].sha256.is_none());
        assert!(parse_catalog(b"not json").is_err());
    }

    #[test]
    fn min_app_version_compares_as_semver() {
        let mut e = entry("m");
        assert!(compatible(&e, &v("2.0.0-a4")));
        e.min_app_version = Some("2.0.0-a5".into());
        assert!(!compatible(&e, &v("2.0.0-a4")));
        assert!(compatible(&e, &v("2.0.0-a5")));
        assert!(compatible(&e, &v("2.0.0")));
        e.min_app_version = Some("garbage".into());
        assert!(!compatible(&e, &v("9.9.9")));
    }

    #[test]
    fn rows_badge_new_models_and_updates_only() {
        let mut too_new = entry("too_new");
        too_new.min_app_version = Some("3.0.0".into());
        let catalog = Catalog {
            models: vec![
                entry("bundled"),
                entry("imported_stale"),
                entry("imported_fresh"),
                entry("new"),
                entry("ignored"),
                too_new,
            ],
        };
        let local = [
            // A bundled model that differs still isn't an update: the
            // bundled copy would shadow the download.
            installed("bundled", true, true),
            installed("imported_stale", false, true),
            installed("imported_fresh", false, false),
            installed("local_only", false, false),
        ];
        let ignored: HashSet<String> = ["ignored".to_string()].into();
        let out = rows(&local, Some(&catalog), &ignored, &v("2.0.0"));
        let get = |n: &str| out.iter().find(|r| r.name == n).unwrap();

        let names: Vec<_> = out.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(
            names,
            ["bundled", "imported_stale", "imported_fresh", "local_only", "new", "ignored", "too_new"]
        );
        assert!(!get("bundled").update && !get("bundled").notify);
        assert!(get("imported_stale").update && get("imported_stale").notify);
        assert!(!get("imported_fresh").notify);
        assert!(!get("local_only").in_catalog && !get("local_only").notify);
        assert!(get("new").notify && !get("new").installed);
        assert_eq!(get("new").download_size, Some(12));
        assert_eq!(get("new").description.as_deref(), Some("new from the catalog"));
        assert!(get("ignored").ignored && !get("ignored").notify);
        assert!(!get("too_new").compatible && !get("too_new").notify);
    }

    #[test]
    fn rows_without_a_catalog_are_just_the_installed_models() {
        let local = [installed("a", true, false), installed("b", false, false)];
        let out = rows(&local, None, &HashSet::new(), &v("2.0.0"));
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|r| r.installed && !r.in_catalog && !r.notify));
    }

    fn scratch(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("buzzdetect-catalog-{tag}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A published model on disk, and the catalog entry pointing at it.
    fn published(dir: &Path) -> CatalogEntry {
        let files = [
            ("model.onnx", "weights"),
            (
                "config_model.json",
                r#"{"classes": ["ins_buzz"], "samplerate": 16000, "framelength_s": 0.96,
                    "digits_time": 2, "digits_results": 2, "samples_hop": 1, "samples_min": 1}"#,
            ),
            ("README.md", "# hi"),
        ];
        let mut e = entry("m");
        e.url = dir.to_string_lossy().into_owned();
        e.files.clear();
        for (name, body) in files {
            std::fs::write(dir.join(name), body).unwrap();
            let sha = (name != README).then(|| hex(&Sha256::digest(body.as_bytes())));
            e.files.insert(name.into(), CatalogFile { sha256: sha, size: None });
        }
        e
    }

    #[test]
    fn download_verifies_and_installs_over_an_old_copy() {
        let release = scratch("release");
        let user = scratch("user");
        let entry = published(&release);
        let rt = tauri::async_runtime::block_on;

        let dest = user.join("m");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("model.onnx"), "old weights").unwrap();
        assert!(differs(&dest, &entry));

        let staging = user.join(".m.download");
        std::fs::create_dir_all(&staging).unwrap();
        rt(download_into(&entry, &staging)).unwrap();
        validate_model_dir(&staging).unwrap();
        install(&staging, &dest, &user.join(".m.old")).unwrap();

        assert!(!staging.exists() && !user.join(".m.old").exists());
        assert_eq!(std::fs::read_to_string(dest.join("model.onnx")).unwrap(), "weights");
        assert!(!differs(&dest, &entry));

        // The README isn't hashed, so editing it isn't an update.
        std::fs::write(dest.join(README), "# edited").unwrap();
        assert!(!differs(&dest, &entry));

        let _ = std::fs::remove_dir_all(release);
        let _ = std::fs::remove_dir_all(user);
    }

    #[test]
    fn download_refuses_a_file_that_does_not_match_its_hash() {
        let release = scratch("tampered");
        let staging = scratch("staging");
        let entry = published(&release);
        std::fs::write(release.join("model.onnx"), "something else").unwrap();
        let err = tauri::async_runtime::block_on(download_into(&entry, &staging)).unwrap_err();
        assert!(err.contains("checksum"), "{err}");
        let _ = std::fs::remove_dir_all(release);
        let _ = std::fs::remove_dir_all(staging);
    }
}

#[cfg(test)]
mod network_tests {
    /// Needs the network and the published catalog, so it only runs when asked
    /// for: `cargo test -- --ignored`. Checks the TLS setup reaches GitHub,
    /// follows its redirect to the asset host, and the catalog there parses
    /// with every entry intact.
    #[test]
    #[ignore]
    fn the_published_catalog_is_reachable_and_valid() {
        let bytes =
            tauri::async_runtime::block_on(super::fetch(super::CATALOG_URL, super::CATALOG_TIMEOUT))
                .unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let parsed = super::parse_catalog(&bytes).unwrap();
        assert_eq!(parsed.models.len(), raw["models"].as_array().unwrap().len());
    }
}
