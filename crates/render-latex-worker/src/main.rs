mod pixel_sizing;
use pixel_sizing::PixelSizingOptions;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, anyhow};
use ratex_layout::{LayoutOptions, layout, to_display_list, validate_layout};
use ratex_parser::parser::parse;
use ratex_render::{RenderOptions, render_to_png};
use ratex_types::color::Color;
use ratex_types::math_style::MathStyle;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
struct Request {
    id: u64,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct Response {
    id: u64,
    result: Option<Value>,
    error: Option<ResponseError>,
}

#[derive(Debug, Serialize)]
struct ResponseError {
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseValue {
    cache_key: String,
    png_path: String,
    width_px: u32,
    height_px: u32,
    diagnostics: Option<String>,
    #[serde(default)]
    sizing: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DiskCacheEntry {
    #[serde(flatten)]
    value: ResponseValue,
    png_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct RenderParams {
    formula: String,
    #[serde(default = "default_true")]
    display_mode: bool,
    foreground_color: Option<String>,
    background: Option<String>,
    font_size: Option<f32>,
    padding: Option<f32>,
    scale: Option<f32>,
    theme_fingerprint: Option<String>,
    #[serde(default)]
    pixel_sizing: Option<PixelSizingOptions>,
}

#[derive(Debug, Deserialize)]
struct InvalidateParams {
    cache_key: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RenderBatchParams {
    items: Vec<RenderParams>,
}

#[derive(Clone)]
struct CacheEntry {
    value: ResponseValue,
    last_used: u64,
    file_stamp: Option<(SystemTime, u64)>,
}

struct Worker {
    cache_dir: PathBuf,
    memory_cache: HashMap<String, CacheEntry>,
    access_tick: u64,
    disk_prune_tick: u64,
}

const MAX_MEMORY_CACHE_ENTRIES: usize = 256;
const MAX_DISK_CACHE_ENTRIES: usize = 2048;
const DISK_PRUNE_INTERVAL: u64 = 32;
const RENDER_CACHE_VERSION: &str = "ratex-7d9f9a3-pixel-sizing-experiment-v3";
const MAX_CACHED_PNG_BYTES: u64 = 128 * 1024 * 1024;

fn default_true() -> bool {
    true
}

impl Worker {
    fn new() -> Result<Self> {
        let cache_dir = resolve_cache_dir()?;
        fs::create_dir_all(&cache_dir)?;

        let mut worker = Self {
            cache_dir,
            memory_cache: HashMap::new(),
            access_tick: 0,
            disk_prune_tick: 0,
        };
        worker.warm_fonts()?;
        Ok(worker)
    }

    fn next_tick(&mut self) -> u64 {
        self.access_tick = self.access_tick.saturating_add(1);
        self.access_tick
    }

    fn cache_get(&mut self, cache_key: &str) -> Option<ResponseValue> {
        let tick = self.next_tick();
        self.memory_cache.get_mut(cache_key).and_then(|entry| {
            if entry.file_stamp.is_some()
                && entry.file_stamp == png_file_stamp(&entry.value.png_path)
            {
                entry.last_used = tick;
                Some(entry.value.clone())
            } else {
                None
            }
        })
    }

    fn cache_insert(&mut self, cache_key: String, value: ResponseValue) {
        let tick = self.next_tick();
        self.memory_cache.insert(
            cache_key,
            CacheEntry {
                file_stamp: png_file_stamp(&value.png_path),
                value,
                last_used: tick,
            },
        );
        self.prune_memory_cache();
    }

    fn prune_memory_cache(&mut self) {
        while self.memory_cache.len() > MAX_MEMORY_CACHE_ENTRIES {
            let oldest_key = self
                .memory_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());

            if let Some(key) = oldest_key {
                self.memory_cache.remove(&key);
            } else {
                break;
            }
        }
    }

    fn maybe_prune_disk_cache(&mut self) -> Result<()> {
        self.disk_prune_tick = self.disk_prune_tick.saturating_add(1);
        if !self.disk_prune_tick.is_multiple_of(DISK_PRUNE_INTERVAL) {
            return Ok(());
        }

        let mut entries = Vec::new();
        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            let modified = entry
                .metadata()?
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH);
            entries.push((path, modified));
        }

        if entries.len() <= MAX_DISK_CACHE_ENTRIES {
            return Ok(());
        }

        entries.sort_by_key(|(_, modified)| *modified);
        let remove_count = entries.len() - MAX_DISK_CACHE_ENTRIES;
        for (metadata_path, _) in entries.into_iter().take(remove_count) {
            if let Some(stem) = metadata_path.file_stem().and_then(|stem| stem.to_str()) {
                self.memory_cache.remove(stem);
                let png_path = self.cache_dir.join(format!("{}.png", stem));
                let _ = fs::remove_file(&metadata_path);
                let _ = fs::remove_file(png_path);
            }
        }

        Ok(())
    }

    fn warm_fonts(&mut self) -> Result<()> {
        let params = RenderParams {
            formula: "x".to_string(),
            display_mode: true,
            foreground_color: Some("#ffffff".to_string()),
            background: Some("transparent".to_string()),
            font_size: Some(34.0),
            padding: Some(4.0),
            scale: Some(1.0),
            theme_fingerprint: Some("warmup".to_string()),
            pixel_sizing: None,
        };
        let _ = render_formula(&params)?;
        Ok(())
    }

    fn render(&mut self, params: RenderParams) -> Result<ResponseValue> {
        let cache_key = build_cache_key(&params)?;
        if let Some(cached) = self.cache_get(&cache_key) {
            return Ok(cached);
        }

        if let Some(cached) = self.load_disk_cache(&cache_key) {
            self.cache_insert(cache_key.clone(), cached.clone());
            return Ok(cached);
        }

        let (png, sizing) = render_formula_measured(&params)?;

        let png_path = self.cache_dir.join(format!("{}.png", cache_key));
        let metadata_path = self.cache_dir.join(format!("{}.json", cache_key));
        fs::write(&png_path, &png)?;
        let (width_px, height_px) = png_dimensions(&png)?;

        let response = ResponseValue {
            cache_key: cache_key.clone(),
            png_path: png_path.to_string_lossy().into_owned(),
            width_px,
            height_px,
            diagnostics: None,
            sizing,
        };

        let entry = DiskCacheEntry {
            value: response.clone(),
            png_sha256: hex::encode(Sha256::digest(&png)),
        };
        fs::write(&metadata_path, serde_json::to_vec(&entry)?)?;
        self.cache_insert(cache_key, response.clone());
        self.maybe_prune_disk_cache()?;

        Ok(response)
    }

    fn invalidate(&mut self, params: InvalidateParams) -> Result<ResponseValue> {
        if let Some(cache_key) = params.cache_key {
            self.memory_cache.remove(&cache_key);
            let png_path = self.cache_dir.join(format!("{}.png", cache_key));
            let metadata_path = self.cache_dir.join(format!("{}.json", cache_key));
            let _ = fs::remove_file(png_path);
            let _ = fs::remove_file(metadata_path);
            return Ok(ResponseValue {
                cache_key,
                png_path: String::new(),
                width_px: 0,
                height_px: 0,
                diagnostics: Some("invalidated cache entry".to_string()),
                sizing: None,
            });
        }

        self.memory_cache.clear();
        Ok(ResponseValue {
            cache_key: "*".to_string(),
            png_path: String::new(),
            width_px: 0,
            height_px: 0,
            diagnostics: Some("cleared memory cache".to_string()),
            sizing: None,
        })
    }

    fn load_disk_cache(&self, cache_key: &str) -> Option<ResponseValue> {
        let metadata_path = self.cache_dir.join(format!("{}.json", cache_key));
        let bytes = fs::read(metadata_path).ok()?;
        let entry: DiskCacheEntry = serde_json::from_slice(&bytes).ok()?;
        let value = entry.value;
        let png_path = self.cache_dir.join(format!("{}.png", cache_key));
        if value.cache_key != cache_key
            || Path::new(&value.png_path) != png_path
            || !cached_png_valid(&value, &entry.png_sha256)
        {
            return None;
        }
        Some(value)
    }
}

fn png_file_stamp(path: &str) -> Option<(SystemTime, u64)> {
    let metadata = fs::metadata(path).ok()?;
    Some((metadata.modified().ok()?, metadata.len()))
}

fn cached_png_valid(value: &ResponseValue, expected_digest: &str) -> bool {
    let Ok(file) = fs::File::open(&value.png_path) else {
        return false;
    };
    let mut bytes = Vec::new();
    if file
        .take(MAX_CACHED_PNG_BYTES + 1)
        .read_to_end(&mut bytes)
        .is_err()
        || bytes.len() as u64 > MAX_CACHED_PNG_BYTES
        || hex::encode(Sha256::digest(&bytes)) != expected_digest
    {
        return false;
    }
    // Verify the stored PNG bytes without allocating or decoding their pixel buffer.
    png_dimensions(&bytes).is_ok_and(|size| size == (value.width_px, value.height_px))
}

fn resolve_cache_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("RENDER_LATEX_CACHE_DIR") {
        return Ok(PathBuf::from(dir));
    }

    let mut base = if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache")
    } else if let Ok(localappdata) = std::env::var("LocalAppData") {
        PathBuf::from(localappdata)
    } else {
        return Err(anyhow!("unable to resolve cache directory"));
    };

    base.push("render-latex");
    base.push("worker");
    Ok(base)
}

fn build_cache_key(params: &RenderParams) -> Result<String> {
    let json = serde_json::to_vec(params)?;
    let mut hasher = Sha256::new();
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update(RENDER_CACHE_VERSION.as_bytes());
    // Experimental font overrides must not reuse measurements/raster data from
    // a different face. Runtime fontconfig points at an immutable Nix font set.
    for name in ["RATEX_UNICODE_FONT", "FONTCONFIG_FILE", "LANG", "LC_ALL"] {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        if let Ok(value) = std::env::var(name) {
            hasher.update(value.as_bytes());
            if let Some(stamp) = png_file_stamp(&value) {
                hasher.update(format!("{stamp:?}").as_bytes());
            }
        }
        hasher.update([0]);
    }
    hasher.update(json);
    Ok(hex::encode(hasher.finalize()))
}

fn render_formula(params: &RenderParams) -> Result<Vec<u8>> {
    Ok(render_formula_measured(params)?.0)
}

fn render_formula_measured(params: &RenderParams) -> Result<(Vec<u8>, Option<Value>)> {
    let foreground = parse_color(params.foreground_color.as_deref().unwrap_or("#ffffff"))?;
    let background = parse_color(params.background.as_deref().unwrap_or("transparent"))?;
    let ast = parse(&params.formula).map_err(|err| anyhow!(err.to_string()))?;
    let options = LayoutOptions {
        style: if params.display_mode {
            MathStyle::Display
        } else {
            MathStyle::Text
        },
        color: foreground,
        ..LayoutOptions::default()
    };
    let layout_box = layout(&ast, &options);
    validate_layout(&layout_box).map_err(|err| anyhow!(err))?;
    let mut display_list = to_display_list(&layout_box);
    let mut render_options = RenderOptions {
        font_size: params.font_size.unwrap_or(34.0),
        padding: params.padding.unwrap_or(10.0),
        background_color: background,
        font_dir: String::new(),
        device_pixel_ratio: params.scale.unwrap_or(1.5),
    };
    let sizing = params
        .pixel_sizing
        .as_ref()
        .map(|policy| {
            pixel_sizing::select(
                &mut display_list,
                &mut render_options,
                policy,
                params.font_size.unwrap_or(0.01),
            )
        })
        .transpose()?;
    let png = render_to_png(&display_list, &render_options).map_err(|err| anyhow!(err))?;
    Ok((png, sizing))
}

fn parse_color(value: &str) -> Result<Color> {
    Color::parse(value).ok_or_else(|| anyhow!("invalid color: {}", value))
}

fn png_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let reader = decoder.read_info()?;
    let info = reader.info();
    Ok((info.width, info.height))
}

fn read_message<R: BufRead>(reader: &mut R) -> Result<Option<Request>> {
    let mut header = String::new();
    if reader.read_line(&mut header)? == 0 {
        return Ok(None);
    }
    if header.trim().is_empty() {
        return Ok(None);
    }

    let length = header
        .strip_prefix("Content-Length: ")
        .ok_or_else(|| anyhow!("missing content-length header"))?
        .trim()
        .parse::<usize>()?;

    let mut blank = String::new();
    reader.read_line(&mut blank)?;

    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(serde_json::from_slice(&body)?))
}

fn write_message<W: Write>(writer: &mut W, response: &Response) -> Result<()> {
    let body = serde_json::to_vec(response)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}

fn handle_request(worker: &mut Worker, request: Request) -> Response {
    match request.method.as_str() {
        "ping" => Response {
            id: request.id,
            result: Some(json!(ResponseValue {
                cache_key: "ping".to_string(),
                png_path: String::new(),
                width_px: 0,
                height_px: 0,
                diagnostics: Some("pong".to_string()),
                sizing: None,
            })),
            error: None,
        },
        "render" => match serde_json::from_value::<RenderParams>(request.params)
            .context("invalid render params")
            .and_then(|params| worker.render(params))
        {
            Ok(result) => Response {
                id: request.id,
                result: Some(json!(result)),
                error: None,
            },
            Err(err) => Response {
                id: request.id,
                result: None,
                error: Some(ResponseError {
                    message: err.to_string(),
                }),
            },
        },
        "render_batch" => match serde_json::from_value::<RenderBatchParams>(request.params)
            .context("invalid render_batch params")
        {
            Ok(params) => {
                let mut results = Vec::with_capacity(params.items.len());
                for item in params.items {
                    match worker.render(item) {
                        Ok(result) => results.push(json!({ "result": result, "error": Value::Null })),
                        Err(err) => results.push(json!({ "result": Value::Null, "error": { "message": err.to_string() } })),
                    }
                }

                Response {
                    id: request.id,
                    result: Some(Value::Array(results)),
                    error: None,
                }
            }
            Err(err) => Response {
                id: request.id,
                result: None,
                error: Some(ResponseError {
                    message: err.to_string(),
                }),
            },
        },
        "invalidate" => match serde_json::from_value::<InvalidateParams>(request.params)
            .context("invalid invalidate params")
            .and_then(|params| worker.invalidate(params))
        {
            Ok(result) => Response {
                id: request.id,
                result: Some(json!(result)),
                error: None,
            },
            Err(err) => Response {
                id: request.id,
                result: None,
                error: Some(ResponseError {
                    message: err.to_string(),
                }),
            },
        },
        "shutdown" => Response {
            id: request.id,
            result: Some(json!(ResponseValue {
                cache_key: "shutdown".to_string(),
                png_path: String::new(),
                width_px: 0,
                height_px: 0,
                diagnostics: Some("shutting down".to_string()),
                sizing: None,
            })),
            error: None,
        },
        _ => Response {
            id: request.id,
            result: None,
            error: Some(ResponseError {
                message: "unknown method".to_string(),
            }),
        },
    }
}

fn main() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut writer = io::BufWriter::new(stdout.lock());
    let mut worker = Worker::new()?;

    while let Some(request) = read_message(&mut reader)? {
        let shutdown = request.method == "shutdown";
        let response = handle_request(&mut worker, request);
        write_message(&mut writer, &response)?;
        if shutdown {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_params(formula: &str) -> RenderParams {
        RenderParams {
            formula: formula.to_string(),
            display_mode: true,
            foreground_color: Some("#ffffff".to_string()),
            background: Some("transparent".to_string()),
            font_size: Some(20.0),
            padding: Some(2.0),
            scale: Some(1.0),
            theme_fingerprint: None,
            pixel_sizing: None,
        }
    }

    #[test]
    fn reads_framed_requests() {
        let body = br#"{"id":7,"method":"ping","params":{}}"#;
        let frame = format!(
            "Content-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        let request = read_message(&mut io::Cursor::new(frame)).unwrap().unwrap();

        assert_eq!(request.id, 7);
        assert_eq!(request.method, "ping");
    }

    #[test]
    fn renders_transparent_pngs() {
        let png = render_formula(&render_params(r"\frac{1}{2} + \sqrt{x}")).unwrap();
        let (width, height) = png_dimensions(&png).unwrap();

        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(width > 0);
        assert!(height > 0);
    }

    #[test]
    fn rejects_excessive_layout_before_display_list_expansion() {
        let error = render_formula(&render_params(
            r"\begin{array}{c:c}\rule{1pt}{1000000000em}&x\end{array}",
        ))
        .unwrap_err();
        assert!(error.to_string().contains("Layout resource limit exceeded"));
    }

    struct TestCache(PathBuf);

    impl Drop for TestCache {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn test_worker() -> (Worker, TestCache) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "render-latex-test-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        let worker = Worker {
            cache_dir: dir.clone(),
            memory_cache: HashMap::new(),
            access_tick: 0,
            disk_prune_tick: 0,
        };
        (worker, TestCache(dir))
    }

    #[test]
    fn regenerates_corrupt_disk_metadata() {
        let (mut worker, _cache) = test_worker();
        let first = worker.render(render_params("x")).unwrap();
        worker.memory_cache.clear();
        let path = worker.cache_dir.join(format!("{}.json", first.cache_key));
        fs::write(&path, b"{broken").unwrap();

        let second = worker.render(render_params("x")).unwrap();
        assert_eq!(first.cache_key, second.cache_key);
        let restored: ResponseValue = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(restored.cache_key, first.cache_key);
    }

    #[test]
    fn regenerates_corrupt_pngs_in_memory_and_disk_caches() {
        let (mut worker, _cache) = test_worker();
        let first = worker.render(render_params("x")).unwrap();
        let original = fs::read(&first.png_path).unwrap();
        for memory_hit in [true, false] {
            if !memory_hit {
                worker.memory_cache.clear();
            }
            fs::write(&first.png_path, b"not a PNG").unwrap();
            let restored = worker.render(render_params("x")).unwrap();
            assert_eq!(fs::read(restored.png_path).unwrap(), original);
        }
    }

    #[test]
    fn regenerates_pngs_with_valid_headers_but_truncated_pixels() {
        let (mut worker, _cache) = test_worker();
        let first = worker.render(render_params("x")).unwrap();
        let original = fs::read(&first.png_path).unwrap();
        worker.memory_cache.clear();
        fs::write(&first.png_path, &original[..original.len() / 2]).unwrap();
        worker.render(render_params("x")).unwrap();
        assert_eq!(fs::read(first.png_path).unwrap(), original);
    }

    #[test]
    fn regenerates_same_length_corruption_and_mismatched_metadata() {
        let (mut worker, _cache) = test_worker();
        let first = worker.render(render_params("x")).unwrap();
        let original = fs::read(&first.png_path).unwrap();
        let mut damaged = original.clone();
        let middle = damaged.len() / 2;
        damaged[middle] ^= 1;
        worker.memory_cache.clear();
        fs::write(&first.png_path, damaged).unwrap();
        worker.render(render_params("x")).unwrap();
        assert_eq!(fs::read(&first.png_path).unwrap(), original);

        let path = worker.cache_dir.join(format!("{}.json", first.cache_key));
        let mut metadata: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        metadata["width_px"] = json!(first.width_px + 1);
        fs::write(&path, serde_json::to_vec(&metadata).unwrap()).unwrap();
        worker.memory_cache.clear();
        let restored = worker.render(render_params("x")).unwrap();
        let metadata: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(restored.width_px, first.width_px);
        assert_eq!(metadata["width_px"], first.width_px);
        assert!(
            serde_json::to_value(restored)
                .unwrap()
                .get("png_sha256")
                .is_none()
        );
    }

    #[test]
    fn regenerates_cache_entries_without_checksums() {
        let (mut worker, _cache) = test_worker();
        let first = worker.render(render_params("x")).unwrap();
        let path = worker.cache_dir.join(format!("{}.json", first.cache_key));
        fs::write(&path, serde_json::to_vec(&first).unwrap()).unwrap();
        worker.memory_cache.clear();
        worker.render(render_params("x")).unwrap();
        let metadata: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let png = fs::read(&first.png_path).unwrap();
        assert_eq!(metadata["png_sha256"], hex::encode(Sha256::digest(png)));
    }

    #[test]
    fn rejects_invalid_colors() {
        let mut params = render_params("x");
        params.foreground_color = Some("not-a-color".to_string());

        assert!(render_formula(&params).is_err());
    }
}
