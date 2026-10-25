//! Experimental discrete sizing using real per-glyph raster coverage.
use anyhow::{Result, anyhow};
use ratex_render::{
    GlyphInkMetrics, RenderOptions, measure_alphanumeric_glyphs, output_dimensions,
};
use ratex_types::display_item::{DisplayItem, DisplayList};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Serialize, Deserialize)]
pub struct PixelSizingOptions {
    pub font_sizes: Vec<f32>,
    pub min_height_px: u32,
    #[serde(default = "default_threshold")]
    pub coverage_threshold: u8,
}
fn default_threshold() -> u8 {
    32
}

fn glyph_json(g: &GlyphInkMetrics) -> Value {
    json!({
        "character": g.character.to_string(), "logical_font": g.logical_font,
        "resolved_font": g.resolved_font, "scale": g.scale, "clipped": g.clipped,
        "bounds": g.bounds.map(|b| json!({"x":b.x,"y":b.y,"width":b.width,"height":b.height})),
    })
}

// A fallback font may have taller ink than the layout engine's provisional
// metrics. Grow/translate the canvas (never stretch the glyphs) so selection
// does not falsely succeed on ink that the final PNG would clip.
fn fit_ink(
    display: &mut DisplayList,
    render: &RenderOptions,
    glyphs: &[GlyphInkMetrics],
) -> Result<bool> {
    if !glyphs.iter().any(|g| g.clipped) {
        return Ok(false);
    }
    let (w, h) = output_dimensions(display, render).map_err(|e| anyhow!(e))?;
    let dpr = render.device_pixel_ratio.clamp(0.01, 16.0);
    let em = f64::from(render.font_size * dpr);
    let margin = (render.padding * dpr).ceil() as i32 + 1;
    let mut extra = [0i32; 4]; // left, top, right, bottom
    for b in glyphs.iter().filter_map(|g| g.bounds) {
        extra[0] = extra[0].max(margin - b.x);
        extra[1] = extra[1].max(margin - b.y);
        extra[2] = extra[2].max(b.x + b.width as i32 + margin - w as i32);
        extra[3] = extra[3].max(b.y + b.height as i32 + margin - h as i32);
    }
    if extra.iter().all(|v| *v == 0) {
        return Ok(false);
    }
    let dx = f64::from(extra[0]) / em;
    let dy = f64::from(extra[1]) / em;
    for item in &mut display.items {
        match item {
            DisplayItem::GlyphPath { x, y, .. }
            | DisplayItem::Line { x, y, .. }
            | DisplayItem::Rect { x, y, .. }
            | DisplayItem::Path { x, y, .. } => {
                *x += dx;
                *y += dy;
            }
        }
    }
    display.width += f64::from(extra[0] + extra[2]) / em;
    display.height += dy;
    display.depth += f64::from(extra[3]) / em;
    Ok(true)
}

pub fn select(
    display: &mut DisplayList,
    render: &mut RenderOptions,
    policy: &PixelSizingOptions,
    base_size: f32,
) -> Result<Value> {
    if policy.font_sizes.is_empty() || policy.font_sizes.len() > 16 {
        return Err(anyhow!(
            "pixel_sizing.font_sizes must contain 1..=16 buckets"
        ));
    }
    if policy.min_height_px > 512 || policy.coverage_threshold == 0 {
        return Err(anyhow!("invalid pixel sizing minimum/coverage threshold"));
    }
    if !base_size.is_finite() || base_size <= 0.0 {
        return Err(anyhow!("font_size must be positive and finite"));
    }
    let mut sizes = policy.font_sizes.clone();
    if sizes
        .iter()
        .any(|s| !s.is_finite() || *s <= 0.0 || *s > 256.0)
    {
        return Err(anyhow!(
            "font buckets must be finite and in (0, 256] pixels/em"
        ));
    }
    sizes.sort_by(f32::total_cmp);
    sizes.dedup();
    sizes.retain(|s| *s >= base_size);
    if sizes.is_empty() {
        return Err(anyhow!(
            "no allowed font size at or above the requested base size"
        ));
    }
    let original = display.clone();
    let mut candidates = Vec::new();
    let mut selected = Vec::new();
    let mut met = false;
    for size in sizes {
        render.font_size = size;
        let mut candidate = original.clone();
        let mut glyphs = measure_alphanumeric_glyphs(&candidate, render, policy.coverage_threshold)
            .map_err(|error| anyhow!(error))?;
        let mut fitted = 0;
        while fitted < 3 && fit_ink(&mut candidate, render, &glyphs)? {
            fitted += 1;
            glyphs = measure_alphanumeric_glyphs(&candidate, render, policy.coverage_threshold)
                .map_err(|e| anyhow!(e))?;
        }
        let min_height = glyphs
            .iter()
            .filter_map(|g| g.bounds.map(|b| b.height))
            .min();
        let missing = glyphs.iter().filter(|g| g.bounds.is_none()).count();
        let clipped = glyphs.iter().filter(|g| g.clipped).count();
        met = missing == 0
            && clipped == 0
            && glyphs
                .iter()
                .all(|g| g.bounds.is_some_and(|b| b.height >= policy.min_height_px));
        candidates.push(
            json!({ "font_size":size, "minimum_glyph_height_px":min_height,
            "eligible_glyphs":glyphs.len(), "missing_glyphs":missing,
            "clipped_glyphs":clipped, "canvas_fit_passes":fitted, "constraint_met":met }),
        );
        *display = candidate;
        selected = glyphs;
        if met {
            break;
        }
    }
    Ok(json!({
        "method":"glyph_raster_coverage", "glyph_set":"unicode_alphanumeric",
        "chosen_font_size":render.font_size, "requested_min_height_px":policy.min_height_px,
        "minimum_glyph_height_px":selected.iter().filter_map(|g|g.bounds.map(|b|b.height)).min(),
        "coverage_threshold":policy.coverage_threshold, "constraint_met":met,
        "candidates":candidates, "glyphs":selected.iter().map(glyph_json).collect::<Vec<_>>(),
        "full_png_renders":1,
        "warning":if met { None } else { Some("maximum bucket reached; minimum not satisfied or glyph missing/clipped") },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(formula: &str, minimum: u32, sizes: Value) -> crate::RenderParams {
        serde_json::from_value(json!({ "formula":formula, "font_size":18.0,
        "padding":3.0, "scale":1.0, "pixel_sizing":{
            "min_height_px":minimum,"font_sizes":sizes,"coverage_threshold":32
        }}))
        .unwrap()
    }
    #[test]
    fn chooses_first_bucket_that_actually_meets_pixel_minimum() {
        let params = request(r"x_{i_2}+\alpha", 10, json!([18, 24, 32, 40, 48, 64]));
        let (png, sizing) = crate::render_formula_measured(&params).unwrap();
        assert!(crate::png_dimensions(&png).unwrap().0 > 0);
        let sizing = sizing.unwrap();
        assert_eq!(sizing["constraint_met"], true);
        assert!(sizing["minimum_glyph_height_px"].as_u64().unwrap() >= 10);
        assert_eq!(sizing["full_png_renders"], 1);
        let candidates = sizing["candidates"].as_array().unwrap();
        assert!(candidates.len() > 1);
        assert!(
            candidates[..candidates.len() - 1]
                .iter()
                .all(|c| c["constraint_met"] == false)
        );
        assert_eq!(
            candidates.last().unwrap()["font_size"],
            sizing["chosen_font_size"]
        );
    }
    #[test]
    fn maximum_bucket_and_cache_policy_are_explicit() {
        let params = request("x", 100, json!([18]));
        let (_, sizing) = crate::render_formula_measured(&params).unwrap();
        let sizing = sizing.unwrap();
        assert_eq!(sizing["constraint_met"], false);
        assert_eq!(sizing["chosen_font_size"], 18.0);
        assert!(!sizing["warning"].is_null());
        assert_ne!(
            crate::build_cache_key(&params).unwrap(),
            crate::build_cache_key(&request("x", 10, json!([18]))).unwrap()
        );
        assert!(crate::render_formula_measured(&request("x", 10, json!([]))).is_err());
        let (_, none) =
            crate::render_formula_measured(&request(r"\rightarrow", 10, json!([18, 24]))).unwrap();
        assert!(none.unwrap()["glyphs"].as_array().unwrap().is_empty());
    }
}
