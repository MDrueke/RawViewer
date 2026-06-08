// ---------------------------------------------------------------------------
// Colormap constants
// ---------------------------------------------------------------------------

pub const COLOR_NEG: [u8; 3] = [0xf4, 0x5d, 0x01];
pub const COLOR_POS: [u8; 3] = [0x97, 0xcc, 0x04];
pub const COLOR_ZERO: [u8; 3] = [0x27, 0x27, 0x27];

// ---------------------------------------------------------------------------

#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
    ]
}

#[inline]
pub fn voltage_to_rgba(v: f32, vmax: f32) -> [u8; 4] {
    let t = (v / vmax).clamp(-1.0, 1.0);
    let [r, g, b] = if t < 0.0 {
        lerp_rgb(COLOR_ZERO, COLOR_NEG, -t)
    } else {
        lerp_rgb(COLOR_ZERO, COLOR_POS, t)
    };
    [r, g, b, 255]
}

/// Render a heatmap directly into a flat RGBA byte buffer (reused across frames).
///
/// * `out`          — reused RGBA buffer; resized to `pixel_w * pixel_h * 4`
/// * `data`         — preprocessed samples, layout: `data[ch * data_stride + data_offset + t]`
/// * `data_stride`  — samples per channel row in `data` (may be larger than `n_view`)
/// * `data_offset`  — first sample index within each channel row to display
/// * `n_view`       — number of samples to display (the visible window)
pub fn build_heatmap_into(
    out: &mut Vec<u8>,
    data: &[f32],
    n_ap: usize,
    data_stride: usize,
    data_offset: usize,
    n_view: usize,
    ch_first: usize,
    ch_last: usize,
    pixel_w: usize,
    pixel_h: usize,
    vmax: f32,
) {
    use rayon::prelude::*;

    let total = pixel_w * pixel_h * 4;
    out.resize(total, 0);

    let n_ch_display = (ch_last + 1).saturating_sub(ch_first).min(n_ap);
    if n_ch_display == 0 || pixel_w == 0 || pixel_h == 0 || n_view == 0 {
        return;
    }

    let pixels_per_ch = pixel_h as f32 / n_ch_display as f32;
    let row_bytes = pixel_w * 4;

    out.par_chunks_mut(row_bytes).enumerate().for_each(|(py, row)| {
        let ch_display_idx = (py as f32 / pixels_per_ch) as usize;
        let ch_abs = ch_last.saturating_sub(ch_display_idx);

        if ch_abs < ch_first || ch_abs >= n_ap {
            for px in row.chunks_exact_mut(4) {
                px[0] = COLOR_ZERO[0];
                px[1] = COLOR_ZERO[1];
                px[2] = COLOR_ZERO[2];
                px[3] = 255;
            }
            return;
        }

        let base = ch_abs * data_stride + data_offset;
        let ch_data = &data[base..base + n_view];

        for (px_col, px) in row.chunks_exact_mut(4).enumerate() {
            let t0 = (px_col * n_view) / pixel_w;
            let t1 = (((px_col + 1) * n_view) / pixel_w).min(n_view);
            let v = if t1 > t0 {
                ch_data[t0..t1].iter().copied().sum::<f32>() / (t1 - t0) as f32
            } else if t0 < n_view {
                ch_data[t0]
            } else {
                0.0
            };
            let rgba = voltage_to_rgba(v, vmax);
            px[0] = rgba[0];
            px[1] = rgba[1];
            px[2] = rgba[2];
            px[3] = rgba[3];
        }
    });
}
