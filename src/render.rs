use crate::data::DisplayRow;

// ---------------------------------------------------------------------------
// 5-stop colormap (high → low):
//   +vmax  →  #0320fc
//   +mid   →  #1e2b8a
//    0     →  #272727  (background grey)
//   -mid   →  #8a251e
//   -vmax  →  #ff1100
// ---------------------------------------------------------------------------

const C_POS_HI: [u8; 3] = [0x03, 0x20, 0xfc];
const C_POS_LO: [u8; 3] = [0x1e, 0x2b, 0x8a];
pub const C_ZERO: [u8; 3] = [0x18, 0x18, 0x18]; // #181818 grey
const C_NEG_LO: [u8; 3] = [0x8a, 0x25, 0x1e];
const C_NEG_HI: [u8; 3] = [0xff, 0x11, 0x00];

const C_GAP: [u8; 3] = [0x50, 0x50, 0x50]; // light grey gap indicator

#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
    ]
}

#[inline]
fn interpolate_stops(t: f32, stops: &[[u8; 3]]) -> [u8; 3] {
    let n = stops.len() - 1;
    let scaled_t = t * n as f32;
    let idx = scaled_t.floor() as usize;
    if idx >= n {
        return stops[n];
    }
    let local_t = scaled_t - idx as f32;
    lerp_rgb(stops[idx], stops[idx + 1], local_t)
}

#[inline]
pub fn voltage_to_rgba(v: f32, vmax: f32, cmap: &crate::app::ColorMapChoice) -> [u8; 4] {
    let t = (v / vmax).clamp(-1.0, 1.0); // -1..1
    
    let [r, g, b] = match cmap {
        crate::app::ColorMapChoice::Default => {
            if t >= 0.0 { interpolate_stops(t, &[C_ZERO, C_POS_LO, C_POS_HI]) }
            else { interpolate_stops(-t, &[C_ZERO, C_NEG_LO, C_NEG_HI]) }
        },
        crate::app::ColorMapChoice::OrangeBlue => {
            if t >= 0.0 { interpolate_stops(t, &[C_ZERO, [0x40, 0x64, 0x94], [0x4e, 0x82, 0xc7]]) }
            else { interpolate_stops(-t, &[C_ZERO, [0x80, 0x4f, 0x32], [0xbf, 0x71, 0x43]]) }
        },
        crate::app::ColorMapChoice::IceFire => {
            if t >= 0.0 { interpolate_stops(t, &[C_ZERO, [0x46, 0x27, 0x8a], [0x20, 0x5f, 0x9e], [0x71, 0xb5, 0xbd], [0x93, 0xcf, 0xc9]]) }
            else { interpolate_stops(-t, &[C_ZERO, [0x54, 0x19, 0x22], [0x8a, 0x24, 0x1d], [0xba, 0x4f, 0x22], [0xd9, 0xa2, 0x73]]) }
        },
        crate::app::ColorMapChoice::Vanimo => {
            if t >= 0.0 { interpolate_stops(t, &[C_ZERO, [0x59, 0x26, 0x66], [0xa6, 0x56, 0xba], [0xe0, 0xa1, 0xf0]]) }
            else { interpolate_stops(-t, &[C_ZERO, [0x28, 0x5c, 0x1f], [0x49, 0x8c, 0x3e], [0x7f, 0xbd, 0x75]]) }
        }
    };
    [r, g, b, 255]
}

// ---------------------------------------------------------------------------
// Heatmap renderer
//
// `display_rows` — the full ordered list of rows to render (Data + Gap variants).
//   Data rows carry a `data_idx` into the flat `data` buffer.
//   Gap rows are rendered as a solid light-grey stripe.
//
// `first_row_idx` / `last_row_idx` — indices into `display_rows` to render
//   (the channel-range selection from the UI sliders).
// ---------------------------------------------------------------------------

pub fn build_heatmap_into(
    out: &mut Vec<u8>,
    data: &[f32],
    display_rows: &[DisplayRow],
    first_row_idx: usize,   // first display_row index to render
    last_row_idx: usize,    // last display_row index to render (inclusive)
    data_stride: usize,     // n_samp in the buffer
    data_offset: usize,     // first sample within each row to display
    n_view: usize,          // number of samples to display
    pixel_w: usize,
    pixel_h: usize,
    vmax: f32,
    cmap: &crate::app::ColorMapChoice,
) {
    use rayon::prelude::*;

    let total = pixel_w * pixel_h * 4;
    out.resize(total, 0);

    if pixel_w == 0 || pixel_h == 0 || n_view == 0 { return; }

    let first = first_row_idx.min(display_rows.len().saturating_sub(1));
    let last = last_row_idx.min(display_rows.len().saturating_sub(1));
    let visible = &display_rows[first..=last];
    let n_rows = visible.len();
    if n_rows == 0 { return; }

    let row_bytes = pixel_w * 4;

    out.par_chunks_mut(row_bytes).enumerate().for_each(|(py, row)| {
        // map pixel row → display row (ch_last at top, ch_first at bottom)
        let disp_idx = n_rows.saturating_sub(1)
            .saturating_sub((py * n_rows) / pixel_h);
        let disp_idx = disp_idx.min(n_rows - 1);

        match &visible[disp_idx] {
            DisplayRow::Gap => {
                for px in row.chunks_exact_mut(4) {
                    px[0] = C_GAP[0]; px[1] = C_GAP[1]; px[2] = C_GAP[2]; px[3] = 255;
                }
            }
            DisplayRow::Data { data_idx, .. } => {
                let base = data_idx * data_stride + data_offset;
                if base + n_view > data.len() {
                    // out-of-range: fill grey
                    for px in row.chunks_exact_mut(4) {
                        px[0] = C_ZERO[0]; px[1] = C_ZERO[1]; px[2] = C_ZERO[2]; px[3] = 255;
                    }
                    return;
                }
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
                    let rgba = voltage_to_rgba(v, vmax, cmap);
                    px[0] = rgba[0]; px[1] = rgba[1]; px[2] = rgba[2]; px[3] = rgba[3];
                }
            }
        }
    });
}
