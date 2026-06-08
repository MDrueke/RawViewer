use anyhow::{Context, Result, bail};
use std::path::Path;

/// Probe geometry for one channel
#[derive(Clone, Debug)]
pub struct ChannelGeom {
    pub x_um: f32,
    pub y_um: f32,
}

/// Everything parsed from the .meta file that we need
#[derive(Clone, Debug)]
pub struct Meta {
    /// Total channels saved in the .bin file (includes sync)
    pub n_saved_chans: usize,
    /// Number of AP channels (n_saved_chans - n_sync)
    pub n_ap_chans: usize,
    /// Number of sync channels
    pub n_sync_chans: usize,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Total number of samples in the file
    pub n_samples: usize,
    /// Scale factor: int16 raw → µV
    pub uv_per_bit: f32,
    /// Per-channel probe geometry (384 entries for NP1)
    pub channel_geom: Vec<ChannelGeom>,
}

impl Meta {
    pub fn from_file(meta_path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(meta_path)
            .with_context(|| format!("reading meta file: {}", meta_path.display()))?;

        let mut n_saved_chans: Option<usize> = None;
        let mut sample_rate: Option<f64> = None;
        let mut file_size_bytes: Option<u64> = None;
        let mut ai_range_max: f64 = 0.6;
        let mut ap_gain: f64 = 500.0;
        let mut max_int: f64 = 512.0; // NP1 default; NP2 uses 8192
        let mut n_ap: Option<usize> = None;
        let mut n_lf: Option<usize> = None;
        let mut n_sy: Option<usize> = None;
        let mut geom_str: Option<String> = None;

        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if let Some((key, val)) = line.split_once('=') {
                match key {
                    "nSavedChans" => n_saved_chans = val.parse().ok(),
                    "imSampRate" => sample_rate = val.parse().ok(),
                    "fileSizeBytes" => file_size_bytes = val.parse().ok(),
                    "imAiRangeMax" => ai_range_max = val.parse().unwrap_or(0.6),
                    "imChan0apGain" => ap_gain = val.parse().unwrap_or(500.0),
                    "imMaxInt" => max_int = val.parse().unwrap_or(512.0),
                    "snsApLfSy" => {
                        // e.g. "384,384,1"
                        let parts: Vec<&str> = val.split(',').collect();
                        if parts.len() >= 3 {
                            n_ap = parts[0].parse().ok();
                            n_lf = parts[1].parse().ok();
                            n_sy = parts[2].parse().ok();
                        }
                    }
                    "snsGeomMap" => geom_str = Some(val.to_string()),
                    _ => {}
                }
            }
        }

        let n_saved_chans = n_saved_chans.context("missing nSavedChans")?;
        let sample_rate = sample_rate.context("missing imSampRate")?;
        let file_size_bytes = file_size_bytes.context("missing fileSizeBytes")?;

        let n_ap = n_ap.unwrap_or(n_saved_chans.saturating_sub(1));
        let n_sy = n_sy.unwrap_or(1);
        let _ = n_lf; // not used for AP

        let n_ap_chans = n_ap;
        let n_sync_chans = n_sy;

        // bytes per sample = n_saved_chans * 2 (int16)
        let n_samples = (file_size_bytes / (n_saved_chans as u64 * 2)) as usize;

        // µV per raw int16 bit
        let uv_per_bit = (ai_range_max / max_int / ap_gain * 1e6) as f32;

        let channel_geom = parse_geom_map(geom_str.as_deref(), n_ap_chans);

        Ok(Meta {
            n_saved_chans,
            n_ap_chans,
            n_sync_chans,
            sample_rate,
            n_samples,
            uv_per_bit,
            channel_geom,
        })
    }
}

/// Parse snsGeomMap into per-channel (x, y) in µm.
/// Format: (header)(ch:x:y:used)...
fn parse_geom_map(s: Option<&str>, n_ap: usize) -> Vec<ChannelGeom> {
    let default = || {
        // fallback: linear layout at x=0, y = ch * 20 µm
        (0..n_ap)
            .map(|i| ChannelGeom {
                x_um: 0.0,
                y_um: i as f32 * 20.0,
            })
            .collect()
    };

    let s = match s {
        Some(s) => s,
        None => return default(),
    };

    // entries are parenthesised: (shank:x:y:used)
    let mut geoms: Vec<(usize, ChannelGeom)> = Vec::new();
    let mut idx: usize = 0;
    let mut ch_idx: usize = 0;
    for token in s.split(')') {
        let token = token.trim_start_matches('(');
        if token.is_empty() {
            continue;
        }
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() == 4 {
            // shank:x:y:used  — channel index matches order of appearance
            let x: f32 = parts[1].parse().unwrap_or(0.0);
            let y: f32 = parts[2].parse().unwrap_or(0.0);
            geoms.push((ch_idx, ChannelGeom { x_um: x, y_um: y }));
            ch_idx += 1;
            idx += 1;
        } else {
            // the first token is the header "(probe_type,n_col,n_row,...)"
            // skip it; it contains colons but different format
        }
    }

    if geoms.is_empty() {
        return default();
    }

    let mut out = vec![ChannelGeom { x_um: 0.0, y_um: 0.0 }; n_ap];
    for (i, g) in geoms.into_iter().take(n_ap) {
        out[i] = g;
    }
    out
}

// ---------------------------------------------------------------------------
// Raw data access
// ---------------------------------------------------------------------------

/// Flat buffer of i16 samples, shape: [n_samples × n_saved_chans]
pub enum RawData {
    Loaded(Vec<i16>),
    Mmap(memmap2::Mmap),
}

impl RawData {
    /// Return a slice of int16 samples for the given sample range and the AP channels only.
    /// Output shape: n_ap × n_samp, stored as a flat Vec<f32> in µV.
    /// Channels are ordered 0..n_ap (sync channel excluded).
    pub fn read_chunk_uv(
        &self,
        first_sample: usize,
        n_samp: usize,
        meta: &Meta,
    ) -> Vec<f32> {
        let n_ch = meta.n_saved_chans;
        let n_ap = meta.n_ap_chans;
        let scale = meta.uv_per_bit;

        // clamp to available samples
        let n_samp = n_samp.min(meta.n_samples.saturating_sub(first_sample));

        let raw = self.as_i16_slice();
        let start = first_sample * n_ch;
        let end = (first_sample + n_samp) * n_ch;
        let src = &raw[start..end.min(raw.len())];

        // output: [n_ap][n_samp], row-major
        let mut out = vec![0.0f32; n_ap * n_samp];
        // rayon parallel over channels
        use rayon::prelude::*;
        out.par_chunks_mut(n_samp)
            .enumerate()
            .for_each(|(ch, row)| {
                for t in 0..n_samp {
                    let idx = t * n_ch + ch;
                    row[t] = if idx < src.len() {
                        src[idx] as f32 * scale
                    } else {
                        0.0
                    };
                }
            });
        out
    }

    fn as_i16_slice(&self) -> &[i16] {
        match self {
            RawData::Loaded(v) => v.as_slice(),
            RawData::Mmap(m) => bytemuck::cast_slice(m.as_ref()),
        }
    }

    pub fn n_i16_samples(&self) -> usize {
        self.as_i16_slice().len()
    }
}

/// Open the raw data file via memory-mapped I/O.
/// The OS page-cache handles read-ahead and eviction — no OOM risk.
pub fn open_data(bin_path: &Path, meta: &Meta) -> Result<(RawData, usize)> {
    let file = std::fs::File::open(bin_path)
        .with_context(|| format!("opening {}", bin_path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok((RawData::Mmap(mmap), meta.n_samples))
}

/// Load an additional chunk starting at `from_sample` up to `max_bytes`.
/// Extends an existing Loaded buffer.
pub fn extend_data(
    bin_path: &Path,
    meta: &Meta,
    existing: &mut RawData,
    already_loaded_samples: usize,
    headroom_bytes: u64,
) -> Result<usize> {
    use std::io::{Read, Seek, SeekFrom};

    let file_size = std::fs::metadata(bin_path)?.len();
    let bytes_per_sample = (meta.n_saved_chans * 2) as u64;
    let remaining_samples = meta.n_samples.saturating_sub(already_loaded_samples);
    if remaining_samples == 0 {
        return Ok(already_loaded_samples);
    }

    let available = free_ram_bytes().saturating_sub(headroom_bytes);
    let n_new = ((available / bytes_per_sample) as usize).min(remaining_samples);
    if n_new == 0 {
        bail!("Not enough RAM to load more data");
    }

    let byte_offset = already_loaded_samples as u64 * bytes_per_sample;
    let load_bytes = (n_new as u64 * bytes_per_sample).min(file_size - byte_offset);

    let mut file = std::fs::File::open(bin_path)?;
    file.seek(SeekFrom::Start(byte_offset))?;
    let mut buf = vec![0u8; load_bytes as usize];
    file.read_exact(&mut buf)?;
    let new_i16: Vec<i16> = bytemuck::cast_vec(buf);

    match existing {
        RawData::Loaded(v) => v.extend_from_slice(&new_i16),
        RawData::Mmap(_) => {
            // should not be called on mmap, but handle gracefully
            *existing = RawData::Loaded(new_i16);
        }
    }

    Ok(already_loaded_samples + n_new)
}

/// Query free physical RAM in bytes via /proc/meminfo
pub fn free_ram_bytes() -> u64 {
    // MemAvailable is the best estimate of usable free RAM
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            if line.starts_with("MemAvailable:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return kb * 1024;
            }
        }
    }
    4 * 1024 * 1024 * 1024 // fallback: 4 GB
}
