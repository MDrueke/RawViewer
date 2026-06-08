use rayon::prelude::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// SOS biquad — f32 arithmetic
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Sos {
    pub b0: f32, pub b1: f32, pub b2: f32,
    pub a1: f32, pub a2: f32,
}

pub fn sosfilt_inplace(sos: &[Sos], x: &mut [f32]) {
    for s in sos {
        let (b0, b1, b2, a1, a2) = (s.b0, s.b1, s.b2, s.a1, s.a2);
        let (mut z0, mut z1) = (0.0f32, 0.0f32);
        for v in x.iter_mut() {
            let xi = *v;
            let yi = b0 * xi + z0;
            z0 = b1 * xi - a1 * yi + z1;
            z1 = b2 * xi - a2 * yi;
            *v = yi;
        }
    }
}

pub fn sosfiltfilt_inplace(sos: &[Sos], x: &mut [f32]) {
    sosfilt_inplace(sos, x);
    x.reverse();
    sosfilt_inplace(sos, x);
    x.reverse();
}

// ---------------------------------------------------------------------------
// Butterworth highpass — correct bilinear transform with LP→HP conversion
// Verified: H(z=1)=0 (blocks DC), H(z=-1)=1 (passes Nyquist)
// ---------------------------------------------------------------------------

pub fn butter_highpass_sos(n: usize, wn: f64) -> Vec<Sos> {
    use std::f64::consts::PI;
    // prewarped analog cutoff (normalized fs=1, so 2/T = 2)
    let wd = 2.0 * (PI * wn / 2.0).tan();
    let ni = n as i32;
    let mut sections = Vec::new();

    for k in 0..n {
        let angle = PI * (2 * k as i32 + ni + 1) as f64 / (2.0 * ni as f64);
        let lp_re = angle.cos(); // always ≤ 0 for Butterworth LP prototype
        let lp_im = angle.sin();

        if lp_im.abs() < 1e-10 {
            // real pole → 1st order HP section (zero at z=1)
            // HP pole: pr = wd * lp_re  (< 0 since lp_re = -1 at angle=π)
            let pr = wd * lp_re;
            let d = 2.0 - pr;
            sections.push(Sos {
                b0: (2.0 / d) as f32,
                b1: (-2.0 / d) as f32,
                b2: 0.0,
                a1: ((-2.0 - pr) / d) as f32,
                a2: 0.0,
            });
        } else if lp_im > 0.0 {
            // complex conjugate pair → 2nd order HP section (double zero at z=1)
            let pr   = wd * lp_re; // < 0
            let pi_v = wd * lp_im; // > 0
            // analog quadratic denom: s² - 2*pr*s + (pr²+pi_v²)
            let a_c = -2.0 * pr;
            let b_c = pr * pr + pi_v * pi_v;
            let d = 4.0 + 2.0 * a_c + b_c;
            sections.push(Sos {
                b0: ( 4.0 / d) as f32,
                b1: (-8.0 / d) as f32,
                b2: ( 4.0 / d) as f32,
                a1: ((-8.0 + 2.0 * b_c) / d) as f32,
                a2: (( 4.0 - 2.0 * a_c + b_c) / d) as f32,
            });
            // conjugate (lp_im < 0) will be skipped in the next iteration
        }
        // lp_im < 0 → conjugate already handled, skip
    }
    sections
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialMode { None, Destripe }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmrMode { None, Global }

#[derive(Clone, Debug, PartialEq)]
pub struct PreprocConfig {
    pub dc_removal: bool,
    pub cmr: CmrMode,
    pub highpass: bool,
    pub spatial: SpatialMode,
    pub sample_rate: f64,
}

#[derive(Clone)]
pub struct Filters {
    pub hp_sos: Vec<Sos>,
    pub kfilt_sos: Vec<Sos>,
    pub kfilt_ntr_pad: usize,
    pub kfilt_lagc: usize,
}

impl Filters {
    pub fn new(cfg: &PreprocConfig) -> Self {
        let fs = cfg.sample_rate;
        Filters {
            hp_sos: butter_highpass_sos(3, 300.0 / fs * 2.0),
            kfilt_sos: butter_highpass_sos(3, 0.01),
            kfilt_ntr_pad: 60,
            kfilt_lagc: (fs / 10.0).round() as usize,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level entry point
// Order matches Python surface-detection tool:
//   DC removal → temporal HP → global CMR → spatial kfilt
// ---------------------------------------------------------------------------

pub fn preprocess(
    data: &mut Vec<f32>,
    n_ap: usize,
    n_samp: usize,
    cfg: &PreprocConfig,
    filt: &Filters,
) {
    if n_ap == 0 || n_samp == 0 { return; }

    if cfg.dc_removal {
        apply_dc_removal(data, n_samp);
    }

    // temporal HP first (same order as Python tool)
    let need_hp = cfg.highpass || cfg.spatial == SpatialMode::Destripe;
    if need_hp {
        apply_temporal_hp(data, n_samp, &filt.hp_sos);
    }

    // CMR after HP (removes common noise that remains after HP)
    match cfg.cmr {
        CmrMode::Global => apply_global_cmr(data, n_ap, n_samp),
        CmrMode::None => {}
    }

    if cfg.spatial == SpatialMode::Destripe {
        apply_kfilt(data, n_ap, n_samp, filt);
    }
}

// ---------------------------------------------------------------------------
// DC offset removal — subtract per-channel mean (parallel, cache-friendly)
// ---------------------------------------------------------------------------

fn apply_dc_removal(data: &mut Vec<f32>, n_samp: usize) {
    data.par_chunks_mut(n_samp).for_each(|row| {
        let mean = row.iter().sum::<f32>() / n_samp as f32;
        row.iter_mut().for_each(|v| *v -= mean);
    });
}

// ---------------------------------------------------------------------------
// Temporal HP — parallel over channels
// ---------------------------------------------------------------------------

fn apply_temporal_hp(data: &mut Vec<f32>, n_samp: usize, sos: &[Sos]) {
    data.par_chunks_mut(n_samp).for_each(|ch| {
        sosfiltfilt_inplace(sos, ch);
    });
}

// ---------------------------------------------------------------------------
// Global CMR — subtract median across channels per time step (parallel)
// ---------------------------------------------------------------------------

thread_local! {
    static CMR_COL: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

fn apply_global_cmr(data: &mut Vec<f32>, n_ap: usize, n_samp: usize) {
    let data_ptr = SendPtr(data.as_mut_ptr());
    let half = n_ap / 2;

    (0..n_samp).into_par_iter().for_each(|t| {
        let dp = data_ptr.0;
        let _ = &data_ptr;

        CMR_COL.with(|cell| {
            let mut col = cell.borrow_mut();
            col.resize(n_ap, 0.0);
            for ch in 0..n_ap {
                col[ch] = unsafe { *dp.add(ch * n_samp + t) };
            }
            col.select_nth_unstable_by(half, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let median = if n_ap % 2 == 0 {
                let upper = col[half];
                col[..half].select_nth_unstable_by(half - 1, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                (upper + col[half - 1]) / 2.0
            } else {
                col[half]
            };
            for ch in 0..n_ap {
                unsafe { *dp.add(ch * n_samp + t) -= median; }
            }
        });
    });
}

// ---------------------------------------------------------------------------
// AGC — sliding RMS envelope per channel
// ---------------------------------------------------------------------------

fn compute_agc_gain(data: &[f32], n_ap: usize, n_samp: usize, win: usize) -> Vec<f32> {
    let half = win / 2;
    let mut gain = vec![1.0f32; n_ap * n_samp];
    gain.par_chunks_mut(n_samp).enumerate().for_each(|(ch, g)| {
        let src = &data[ch * n_samp..(ch + 1) * n_samp];
        let mut sum_sq = 0.0f32;
        let mut count = 0usize;
        for t in 0..n_samp {
            if t + half < n_samp {
                let v = src[t + half]; sum_sq += v * v; count += 1;
            }
            if t > half && t - half - 1 < n_samp {
                let v = src[t - half - 1];
                sum_sq = (sum_sq - v * v).max(0.0);
                count = count.saturating_sub(1);
            }
            g[t] = if count > 0 { (sum_sq / count as f32).sqrt().max(1e-6) } else { 1.0 };
        }
    });
    gain
}

// ---------------------------------------------------------------------------
// Spatial kfilt — column-by-column, parallel over time
// ---------------------------------------------------------------------------

thread_local! {
    static KFILT_COL: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

fn apply_kfilt(data: &mut Vec<f32>, n_ap: usize, n_samp: usize, filt: &Filters) {
    let pad = filt.kfilt_ntr_pad;
    let n_padded = n_ap + 2 * pad;
    let gain = compute_agc_gain(data, n_ap, n_samp, filt.kfilt_lagc);

    for (ch, g) in gain.chunks(n_samp).enumerate() {
        let row = &mut data[ch * n_samp..(ch + 1) * n_samp];
        for (v, &gv) in row.iter_mut().zip(g.iter()) { *v /= gv; }
    }

    let data_ptr = SendPtr(data.as_mut_ptr());
    let gain_ptr = SendPtr(gain.as_ptr() as *mut f32);
    let sos = filt.kfilt_sos.clone();

    (0..n_samp).into_par_iter().for_each(|t| {
        let dp = data_ptr.0;
        let gp = gain_ptr.0;
        let _ = (&data_ptr, &gain_ptr);

        KFILT_COL.with(|cell| {
            let mut col = cell.borrow_mut();
            col.resize(n_padded, 0.0);
            for p in 0..pad {
                col[p] = unsafe { *dp.add((pad - 1 - p).min(n_ap - 1) * n_samp + t) };
            }
            for ch in 0..n_ap {
                col[pad + ch] = unsafe { *dp.add(ch * n_samp + t) };
            }
            for p in 0..pad {
                col[pad + n_ap + p] = unsafe { *dp.add(n_ap.saturating_sub(1 + p) * n_samp + t) };
            }
            sosfiltfilt_inplace(&sos, &mut col);
            for ch in 0..n_ap {
                let gv = unsafe { *gp.add(ch * n_samp + t) };
                unsafe { *dp.add(ch * n_samp + t) = col[pad + ch] * gv; }
            }
        });
    });
}
