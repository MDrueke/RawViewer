use egui::{CentralPanel, TextureHandle, TextureOptions, TopBottomPanel, Ui, Vec2};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::data::{Meta, RawData, open_data};
use crate::preprocess::{CmrMode, Filters, PreprocConfig, SpatialMode};
use crate::render::build_heatmap_into;
use crate::worker::{SharedWorkerState, WorkerRequest, WorkerState, WorkerStatus, spawn_worker};

// half-buffer for background worker in seconds
const WORKER_HALF_WINDOW_SECS: f64 = 10.0;
// fraction of loaded region at which to warn about reaching the end
const LOAD_WARN_FRACTION: f64 = 0.95;

pub struct RawViewerApp {
    // --- file info ---
    bin_path: PathBuf,
    meta: Arc<Meta>,
    raw: Arc<RawData>,
    /// How many samples are currently loaded in RAM (< meta.n_samples if partial)
    loaded_samples: usize,
    /// true if the whole file is loaded
    fully_loaded: bool,
    /// true if data is mmap (not fully in RAM)
    is_mmap: bool,

    // --- view state ---
    /// Current view start in seconds
    view_start_s: f64,
    /// Current view duration in seconds
    view_dur_s: f64,
    /// First channel to display (inclusive)
    ch_first: usize,
    /// Last channel to display (inclusive)
    ch_last: usize,
    /// Amplitude scale (±µV displayed)
    vmax: f32,

    // --- preprocessing config ---
    preproc_cfg: PreprocConfig,
    preproc_filters: Arc<Mutex<Filters>>,
    // if destripe is on, HPF checkbox is locked
    hp_locked_by_destripe: bool,
    hp_before_destripe: bool,   // saved HP state before destripe was enabled

    // --- async worker ---
    worker_state: SharedWorkerState,
    _worker_handle: std::thread::JoinHandle<()>,

    // --- rendering ---
    heatmap_texture: Option<TextureHandle>,
    pixel_buf: Vec<u8>,
    /// view_first of the last successfully rendered frame
    last_rendered_first: usize,
    /// config of the last successfully rendered frame
    last_rendered_cfg: Option<PreprocConfig>,
    /// number of samples in last render
    last_rendered_n: usize,

    // --- smooth-scroll state ---
    /// when we first noticed the view is outside the worker buffer
    waiting_since: Option<Instant>,
    /// center sample of the last request sent to the worker
    last_requested_center: usize,

    // --- UI state ---
    show_load_dialog: bool,
    #[allow(dead_code)]
    status_msg: String,
    pending_cfg_recompute: bool,
}

impl RawViewerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, bin_path: PathBuf) -> anyhow::Result<Self> {
        let meta_path = bin_path.with_extension("meta");
        let meta = Arc::new(Meta::from_file(&meta_path)?);

        let (raw, loaded_samples) = open_data(&bin_path, &meta)?;
        let fully_loaded = true; // mmap always covers the whole file
        let is_mmap = true;
        let raw = Arc::new(raw);

        let fs = meta.sample_rate;
        let preproc_cfg = PreprocConfig {
            dc_removal: true,  // on by default
            cmr: CmrMode::None,
            highpass: false,
            spatial: SpatialMode::None,
            sample_rate: fs,
        };
        let filters = Arc::new(Mutex::new(Filters::new(&preproc_cfg)));
        let shared = Arc::new((Mutex::new(WorkerState::new()), Condvar::new()));

        let handle = spawn_worker(
            Arc::clone(&raw),
            Arc::clone(&meta),
            Arc::clone(&filters),
            Arc::clone(&shared),
        );

        let n_ap = meta.n_ap_chans;

        // send initial request so the worker pre-computes the first view
        {
            let (lock, cvar) = &*shared;
            lock.lock().unwrap().request = Some(WorkerRequest {
                center_sample: (WORKER_HALF_WINDOW_SECS * fs) as usize,
                half_window: (WORKER_HALF_WINDOW_SECS * fs) as usize,
                cfg: preproc_cfg.clone(),
            });
            cvar.notify_one();
        }

        Ok(Self {
            bin_path,
            meta,
            raw,
            loaded_samples,
            fully_loaded,
            is_mmap,
            view_start_s: 0.0,
            view_dur_s: 0.5,
            ch_first: 0,
            ch_last: n_ap.saturating_sub(1),
            vmax: 250.0,
            preproc_cfg: preproc_cfg.clone(),
            preproc_filters: filters,
            hp_locked_by_destripe: false,
            hp_before_destripe: false,
            worker_state: shared,
            _worker_handle: handle,
            heatmap_texture: None,
            pixel_buf: Vec::new(),
            last_rendered_first: usize::MAX,
            last_rendered_cfg: None,
            last_rendered_n: 0,
            waiting_since: None,
            last_requested_center: 0,
            show_load_dialog: false,
            status_msg: String::new(),
            pending_cfg_recompute: false,
        })
    }

    /// Send a new request to the worker if the view/config changed.
    /// The Condvar wakes the worker immediately (no 20 ms poll delay).
    fn request_recompute(&self) {
        let fs = self.meta.sample_rate;
        let center = ((self.view_start_s + self.view_dur_s / 2.0) * fs) as usize;
        let half = (WORKER_HALF_WINDOW_SECS * fs) as usize;
        let req = WorkerRequest {
            center_sample: center,
            half_window: half,
            cfg: self.preproc_cfg.clone(),
        };
        let (lock, cvar) = &*self.worker_state;
        lock.lock().unwrap().request = Some(req);
        cvar.notify_one();
    }

    // -----------------------------------------------------------------------
    // UI panels
    // -----------------------------------------------------------------------

    fn draw_toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("RawViewer").strong());
            ui.separator();
            ui.label(format!(
                "{}",
                self.bin_path.file_name().unwrap_or_default().to_string_lossy()
            ));
            ui.separator();

            // time display
            ui.label(format!("t = {:.3} s", self.view_start_s));

            // window duration — text field limited to 10 s
            ui.label("Window:");
            let dur_display = format!("{:.3}", self.view_dur_s);
            let mut dur_str = dur_display.clone();
            let resp = ui.add(
                egui::TextEdit::singleline(&mut dur_str)
                    .desired_width(55.0)
                    .hint_text("s")
            );
            if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let Ok(v) = dur_str.trim().parse::<f64>() {
                    let new_dur = v.clamp(0.01, 10.0);
                    if (new_dur - self.view_dur_s).abs() > 1e-6 {
                        self.view_dur_s = new_dur;
                        self.heatmap_texture = None; // force immediate redraw
                        self.pending_cfg_recompute = true;
                    }
                }
            }

            ui.separator();

            // amplitude scale
            ui.label("±µV:");
            if ui
                .add(egui::Slider::new(&mut self.vmax, 10.0..=5000.0).logarithmic(true))
                .changed()
            {
                // no recompute needed, just re-render
                self.heatmap_texture = None;
            }
        });
    }

    fn draw_preproc_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Preprocessing:");

            // DC offset removal
            let mut dc = self.preproc_cfg.dc_removal;
            if ui.checkbox(&mut dc, "DC").changed() {
                self.preproc_cfg.dc_removal = dc;
                self.pending_cfg_recompute = true;
            }

            ui.separator();

            // CMR
            ui.label("CMR:");
            let mut cmr = self.preproc_cfg.cmr;
            let cmr_changed = ui.radio_value(&mut cmr, CmrMode::None, "Off").changed()
                || ui.radio_value(&mut cmr, CmrMode::Global, "Global").changed();
            if cmr_changed {
                self.preproc_cfg.cmr = cmr;
                self.pending_cfg_recompute = true;
            }

            ui.separator();

            // HPF checkbox (greyed out if destripe is active)
            let hp_enabled = !self.hp_locked_by_destripe;
            let mut hp = self.preproc_cfg.highpass;
            if ui
                .add_enabled(hp_enabled, egui::Checkbox::new(&mut hp, "300 Hz HP"))
                .on_disabled_hover_text("Included in destripe")
                .changed()
            {
                self.preproc_cfg.highpass = hp;
                self.pending_cfg_recompute = true;
            }

            ui.separator();
            ui.label("Spatial:");

            let mut spatial = self.preproc_cfg.spatial;
            let changed = ui.radio_value(&mut spatial, SpatialMode::None, "None").changed()
                || ui.radio_value(&mut spatial, SpatialMode::Destripe, "Destripe (IBL)").changed();
            if changed {
                if spatial == SpatialMode::Destripe && !self.hp_locked_by_destripe {
                    // enabling destripe: save current HP state
                    self.hp_before_destripe = self.preproc_cfg.highpass;
                    self.preproc_cfg.highpass = true;
                    self.hp_locked_by_destripe = true;
                } else if spatial != SpatialMode::Destripe && self.hp_locked_by_destripe {
                    // disabling destripe: restore HP state
                    self.preproc_cfg.highpass = self.hp_before_destripe;
                    self.hp_locked_by_destripe = false;
                }
                self.preproc_cfg.spatial = spatial;
                {
                    let mut f = self.preproc_filters.lock().unwrap();
                    *f = Filters::new(&self.preproc_cfg);
                }
                self.heatmap_texture = None; // show raw immediately while recomputing
                self.pending_cfg_recompute = true;
            }

            // status indicator
            let status = {
                let (lock, _) = &*self.worker_state;
                lock.lock().unwrap().status.clone()
            };
            if status == WorkerStatus::Computing {
                if self.waiting_since.map(|t| t.elapsed().as_millis() > 150).unwrap_or(false) {
                    ui.spinner();
                    ui.label("Computing…");
                }
            }
        });
    }

    fn draw_channel_controls(&mut self, ui: &mut Ui) {
        let n_ap = self.meta.n_ap_chans;
        ui.horizontal(|ui| {
            ui.label("Channels:");
            // display as 1-indexed; store 0-indexed internally
            let mut cf_1 = self.ch_first + 1;
            let mut cl_1 = self.ch_last + 1;
            let mut changed = false;
            changed |= ui
                .add(egui::Slider::new(&mut cf_1, 1..=n_ap).text("First"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut cl_1, 1..=n_ap).text("Last"))
                .changed();
            if changed {
                let cf = (cf_1 - 1).min(cl_1 - 1);
                let cl = (cl_1 - 1).max(cf_1 - 1);
                self.ch_first = cf;
                self.ch_last = cl;
                self.heatmap_texture = None;
            }

            ui.separator();
            ui.label("Jump to (s):");
            let mut jump_str = format!("{:.3}", self.view_start_s);
            let resp = ui.add(egui::TextEdit::singleline(&mut jump_str).desired_width(70.0));
            if resp.lost_focus() {
                if let Ok(t) = jump_str.parse::<f64>() {
                    let max_t = self.meta.n_samples as f64 / self.meta.sample_rate - self.view_dur_s;
                    self.view_start_s = t.clamp(0.0, max_t.max(0.0));
                    // buffer coverage check will handle recompute
                }
            }
        });
    }

    fn draw_nav_bar(&mut self, ui: &mut Ui) {
        let total_s = self.meta.n_samples as f64 / self.meta.sample_rate;
        let loaded_s = self.loaded_samples as f64 / self.meta.sample_rate;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(ui.available_width(), 32.0),
            egui::Sense::click(),
        );
        let rect = response.rect;
        let w = rect.width();

        // background
        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x18, 0x18, 0x18));

        // hatched region = loaded in RAM
        let loaded_frac = (loaded_s / total_s) as f32;
        let loaded_rect = egui::Rect::from_min_size(
            rect.min,
            Vec2::new(w * loaded_frac, rect.height()),
        );

        // hatched fill (draw diagonal lines)
        if self.is_mmap {
            // mmap: show solid grey to distinguish from true RAM
            painter.rect_filled(loaded_rect, 0.0, egui::Color32::from_rgba_premultiplied(60, 60, 80, 100));
        } else {
            // RAM: hatch pattern
            painter.rect_filled(loaded_rect, 0.0, egui::Color32::from_rgba_premultiplied(50, 80, 50, 60));
            let step = 8.0;
            let mut x = rect.min.x;
            while x < rect.min.x + w * loaded_frac {
                painter.line_segment(
                    [egui::pos2(x, rect.min.y), egui::pos2(x + rect.height(), rect.max.y)],
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(80, 140, 80, 150)),
                );
                x += step;
            }
        }

        // current view marker
        let view_frac = (self.view_start_s / total_s) as f32;
        let view_w_frac = (self.view_dur_s / total_s) as f32;
        let view_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + w * view_frac, rect.min.y),
            Vec2::new((w * view_w_frac).max(2.0), rect.height()),
        );
        painter.rect_filled(view_rect, 1.0, egui::Color32::from_rgba_premultiplied(200, 200, 255, 180));

        // time labels
        let n_labels = 8;
        for i in 0..=n_labels {
            let frac = i as f32 / n_labels as f32;
            let t = frac as f64 * total_s;
            let x = rect.min.x + w * frac;
            painter.line_segment(
                [egui::pos2(x, rect.max.y - 6.0), egui::pos2(x, rect.max.y)],
                egui::Stroke::new(1.0, egui::Color32::GRAY),
            );
            painter.text(
                egui::pos2(x, rect.max.y - 8.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{:.0}s", t),
                egui::FontId::proportional(9.0),
                egui::Color32::GRAY,
            );
        }

        // click to seek
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let frac = ((pos.x - rect.min.x) / w).clamp(0.0, 1.0) as f64;
                let max_t = total_s - self.view_dur_s;
                self.view_start_s = (frac * total_s).clamp(0.0, max_t.max(0.0));
                // buffer coverage check will handle recompute
            }
        }

        // warn near end of loaded region
        if !self.fully_loaded {
            let loaded_end = self.loaded_samples as f64 / self.meta.sample_rate;
            let view_end = self.view_start_s + self.view_dur_s;
            if view_end > loaded_end * LOAD_WARN_FRACTION {
                self.show_load_dialog = true;
            }
        }
    }
}

impl eframe::App for RawViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // request repaint while worker is computing
        {
            let (lock, _) = &*self.worker_state;
            let st = lock.lock().unwrap();
            if st.status == WorkerStatus::Computing || st.request.is_some() {
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
            }
        }

        // ---- discrete mouse-wheel scroll (one tick = 90% of window) ----
        let ticks = ctx.input(|i| {
            i.events.iter().filter_map(|e| match e {
                egui::Event::MouseWheel { delta, .. } => Some(-delta.y.signum()),
                _ => None,
            }).sum::<f32>()
        });
        if ticks != 0.0 {
            let fs = self.meta.sample_rate;
            let total_s = self.meta.n_samples as f64 / fs;
            let max_start = (total_s - self.view_dur_s).max(0.0);
            let step = ticks as f64 * self.view_dur_s * 0.9;
            self.view_start_s = (self.view_start_s + step).clamp(0.0, max_start);
        }

        // keyboard scrolling
        ctx.input(|i| {
            let fs = self.meta.sample_rate;
            let total_s = self.meta.n_samples as f64 / fs;
            let step = self.view_dur_s * 0.5;
            let max_start = (total_s - self.view_dur_s).max(0.0);
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::D) {
                self.view_start_s = (self.view_start_s + step).min(max_start);
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::A) {
                self.view_start_s = (self.view_start_s - step).max(0.0);
            }
        });

        // ---- top bar ----
        TopBottomPanel::top("toolbar").show(ctx, |ui| {
            self.draw_toolbar(ui);
        });
        TopBottomPanel::top("preproc").show(ctx, |ui| {
            self.draw_preproc_panel(ui);
        });
        TopBottomPanel::top("chan_ctrl").show(ctx, |ui| {
            self.draw_channel_controls(ui);
        });

        // ---- nav bar at bottom ----
        TopBottomPanel::bottom("nav_bar").show(ctx, |ui| {
            self.draw_nav_bar(ui);
        });

        // ---- "load next chunk?" dialog ----
        if self.show_load_dialog {
            let mut open = true;
            egui::Window::new("Load more data?")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("You are near the end of the portion loaded in RAM.");
                    ui.label("Load the next chunk?");
                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            self.show_load_dialog = false;
                            // TODO: call extend_data on a background thread
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_load_dialog = false;
                        }
                    });
                });
            if !open {
                self.show_load_dialog = false;
            }
        }

        // ---- main heatmap panel ----
        CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(0x18, 0x18, 0x18)))
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let pw = avail.x as usize;
                let ph = avail.y as usize;

                if pw < 2 || ph < 2 {
                    return;
                }

                // no smooth scroll — discrete ticks handled above

                let fs = self.meta.sample_rate;
                let view_first = (self.view_start_s * fs) as usize;
                let view_n = (self.view_dur_s * fs) as usize;
                let n_ap = self.meta.n_ap_chans;
                let center = view_first + view_n / 2;

                // ---- check buffer coverage (zero-copy Arc grab) ----
                let covered = {
                    let (lock, _) = &*self.worker_state;
                    let st = lock.lock().unwrap();
                    if let Some(buf) = &st.buffer {
                        let buf_end = buf.first_sample + buf.n_samp;
                        buf.first_sample <= view_first
                            && view_first + view_n <= buf_end
                            && buf.cfg == self.preproc_cfg
                    } else { false }
                };

                if covered {
                    // inside buffer — clear waiting state
                    self.waiting_since = None;

                    // rebuild texture when view or config changed
                    let cfg_changed = self.last_rendered_cfg.as_ref() != Some(&self.preproc_cfg);
                    let pos_changed = self.last_rendered_first != view_first;
                    let need_rebuild = self.heatmap_texture.is_none() || pos_changed
                        || cfg_changed || view_n != self.last_rendered_n;

                    if need_rebuild && view_n > 0 {
                        // grab Arc clone with lock held, then release before heavy work
                        let (data_arc, stride, offset, n) = {
                            let (lock, _) = &*self.worker_state;
                            let st = lock.lock().unwrap();
                            let buf = st.buffer.as_ref().unwrap();
                            let off = view_first - buf.first_sample;
                            let n = view_n.min(buf.n_samp - off);
                            (Arc::clone(&buf.data), buf.n_samp, off, n)
                        };
                        build_heatmap_into(
                            &mut self.pixel_buf,
                            &data_arc,
                            n_ap, stride, offset, n,
                            self.ch_first, self.ch_last,
                            pw, ph, self.vmax,
                        );
                        let img = egui::ColorImage::from_rgba_unmultiplied(
                            [pw, ph], &self.pixel_buf);
                        self.heatmap_texture = Some(
                            ctx.load_texture("heatmap", img, TextureOptions::NEAREST));
                        self.last_rendered_first = view_first;
                        self.last_rendered_n = view_n;
                        self.last_rendered_cfg = Some(self.preproc_cfg.clone());
                    }

                    // ---- prefetch: silently request a new buffer if within 5 s of edge ----
                    let prefetch_margin = (5.0 * fs) as usize;
                    let (buf_first, buf_end) = {
                        let (lock, _) = &*self.worker_state;
                        let st = lock.lock().unwrap();
                        if let Some(buf) = &st.buffer {
                            (buf.first_sample, buf.first_sample + buf.n_samp)
                        } else { (0, 0) }
                    };
                    let near_edge = view_first < buf_first + prefetch_margin
                        || view_first + view_n + prefetch_margin > buf_end;
                    if near_edge && self.last_requested_center != center {
                        self.request_recompute();
                        self.last_requested_center = center;
                    }

                } else {
                    // outside buffer — request recompute if center changed
                    if self.last_requested_center != center || self.pending_cfg_recompute {
                        self.request_recompute();
                        self.last_requested_center = center;
                        if self.waiting_since.is_none() {
                            self.waiting_since = Some(Instant::now());
                        }
                    }

                    // keep existing texture — do NOT clear it
                    // if no texture yet, fall back to raw for immediate display
                    if self.heatmap_texture.is_none() && view_n > 0 {
                        let raw_data = Arc::new(
                            self.raw.read_chunk_uv(view_first, view_n, &self.meta));
                        build_heatmap_into(
                            &mut self.pixel_buf, &raw_data,
                            n_ap, view_n, 0, view_n,
                            self.ch_first, self.ch_last,
                            pw, ph, self.vmax,
                        );
                        let img = egui::ColorImage::from_rgba_unmultiplied(
                            [pw, ph], &self.pixel_buf);
                        self.heatmap_texture = Some(
                            ctx.load_texture("heatmap", img, TextureOptions::NEAREST));
                    }

                    // show raw warning if preprocessing is active
                    let preproc_active = self.preproc_cfg.highpass
                        || self.preproc_cfg.spatial != SpatialMode::None
                        || self.preproc_cfg.cmr != CmrMode::None
                        || self.preproc_cfg.dc_removal;
                    if preproc_active
                        && self.waiting_since.map(|t| t.elapsed().as_millis() > 150).unwrap_or(false)
                    {
                        ui.painter().text(
                            ui.clip_rect().center_top() + Vec2::new(0.0, 4.0),
                            egui::Align2::CENTER_TOP,
                            "⚠ Showing stale data — preprocessing…",
                            egui::FontId::proportional(13.0),
                            egui::Color32::YELLOW,
                        );
                    }
                }

                self.pending_cfg_recompute = false;

                // draw texture
                if let Some(tex) = &self.heatmap_texture {
                    let img_widget = egui::Image::new(tex)
                        .fit_to_exact_size(avail)
                        .sense(egui::Sense::hover());
                    let resp = ui.add(img_widget);

                    // bottom-left overlay: ch / time / voltage
                    if let Some(pos) = resp.hover_pos() {
                        let n_ch_display = self.ch_last + 1 - self.ch_first;
                        let frac_y = ((pos.y - resp.rect.top()) / resp.rect.height()).clamp(0.0, 1.0);
                        // ch_last is at top, ch_first at bottom
                        let ch_0 = self.ch_last.saturating_sub((frac_y * n_ch_display as f32) as usize);
                        let frac_x = ((pos.x - resp.rect.left()) / resp.rect.width()).clamp(0.0, 1.0);
                        let t = self.view_start_s + frac_x as f64 * self.view_dur_s;

                        // read voltage from displayed buffer
                        let voltage_uv: Option<f32> = {
                            let (lock, _) = &*self.worker_state;
                            let st = lock.lock().unwrap();
                            if let Some(buf) = &st.buffer {
                                let t_sample = (t * self.meta.sample_rate) as usize;
                                if t_sample >= buf.first_sample {
                                    let off = t_sample - buf.first_sample;
                                    if off < buf.n_samp && ch_0 < n_ap {
                                        Some(buf.data[ch_0 * buf.n_samp + off])
                                    } else { None }
                                } else { None }
                            } else { None }
                        };

                        let volt_str = voltage_uv
                            .map(|v| format!("  {:.1} µV", v))
                            .unwrap_or_default();

                        // 1-indexed channel display
                        let label = format!("Ch {}  t = {:.4} s{}", ch_0 + 1, t, volt_str);
                        ui.painter().text(
                            resp.rect.left_bottom() + Vec2::new(6.0, -6.0),
                            egui::Align2::LEFT_BOTTOM,
                            label,
                            egui::FontId::proportional(12.0),
                            egui::Color32::from_rgba_unmultiplied(220, 220, 220, 200),
                        );
                    }
                }
            });
    }
}
