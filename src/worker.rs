use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::data::{Meta, RawData};
use crate::preprocess::{Filters, PreprocConfig, preprocess};

// ---------------------------------------------------------------------------
// Shared state between UI thread and worker thread
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WorkerStatus {
    Idle,
    Computing,
    Done,
    Error(String),
}

/// A preprocessed segment. `data` is Arc so the UI thread can take a cheap
/// reference without copying the buffer.
#[derive(Clone)]
pub struct PreprocBuffer {
    pub first_sample: usize,
    pub n_samp: usize,
    /// Layout: data[ch * n_samp .. (ch+1) * n_samp], µV
    pub data: Arc<Vec<f32>>,
    pub cfg: PreprocConfig,
}

pub struct WorkerState {
    pub buffer: Option<PreprocBuffer>,
    pub status: WorkerStatus,
    pub request: Option<WorkerRequest>,
}

impl WorkerState {
    pub fn new() -> Self {
        WorkerState { buffer: None, status: WorkerStatus::Idle, request: None }
    }
}

#[derive(Clone)]
pub struct WorkerRequest {
    pub center_sample: usize,
    pub half_window: usize,
    pub cfg: PreprocConfig,
}

/// `(Mutex<WorkerState>, Condvar)` — standard Rust pattern for waking the worker.
pub type SharedWorkerState = Arc<(Mutex<WorkerState>, Condvar)>;

pub fn spawn_worker(
    raw: Arc<RawData>,
    meta: Arc<Meta>,
    filt: Arc<Mutex<Filters>>,
    shared: SharedWorkerState,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let (lock, cvar) = &*shared;
        loop {
            // wait for a request (Condvar replaces 20 ms sleep)
            let req = {
                let mut st = lock.lock().unwrap();
                loop {
                    if let Some(r) = st.request.take() {
                        st.status = WorkerStatus::Computing;
                        break r;
                    }
                    st = cvar.wait(st).unwrap();
                }
            };

            let first = req.center_sample.saturating_sub(req.half_window);
            let n_samp =
                (req.half_window * 2).min(meta.n_samples.saturating_sub(first));

            let mut data = raw.read_chunk_uv(first, n_samp, &meta);
            let filt_g = filt.lock().unwrap().clone();
            preprocess(&mut data, meta.n_ap_chans, n_samp, &req.cfg, &filt_g);

            let buf = PreprocBuffer {
                first_sample: first,
                n_samp,
                data: Arc::new(data),
                cfg: req.cfg,
            };

            let mut st = lock.lock().unwrap();
            if st.request.is_some() {
                // a newer request arrived while computing — loop again
                st.status = WorkerStatus::Idle;
            } else {
                st.buffer = Some(buf);
                st.status = WorkerStatus::Done;
            }
        }
    })
}
