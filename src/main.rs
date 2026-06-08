mod app;
mod data;
mod geometry;
mod preprocess;
mod render;
mod worker;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rawviewer", about = "Neuropixels raw data viewer")]
struct Args {
    /// Path to the .ap.bin SpikeGLX file
    #[arg(short, long)]
    file: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // if no file given, open a native file dialog
    let bin_path = match args.file {
        Some(p) => p,
        None => {
            // simple fallback: ask via stdin
            eprintln!("No file specified with --file. Usage: rawviewer --file <path.ap.bin>");
            std::process::exit(1);
        }
    };

    if !bin_path.exists() {
        anyhow::bail!("File not found: {}", bin_path.display());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("RawViewer")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RawViewer",
        options,
        Box::new(move |cc| {
            // set dark theme
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            match app::RawViewerApp::new(cc, bin_path) {
                Ok(a) => Ok(Box::new(a) as Box<dyn eframe::App>),
                Err(e) => {
                    eprintln!("Error opening file: {e}");
                    std::process::exit(1);
                }
            }
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}
