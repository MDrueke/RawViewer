/// Probe geometry helpers — distances between channels for lCMR etc.
use crate::data::ChannelGeom;

/// Euclidean distance in µm between two channels
pub fn dist_um(a: &ChannelGeom, b: &ChannelGeom) -> f32 {
    let dx = a.x_um - b.x_um;
    let dy = a.y_um - b.y_um;
    (dx * dx + dy * dy).sqrt()
}

/// For each channel, find the indices of channels whose distance is in [inner_um, outer_um].
/// Used for local CMR (future feature).
pub fn annular_neighbors(
    geom: &[ChannelGeom],
    inner_um: f32,
    outer_um: f32,
) -> Vec<Vec<usize>> {
    geom.iter()
        .map(|g| {
            geom.iter()
                .enumerate()
                .filter(|(_, other)| {
                    let d = dist_um(g, other);
                    d >= inner_um && d <= outer_um
                })
                .map(|(i, _)| i)
                .collect()
        })
        .collect()
}
