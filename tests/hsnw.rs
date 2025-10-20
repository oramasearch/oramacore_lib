use rallo::RalloAllocator;

// This is the maximum length of a frame
const MAX_FRAME_LENGTH: usize = 512;
// Maximum number of allocations to keep
const MAX_LOG_COUNT: usize = 1_024 * 256;
#[global_allocator]
static ALLOCATOR: RalloAllocator<MAX_FRAME_LENGTH, MAX_LOG_COUNT> = RalloAllocator::new();

#[cfg(feature = "track_allocations")]
#[test]
fn test_allocation() {
    use oramacore_lib::data_structures::hnsw2::HNSW2Index;
    use rand::distr::{Distribution, Uniform};
    use std::time::Instant;

    let n = 1_000;
    let dimension = 64;

    let normal = Uniform::new(0.0, 10.0).unwrap();
    let samples = (0..n)
        .map(|_| {
            (0..dimension)
                .map(|_| normal.sample(&mut rand::rng()))
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();
    let mut index = HNSW2Index::new(dimension);

    let now = Instant::now();
    for (i, sample) in samples.into_iter().enumerate() {
        index.add_owned(sample, i).unwrap();
    }

    println!("Time to add: {:?}", now.elapsed());
    unsafe { ALLOCATOR.start_track() };
    println!("Tracking: {:?}", now.elapsed());
    index.build().unwrap();
    ALLOCATOR.stop_track();

    // Safety: it is called after `stop_track`
    let stats = unsafe { ALLOCATOR.calculate_stats() };
    let tree = stats.into_tree().unwrap();

    let file_name = "simple-memory-flamegraph.html";
    let path = std::env::current_dir().unwrap().join(file_name);
    tree.print_flamegraph(&path);

    println!("Flamegraph saved to {}", path.display());
}
