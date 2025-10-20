use std::{io::Write, time::Instant};

use oramacore_lib::data_structures::hnsw2::HNSW2Index;
use rand::distr::{Distribution, Uniform};

fn main() {
    let n = 50_000;
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
    println!("adding {} points took: {:.2?}", n, now.elapsed());
    index.build().unwrap();
    println!("building index took: {:.2?}", now.elapsed());

    for j in [
        8 * 1024,
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
    ]
    .iter()
    {
        let f = format!("file_{j}.hsnw");
        let mut f = std::fs::File::create(&f).unwrap();
        let buf = std::io::BufWriter::with_capacity(*j, &mut f);

        let before = Instant::now();
        bincode::serialize_into(buf, &index).unwrap();
        f.flush().unwrap();
        f.sync_data().unwrap();
        let elapsed = before.elapsed();

        println!("serialize to {j} bytes buffer took: {elapsed:.2?}");
    }
}
