use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use oramacore_lib::data_structures::basic::BasicIndex;
use oramacore_lib::data_structures::basic2::BasicIndex2;
use oramacore_lib::data_structures::hnsw2::HNSW2Index;
use rand::distr::{Distribution, Uniform};

const DIM: usize = 128;
const N: usize = 10_000;

fn generate_sample() -> Vec<Vec<f32>> {
    let normal = Uniform::new(0.0, 10.0).unwrap();
    let n = N;
    let dimension = DIM;
    (0..n)
        .map(|_| {
            (0..dimension)
                .map(|_| normal.sample(&mut rand::rng()))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn build_basic(dim: usize, data: Vec<Vec<f32>>) -> BasicIndex<u64> {
    let mut index = BasicIndex::new(dim);
    for (i, sample) in data.into_iter().enumerate() {
        index.add_owned(sample, i as u64);
    }

    index
}

fn build_basic2(dim: usize, data: Vec<Vec<f32>>) -> BasicIndex2<u64> {
    let mut index = BasicIndex2::new(dim);
    for (i, sample) in data.into_iter().enumerate() {
        index.add_owned(&sample, i as u64);
    }

    index
}

fn build_hnsw2(dim: usize, data: Vec<Vec<f32>>) -> HNSW2Index<u64> {
    let mut index = HNSW2Index::new(dim);
    for (i, sample) in data.into_iter().enumerate() {
        index.add(&sample, i as u64).unwrap();
    }
    index.build().unwrap();

    index
}

pub fn hnsw_vs_basic(c: &mut Criterion) {
    let data = generate_sample();
    let data2 = data.clone();

    c.bench_function("hnsw_vs_basic - build hnsw2", |b| {
        b.iter(|| {
            let _ = build_hnsw2(DIM, black_box(data2.clone()));
        });
    });

    c.bench_function("hnsw_vs_basic - build basic", |b| {
        b.iter(|| {
            let _ = build_basic(DIM, black_box(data2.clone()));
        });
    });

    c.bench_function("hnsw_vs_basic - build basic2", |b| {
        b.iter(|| {
            let _ = build_basic2(DIM, black_box(data2.clone()));
        });
    });

    c.bench_function("hnsw_vs_basic - search hnsw2", |b| {
        let index = build_hnsw2(DIM, data.clone());
        b.iter(|| {
            let _ = index.search(black_box(&data[0]), 10);
        });
    });
    c.bench_function("hnsw_vs_basic - search basic", |b| {
        let index = build_basic(DIM, data.clone());
        b.iter(|| {
            let _ = index.search(black_box(&data[0]), 10);
        });
    });
    c.bench_function("hnsw_vs_basic - search basic2", |b| {
        let index = build_basic2(DIM, data.clone());
        b.iter(|| {
            let _ = index.search(black_box(&data[0]), 10);
        });
    });
}

criterion_group!(benches, hnsw_vs_basic);
criterion_main!(benches);
