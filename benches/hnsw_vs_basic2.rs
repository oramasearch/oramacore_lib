use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use oramacore_lib::data_structures::basic::BasicIndex;
use oramacore_lib::data_structures::basic2::BasicIndex2;
use oramacore_lib::data_structures::basic3::BasicIndex3;
use rand::distr::{Distribution, Uniform};

fn generate_sample(dim: usize, n: usize) -> Vec<Vec<f32>> {
    let normal = Uniform::new(0.0, 10.0).unwrap();
    (0..n)
        .map(|_| {
            (0..dim)
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

fn build_basic3(dim: usize, data: Vec<Vec<f32>>) -> BasicIndex3<u64> {
    let mut index = BasicIndex3::new(dim);
    for (i, sample) in data.into_iter().enumerate() {
        index.add_owned(sample, i as u64);
    }

    index
}

pub fn hnsw_vs_basic2(c: &mut Criterion) {
    let data = [
        (128, 1_000),
        (384, 1_000),
        (512, 1_000),
        (128, 10_000),
        (384, 10_000),
        (512, 10_000),
        (128, 20_000),
        (384, 20_000),
        (512, 20_000),
        (128, 50_000),
        (384, 50_000),
        (512, 50_000),
    ];

    for d in &data {
        let dim = d.0;
        let n = d.1;

        let points = generate_sample(dim, n);
        let points2 = points.clone();

        let id = format!("hnsw_vs_basic2 - basic - build dim={} n={}", dim, n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _ = build_basic(dim, black_box(points2.clone()));
            });
        });
        let id = format!("hnsw_vs_basic2 - basic2 - build dim={} n={}", dim, n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _ = build_basic2(dim, black_box(points2.clone()));
            });
        });
        let id = format!("hnsw_vs_basic2 - basic3 - build dim={} n={}", dim, n);
        c.bench_function(&id, |b| {
            b.iter(|| {
                let _ = build_basic3(dim, black_box(points2.clone()));
            });
        });
        let id = format!("hnsw_vs_basic2 - basic - search dim={} n={}", dim, n);
        c.bench_function(&id, |b| {
            let index = build_basic(dim, points2.clone());
            b.iter(|| {
                let _ = index.search(black_box(&points2[0]), 10);
            });
        });
        let id = format!("hnsw_vs_basic2 - basic2 - search dim={} n={}", dim, n);
        c.bench_function(&id, |b| {
            let index = build_basic2(dim, points2.clone());
            b.iter(|| {
                let _ = index.search(black_box(&points2[0]), 10);
            });
        });
        let id = format!("hnsw_vs_basic2 - basic3 - search dim={} n={}", dim, n);
        c.bench_function(&id, |b| {
            let index = build_basic3(dim, points2.clone());
            b.iter(|| {
                let _ = index.search(black_box(&points2[0]), 10);
            });
        });
    }
}

criterion_group!(benches, hnsw_vs_basic2);
criterion_main!(benches);
