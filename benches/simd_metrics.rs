use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use oramacore_lib::hnsw2::core::{simd_metrics::SIMDOptmized, simd_metrics_old};
use rand::distr::{Distribution, Uniform};

fn generate_vectors_f32(size: usize) -> (Vec<f32>, Vec<f32>) {
    let normal = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let a: Vec<f32> = (0..size).map(|_| normal.sample(&mut rand::rng())).collect();
    let b: Vec<f32> = (0..size).map(|_| normal.sample(&mut rand::rng())).collect();
    (a, b)
}

fn generate_vectors_f64(size: usize) -> (Vec<f64>, Vec<f64>) {
    let normal = Uniform::new(-1.0f64, 1.0f64).unwrap();
    let a: Vec<f64> = (0..size).map(|_| normal.sample(&mut rand::rng())).collect();
    let b: Vec<f64> = (0..size).map(|_| normal.sample(&mut rand::rng())).collect();
    (a, b)
}

fn bench_dot_product_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_f32");

    for size in [64, 256, 1024, 4096].iter() {
        let (a, b) = generate_vectors_f32(*size);

        group.bench_with_input(BenchmarkId::new("old", size), size, |bencher, _| {
            bencher.iter(|| {
                <f32 as simd_metrics_old::SIMDOptmizedOld>::dot_product(
                    black_box(&a),
                    black_box(&b),
                )
                .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("new", size), size, |bencher, _| {
            bencher
                .iter(|| <f32 as SIMDOptmized>::dot_product(black_box(&a), black_box(&b)).unwrap());
        });
    }

    group.finish();
}

fn bench_dot_product_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_f64");

    for size in [64, 256, 1024, 4096].iter() {
        let (a, b) = generate_vectors_f64(*size);

        group.bench_with_input(BenchmarkId::new("old", size), size, |bencher, _| {
            bencher.iter(|| {
                <f64 as simd_metrics_old::SIMDOptmizedOld>::dot_product(
                    black_box(&a),
                    black_box(&b),
                )
                .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("new", size), size, |bencher, _| {
            bencher
                .iter(|| <f64 as SIMDOptmized>::dot_product(black_box(&a), black_box(&b)).unwrap());
        });
    }

    group.finish();
}

fn bench_euclidean_distance_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance_f32");

    for size in [64, 256, 1024, 4096].iter() {
        let (a, b) = generate_vectors_f32(*size);

        group.bench_with_input(BenchmarkId::new("old", size), size, |bencher, _| {
            bencher.iter(|| {
                <f32 as simd_metrics_old::SIMDOptmizedOld>::euclidean_distance(
                    black_box(&a),
                    black_box(&b),
                )
                .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("new", size), size, |bencher, _| {
            bencher.iter(|| {
                <f32 as SIMDOptmized>::euclidean_distance(black_box(&a), black_box(&b)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_euclidean_distance_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance_f64");

    for size in [64, 256, 1024, 4096].iter() {
        let (a, b) = generate_vectors_f64(*size);

        group.bench_with_input(BenchmarkId::new("old", size), size, |bencher, _| {
            bencher.iter(|| {
                <f64 as simd_metrics_old::SIMDOptmizedOld>::euclidean_distance(
                    black_box(&a),
                    black_box(&b),
                )
                .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("new", size), size, |bencher, _| {
            bencher.iter(|| {
                <f64 as SIMDOptmized>::euclidean_distance(black_box(&a), black_box(&b)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_manhattan_distance_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("manhattan_distance_f32");

    for size in [64, 256, 1024, 4096].iter() {
        let (a, b) = generate_vectors_f32(*size);

        group.bench_with_input(BenchmarkId::new("old", size), size, |bencher, _| {
            bencher.iter(|| {
                <f32 as simd_metrics_old::SIMDOptmizedOld>::manhattan_distance(
                    black_box(&a),
                    black_box(&b),
                )
                .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("new", size), size, |bencher, _| {
            bencher.iter(|| {
                <f32 as SIMDOptmized>::manhattan_distance(black_box(&a), black_box(&b)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_manhattan_distance_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("manhattan_distance_f64");

    for size in [64, 256, 1024, 4096].iter() {
        let (a, b) = generate_vectors_f64(*size);

        group.bench_with_input(BenchmarkId::new("old", size), size, |bencher, _| {
            bencher.iter(|| {
                <f64 as simd_metrics_old::SIMDOptmizedOld>::manhattan_distance(
                    black_box(&a),
                    black_box(&b),
                )
                .unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("new", size), size, |bencher, _| {
            bencher.iter(|| {
                <f64 as SIMDOptmized>::manhattan_distance(black_box(&a), black_box(&b)).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product_f32,
    bench_dot_product_f64,
    bench_euclidean_distance_f32,
    bench_euclidean_distance_f64,
    bench_manhattan_distance_f32,
    bench_manhattan_distance_f64
);
criterion_main!(benches);
