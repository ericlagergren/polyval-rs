//! Benchmarks.

use core::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polyhash::*;

fn benchmark<B: Backend>(c: &mut Criterion, name: &str) {
    let key = Key::new_unchecked(&[1u8; 16]);
    let mut m = Polyval::<B>::new(&key);

    let sizes = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

    let mut g = c.benchmark_group(name);

    for size in sizes {
        g.throughput(Throughput::Bytes(size as u64));
        g.bench_with_input(
            BenchmarkId::new("aligned/update_padded", size),
            &size,
            |b, &size| {
                let data = vec![0; size];
                b.iter(|| {
                    black_box(black_box(&mut m).update_padded(black_box(&data)));
                });
            },
        );
    }

    for size in sizes {
        let size = size - 1;
        g.throughput(Throughput::Bytes(size as u64));
        g.bench_with_input(
            BenchmarkId::new("unaligned/update_padded", size),
            &size,
            |b, &size| {
                let data = vec![0; size];
                b.iter(|| {
                    black_box(black_box(&mut m).update_padded(black_box(&data)));
                });
            },
        );
    }

    g.finish();
}

fn benchmarks(c: &mut Criterion) {
    benchmark::<Lite>(c, "lite");
    benchmark::<Precomputed>(c, "precomputed");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
