use core::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polyhash::*;

fn benchmark(c: &mut Criterion) {
    let key = Key::new_unchecked([1u8; 16]);
    let mut m = Polyval::new(&key);

    let sizes = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

    let mut g = c.benchmark_group("aligned");
    for size in sizes {
        g.throughput(Throughput::Bytes(size as u64));
        g.bench_with_input(
            BenchmarkId::new("update_padded", size),
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

    let mut g = c.benchmark_group("unaligned");
    for size in sizes {
        let size = size - 1;
        g.throughput(Throughput::Bytes(size as u64));
        g.bench_with_input(
            BenchmarkId::new("update_padded", size),
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

criterion_group!(benches, benchmark);
criterion_main!(benches);
