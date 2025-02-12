//! Benchmarks.

use core::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polyhash::{
    ghash::{GHash, GHashLite},
    Polyval, PolyvalLite, KEY_SIZE,
};

trait Hash {
    fn new(key: &[u8; 16]) -> Self;
    fn update_padded(&mut self, data: &[u8]);
}
macro_rules! impl_hash {
    ($($name:ident),+ $(,)?) => {
        $(
            impl Hash for $name {
                fn new(key: &[u8; 16]) -> Self {
                    $name::new_unchecked(key)
                }
                fn update_padded(&mut self, data: &[u8]) {
                    self.update_padded(data);
                }
            }
        )+
    };
}
impl_hash!(Polyval, PolyvalLite, GHash, GHashLite);

fn benchmark<H: Hash>(c: &mut Criterion, name: &str) {
    let mut m = <H>::new(&[0; KEY_SIZE]);

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
    benchmark::<Polyval>(c, "polyval/default");
    benchmark::<PolyvalLite>(c, "polyval/lite");
    benchmark::<GHash>(c, "ghash/default");
    benchmark::<GHashLite>(c, "ghash/lite");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
