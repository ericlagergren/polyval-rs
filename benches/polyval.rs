#![feature(test)]

extern crate test;

use {polyhash::*, test::Bencher};

macro_rules! bench {
    ($name:ident, $bs:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let key = Key::new(&[1u8; 16]).expect("impossible");
            let mut m = Polyval::new(&key);
            let data = [0; $bs];

            b.iter(|| {
                m.update_padded(&data);
            });

            b.bytes = $bs;
        }
    };
}

bench!(bench1_16, 16);
bench!(bench2_64, 64);
bench!(bench3_128, 128);
bench!(bench4_256, 256);
bench!(bench5_512, 512);
bench!(bench6_1024, 1024);
bench!(bench7_2048, 2048);
bench!(bench8_4096, 4096);
bench!(bench9_8192, 8192);
