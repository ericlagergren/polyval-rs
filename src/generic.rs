//! The generic (software) implementation.
//!
//! It's used by the other backends (`soft`, `aarch64`, etc.) if
//! hardware support is not detected.

#![forbid(unsafe_code)]

use {
    crate::poly::BLOCK_SIZE,
    core::ops::{BitXor, BitXorAssign, Mul},
};

/// An element in the field
///
/// ```text
/// x^128 + x^127 + x^126 + x^121 + 1
/// ```
#[derive(Copy, Clone, Default, Debug)]
#[cfg_attr(test, derive(Eq, PartialEq))]
#[cfg_attr(feature = "zeroize", derive(zeroize::Zeroize))]
pub(crate) struct FieldElement(u64, u64);

impl FieldElement {
    pub(crate) fn from_le_bytes(b: &[u8]) -> Self {
        debug_assert!(b.len() >= 16);

        Self(
            u64::from_le_bytes(b[0..8].try_into().expect("should be 16 bytes")),
            u64::from_le_bytes(
                b[8..16].try_into().expect("should be 16 bytes"),
            ),
        )
    }

    pub(crate) fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.0.to_le_bytes());
        out[8..16].copy_from_slice(&self.1.to_le_bytes());
        out
    }
}

impl BitXor for FieldElement {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0, self.1 ^ rhs.1)
    }
}

impl BitXorAssign for FieldElement {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        polymul(self, rhs)
    }
}

pub(crate) fn polymul(x: FieldElement, y: FieldElement) -> FieldElement {
    // We perform schoolbook multiplication of x and y:
    //
    // (x1,x0)*(y1,y0) = (x1*y1) + (x1*y0 + x0*y1) + (x0*y0)
    //                      H         M       M         L
    //
    // The middle result (M) can be simplified with Karatsuba
    // multiplication:
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // This requires one less 64-bit multiplication and reuses
    // the existing results H and L. (H and L are added to M in
    // the montgomery reduction; see x1 and x2.)
    //
    // This gives us a 256-bit product, X.
    //
    // Use the "Shift-XOR reflected reduction" method to reduce
    // it modulo x^128 + x^127 + x^126 + x^121 + 1.
    //
    // This is faster than Gueron's "Fast reduction ..." method
    // without CMUL/PMULL intrinsics.
    //
    // See [gueron] page 17-19.
    //
    // [gueron]: https://crypto.stanford.edu/RealWorldCrypto/slides/gueron.pdf]
    let mut h = gf128_mul(x.1, y.1); // H
    let mut m = gf128_mul(x.1 ^ x.0, y.1 ^ y.0); // M
    let mut l = gf128_mul(x.0, y.0); // L

    m ^= FieldElement(l.0 ^ h.0, l.1 ^ h.1);

    l.1 ^= m.0 ^ (l.0 << 63) ^ (l.0 << 62) ^ (l.0 << 57);
    h.0 ^= l.0 ^ (l.0 >> 1) ^ (l.0 >> 2) ^ (l.0 >> 7);
    h.0 ^= m.1 ^ (l.1 << 63) ^ (l.1 << 62) ^ (l.1 << 57);
    h.1 ^= l.1 ^ (l.1 >> 1) ^ (l.1 >> 2) ^ (l.1 >> 7);

    FieldElement(h.0, h.1)
}

/// Multiplies `acc` with the series of field elements in
/// `blocks`.
pub(crate) fn polymul_series(
    mut acc: FieldElement,
    pow: &[FieldElement; 8],
    blocks: &[u8],
) -> FieldElement {
    debug_assert!(blocks.len() % BLOCK_SIZE == 0);

    // Handle wide chunks.
    let mut blocks = blocks.chunks_exact(BLOCK_SIZE * 8);
    for chunk in blocks.by_ref() {
        let mut h = FieldElement::default();
        let mut m = FieldElement::default();
        let mut l = FieldElement::default();

        macro_rules! karatsuba_xor {
            ($i:expr) => {
                let mut y =
                    FieldElement::from_le_bytes(&chunk[$i * BLOCK_SIZE..]);
                if $i == 0 {
                    y ^= acc
                };
                let x = &pow[$i];
                h ^= gf128_mul(x.1, y.1);
                l ^= gf128_mul(x.0, y.0);
                m ^= gf128_mul(x.1 ^ x.0, y.1 ^ y.0);
            };
        }
        karatsuba_xor!(7);
        karatsuba_xor!(6);
        karatsuba_xor!(5);
        karatsuba_xor!(4);
        karatsuba_xor!(3);
        karatsuba_xor!(2);
        karatsuba_xor!(1);
        karatsuba_xor!(0);

        m ^= FieldElement(l.0 ^ h.0, l.1 ^ h.1);

        l.1 ^= m.0 ^ (l.0 << 63) ^ (l.0 << 62) ^ (l.0 << 57);
        h.0 ^= l.0 ^ (l.0 >> 1) ^ (l.0 >> 2) ^ (l.0 >> 7);
        h.0 ^= m.1 ^ (l.1 << 63) ^ (l.1 << 62) ^ (l.1 << 57);
        h.1 ^= l.1 ^ (l.1 >> 1) ^ (l.1 >> 2) ^ (l.1 >> 7);

        acc = h
    }

    // Handle singles.
    for block in blocks.remainder().chunks_exact(BLOCK_SIZE) {
        let y = FieldElement::from_le_bytes(block);
        acc = (acc ^ y) * pow[7];
    }

    acc
}

/// Returns the constant time 128-bit product of `x` and `y` in
/// GF(2¹²⁸).
///
/// The idea comes from [Thomas Pornin]'s constant-time blog post
/// with 64-bit fixes from [Tim Taubert]'s blog post on formally
/// verified GHASH.
///
/// [Thomas Pornin]: https://www.bearssl.org/constanttime.html
/// [Tim Taubert]: https://timtaubert.de/blog/2017/06/verified-binary-multiplication-for-ghash/
pub(crate) const fn gf128_mul(x: u64, y: u64) -> FieldElement {
    const MASK0: u128 = 0x21084210842108421084210842108421;
    const MASK1: u128 = 0x42108421084210842108421084210842;
    const MASK2: u128 = 0x84210842108421084210842108421084;
    const MASK3: u128 = 0x08421084210842108421084210842108;
    const MASK4: u128 = 0x10842108421084210842108421084210;

    // Split both x and y into 5 words with four-bit holes.
    let x0 = (x as u128) & MASK0;
    let y0 = (y as u128) & MASK0;
    let x1 = (x as u128) & MASK1;
    let y1 = (y as u128) & MASK1;
    let x2 = (x as u128) & MASK2;
    let y2 = (y as u128) & MASK2;
    let x3 = (x as u128) & MASK3;
    let y3 = (y as u128) & MASK3;
    let x4 = (x as u128) & MASK4;
    let y4 = (y as u128) & MASK4;

    let t0 = (x0 * y0) ^ (x1 * y4) ^ (x2 * y3) ^ (x3 * y2) ^ (x4 * y1);
    let t1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y4) ^ (x3 * y3) ^ (x4 * y2);
    let t2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y4) ^ (x4 * y3);
    let t3 = (x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0) ^ (x4 * y4);
    let t4 = (x0 * y4) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1) ^ (x4 * y0);

    let z = (t0 & MASK0)
        | (t1 & MASK1)
        | (t2 & MASK2)
        | (t3 & MASK3)
        | (t4 & MASK4);

    FieldElement(z as u64, (z >> 64) as u64)
}
