//! The generic (software) implementation.
//!
//! It's used by the other backends (`soft`, `aarch64`, etc.) if
//! hardware support is not detected.

#![allow(clippy::expect_used)]
#![allow(clippy::indexing_slicing)]

use core::ops::{BitXor, BitXorAssign, Mul, MulAssign, Shl, Shr};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

use crate::poly::BLOCK_SIZE;

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "zeroize", derive(Zeroize))]
#[cfg_attr(test, derive(Eq, PartialEq))]
#[repr(transparent)]
pub(crate) struct FieldElement(u128);

impl FieldElement {
    pub const fn from_le_bytes(b: &[u8; 16]) -> Self {
        Self(u128::from_le_bytes(*b))
    }

    pub const fn to_le_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    const fn pack(lo: u64, hi: u64) -> Self {
        Self(((hi as u128) << 64) | (lo as u128))
    }

    const fn unpack(self) -> (u64, u64) {
        let lo = self.0 as u64;
        let hi = (self.0 >> 64) as u64;
        (lo, hi)
    }

    /// Multiplies `acc` with the series of field elements in
    /// `blocks`.
    #[must_use = "this returns the result of the operation \
                      without modifying the original"]
    pub fn mul_series(self, pow: &[Self; 8], blocks: &[u8]) -> Self {
        polymul_series(self, pow, blocks)
    }
}

impl BitXor for FieldElement {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}
impl BitXorAssign for FieldElement {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        polymul(self, rhs)
    }
}
impl MulAssign for FieldElement {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Shl<u32> for FieldElement {
    type Output = Self;

    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0 << rhs)
    }
}

impl Shr<u32> for FieldElement {
    type Output = Self;

    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0 >> rhs)
    }
}

pub(super) const fn polymul(x: FieldElement, y: FieldElement) -> FieldElement {
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
    let (x0, x1) = x.unpack();
    let (y0, y1) = y.unpack();

    let (mut h0, mut h1) = gf128_mul(x1, y1).unpack(); // H
    let (mut m0, mut m1) = gf128_mul(x1 ^ x0, y1 ^ y0).unpack(); // M
    let (l0, mut l1) = gf128_mul(x0, y0).unpack(); // L

    m0 ^= l0 ^ h0;
    m1 ^= l1 ^ h1;

    l1 ^= m0 ^ (l0 << 63) ^ (l0 << 62) ^ (l0 << 57);
    h0 ^= l0 ^ (l0 >> 1) ^ (l0 >> 2) ^ (l0 >> 7);
    h0 ^= m1 ^ (l1 << 63) ^ (l1 << 62) ^ (l1 << 57);
    h1 ^= l1 ^ (l1 >> 1) ^ (l1 >> 2) ^ (l1 >> 7);

    FieldElement::pack(h0, h1)
}

/// Multiplies `acc` with the series of field elements in
/// `blocks`.
fn polymul_series(
    mut acc: FieldElement,
    pow: &[FieldElement; 8],
    mut blocks: &[u8],
) -> FieldElement {
    debug_assert!(blocks.len() % BLOCK_SIZE == 0);

    // Handle wide chunks.
    while let Some((chunk, rest)) = blocks.split_first_chunk::<{ BLOCK_SIZE * 8 }>() {
        let mut h = FieldElement(0);
        let mut m = FieldElement(0);
        let mut l = FieldElement(0);

        macro_rules! karatsuba_xor {
            ($i:literal) => {
                let block = &chunk[$i * BLOCK_SIZE..($i * BLOCK_SIZE) + 16]
                    .try_into()
                    .expect("should be exactly 16 bytes");
                let mut y = FieldElement::from_le_bytes(*block);
                if $i == 0 {
                    y ^= acc
                };
                let (y0, y1) = y.unpack();
                let (x0, x1) = pow[$i].unpack();
                h ^= gf128_mul(x1, y1);
                l ^= gf128_mul(x0, y0);
                m ^= gf128_mul(x0 ^ x1, y0 ^ y1);
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

        let (mut h0, mut h1) = h.unpack();
        let (mut m0, mut m1) = m.unpack();
        let (l0, mut l1) = l.unpack();

        m0 ^= l0 ^ h0;
        m1 ^= l1 ^ h1;

        l1 ^= m0 ^ (l0 << 63) ^ (l0 << 62) ^ (l0 << 57);
        h0 ^= l0 ^ (l0 >> 1) ^ (l0 >> 2) ^ (l0 >> 7);
        h0 ^= m1 ^ (l1 << 63) ^ (l1 << 62) ^ (l1 << 57);
        h1 ^= l1 ^ (l1 >> 1) ^ (l1 >> 2) ^ (l1 >> 7);

        acc = FieldElement::pack(h0, h1);
        blocks = rest;
    }

    // Handle singles.
    while let Some((block, rest)) = blocks.split_first_chunk() {
        let y = FieldElement::from_le_bytes(block);
        acc = (acc ^ y) * pow[7];
        blocks = rest;
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

    let fe = (t0 & MASK0) | (t1 & MASK1) | (t2 & MASK2) | (t3 & MASK3) | (t4 & MASK4);
    FieldElement(fe)
}
