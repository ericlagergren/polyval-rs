//! The generic (software) implementation.
//!
//! It's used by the other backends (`soft`, `aarch64`, etc.) if
//! hardware support is not detected.

use core::{
    array,
    ops::{BitXor, BitXorAssign, Mul, MulAssign, Shl, Shr},
};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

use crate::{BLOCK_SIZE, KEY_SIZE};

#[derive(Copy, Clone, Debug)]
#[allow(dead_code, reason = "Depends on the backend")]
pub(super) struct Token;

#[allow(dead_code, reason = "Depends on the backend")]
impl Token {
    #[inline]
    pub fn new() -> (Self, bool) {
        (Self, true)
    }

    #[inline]
    pub fn supported(&self) -> bool {
        true
    }
}

pub(super) type Big<const GHASH: bool> = Backend<GHASH, 8>;
pub(super) type Small<const GHASH: bool> = Backend<GHASH, 1>;

#[derive(Clone, Debug)]
pub struct Backend<const GHASH: bool, const N: usize> {
    /// The running state.
    y: FieldElement,
    /// The key, possibly precomputed for batched computations.
    h: [FieldElement; N],
}

impl<const GHASH: bool, const N: usize> Backend<GHASH, N> {
    #[cfg_attr(feature = "soft", inline)]
    #[cfg_attr(not(feature = "soft"), cold)]
    #[allow(clippy::arithmetic_side_effects, reason = "It's all in GF(2¹²⁸)")]
    pub fn new(key: &[u8; KEY_SIZE]) -> Self {
        let h = if GHASH {
            FieldElement::from_be_bytes(key).mulx()
        } else {
            FieldElement::from_le_bytes(key)
        };
        let h = {
            let mut prev = h;
            let mut pow: [FieldElement; N] = array::from_fn(|_| FieldElement(0));
            for (i, v) in pow.iter_mut().rev().enumerate() {
                *v = h;
                if i > 0 {
                    *v *= prev;
                }
                prev = *v;
            }
            pow
        };
        Self {
            y: FieldElement(0),
            h,
        }
    }

    #[cfg_attr(feature = "soft", inline)]
    #[cfg_attr(not(feature = "soft"), cold)]
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        reason = "N - 1 is constant and N > 0"
    )]
    pub fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        let x = if GHASH {
            FieldElement::from_be_bytes(block)
        } else {
            FieldElement::from_le_bytes(block)
        };
        self.y = (self.y ^ x) * self.h[N - 1];
    }

    #[cfg_attr(feature = "soft", inline)]
    #[cfg_attr(not(feature = "soft"), cold)]
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "N - 1 is constant and N > 0"
    )]
    pub fn update_blocks(&mut self, mut blocks: &[[u8; BLOCK_SIZE]]) {
        if N > 1 {
            let (head, tail) = super::as_chunks::<_, N>(blocks);

            for chunk in head {
                let mut h = FieldElement(0);
                let mut m = FieldElement(0);
                let mut l = FieldElement(0);

                for (i, (block, y)) in chunk.iter().rev().zip(self.h.iter().rev()).enumerate() {
                    let mut x = if GHASH {
                        FieldElement::from_be_bytes(block)
                    } else {
                        FieldElement::from_le_bytes(block)
                    };
                    if i == N - 1 {
                        // Fold in the accumulator.
                        x ^= self.y;
                    };
                    let (x0, x1) = y.unpack();
                    let (y0, y1) = x.unpack();
                    h ^= gf128_mul(x1, y1);
                    l ^= gf128_mul(x0, y0);
                    m ^= gf128_mul(x0 ^ x1, y0 ^ y1);
                }

                let (mut h0, mut h1) = h.unpack();
                let (mut m0, mut m1) = m.unpack();
                let (l0, mut l1) = l.unpack();

                m0 ^= l0 ^ h0;
                m1 ^= l1 ^ h1;

                l1 ^= m0 ^ (l0 << 63) ^ (l0 << 62) ^ (l0 << 57);
                h0 ^= l0 ^ (l0 >> 1) ^ (l0 >> 2) ^ (l0 >> 7);
                h0 ^= m1 ^ (l1 << 63) ^ (l1 << 62) ^ (l1 << 57);
                h1 ^= l1 ^ (l1 >> 1) ^ (l1 >> 2) ^ (l1 >> 7);

                self.y = FieldElement::pack(h0, h1);
            }

            blocks = tail;
        }

        // Handle singles.
        for block in blocks {
            self.update_block(block);
        }
    }

    #[inline]
    pub fn tag(&self) -> [u8; 16] {
        if GHASH {
            self.y.to_be_bytes()
        } else {
            self.y.to_le_bytes()
        }
    }

    #[inline]
    #[cfg(feature = "experimental")]
    pub fn export(&self) -> FieldElement {
        self.y
    }

    #[inline]
    #[cfg(feature = "experimental")]
    pub fn reset(&mut self, y: FieldElement) {
        self.y = y;
    }
}

#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(test, derive(Eq, PartialEq))]
#[repr(transparent)]
pub(crate) struct FieldElement(u128);

impl FieldElement {
    /// Creates a field element from little-endian bytes.
    #[inline]
    pub const fn from_le_bytes(b: &[u8; BLOCK_SIZE]) -> Self {
        Self(u128::from_le_bytes(*b))
    }

    /// Converts the field element to little-endian bytes.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; BLOCK_SIZE] {
        self.0.to_le_bytes()
    }

    /// Creates a field element from big-endian bytes.
    #[inline]
    const fn from_be_bytes(b: &[u8; BLOCK_SIZE]) -> Self {
        Self(u128::from_be_bytes(*b))
    }

    /// Converts the field element to big-endian bytes.
    #[inline]
    const fn to_be_bytes(self) -> [u8; BLOCK_SIZE] {
        self.0.to_be_bytes()
    }

    const fn pack(lo: u64, hi: u64) -> Self {
        Self(((hi as u128) << 64) | (lo as u128))
    }

    const fn unpack(self) -> (u64, u64) {
        let lo = self.0 as u64;
        let hi = (self.0 >> 64) as u64;
        (lo, hi)
    }

    /// Doubles `self` in GF(2¹²⁸).
    #[must_use = "this returns the result of the operation \
                      without modifying the original"]
    const fn mulx(self) -> Self {
        Self(super::mulx(self.0))
    }
}

impl BitXor for FieldElement {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for FieldElement {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Mul for FieldElement {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        polymul(self, rhs)
    }
}

impl MulAssign for FieldElement {
    #[inline(always)]
    #[allow(clippy::arithmetic_side_effects)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Shl<u32> for FieldElement {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0 << rhs)
    }
}

impl Shr<u32> for FieldElement {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0 >> rhs)
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for FieldElement {
    fn zeroize(&mut self) {
        self.0.zeroize();
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

/// Returns the constant time 128-bit product of `x` and `y` in
/// `GF(2¹²⁸)`.
///
/// The idea comes from [Thomas Pornin]'s constant-time blog post
/// with 64-bit fixes from [Tim Taubert]'s blog post on formally
/// verified GHASH.
///
/// [Thomas Pornin]: https://www.bearssl.org/constanttime.html
/// [Tim Taubert]: https://timtaubert.de/blog/2017/06/verified-binary-multiplication-for-ghash/
#[allow(clippy::arithmetic_side_effects)]
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

#[cfg(test)]
mod tests {
    use hex_literal::hex;

    use super::*;

    macro_rules! fe {
        ($s:expr) => {{
            FieldElement::from_le_bytes(&hex!($s))
        }};
    }

    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn test_fe_ops() {
        let a = fe!("66e94bd4ef8a2c3b884cfa59ca342b2e");
        let b = fe!("ff000000000000000000000000000000");

        let want = fe!("99e94bd4ef8a2c3b884cfa59ca342b2e");
        assert_eq!(a ^ b, want);
        assert_eq!(b ^ a, want);

        let want = fe!("ebe563401e7e91ea3ad6426b8140c394");
        assert_eq!(a * b, want);
        assert_eq!(b * a, want);
    }
}
