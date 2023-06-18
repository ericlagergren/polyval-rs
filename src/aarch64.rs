//! The AArch64 implementation.

#![cfg(all(not(feature = "soft"), target_arch = "aarch64"))]

use {
    crate::{
        generic,
        poly::{Polyval, BLOCK_SIZE},
    },
    core::{
        arch::aarch64::*,
        mem,
        ops::{BitXor, BitXorAssign, Mul},
    },
};

const fn have_asm() -> bool {
    cfg!(all(target_feature = "aes", target_feature = "neon"))
}

impl Polyval {
    pub(crate) fn update_blocks(&mut self, blocks: &[u8]) {
        self.y = polymul_series(self.y, &self.pow, blocks)
    }
}

/// An element in the field
///
/// ```text
/// x^128 + x^127 + x^126 + x^121 + 1
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct FieldElement(uint8x16_t);

impl FieldElement {
    pub(crate) fn from_le_bytes(data: &[u8]) -> Self {
        Self(unsafe { vld1q_u8(data.as_ptr()) })
    }

    pub(crate) fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        unsafe { vst1q_u8(out.as_mut_ptr(), self.0) }
        out
    }
}

impl From<FieldElement> for generic::FieldElement {
    fn from(fe: FieldElement) -> Self {
        Self::from_le_bytes(&fe.to_le_bytes())
    }
}

impl From<generic::FieldElement> for FieldElement {
    fn from(fe: generic::FieldElement) -> Self {
        Self::from_le_bytes(&fe.to_le_bytes())
    }
}

#[cfg(feature = "zeroize")]
impl zeroize::Zeroize for FieldElement {
    fn zeroize(&mut self) {
        unsafe { self.0 = veorq_u8(self.0, self.0) }
    }
}

#[cfg(test)]
impl Eq for FieldElement {}

#[cfg(test)]
impl PartialEq for FieldElement {
    fn eq(&self, other: &Self) -> bool {
        let v = unsafe { vreinterpretq_p128_u8(vceqq_u8(self.0, other.0)) };
        v == u128::MAX
    }
}

impl Default for FieldElement {
    fn default() -> Self {
        unsafe { Self(vdupq_n_u8(0)) }
    }
}

impl BitXor for FieldElement {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        Self(unsafe { veorq_u8(self.0, rhs.0) })
    }
}

impl BitXorAssign for FieldElement {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = Self(unsafe { veorq_u8(self.0, rhs.0) })
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        polymul(self, rhs)
    }
}

pub(crate) fn polymul(acc: FieldElement, key: FieldElement) -> FieldElement {
    if have_asm() {
        unsafe { polymul_asm(acc, key) }
    } else {
        generic::polymul(acc.into(), key.into()).into()
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn polymul_asm(x: FieldElement, y: FieldElement) -> FieldElement {
    let (h, m, l) = karatsuba1(x.0, y.0);
    let (h, l) = karatsuba2(h, m, l);
    let d = mont_reduce(h, l);
    FieldElement(d)
}

/// Multiplies `acc` with the series of field elements in
/// `blocks`.
pub(crate) fn polymul_series(
    acc: FieldElement,
    pow: &[FieldElement; 8],
    blocks: &[u8],
) -> FieldElement {
    if have_asm() {
        unsafe { polymul_series_asm(acc, pow, blocks) }
    } else {
        let pow = pow.map(|fe| fe.into());
        generic::polymul_series(acc.into(), &pow, blocks).into()
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn polymul_series_asm(
    mut acc: FieldElement,
    pow: &[FieldElement; 8],
    blocks: &[u8],
) -> FieldElement {
    debug_assert!(blocks.len() % BLOCK_SIZE == 0);

    let mut blocks = blocks.chunks_exact(BLOCK_SIZE * pow.len());
    if blocks.len() > 0 {
        let (lhs, rhs) = pow.split_at(pow.len() / 2);
        let uint8x16x4_t(h0, h1, h2, h3) =
            vld1q_u8_x4(lhs.as_ptr() as *const u8);
        let uint8x16x4_t(h4, h5, h6, h7) =
            vld1q_u8_x4(rhs.as_ptr() as *const u8);

        for chunk in blocks.by_ref() {
            let (lhs, rhs) = chunk.split_at(chunk.len() / 2);
            let uint8x16x4_t(m0, m1, m2, m3) = vld1q_u8_x4(lhs.as_ptr());
            let uint8x16x4_t(m4, m5, m6, m7) = vld1q_u8_x4(rhs.as_ptr());

            let mut h = vdupq_n_u8(0);
            let mut m = vdupq_n_u8(0);
            let mut l = vdupq_n_u8(0);

            macro_rules! karatsuba_xor {
                ($m:expr, $h:expr) => {
                    let (hh, mm, ll) = karatsuba1($m, $h);
                    h = veorq_u8(h, hh);
                    m = veorq_u8(m, mm);
                    l = veorq_u8(l, ll);
                };
            }
            karatsuba_xor!(m7, h7);
            karatsuba_xor!(m6, h6);
            karatsuba_xor!(m5, h5);
            karatsuba_xor!(m4, h4);
            karatsuba_xor!(m3, h3);
            karatsuba_xor!(m2, h2);
            karatsuba_xor!(m1, h1);
            let m0 = veorq_u8(m0, acc.0); // fold in accumulator
            karatsuba_xor!(m0, h0);

            let (h, l) = karatsuba2(h, m, l);
            acc = FieldElement(mont_reduce(h, l));
        }
    }

    // Handle singles.
    for block in blocks.remainder().chunks_exact(BLOCK_SIZE) {
        let y = FieldElement::from_le_bytes(block);
        acc = (acc ^ y) * pow[7];
    }

    acc
}

/// Karatsuba decomposition for `x*y`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn karatsuba1(
    x: uint8x16_t,
    y: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = pmull(
        veorq_u8(x, vextq_u8(x, x, 8)), // x.hi^x.lo
        veorq_u8(y, vextq_u8(y, y, 8)), // y.hi^y.lo
    );
    let h = pmull2(x, y); // h = x.hi * y.hi
    let l = pmull(x, y); // l = x.lo * y.lo
    (h, m, l)
}

/// Karatsuba combine.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn karatsuba2(
    h: uint8x16_t,
    m: uint8x16_t,
    l: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    // Second Karatsuba step: combine into a 2n-bit product.
    //
    // m0 ^= l0 ^ h0 // = m0^(l0^h0)
    // m1 ^= l1 ^ h1 // = m1^(l1^h1)
    // l1 ^= m0      // = l1^(m0^l0^h0)
    // h0 ^= l0 ^ m1 // = h0^(l0^m1^l1^h1)
    // h1 ^= l1      // = h1^(l1^m0^l0^h0)
    let t = {
        //   {m0, m1} ^ {l1, h0}
        // = {m0^l1, m1^h0}
        let t0 = veorq_u8(m, vextq_u8(l, h, 8));

        //   {h0, h1} ^ {l0, l1}
        // = {h0^l0, h1^l1}
        let t1 = veorq_u8(h, l);

        //   {m0^l1, m1^h0} ^ {h0^l0, h1^l1}
        // = {m0^l1^h0^l0, m1^h0^h1^l1}
        veorq_u8(t0, t1)
    };

    // {m0^l1^h0^l0, l0}
    let x01 = vextq_u8(
        vextq_u8(l, l, 8), // {l1, l0}
        t,
        8,
    );

    // {h1, m1^h0^h1^l1}
    let x23 = vextq_u8(
        t,
        vextq_u8(h, h, 8), // {h1, h0}
        8,
    );

    (x23, x01)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn mont_reduce(x23: uint8x16_t, x01: uint8x16_t) -> uint8x16_t {
    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    let poly = vreinterpretq_u8_p128(
        1 << 127 | 1 << 126 | 1 << 121 | 1 << 63 | 1 << 62 | 1 << 57,
    );
    let a = pmull(x01, poly);
    let b = veorq_u8(x01, vextq_u8(a, a, 8));
    let c = pmull2(b, poly);
    veorq_u8(x23, veorq_u8(c, b))
}

/// Multiplies the low bits in `a` and `b`.
#[inline(always)]
unsafe fn pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    mem::transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 0),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 0),
    ))
}

/// Multiplies the high bits in `a` and `b`.
#[inline(always)]
unsafe fn pmull2(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    mem::transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 1),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 1),
    ))
}

#[cfg(test)]
pub(crate) fn gf128_mul(x: u64, y: u64) -> FieldElement {
    let z = unsafe { vreinterpretq_u8_p128(vmull_p64(x, y)) };
    FieldElement(z)
}
