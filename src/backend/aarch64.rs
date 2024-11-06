//! The AArch64 implementation.

#![cfg(all(
    not(feature = "soft"),
    target_arch = "aarch64",
    target_feature = "neon",
))]

use core::{
    arch::aarch64::{
        uint8x16_t, uint8x16x4_t, vdupq_n_u8, veorq_u8, vextq_u8, vgetq_lane_u64, vld1q_u8,
        vld1q_u8_x4, vmull_p64, vreinterpretq_u64_u8, vreinterpretq_u8_p128, vst1q_u8,
    },
    mem,
    ops::{BitXor, BitXorAssign, Mul, MulAssign},
};

use super::generic;
use crate::poly::BLOCK_SIZE;

cpufeatures::new!(have_aes, "aes");

fn have_aes() -> bool {
    have_aes::get()
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct FieldElement(uint8x16_t);

impl FieldElement {
    pub fn from_le_bytes(data: &[u8; 16]) -> Self {
        // SAFETY: This intrinsic requires the `neon` target feature,
        // which we have.
        let fe = unsafe { vld1q_u8(data.as_ptr()) };
        Self(fe)
    }

    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        unsafe { vst1q_u8(out.as_mut_ptr(), self.0) }
        out
    }

    /// Multiplies `acc` with the series of field elements in
    /// `blocks`.
    #[must_use = "this returns the result of the operation \
                      without modifying the original"]
    pub fn mul_series(self, pow: &[Self; 8], blocks: &[u8]) -> Self {
        if have_aes() {
            // SAFETY: `uint8x16_t` and `FieldElement` have the same
            // layout in memory.
            let pow = unsafe { mem::transmute::<&[FieldElement; 8], &[uint8x16_t; 8]>(pow) };
            // SAFETY: `polymul_series_asm` requires the `neon` and
            // `aes` target features, which we have.
            let fe = unsafe { polymul_series_asm(self.0, pow, blocks) };
            FieldElement(fe)
        } else {
            let pow = pow.map(Into::into);
            generic::FieldElement::from(self)
                .mul_series(&pow, blocks)
                .into()
        }
    }
}

impl Default for FieldElement {
    fn default() -> Self {
        // SAFETY: This intrinsic requires the `neon` target feature,
        // which we have.
        let fe = unsafe { vdupq_n_u8(0) };
        Self(fe)
    }
}

impl BitXor for FieldElement {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        let fe = unsafe { veorq_u8(self.0, rhs.0) };
        Self(fe)
    }
}
impl BitXorAssign for FieldElement {
    fn bitxor_assign(&mut self, rhs: Self) {
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        self.0 = unsafe { veorq_u8(self.0, rhs.0) };
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if have_aes() {
            // SAFETY: `polymul_asm` requires the `neon` and
            // `aes` target features, which we have.
            let fe = unsafe { polymul_asm(self.0, rhs.0) };
            Self(fe)
        } else {
            let fe = generic::FieldElement::from(self) * generic::FieldElement::from(rhs);
            fe.into()
        }
    }
}
impl MulAssign for FieldElement {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(feature = "zeroize")]
impl zeroize::Zeroize for FieldElement {
    fn zeroize(&mut self) {
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        self.0 = unsafe { veorq_u8(self.0, self.0) };
    }
}

#[cfg(test)]
impl Eq for FieldElement {}

#[cfg(test)]
impl PartialEq for FieldElement {
    fn eq(&self, other: &Self) -> bool {
        use core::arch::aarch64::vceqq_u8;

        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        let v = unsafe { vceqq_u8(self.0, other.0) };

        // SAFETY: `uint8x16_t` has the same size as `u128`.
        let v = unsafe { mem::transmute::<uint8x16_t, u128>(v) };

        v == u128::MAX
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

/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn polymul_asm(x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
    debug_assert!(have_aes());

    let (h, m, l) = karatsuba1(x, y);
    let (h, l) = karatsuba2(h, m, l);
    mont_reduce(h, l)
}

/// Multiplies `acc` with the series of field elements in
/// `blocks`.
///
/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[target_feature(enable = "neon,aes")]
unsafe fn polymul_series_asm(
    mut acc: uint8x16_t,
    pow: &[uint8x16_t; 8],
    blocks: &[u8],
) -> uint8x16_t {
    debug_assert!(have_aes());
    debug_assert!(blocks.len() % BLOCK_SIZE == 0);

    let mut blocks = blocks.chunks_exact(BLOCK_SIZE * pow.len());
    if blocks.len() > 0 {
        let (lhs, rhs) = pow.split_at(pow.len() / 2);
        let uint8x16x4_t(h0, h1, h2, h3) = vld1q_u8_x4(lhs.as_ptr().cast::<u8>());
        let uint8x16x4_t(h4, h5, h6, h7) = vld1q_u8_x4(rhs.as_ptr().cast::<u8>());

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
            let m0 = veorq_u8(m0, acc); // fold in accumulator
            karatsuba_xor!(m0, h0);

            let (h, l) = karatsuba2(h, m, l);
            acc = mont_reduce(h, l);
        }
    }

    // Handle singles.
    for block in blocks.remainder().chunks_exact(BLOCK_SIZE) {
        let y = vld1q_u8(block.as_ptr());
        // acc = (acc ^ y) * pow[7];
        acc = veorq_u8(acc, y);
        acc = polymul_asm(acc, pow[7]);
    }

    acc
}

/// Karatsuba decomposition for `x*y`.
///
/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn karatsuba1(x: uint8x16_t, y: uint8x16_t) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    debug_assert!(have_aes());

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
///
/// # Safety
///
/// The NEON architectural feature must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn karatsuba2(h: uint8x16_t, m: uint8x16_t, l: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    debug_assert!(have_aes());

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

/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn mont_reduce(x23: uint8x16_t, x01: uint8x16_t) -> uint8x16_t {
    debug_assert!(have_aes());

    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    let poly = vreinterpretq_u8_p128(1 << 127 | 1 << 126 | 1 << 121 | 1 << 63 | 1 << 62 | 1 << 57);
    let a = pmull(x01, poly);
    let b = veorq_u8(x01, vextq_u8(a, a, 8));
    let c = pmull2(b, poly);
    veorq_u8(x23, veorq_u8(c, b))
}

/// Multiplies the low bits in `a` and `b`.
///
/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    debug_assert!(have_aes());

    let p = vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 0),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 0),
    );
    vreinterpretq_u8_p128(p)
}

/// Multiplies the high bits in `a` and `b`.
///
/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn pmull2(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    debug_assert!(have_aes());

    let p = vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 1),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 1),
    );
    vreinterpretq_u8_p128(p)
}
