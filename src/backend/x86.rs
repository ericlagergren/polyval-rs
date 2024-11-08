//! The x86 implementation.

#![cfg(all(
    not(feature = "soft"),
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
))]

use core::{
    ops::{BitXor, BitXorAssign, Mul, MulAssign},
    ptr,
};

use cfg_if::cfg_if;

use super::generic;
use crate::poly::BLOCK_SIZE;

cfg_if! {
    if #[cfg(target_arch = "x86")] {
        use core::arch::x86::*;
    } else {
        use core::arch::x86_64::*;
    }
}

// NB: `pclmulqdq` implies `sse2`.
cpufeatures::new!(have_pclmulqdq, "pclmulqdq");

pub(crate) fn have_pclmulqdq() -> bool {
    // For some reason we can't detect pclmulqdq on an M1.
    // However, Go has no problem doing this.
    if cfg!(target_os = "macos") {
        true
    } else {
        have_pclmulqdq::get()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct FieldElement(__m128i);

impl FieldElement {
    pub fn from_le_bytes(data: &[u8; 16]) -> Self {
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        let fe = unsafe { _mm_loadu_si128(data.as_ptr().cast()) };
        Self(fe)
    }

    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), self.0) }
        out
    }

    /// Multiplies `acc` with the series of field elements in
    /// `blocks`.
    #[must_use = "this returns the result of the operation \
                      without modifying the original"]
    pub fn mul_series(self, pow: &[Self; 8], blocks: &[u8]) -> Self {
        if have_pclmulqdq() {
            // SAFETY: `__m128i` and `FieldElement` have the same
            // layout in memory.
            let pow = unsafe { &*(pow as *const [FieldElement; 8]).cast() };
            // SAFETY: `polymul_series_asm` requires the `sse2`
            // and `pclmulqdq` target features, which we have.
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
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        let fe = unsafe { _mm_setzero_si128() };
        Self(fe)
    }
}

impl BitXor for FieldElement {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        let fe = unsafe { _mm_xor_si128(self.0, rhs.0) };
        Self(fe)
    }
}
impl BitXorAssign for FieldElement {
    fn bitxor_assign(&mut self, rhs: Self) {
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        self.0 = unsafe { _mm_xor_si128(self.0, rhs.0) };
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if have_pclmulqdq() {
            // SAFETY: `polymul_asm` requires the `sse2` and
            // `pclmulqdq` target features, which we have.
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
        unsafe { self.0 = _mm_xor_si128(self.0, self.0) }
    }
}

#[cfg(test)]
impl Eq for FieldElement {}

#[cfg(test)]
impl PartialEq for FieldElement {
    fn eq(&self, other: &Self) -> bool {
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        let v = unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, other.0)) };
        v == 0xffff
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
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn polymul_asm(x: __m128i, y: __m128i) -> __m128i {
    let (h, m, l) = karatsuba1(x, y);
    let (h, l) = karatsuba2(h, m, l);
    mont_reduce(h, l) // d
}

/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
pub(crate) unsafe fn polymul_series_asm(
    mut acc: __m128i,
    pow: &[__m128i; 8],
    blocks: &[u8],
) -> __m128i {
    debug_assert!(have_pclmulqdq());
    debug_assert!(blocks.len() % BLOCK_SIZE == 0);

    let mut blocks = blocks.chunks_exact(BLOCK_SIZE * pow.len());
    for chunk in blocks.by_ref() {
        let mut h = _mm_setzero_si128();
        let mut m = _mm_setzero_si128();
        let mut l = _mm_setzero_si128();

        macro_rules! karatsuba_xor {
            ($i:expr) => {
                let mut y = _mm_loadu_si128((&chunk[$i * BLOCK_SIZE..]).as_ptr() as *const __m128i);
                if $i == 0 {
                    y = _mm_xor_si128(y, acc); // fold in accumulator
                }
                let x = _mm_loadu_si128(ptr::addr_of!(pow[$i]));
                let (hh, mm, ll) = karatsuba1(x, y);
                h = _mm_xor_si128(h, hh);
                m = _mm_xor_si128(m, mm);
                l = _mm_xor_si128(l, ll);
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

        let (h, l) = karatsuba2(h, m, l);
        acc = mont_reduce(h, l);
    }

    // Handle singles.
    for block in blocks.remainder().chunks_exact(BLOCK_SIZE) {
        let y = _mm_loadu_si128(block.as_ptr().cast());
        // acc = (acc ^ y) * pow[7];
        acc = _mm_xor_si128(acc, y);
        acc = polymul_asm(acc, pow[7]);
    }

    acc
}

/// Karatsuba decomposition for `x*y`.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn karatsuba1(x: __m128i, y: __m128i) -> (__m128i, __m128i, __m128i) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = pmull(
        _mm_xor_si128(x, _mm_shuffle_epi32(x, 0xee)),
        _mm_xor_si128(y, _mm_shuffle_epi32(y, 0xee)),
    );
    let h = pmull2(y, x); // h = x.hi * y.hi
    let l = pmull(y, x); // l = x.lo * y.lo
    (h, m, l)
}

/// Karatsuba combine.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn karatsuba2(h: __m128i, m: __m128i, l: __m128i) -> (__m128i, __m128i) {
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
        let t0 = _mm_xor_si128(
            m,
            _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(l),
                _mm_castsi128_ps(h),
                0x4e,
            )),
        );

        //   {h0, h1} ^ {l0, l1}
        // = {h0^l0, h1^l1}
        let t1 = _mm_xor_si128(h, l);

        //   {m0^l1, m1^h0} ^ {h0^l0, h1^l1}
        // = {m0^l1^h0^l0, m1^h0^h1^l1}
        _mm_xor_si128(t0, t1)
    };

    // {m0^l1^h0^l0, l0}
    let x01 = _mm_unpacklo_epi64(l, t);

    // {h1, m1^h0^h1^l1}
    let x23 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(h), _mm_castsi128_ps(t)));

    (x23, x01)
}

/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn mont_reduce(x23: __m128i, x01: __m128i) -> __m128i {
    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    static POLY: u128 = 1 << 127 | 1 << 126 | 1 << 121 | 1 << 63 | 1 << 62 | 1 << 57;
    let poly = _mm_loadu_si128(ptr::addr_of!(POLY) as *const __m128i);
    let a = pmull(x01, poly);
    let b = _mm_xor_si128(x01, _mm_shuffle_epi32(a, 0x4e));
    let c = pmull2(b, poly);
    _mm_xor_si128(x23, _mm_xor_si128(c, b))
}

/// Multiplies the low bits in `a` and `b`.
///
/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn pmull(a: __m128i, b: __m128i) -> __m128i {
    _mm_clmulepi64_si128(a, b, 0x00)
}

/// Multiplies the high bits in `a` and `b`.
///
/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn pmull2(a: __m128i, b: __m128i) -> __m128i {
    _mm_clmulepi64_si128(a, b, 0x11)
}
