//! The x86 implementation.

#![cfg(all(
    not(feature = "soft"),
    any(target_arch = "x86", target_arch = "x86_64")
))]

use core::{
    ops::{BitXor, BitXorAssign, Mul},
    ptr,
};

use cfg_if::cfg_if;

use crate::{generic, poly::BLOCK_SIZE};

cfg_if! {
    if #[cfg(target_arch = "x86")] {
        use core::arch::x86::*;
    } else {
        use core::arch::x86_64::*;
    }
}

cpufeatures::new!(have_pclmulqdq, "pclmulqdq");

pub(crate) fn enabled() -> bool {
    // For some reason we can't detect pclmulqdq on an M1.
    // However, Go has no problem doing this.
    if cfg!(target_os = "macos") {
        true
    } else {
        have_pclmulqdq::get()
    }
}

/// An element in the field
///
/// ```text
/// x^128 + x^127 + x^126 + x^121 + 1
/// ```
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct FieldElement(__m128i);

impl FieldElement {
    pub(crate) fn from_le_bytes(data: &[u8]) -> Self {
        Self(unsafe { _mm_loadu_si128(data.as_ptr() as *const __m128i) })
    }

    pub(crate) fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, self.0) }
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
        unsafe { self.0 = _mm_xor_si128(self.0, self.0) }
    }
}

#[cfg(test)]
impl Eq for FieldElement {}

#[cfg(test)]
impl PartialEq for FieldElement {
    fn eq(&self, other: &Self) -> bool {
        let v = unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, other.0)) };
        v == 0xffff
    }
}

impl Default for FieldElement {
    fn default() -> Self {
        unsafe { Self(_mm_setzero_si128()) }
    }
}

impl BitXor for FieldElement {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
    }
}

impl BitXorAssign for FieldElement {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
    }
}

impl Mul for FieldElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // SAFETY: TODO
        unsafe { polymul(self, rhs) }
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
#[target_feature(enable = "pclmulqdq")]
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

#[target_feature(enable = "pclmulqdq")]
pub(crate) unsafe fn polymul_series_asm(
    mut acc: FieldElement,
    pow: &[FieldElement; 8],
    blocks: &[u8],
) -> FieldElement {
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
                    y = _mm_xor_si128(y, acc.0); // fold in accumulator
                }
                let x = _mm_loadu_si128(ptr::addr_of!(pow[$i].0));
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
        acc = FieldElement(mont_reduce(h, l));
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
#[target_feature(enable = "pclmulqdq")]
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
#[target_feature(enable = "pclmulqdq")]
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

#[inline]
#[target_feature(enable = "pclmulqdq")]
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

/*
#[target_feature(enable = "pclmulqdq")]
unsafe fn mul(x: __m128i, y: __m128i) -> __m128i {
    let y = _mm_xor_si128(y, x);

    let x0 = x;
    let x1 = _mm_shuffle_epi32(x, 0x0E);
    let x2 = _mm_xor_si128(x0, x1);
    let y0 = y;

    // Multiply values partitioned to 64-bit parts
    let y1 = _mm_shuffle_epi32(y, 0x0E);
    let y2 = _mm_xor_si128(y0, y1);
    let t0 = _mm_clmulepi64_si128(y0, x0, 0x00);
    let t1 = _mm_clmulepi64_si128(y, x, 0x11);
    let t2 = _mm_clmulepi64_si128(y2, x2, 0x00);
    let t2 = _mm_xor_si128(t2, _mm_xor_si128(t0, t1));
    let v0 = t0;
    let v1 = _mm_xor_si128(_mm_shuffle_epi32(t0, 0x0E), t2);
    let v2 = _mm_xor_si128(t1, _mm_shuffle_epi32(t2, 0x0E));
    let v3 = _mm_shuffle_epi32(t1, 0x0E);

    // Polynomial reduction
    let v2 = xor5(
        v2,
        v0,
        _mm_srli_epi64(v0, 1),
        _mm_srli_epi64(v0, 2),
        _mm_srli_epi64(v0, 7),
    );

    let v1 = xor4(
        v1,
        _mm_slli_epi64(v0, 63),
        _mm_slli_epi64(v0, 62),
        _mm_slli_epi64(v0, 57),
    );

    let v3 = xor5(
        v3,
        v1,
        _mm_srli_epi64(v1, 1),
        _mm_srli_epi64(v1, 2),
        _mm_srli_epi64(v1, 7),
    );

    let v2 = xor4(
        v2,
        _mm_slli_epi64(v1, 63),
        _mm_slli_epi64(v1, 62),
        _mm_slli_epi64(v1, 57),
    );

    _mm_unpacklo_epi64(v2, v3)
}
*/

/// Multiplies the low bits in `a` and `b`.
#[inline(always)]
unsafe fn pmull(a: __m128i, b: __m128i) -> __m128i {
    _mm_clmulepi64_si128(a, b, 0x00)
}

/// Multiplies the high bits in `a` and `b`.
#[inline(always)]
unsafe fn pmull2(a: __m128i, b: __m128i) -> __m128i {
    _mm_clmulepi64_si128(a, b, 0x11)
}

#[cfg(test)]
pub(crate) unsafe fn gf128_mul(x: u64, y: u64) -> FieldElement {
    generic::gf128_mul(x, y).into()
}
