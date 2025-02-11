//! x86/x86_64 implementation.

#![cfg(all(
    not(feature = "soft"),
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
))]

use core::{array, ptr};

use cfg_if::cfg_if;
#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

use crate::{BLOCK_SIZE, KEY_SIZE};

cfg_if! {
    if #[cfg(target_arch = "x86")] {
        use core::arch::x86 as imp;
    } else {
        use core::arch::x86_64 as imp;
    }
}
use imp::{
    __m128i, _mm_castps_si128, _mm_castsi128_ps, _mm_clmulepi64_si128, _mm_loadu_si128,
    _mm_movehl_ps, _mm_set_epi8, _mm_setzero_si128, _mm_shuffle_epi32, _mm_shuffle_epi8,
    _mm_shuffle_ps, _mm_storeu_si128, _mm_unpacklo_epi64, _mm_xor_si128,
};

// NB: `pclmulqdq` implies `sse2`.
cpufeatures::new!(have_pclmulqdq, "pclmulqdq");

#[derive(Copy, Clone, Debug)]
pub(super) struct Token {
    token: have_pclmulqdq::InitToken,
}

impl Token {
    #[inline]
    pub fn new() -> (Self, bool) {
        let (token, supported) = have_pclmulqdq::init_get();
        (Self { token }, supported)
    }

    #[inline]
    pub fn supported(&self) -> bool {
        self.token.get()
    }
}

/// Reverse the bytes in `x`.
///
/// # Safety
///
/// The SSE2 target feature must be enabled.
#[inline]
#[target_feature(enable = "sse2")]
#[allow(clippy::undocumented_unsafe_blocks)]
unsafe fn swap_bytes(x: __m128i) -> __m128i {
    if cfg!(target_feature = "sse3") {
        // SAFETY: The `sse3` target feature is enabled.
        unsafe { swap_bytes_sse3(x) }
    } else {
        // Otherwise, just do whatever the compiler thinks is
        // reasonable.
        let mut tmp = [0u8; 16];
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        unsafe { _mm_storeu_si128(tmp.as_mut_ptr().cast(), x) };
        tmp.reverse();
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        unsafe { _mm_loadu_si128(tmp.as_ptr().cast()) }
    }
}

/// # Safety
///
/// The SSE3 target feature must be enabled.
#[inline]
#[target_feature(enable = "sse3")]
unsafe fn swap_bytes_sse3(x: __m128i) -> __m128i {
    // SAFETY: This intrinsic requires the `sse3` target feature,
    // which we have.
    unsafe {
        let mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _mm_shuffle_epi8(x, mask)
    }
}

pub(super) type Big<const GHASH: bool> = Backend<GHASH, 8>;
pub(super) type Small<const GHASH: bool> = Backend<GHASH, 1>;

/// Either POLYVAL or GHASH.
///
/// GHASH is implemented in terms of POLYVAL:
///
/// ```text
/// GHASH(H, X_1, ..., X_n) =
///     ByteReverse(POLYVAL(mulX_POLYVAL(ByteReverse(H)),
///         ByteReverse(X_1), ..., ByteReverse(X_n)))
/// ```
#[derive(Clone, Debug)]
pub(super) struct Backend<const GHASH: bool, const N: usize> {
    /// The running state.
    y: __m128i,
    /// `h[N-1]` is the H, the remaining elements (if any) are
    /// powers of `h[n-1]` for batched computations.
    h: [__m128i; N],
}

impl<const GHASH: bool, const N: usize> Backend<GHASH, N> {
    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "sse2,pclmulqdq")]
    #[allow(clippy::undocumented_unsafe_blocks)]
    pub unsafe fn new(key: &[u8; KEY_SIZE]) -> Self {
        const { assert!(N > 0) }

        let h = if GHASH {
            let key = super::mulx(u128::from_be_bytes(*key)).to_le_bytes();
            unsafe { _mm_loadu_si128(key.as_ptr().cast()) }
        } else {
            unsafe { _mm_loadu_si128(key.as_ptr().cast()) }
        };
        let h = {
            let mut prev = h;
            let mut pow: [__m128i; N] = array::from_fn(|_| unsafe { _mm_setzero_si128() });
            for (i, v) in pow.iter_mut().rev().enumerate() {
                *v = h;
                if i > 0 {
                    *v = unsafe { polymul(*v, prev) };
                }
                prev = *v;
            }
            pow
        };
        Self {
            y: unsafe { _mm_setzero_si128() },
            h,
        }
    }

    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "sse2,pclmulqdq")]
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        reason = "N - 1 is constant and N > 0"
    )]
    pub unsafe fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        const { assert!(N > 0) }

        // SAFETY: These require the `sse2` and `pclmuldqd`
        // target features, which we have.
        unsafe {
            let mut x = _mm_loadu_si128(block.as_ptr().cast());
            if GHASH {
                x = swap_bytes(x);
            }
            self.y = polymul(_mm_xor_si128(self.y, x), self.h[N - 1]);
        }
    }

    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "sse2,pclmulqdq")]
    #[allow(clippy::undocumented_unsafe_blocks)]
    pub unsafe fn update_blocks(&mut self, mut blocks: &[[u8; BLOCK_SIZE]]) {
        const { assert!(N > 0) }

        if self.h.len() == 8 {
            let (head, tail) = super::as_chunks::<_, N>(blocks);

            for chunk in head {
                let mut h = unsafe { _mm_setzero_si128() };
                let mut m = unsafe { _mm_setzero_si128() };
                let mut l = unsafe { _mm_setzero_si128() };

                macro_rules! karatsuba_xor {
                    ($i:expr) => {
                        unsafe {
                            let mut x = _mm_loadu_si128(chunk[$i].as_ptr().cast());
                            if GHASH {
                                x = swap_bytes(x);
                            }
                            if $i == 0 {
                                x = _mm_xor_si128(x, self.y); // fold in accumulator
                            }
                            let y = self.h[$i];
                            let (hh, mm, ll) = karatsuba1(x, y);
                            h = _mm_xor_si128(h, hh);
                            m = _mm_xor_si128(m, mm);
                            l = _mm_xor_si128(l, ll);
                        }
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

                let (h, l) = unsafe { karatsuba2(h, m, l) };
                self.y = unsafe { mont_reduce(h, l) };
            }

            blocks = tail;
        }

        // Handle singles.
        for block in blocks {
            // SAFETY: This requires the `sse2` and `pclmulqdq`
            // target features, which we have.
            unsafe { self.update_block(block) }
        }
    }

    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn tag(&self) -> [u8; 16] {
        let mut tag = [0u8; 16];

        let y = if GHASH && cfg!(target_feature = "sse3") {
            // SAFETY: This intrinsic requires the `sse3` target
            // feature, which we have.
            unsafe { swap_bytes_sse3(self.y) }
        } else {
            self.y
        };

        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        unsafe { _mm_storeu_si128(tag.as_mut_ptr().cast(), y) }

        if GHASH && !cfg!(target_feature = "sse3") {
            tag.reverse()
        }

        tag
    }

    #[inline]
    #[cfg(feature = "experimental")]
    pub fn export(&self) -> FieldElement {
        FieldElement(self.y)
    }

    #[inline]
    #[cfg(feature = "experimental")]
    pub fn reset(&mut self, y: FieldElement) {
        self.y = y.0;
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct FieldElement(__m128i);

impl FieldElement {
    #[inline]
    pub fn from_le_bytes(data: &[u8; BLOCK_SIZE]) -> Self {
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        let fe = unsafe { _mm_loadu_si128(data.as_ptr().cast()) };
        Self(fe)
    }

    #[inline]
    pub fn to_le_bytes(self) -> [u8; BLOCK_SIZE] {
        let mut out = [0u8; BLOCK_SIZE];
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), self.0) }
        out
    }
}

impl Default for FieldElement {
    #[inline]
    fn default() -> Self {
        // SAFETY: This intrinsic requires the `sse2` target
        // feature, which we have.
        let fe = unsafe { _mm_setzero_si128() };
        Self(fe)
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for FieldElement {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

#[cfg(test)]
impl Eq for FieldElement {}

#[cfg(test)]
impl PartialEq for FieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.to_le_bytes() == other.to_le_bytes()
    }
}

/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn polymul(x: __m128i, y: __m128i) -> __m128i {
    let (h, m, l) = unsafe { karatsuba1(x, y) };
    let (h, l) = unsafe { karatsuba2(h, m, l) };
    unsafe {
        mont_reduce(h, l) // d
    }
}

/// Karatsuba decomposition for `x*y`.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn karatsuba1(x: __m128i, y: __m128i) -> (__m128i, __m128i, __m128i) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = unsafe {
        pmull(
            _mm_xor_si128(x, _mm_shuffle_epi32(x, 0xee)),
            _mm_xor_si128(y, _mm_shuffle_epi32(y, 0xee)),
        )
    };
    let h = unsafe { pmull2(y, x) }; // h = x.hi * y.hi
    let l = unsafe { pmull(y, x) }; // l = x.lo * y.lo
    (h, m, l)
}

/// Karatsuba combine.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn karatsuba2(h: __m128i, m: __m128i, l: __m128i) -> (__m128i, __m128i) {
    // Second Karatsuba step: combine into a 2n-bit product.
    //
    // m0 ^= l0 ^ h0 // = m0^(l0^h0)
    // m1 ^= l1 ^ h1 // = m1^(l1^h1)
    // l1 ^= m0      // = l1^(m0^l0^h0)
    // h0 ^= l0 ^ m1 // = h0^(l0^m1^l1^h1)
    // h1 ^= l1      // = h1^(l1^m0^l0^h0)
    let t = unsafe {
        //   {m0, m1} ^ {l1, h0}
        // = {m0^l1, m1^h0}
        let t0 = {
            _mm_xor_si128(
                m,
                _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(l),
                    _mm_castsi128_ps(h),
                    0x4e,
                )),
            )
        };

        //   {h0, h1} ^ {l0, l1}
        // = {h0^l0, h1^l1}
        let t1 = _mm_xor_si128(h, l);

        //   {m0^l1, m1^h0} ^ {h0^l0, h1^l1}
        // = {m0^l1^h0^l0, m1^h0^h1^l1}
        _mm_xor_si128(t0, t1)
    };

    // {m0^l1^h0^l0, l0}
    let x01 = unsafe { _mm_unpacklo_epi64(l, t) };

    // {h1, m1^h0^h1^l1}
    let x23 = unsafe { _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(h), _mm_castsi128_ps(t))) };

    (x23, x01)
}

/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn mont_reduce(x23: __m128i, x01: __m128i) -> __m128i {
    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    static POLY: u128 = 1 << 127 | 1 << 126 | 1 << 121 | 1 << 63 | 1 << 62 | 1 << 57;
    let poly = unsafe { _mm_loadu_si128(ptr::addr_of!(POLY).cast()) };
    let a = unsafe { pmull(x01, poly) };
    let b = unsafe { _mm_xor_si128(x01, _mm_shuffle_epi32(a, 0x4e)) };
    let c = unsafe { pmull2(b, poly) };
    unsafe { _mm_xor_si128(x23, _mm_xor_si128(c, b)) }
}

/// Multiplies the low bits in `a` and `b`.
///
/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enabled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn pmull(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: This requires the `sse2` and `pclmulqdq` features
    // which we have.
    unsafe { _mm_clmulepi64_si128(a, b, 0x00) }
}

/// Multiplies the high bits in `a` and `b`.
///
/// # Safety
///
/// The SSE2 and pclmulqdq target features must be enavled.
#[inline]
#[target_feature(enable = "sse2,pclmulqdq")]
unsafe fn pmull2(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: This requires the `sse2` and `pclmulqdq` features
    // which we have.
    unsafe { _mm_clmulepi64_si128(a, b, 0x11) }
}
