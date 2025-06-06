//! AArch64 implementation.

#![cfg(all(
    not(feature = "soft"),
    target_arch = "aarch64",
    target_feature = "neon",
))]

use core::{
    arch::aarch64::{
        uint8x16_t, uint8x16x4_t, vdupq_n_u8, veorq_u8, vextq_u8, vgetq_lane_u64, vld1q_u8,
        vld1q_u8_x4, vmull_p64, vreinterpretq_u64_u8, vreinterpretq_u8_p128, vrev64q_u8, vst1q_u8,
    },
    array,
};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

use crate::{BLOCK_SIZE, KEY_SIZE};

// NB: `aes` implies `neon`.
cpufeatures::new!(have_aes, "aes");

#[derive(Copy, Clone, Debug)]
pub(super) struct Token {
    token: have_aes::InitToken,
}

impl Token {
    #[inline]
    pub fn new() -> (Self, bool) {
        let (token, supported) = have_aes::init_get();
        (Self { token }, supported)
    }

    #[inline]
    pub fn supported(&self) -> bool {
        self.token.get()
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
    y: uint8x16_t,
    /// h[0] = H, h[1] = H^2, h[2] = H^3, etc,.
    h: [uint8x16_t; N],
}

impl<const GHASH: bool, const N: usize> Backend<GHASH, N> {
    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    #[allow(clippy::undocumented_unsafe_blocks)]
    pub unsafe fn new(key: &[u8; KEY_SIZE]) -> Self {
        const { assert!(N > 0) }

        let h = if GHASH {
            let key = super::mulx(u128::from_be_bytes(*key)).to_le_bytes();
            unsafe { vld1q_u8(key.as_ptr()) }
        } else {
            unsafe { vld1q_u8(key.as_ptr()) }
        };
        let h = {
            let mut prev = unsafe { vdupq_n_u8(0) };
            array::from_fn(|i| {
                prev = if i == 0 {
                    h
                } else {
                    unsafe { polymul(h, prev) }
                };
                prev
            })
        };
        Self {
            y: unsafe { vdupq_n_u8(0) },
            h,
        }
    }

    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    pub unsafe fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        const { assert!(N > 0) }

        // SAFETY: These require the `neon` and `aes` target
        // features, which we have.
        unsafe {
            let mut x = vld1q_u8(block.as_ptr());
            if GHASH {
                x = swap_bytes(x);
            }
            self.y = polymul(veorq_u8(self.y, x), self.h[0]);
        }
    }

    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "neon,aes")]
    #[allow(clippy::undocumented_unsafe_blocks)]
    pub unsafe fn update_blocks(&mut self, mut blocks: &[[u8; BLOCK_SIZE]]) {
        const { assert!(N > 0) }

        if N == 8 {
            let (head, tail) = super::as_chunks::<_, N>(blocks);

            for chunk in head {
                let uint8x16x4_t(m0, m1, m2, m3) = unsafe { vld1q_u8_x4(chunk.as_ptr().cast()) };
                let uint8x16x4_t(m4, m5, m6, m7) =
                    unsafe { vld1q_u8_x4(chunk.as_ptr().add(4).cast()) };

                let mut h = unsafe { vdupq_n_u8(0) };
                let mut m = unsafe { vdupq_n_u8(0) };
                let mut l = unsafe { vdupq_n_u8(0) };

                macro_rules! karatsuba_xor {
                    ($m:expr, $idx:expr) => {
                        unsafe {
                            let mut x = if GHASH { swap_bytes($m) } else { $m };
                            #[allow(clippy::arithmetic_side_effects, reason = "N == 8")]
                            if $idx == N - 1 {
                                // Fold in the accumulator.
                                x = veorq_u8(x, self.y);
                            }
                            let y = self.h[$idx];
                            let (hh, mm, ll) = karatsuba1(x, y);
                            h = veorq_u8(h, hh);
                            m = veorq_u8(m, mm);
                            l = veorq_u8(l, ll);
                        }
                    };
                }

                // For some reason we get significantly worse
                // performance if we calculate m0, m1, ..., m7
                // instead of m7, m6, ..., m0. This holds true
                // even if we reverse `self.h` so that we load
                // h[0], h[1], ..., h[7].
                karatsuba_xor!(m7, 0);
                karatsuba_xor!(m6, 1);
                karatsuba_xor!(m5, 2);
                karatsuba_xor!(m4, 3);
                karatsuba_xor!(m3, 4);
                karatsuba_xor!(m2, 5);
                karatsuba_xor!(m1, 6);
                karatsuba_xor!(m0, 7);

                let (h, l) = unsafe { karatsuba2(h, m, l) };
                self.y = unsafe { mont_reduce(h, l) };
            }

            blocks = tail;
        }

        // Handle singles.
        for block in blocks {
            // SAFETY: This requires the `neon` and `aes` target
            // features, which we have.
            unsafe { self.update_block(block) }
        }
    }

    /// # Safety
    ///
    /// [`Token::supported`] must be true.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn tag(&self) -> [u8; 16] {
        let y = if GHASH {
            // SAFETY: This requires the `neon` target feature,
            // which we have.
            unsafe { swap_bytes(self.y) }
        } else {
            self.y
        };
        let mut tag = [0u8; 16];
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        unsafe { vst1q_u8(tag.as_mut_ptr(), y) }
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

/// Reverses byte order of `x`.
///
/// # Safety
///
/// The NEON architectural feature must be enabled.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn swap_bytes(x: uint8x16_t) -> uint8x16_t {
    // SAFETY: These intrinsics require the `neon` target
    // feature, which we have.
    unsafe {
        let x = vrev64q_u8(x);
        vextq_u8(x, x, 8)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct FieldElement(uint8x16_t);

impl FieldElement {
    #[inline]
    pub fn from_le_bytes(data: &[u8; BLOCK_SIZE]) -> Self {
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        let fe = unsafe { vld1q_u8(data.as_ptr()) };
        Self(fe)
    }

    #[inline]
    pub fn to_le_bytes(self) -> [u8; BLOCK_SIZE] {
        let mut out = [0u8; BLOCK_SIZE];
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        unsafe { vst1q_u8(out.as_mut_ptr(), self.0) }
        out
    }
}

impl Default for FieldElement {
    #[inline]
    fn default() -> Self {
        // SAFETY: This intrinsic requires the `neon` target
        // feature, which we have.
        let fe = unsafe { vdupq_n_u8(0) };
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
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn polymul(x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
    let (h, m, l) = unsafe { karatsuba1(x, y) };
    let (h, l) = unsafe { karatsuba2(h, m, l) };
    unsafe {
        mont_reduce(h, l) // d
    }
}

/// Karatsuba decomposition for `x*y`.
///
/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn karatsuba1(x: uint8x16_t, y: uint8x16_t) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = unsafe {
        pmull(
            veorq_u8(x, vextq_u8(x, x, 8)), // x.hi^x.lo
            veorq_u8(y, vextq_u8(y, y, 8)), // y.hi^y.lo
        )
    };
    let h = unsafe { pmull2(x, y) }; // h = x.hi * y.hi
    let l = unsafe { pmull(x, y) }; // l = x.lo * y.lo
    (h, m, l)
}

/// Karatsuba combine.
///
/// # Safety
///
/// The NEON architectural feature must be enabled.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn karatsuba2(h: uint8x16_t, m: uint8x16_t, l: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
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
        let t0 = veorq_u8(m, vextq_u8(l, h, 8));

        //   {h0, h1} ^ {l0, l1}
        // = {h0^l0, h1^l1}
        let t1 = veorq_u8(h, l);

        //   {m0^l1, m1^h0} ^ {h0^l0, h1^l1}
        // = {m0^l1^h0^l0, m1^h0^h1^l1}
        veorq_u8(t0, t1)
    };

    // {m0^l1^h0^l0, l0}
    let x01 = unsafe {
        vextq_u8(
            vextq_u8(l, l, 8), // {l1, l0}
            t,
            8,
        )
    };

    // {h1, m1^h0^h1^l1}
    let x23 = unsafe {
        vextq_u8(
            t,
            vextq_u8(h, h, 8), // {h1, h0}
            8,
        )
    };

    (x23, x01)
}

/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
#[allow(clippy::undocumented_unsafe_blocks, reason = "Too many unsafe blocks.")]
unsafe fn mont_reduce(x23: uint8x16_t, x01: uint8x16_t) -> uint8x16_t {
    // Perform the Montgomery reduction over the 256-bit X.
    //    [A1:A0] = X0 • poly
    //    [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    //    [C1:C0] = B0 • poly
    //    [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // Output: [D1 ⊕ X3 : D0 ⊕ X2]
    let poly = unsafe {
        vreinterpretq_u8_p128(1 << 127 | 1 << 126 | 1 << 121 | 1 << 63 | 1 << 62 | 1 << 57)
    };
    let a = unsafe { pmull(x01, poly) };
    let b = unsafe { veorq_u8(x01, vextq_u8(a, a, 8)) };
    let c = unsafe { pmull2(b, poly) };
    unsafe { veorq_u8(x23, veorq_u8(c, b)) }
}

/// Multiplies the low bits in `a` and `b`.
///
/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    // SAFETY: These intrinsics require the `neon` and `aes`
    // target features, which we have.
    unsafe {
        let p = vmull_p64(
            vgetq_lane_u64(vreinterpretq_u64_u8(a), 0),
            vgetq_lane_u64(vreinterpretq_u64_u8(b), 0),
        );
        vreinterpretq_u8_p128(p)
    }
}

/// Multiplies the high bits in `a` and `b`.
///
/// # Safety
///
/// The NEON and AES architectural features must be enabled.
#[inline]
#[target_feature(enable = "neon,aes")]
unsafe fn pmull2(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    // SAFETY: These intrinsics require the `neon` and `aes`
    // target features, which we have.
    unsafe {
        let p = vmull_p64(
            vgetq_lane_u64(vreinterpretq_u64_u8(a), 1),
            vgetq_lane_u64(vreinterpretq_u64_u8(b), 1),
        );
        vreinterpretq_u8_p128(p)
    }
}

#[cfg(test)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {
    use core::ops::BitXor;

    use hex_literal::hex;

    use super::*;

    macro_rules! fe {
        ($s:expr) => {{
            FieldElement::from_le_bytes(&hex!($s))
        }};
    }

    impl FieldElement {
        #[inline]
        #[must_use]
        #[target_feature(enable = "neon,aes")]
        unsafe fn polymul(self, rhs: Self) -> Self {
            let fe = unsafe { polymul(self.0, rhs.0) };
            Self(fe)
        }
    }

    impl BitXor for FieldElement {
        type Output = Self;
        fn bitxor(self, rhs: Self) -> Self::Output {
            let fe = unsafe { veorq_u8(self.0, rhs.0) };
            Self(fe)
        }
    }

    #[test]
    fn test_fe_ops() {
        let a = fe!("66e94bd4ef8a2c3b884cfa59ca342b2e");
        let b = fe!("ff000000000000000000000000000000");

        let want = fe!("99e94bd4ef8a2c3b884cfa59ca342b2e");
        assert_eq!(a ^ b, want);
        assert_eq!(b ^ a, want);

        if have_aes::get() {
            let want = fe!("ebe563401e7e91ea3ad6426b8140c394");
            assert_eq!(unsafe { a.polymul(b) }, want);
            assert_eq!(unsafe { b.polymul(a) }, want);
        }
    }

    #[test]
    fn test_idk() {
        let v = [[0u8; 16], [1; 16], [2; 16], [3; 16]];
        let uint8x16x4_t(m0, m1, m2, m3) = unsafe { vld1q_u8_x4(v.as_ptr().cast()) };
        println!("{:?} {:?} {:?} {:?}", m0, m1, m2, m3);
        assert!(false);
    }
}
