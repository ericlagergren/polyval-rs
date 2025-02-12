mod aarch64;
mod soft;
mod x86;

use core::{fmt, mem::ManuallyDrop, slice};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

cfg_if::cfg_if! {
    if #[cfg(feature = "soft")] {
        use soft as imp;
    } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
        use aarch64 as imp;
    } else if #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2",
    ))] {
        use x86 as imp;
    } else {
        use soft as imp;
    }
}

use crate::{BLOCK_SIZE, KEY_SIZE};

pub const GHASH: bool = true;
pub const POLYVAL: bool = false;

/// Doubles `x` in GF(2¹²⁸).
#[inline]
pub(crate) const fn mulx(mut x: u128) -> u128 {
    let hi = x >> 127;
    x <<= 1;
    x ^= hi ^ (hi << 127) ^ (hi << 126) ^ (hi << 121);
    x
}

// To/from byte generic impls.
#[allow(dead_code, reason = "Depends on the backend")]
impl imp::FieldElement {
    #[inline(always)]
    pub fn to_soft(self) -> soft::FieldElement {
        soft::FieldElement::from_le_bytes(&self.to_le_bytes())
    }

    #[inline(always)]
    pub fn from_soft(fe: soft::FieldElement) -> Self {
        Self::from_le_bytes(&fe.to_le_bytes())
    }
}

/// An element in the field
///
/// ```text
/// x^128 + x^127 + x^126 + x^121 + 1
/// ```
#[derive(Copy, Clone, Default, Debug)]
#[repr(transparent)]
pub struct FieldElement(imp::FieldElement);

#[cfg(feature = "zeroize")]
impl Zeroize for FieldElement {
    #[inline]
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

union Inner<A, S> {
    asm: ManuallyDrop<A>,
    soft: ManuallyDrop<S>,
}

macro_rules! impl_hash {
    ($name:ident) => {
        pub struct $name<const GHASH: bool> {
            inner: Inner<imp::$name<GHASH>, soft::$name<GHASH>>,
            token: imp::Token,
        }

        impl<const GHASH: bool> $name<GHASH> {
            #[inline]
            fn have_asm(&self) -> bool {
                self.token.supported()
            }

            #[inline]
            pub fn new(key: &[u8; KEY_SIZE]) -> Self {
                let (token, supported) = imp::Token::new();
                let inner = if supported {
                    #[allow(unused_unsafe)]
                    // SAFETY: `have_asm` is true, so it is safe
                    // to call this function.
                    let asm = unsafe { imp::$name::<GHASH>::new(key) };
                    Inner {
                        asm: ManuallyDrop::new(asm),
                    }
                } else {
                    let soft = soft::$name::<GHASH>::new(key);
                    Inner {
                        soft: ManuallyDrop::new(soft),
                    }
                };
                Self { inner, token }
            }

            /// Writes one block to the hash.
            #[inline]
            pub fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` has
                    // been initialized.
                    unsafe { (&mut self.inner.asm).update_block(block) }
                } else {
                    // SAFETY: `have_asm` is true, so `soft` has
                    // been initialized.
                    unsafe { (&mut self.inner.soft).update_block(block) }
                }
            }

            /// Writes one or more blocks to the hash.
            #[inline]
            pub fn update_blocks(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` has
                    // been initialized.
                    unsafe { (&mut self.inner.asm).update_blocks(blocks) }
                } else {
                    // SAFETY: `have_asm` is true, so `soft` has
                    // been initialized.
                    unsafe { (&mut self.inner.soft).update_blocks(blocks) }
                }
            }

            #[inline]
            pub fn tag(&self) -> [u8; 16] {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` has
                    // been initialized.
                    unsafe { (&self.inner.asm).tag() }
                } else {
                    // SAFETY: `have_asm` is true, so `soft` has
                    // been initialized.
                    unsafe { (&self.inner.soft).tag() }
                }
            }

            #[inline]
            #[cfg(feature = "experimental")]
            pub fn export(&self) -> FieldElement {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` has
                    // been initialized.
                    let fe = unsafe { (&self.inner.asm).export() };
                    FieldElement(fe)
                } else {
                    // SAFETY: `have_asm` is true, so `soft` has
                    // been initialized.
                    let fe = unsafe { (&self.inner.soft).export() };
                    FieldElement(imp::FieldElement::from_soft(fe))
                }
            }

            #[inline]
            #[cfg(feature = "experimental")]
            pub fn reset(&mut self, y: FieldElement) {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` has
                    // been initialized.
                    unsafe { (&mut self.inner.asm).reset(y.0) }
                } else {
                    // SAFETY: `have_asm` is true, so `soft` has
                    // been initialized.
                    unsafe { (&mut self.inner.soft).reset(y.0.to_soft()) }
                }
            }
        }

        impl<const GHASH: bool> Clone for $name<GHASH> {
            #[inline]
            fn clone(&self) -> Self {
                let inner = if self.token.supported() {
                    Inner {
                        // SAFETY: `have_asm` is true, so `asm`
                        // has been initialized.
                        asm: unsafe { &self.inner.asm }.clone(),
                    }
                } else {
                    Inner {
                        // SAFETY: `have_asm` is true, so `soft`
                        // has been initialized.
                        soft: unsafe { &self.inner.soft }.clone(),
                    }
                };
                Self {
                    inner,
                    token: self.token,
                }
            }

            #[inline]
            fn clone_from(&mut self, other: &Self) {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` has
                    // been initialized.
                    unsafe { (&mut self.inner.asm).clone_from(&other.inner.asm) }
                } else {
                    // SAFETY: `have_asm` is true, so `soft` has
                    // been initialized.
                    unsafe { (&mut self.inner.soft).clone_from(&other.inner.soft) }
                }
                self.token = other.token;
            }
        }

        impl<const GHASH: bool> Drop for $name<GHASH> {
            #[inline]
            fn drop(&mut self) {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` has
                    // been initialized.
                    unsafe { ManuallyDrop::drop(&mut self.inner.asm) }
                } else {
                    // SAFETY: `have_asm` is true, so `soft` has
                    // been initialized.
                    unsafe { ManuallyDrop::drop(&mut self.inner.soft) }
                }
            }
        }

        impl<const GHASH: bool> fmt::Debug for $name<GHASH> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.have_asm() {
                    // SAFETY: `have_asm` is true, so `asm` is
                    // initialized.
                    unsafe { fmt::Debug::fmt(&self.inner.asm, f) }
                } else {
                    // SAFETY: `have_asm` is false, so `soft` is
                    // initialized.
                    unsafe { fmt::Debug::fmt(&self.inner.soft, f) }
                }
            }
        }
    };
}
impl_hash!(Big);
impl_hash!(Small);

// See https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks
#[inline(always)]
#[allow(clippy::arithmetic_side_effects)]
pub(crate) const fn as_chunks<T, const N: usize>(data: &[T]) -> (&[[T; N]], &[T]) {
    const { assert!(N > 0) }

    let len_rounded_down = (data.len() / N) * N;
    // SAFETY: The rounded-down value is always the same or
    // smaller than the original length, and thus must be
    // in-bounds of the slice.
    let (head, tail) = unsafe { data.split_at_unchecked(len_rounded_down) };
    let new_len = head.len() / N;
    // SAFETY: We cast a slice of `new_len * N` elements into
    // a slice of `new_len` many `N` elements chunks.
    let head = unsafe { slice::from_raw_parts(head.as_ptr().cast(), new_len) };
    (head, tail)
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

    impl FieldElement {
        fn from_le_bytes(data: &[u8; BLOCK_SIZE]) -> Self {
            Self(imp::FieldElement::from_le_bytes(data))
        }

        fn mulx(self) -> Self {
            let x = mulx(u128::from_le_bytes(self.0.to_le_bytes()));
            Self(imp::FieldElement::from_le_bytes(&x.to_le_bytes()))
        }
    }

    #[cfg(test)]
    impl Eq for FieldElement {}

    #[cfg(test)]
    impl PartialEq for FieldElement {
        fn eq(&self, other: &Self) -> bool {
            PartialEq::eq(&self.0, &other.0)
        }
    }

    #[test]
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    fn test_gf128_mul_commutative() {
        use imp::gf128_mul;
        use rand::{rngs::StdRng, RngCore, SeedableRng};

        let mut rng = StdRng::from_entropy();
        for _ in 0..100_000 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let xy = unsafe { gf128_mul(x, y) };
            let yx = unsafe { gf128_mul(y, x) };
            assert_eq!(xy, yx, "{x}*{y}");
        }
    }

    #[test]
    fn test_mulx() {
        let tests = [
            fe!("02000000000000000000000000000000"),
            fe!("04000000000000000000000000000000"),
            fe!("08000000000000000000000000000000"),
            fe!("10000000000000000000000000000000"),
            fe!("20000000000000000000000000000000"),
            fe!("40000000000000000000000000000000"),
            fe!("80000000000000000000000000000000"),
            fe!("00010000000000000000000000000000"),
            fe!("00020000000000000000000000000000"),
            fe!("00040000000000000000000000000000"),
            fe!("00080000000000000000000000000000"),
            fe!("00100000000000000000000000000000"),
            fe!("00200000000000000000000000000000"),
            fe!("00400000000000000000000000000000"),
            fe!("00800000000000000000000000000000"),
            fe!("00000100000000000000000000000000"),
            fe!("00000200000000000000000000000000"),
            fe!("00000400000000000000000000000000"),
            fe!("00000800000000000000000000000000"),
            fe!("00001000000000000000000000000000"),
            fe!("00002000000000000000000000000000"),
            fe!("00004000000000000000000000000000"),
            fe!("00008000000000000000000000000000"),
            fe!("00000001000000000000000000000000"),
            fe!("00000002000000000000000000000000"),
            fe!("00000004000000000000000000000000"),
            fe!("00000008000000000000000000000000"),
            fe!("00000010000000000000000000000000"),
            fe!("00000020000000000000000000000000"),
            fe!("00000040000000000000000000000000"),
            fe!("00000080000000000000000000000000"),
            fe!("00000000010000000000000000000000"),
            fe!("00000000020000000000000000000000"),
            fe!("00000000040000000000000000000000"),
            fe!("00000000080000000000000000000000"),
            fe!("00000000100000000000000000000000"),
            fe!("00000000200000000000000000000000"),
            fe!("00000000400000000000000000000000"),
            fe!("00000000800000000000000000000000"),
            fe!("00000000000100000000000000000000"),
            fe!("00000000000200000000000000000000"),
            fe!("00000000000400000000000000000000"),
            fe!("00000000000800000000000000000000"),
            fe!("00000000001000000000000000000000"),
            fe!("00000000002000000000000000000000"),
            fe!("00000000004000000000000000000000"),
            fe!("00000000008000000000000000000000"),
            fe!("00000000000001000000000000000000"),
            fe!("00000000000002000000000000000000"),
            fe!("00000000000004000000000000000000"),
            fe!("00000000000008000000000000000000"),
            fe!("00000000000010000000000000000000"),
            fe!("00000000000020000000000000000000"),
            fe!("00000000000040000000000000000000"),
            fe!("00000000000080000000000000000000"),
            fe!("00000000000000010000000000000000"),
            fe!("00000000000000020000000000000000"),
            fe!("00000000000000040000000000000000"),
            fe!("00000000000000080000000000000000"),
            fe!("00000000000000100000000000000000"),
            fe!("00000000000000200000000000000000"),
            fe!("00000000000000400000000000000000"),
            fe!("00000000000000800000000000000000"),
            fe!("00000000000000000100000000000000"),
            fe!("00000000000000000200000000000000"),
            fe!("00000000000000000400000000000000"),
            fe!("00000000000000000800000000000000"),
            fe!("00000000000000001000000000000000"),
            fe!("00000000000000002000000000000000"),
            fe!("00000000000000004000000000000000"),
            fe!("00000000000000008000000000000000"),
            fe!("00000000000000000001000000000000"),
            fe!("00000000000000000002000000000000"),
            fe!("00000000000000000004000000000000"),
            fe!("00000000000000000008000000000000"),
            fe!("00000000000000000010000000000000"),
            fe!("00000000000000000020000000000000"),
            fe!("00000000000000000040000000000000"),
            fe!("00000000000000000080000000000000"),
            fe!("00000000000000000000010000000000"),
            fe!("00000000000000000000020000000000"),
            fe!("00000000000000000000040000000000"),
            fe!("00000000000000000000080000000000"),
            fe!("00000000000000000000100000000000"),
            fe!("00000000000000000000200000000000"),
            fe!("00000000000000000000400000000000"),
            fe!("00000000000000000000800000000000"),
            fe!("00000000000000000000000100000000"),
            fe!("00000000000000000000000200000000"),
            fe!("00000000000000000000000400000000"),
            fe!("00000000000000000000000800000000"),
            fe!("00000000000000000000001000000000"),
            fe!("00000000000000000000002000000000"),
            fe!("00000000000000000000004000000000"),
            fe!("00000000000000000000008000000000"),
            fe!("00000000000000000000000001000000"),
            fe!("00000000000000000000000002000000"),
            fe!("00000000000000000000000004000000"),
            fe!("00000000000000000000000008000000"),
            fe!("00000000000000000000000010000000"),
            fe!("00000000000000000000000020000000"),
            fe!("00000000000000000000000040000000"),
            fe!("00000000000000000000000080000000"),
            fe!("00000000000000000000000000010000"),
            fe!("00000000000000000000000000020000"),
            fe!("00000000000000000000000000040000"),
            fe!("00000000000000000000000000080000"),
            fe!("00000000000000000000000000100000"),
            fe!("00000000000000000000000000200000"),
            fe!("00000000000000000000000000400000"),
            fe!("00000000000000000000000000800000"),
            fe!("00000000000000000000000000000100"),
            fe!("00000000000000000000000000000200"),
            fe!("00000000000000000000000000000400"),
            fe!("00000000000000000000000000000800"),
            fe!("00000000000000000000000000001000"),
            fe!("00000000000000000000000000002000"),
            fe!("00000000000000000000000000004000"),
            fe!("00000000000000000000000000008000"),
            fe!("00000000000000000000000000000001"),
            fe!("00000000000000000000000000000002"),
            fe!("00000000000000000000000000000004"),
            fe!("00000000000000000000000000000008"),
            fe!("00000000000000000000000000000010"),
            fe!("00000000000000000000000000000020"),
            fe!("00000000000000000000000000000040"),
            fe!("00000000000000000000000000000080"),
            fe!("010000000000000000000000000000c2"),
        ];

        let mut got = fe!("01000000000000000000000000000000");
        for (i, &want) in tests.iter().enumerate() {
            got = got.mulx();
            assert_eq!(got, want, "#{i}");
        }
    }
}
