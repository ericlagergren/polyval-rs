use core::{error, fmt, slice};

use subtle::{Choice, ConstantTimeEq};
#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::backend::FieldElement;

/// The size in bytes of a POLYVAL key.
pub const KEY_SIZE: usize = 16;

/// The size in bytes of a POLYVAL block.
pub const BLOCK_SIZE: usize = 16;

/// The length of the input is not divisible by [`BLOCK_SIZE`].
#[derive(Copy, Clone, Debug)]
pub struct InvalidInputLength;

impl fmt::Display for InvalidInputLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid input length")
    }
}

impl error::Error for InvalidInputLength {}

/// A POLYVAL key.
#[derive(Clone)]
pub struct Key(FieldElement);

impl Key {
    const ZERO: &[u8; KEY_SIZE] = &[0u8; KEY_SIZE];

    /// Creates a POLYVAL key.
    ///
    /// It returns `None` if the key is all zero.
    pub fn new(key: &[u8; KEY_SIZE]) -> Option<Self> {
        if bool::from(key.ct_eq(Self::ZERO)) {
            None
        } else {
            Some(Self::new_unchecked(key))
        }
    }

    /// Creates a POLYVAL key from a known non-zero key.
    ///
    /// # Warning
    ///
    /// Only use this method if `key` is known to be non-zero.
    /// Using an all zero key fixes the POLYVAL to zero,
    /// regardless of the input.
    pub fn new_unchecked(key: &[u8; KEY_SIZE]) -> Self {
        Self(FieldElement::from_le_bytes(key))
    }
}

#[cfg(feature = "zeroize")]
#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl ZeroizeOnDrop for Key {}

impl Drop for Key {
    fn drop(&mut self) {
        #[cfg(feature = "zeroize")]
        {
            self.0.zeroize();
        }
        #[cfg(not(feature = "zeroize"))]
        {
            self.0 ^= self.0;
        }
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Key").finish_non_exhaustive()
    }
}

/// An implementation of POLYVAL.
///
/// POLYVAL is similar to GHASH. It operates in `GF(2¹²⁸)`
/// defined by the irreducible polynomial
///
/// ```text
/// x^128 + x^127 + x^126 + x^121 + 1
/// ```
///
/// The field has characteristic 2, so addition is performed with
/// XOR. Multiplication is polynomial multiplication reduced
/// modulo the polynomial.
///
/// For more information on POLYVAL, see [RFC 8452].
///
/// [RFC 8452]: https://datatracker.ietf.org/doc/html/rfc8452
pub struct Polyval {
    /// The running state.
    pub(crate) y: FieldElement,
    /// Precomputed table of powers of `h` for batched
    /// computations.
    pub(crate) pow: [FieldElement; 8],
}

impl Polyval {
    /// Creates an instance of POLYVAL.
    pub fn new(key: &Key) -> Self {
        let pow = {
            let h = key.0;
            let mut prev = h;
            let mut pow: [FieldElement; 8] = Default::default();
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
            y: FieldElement::default(),
            pow,
        }
    }

    /// Writes a single block to the running hash.
    pub fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        let fe = FieldElement::from_le_bytes(block);
        self.y = (self.y ^ fe) * self.pow[7];
    }

    /// Writes one or more blocks to the running hash.
    pub fn update(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
        // TODO(eric): Should the backends use `&[[u8; 16]]`
        // instead of `&[u8]`?
        self.y = self.y.mul_series(&self.pow, blocks.as_flattened());
    }

    /// Writes one or more blocks to the running hash.
    ///
    /// If the length of `blocks` is non-zero and is not
    /// a multiple of [`BLOCK_SIZE`], it's padded with zeros.
    pub fn update_padded(&mut self, blocks: &[u8]) {
        let (head, tail) = as_blocks(blocks);
        if !head.is_empty() {
            self.update(head);
        }
        if !tail.is_empty() {
            let mut block = [0u8; BLOCK_SIZE];
            #[allow(
                clippy::indexing_slicing,
                reason = "The compiler can prove the slice is in bounds."
            )]
            block[..tail.len()].copy_from_slice(tail);
            self.update_block(&block);
        }
    }

    /// Returns the current authentication tag.
    #[inline]
    pub fn tag(self) -> Tag {
        Tag(self.y.to_le_bytes())
    }

    /// Reports whether the current authentication tag matches
    /// `expected_tag`.
    pub fn verify(self, expected_tag: &Tag) -> Choice {
        self.tag().ct_eq(expected_tag)
    }
}

impl Clone for Polyval {
    fn clone(&self) -> Self {
        Self {
            y: self.y,
            pow: self.pow,
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.y = other.y;
        self.pow = other.pow;
    }
}

#[cfg(feature = "zeroize")]
#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl ZeroizeOnDrop for Polyval {}

impl Drop for Polyval {
    fn drop(&mut self) {
        #[cfg(feature = "zeroize")]
        {
            self.y.zeroize();
            self.pow.zeroize();
        }
        #[cfg(not(feature = "zeroize"))]
        {
            self.y ^= self.y;
            for h in &mut self.pow {
                *h ^= *h;
            }
        }
    }
}

impl fmt::Debug for Polyval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Polyval").finish_non_exhaustive()
    }
}

/// An authentication tag.
#[derive(Copy, Clone, Debug)]
pub struct Tag(pub(crate) [u8; 16]);

impl ConstantTimeEq for Tag {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl From<Tag> for [u8; 16] {
    fn from(tag: Tag) -> Self {
        tag.0
    }
}

// See https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks
pub(crate) const fn as_blocks(blocks: &[u8]) -> (&[[u8; BLOCK_SIZE]], &[u8]) {
    let len_rounded_down = (blocks.len() / BLOCK_SIZE) * BLOCK_SIZE;
    // SAFETY: The rounded-down value is always the same or
    // smaller than the original length, and thus must be
    // in-bounds of the slice.
    let (head, tail) = unsafe { blocks.split_at_unchecked(len_rounded_down) };
    let new_len = head.len() / BLOCK_SIZE;
    // SAFETY: We cast a slice of `new_len * N` elements into
    // a slice of `new_len` many `N` elements chunks.
    let head = unsafe { slice::from_raw_parts(head.as_ptr().cast(), new_len) };
    (head, tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        hex::decode(s).expect("should be valid hex")
    }

    macro_rules! fe {
        ($s:expr) => {{
            FieldElement::from_le_bytes(unhex($s).as_slice().try_into().unwrap())
        }};
    }

    #[test]
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

    #[test]
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    fn test_gf128_mul_commutative() {
        use rand::{rngs::StdRng, RngCore, SeedableRng};

        use super::backend::gf128_mul;

        let mut rng = StdRng::from_entropy();
        for _ in 0..100_000 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let xy = unsafe { gf128_mul(x, y) };
            let yx = unsafe { gf128_mul(y, x) };
            assert_eq!(xy, yx, "{x}*{y}");
        }
    }
}
