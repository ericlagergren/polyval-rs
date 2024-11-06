use core::{error, fmt};

use subtle::{Choice, ConstantTimeEq};
#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::backend::FieldElement;

/// The size in bytes of a POLYVAL key.
pub const KEY_SIZE: usize = 16;

/// The size in bytes of a POLYVAL digest.
pub const SIZE: usize = 16;

/// The size in bytes of a POLYVAL block.
pub const BLOCK_SIZE: usize = 16;

/// The length of the input is not divisible by
/// [`BLOCK_SIZE`].
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
#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
pub struct Key([u8; KEY_SIZE]);

impl Key {
    const ZERO: &[u8; KEY_SIZE] = &[0u8; KEY_SIZE];

    /// Creates a POLYVAL key.
    ///
    /// It is an error if the key is all zero.
    pub fn new(key: [u8; KEY_SIZE]) -> Option<Self> {
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
    pub const fn new_unchecked(key: [u8; KEY_SIZE]) -> Self {
        Self(key)
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
#[derive(Clone)]
#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
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
        let h = FieldElement::from_le_bytes(&key.0);
        let mut pow: [FieldElement; 8] = Default::default();
        for i in (0..pow.len()).rev() {
            pow[i] = h;
            if i < pow.len() - 1 {
                pow[i] *= pow[i + 1];
            }
        }
        Self {
            y: FieldElement::default(),
            pow,
        }
    }

    /// Writes one or more blocks to the running hash.
    ///
    /// It is an error if `blocks` is not a multiple of
    /// [`BLOCK_SIZE`].
    pub fn update(&mut self, blocks: &[u8]) -> Result<(), InvalidInputLength> {
        if blocks.len() % BLOCK_SIZE != 0 {
            Err(InvalidInputLength)
        } else {
            self.update_blocks(blocks);
            Ok(())
        }
    }

    /// Writes one or more blocks to the running hash.
    ///
    /// If the length of `blocks` is not a multiple of
    /// [`BLOCK_SIZE`], it's padded with zeros.
    pub fn update_padded(&mut self, blocks: &[u8]) {
        let n = (blocks.len() / BLOCK_SIZE) * BLOCK_SIZE;
        let (head, tail) = blocks.split_at(n);
        if !head.is_empty() {
            self.update_blocks(head);
        }
        if !tail.is_empty() {
            let mut block = [0u8; BLOCK_SIZE];
            block[..tail.len()].copy_from_slice(tail);
            self.update_blocks(&block);
        }
    }

    /// Returns the current authentication tag.
    pub fn tag(self) -> Tag {
        Tag(self.y.to_le_bytes())
    }

    /// Reports whether the current authentication tag matches
    /// `expected_tag`.
    pub fn verify(self, expected_tag: &Tag) -> Choice {
        self.tag().ct_eq(expected_tag)
    }
}

impl Polyval {
    fn update_blocks(&mut self, blocks: &[u8]) {
        self.y = self.y.mul_series(&self.pow, blocks);
    }
}

/// An authentication tag.
#[derive(Copy, Clone, Debug)]
pub struct Tag([u8; 16]);

impl ConstantTimeEq for Tag {
    #[inline]
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl From<Tag> for [u8; 16] {
    #[inline]
    fn from(tag: Tag) -> Self {
        tag.0
    }
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
    #[cfg(not(target_arch = "aarch64"))]
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
