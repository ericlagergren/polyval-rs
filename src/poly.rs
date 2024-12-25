use core::{error, fmt, slice};

use subtle::{Choice, ConstantTimeEq};
#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{backend::FieldElement, precomp::Precomputed};

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
pub struct Key(pub(crate) FieldElement);

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
    #[inline]
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
#[derive(Clone)]
pub struct Polyval<B = Precomputed>(pub(crate) B);

impl<B: Backend> Polyval<B> {
    /// Creates an instance of POLYVAL.
    #[inline]
    pub fn new(key: &Key) -> Self {
        Self(B::new(key))
    }

    /// Writes a single block to the running hash.
    #[inline]
    pub fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        self.0.update_block(block);
    }

    /// Writes one or more blocks to the running hash.
    #[inline]
    pub fn update(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
        self.0.update_blocks(blocks);
    }

    /// Writes one or more blocks to the running hash.
    ///
    /// If the length of `blocks` is non-zero and is not
    /// a multiple of [`BLOCK_SIZE`], it's padded with zeros.
    #[inline]
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
        self.0.tag()
    }

    /// Reports whether the current authentication tag matches
    /// `expected_tag`.
    #[inline]
    pub fn verify(self, expected_tag: &Tag) -> Choice {
        self.tag().ct_eq(expected_tag)
    }
}

#[cfg(feature = "zeroize")]
#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl<T> ZeroizeOnDrop for Polyval<T> {}

impl<T> fmt::Debug for Polyval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Polyval").finish_non_exhaustive()
    }
}

/// A POLYVAL backend.
///
/// See [`Lite`][crate::Lite] or [`Precomputed`].
pub trait Backend: Sealed {}

mod private {
    use super::{Key, State, Tag, BLOCK_SIZE};

    #[doc(hidden)]
    pub trait Sealed: Clone + Sized {
        fn new(key: &Key) -> Self;
        fn update_block(&mut self, block: &[u8; BLOCK_SIZE]);
        fn update_blocks(&mut self, blocks: &[[u8; BLOCK_SIZE]]);
        fn tag(&self) -> Tag;

        fn export(&self) -> State;
        fn reset(&mut self, state: &State);
    }
}
pub(crate) use private::Sealed;

/// An authentication tag.
#[derive(Copy, Clone, Debug)]
pub struct Tag(pub(crate) [u8; 16]);

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

/// Saved [`Polyval`] state.
#[derive(Clone, Default)]
pub struct State {
    pub(crate) y: FieldElement,
}

#[cfg(feature = "zeroize")]
#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl ZeroizeOnDrop for State {}

impl Drop for State {
    #[inline]
    fn drop(&mut self) {
        #[cfg(feature = "zeroize")]
        {
            self.y.zeroize();
        }
        #[cfg(not(feature = "zeroize"))]
        {
            self.y ^= self.y;
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("State").finish_non_exhaustive()
    }
}

// See https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks
const fn as_blocks(blocks: &[u8]) -> (&[[u8; BLOCK_SIZE]], &[u8]) {
    #[allow(clippy::arithmetic_side_effects)]
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
    use serde::Deserialize;

    use super::*;
    use crate::{
        lite::Lite,
        poly::{as_blocks, Key, BLOCK_SIZE},
    };

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

    #[test]
    fn test_rfc_vectors() {
        struct TestCase {
            h: &'static str,
            x: &'static str,
            r: &'static str,
        }
        let cases = [
            TestCase {
                h: "25629347589242761d31f826ba4b757b",
                x: "4f4f95668c83dfb6401762bb2d01a262",
                r: "cedac64537ff50989c16011551086d77",
            },
            TestCase {
                h: "25629347589242761d31f826ba4b757b",
                x: "4f4f95668c83dfb6401762bb2d01a262d1a24ddd2721d006bbe45f20d3c9f3\
                62",
                r: "f7a3b47b846119fae5b7866cf5e5b77e",
            },
            TestCase {
                h: "d9b360279694941ac5dbc6987ada7377",
                x: "00000000000000000000000000000000",
                r: "00000000000000000000000000000000",
            },
            TestCase {
                h: "d9b360279694941ac5dbc6987ada7377",
                x: "01000000000000000000000000000000000000000000000040",
                r: "eb93b7740962c5e49d2a90a7dc5cec74",
            },
            TestCase {
                h: "d9b360279694941ac5dbc6987ada7377",
                x: "01000000000000000000000000000000000000000000000060",
                r: "48eb6c6c5a2dbe4a1dde508fee06361b",
            },
            TestCase {
                h: "d9b360279694941ac5dbc6987ada7377",
                x: "01000000000000000000000000000000000000000000000080",
                r: "20806c26e3c1de019e111255708031d6",
            },
            TestCase {
                h: "d9b360279694941ac5dbc6987ada7377",
                x: "01000000000000000000000000000000020000000000000000000000000000\
                0000000000000000000001",
                r: "ce6edc9a50b36d9a98986bbf6a261c3b",
            },
            TestCase {
                h: "0533fd71f4119257361a3ff1469dd4e5",
                x: "489c8fde2be2cf97e74e932d4ed87d00c9882e5386fd9f92ec000000000000\
                00780000000000000048",
                r: "bf160bc9ded8c63057d2c38aae552fb4",
            },
            TestCase {
                h: "64779ab10ee8a280272f14cc8851b727",
                x: "0da55210cc1c1b0abde3b2f204d1e9f8b06bc47f0000000000000000000000\
                001db2316fd568378da107b52b00000000a00000000000000060",
                r: "cc86ee22c861e1fd474c84676b42739c",
            },
            TestCase {
                h: "27c2959ed4daea3b1f52e849478de376",
                x: "f37de21c7ff901cfe8a69615a93fdf7a98cad481796245709f000000000000\
                0021702de0de18baa9c9596291b0846600c80000000000000078",
                r: "c4fa5e5b713853703bcf8e6424505fa5",
            },
            TestCase {
                h: "670b98154076ddb59b7a9137d0dcc0f0",
                x: "9c2159058b1f0fe91433a5bdc20e214eab7fecef4454a10ef0657df21ac700\
                00b202b370ef9768ec6561c4fe6b7e7296fa85000000000000000000000000\
                0000f00000000000000090",
                r: "4e4108f09f41d797dc9256f8da8d58c7",
            },
            TestCase {
                h: "cb8c3aa3f8dbaeb4b28a3e86ff6625f8",
                x: "734320ccc9d9bbbb19cb81b2af4ecbc3e72834321f7aa0f70b7282b4f33df2\
                3f16754100000000000000000000000000ced532ce4159b035277d4dfbb7db\
                62968b13cd4eec00000000000000000000001801000000000000a8",
                r: "ffd503c7dd712eb3791b7114b17bb0cf",
            },
        ];

        for (i, tc) in cases.iter().enumerate() {
            let h = unhex(tc.h);
            let x = unhex(tc.x);
            let r = unhex(tc.r);
            let k = Key::new_unchecked(&h.try_into().expect("should be `KEY_SIZE` bytes"));
            let mut p = Polyval::<Precomputed>::new(&k);
            p.update_padded(&x);
            let got: [u8; 16] = p.tag().into();
            let want = &r[..];
            assert_eq!(got, want, "#{i} (precomp)");

            let mut p = Polyval::<Lite>::new(&k);
            p.update_padded(&x);
            let got: [u8; 16] = p.tag().into();
            let want = &r[..];
            assert_eq!(got, want, "#{i} (lite)");
        }
    }

    #[test]
    fn test_vectors() {
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Lengths {
            block: usize,
            key: usize,
            nonce: usize,
        }

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct BlockCipher {
            cipher: String,
            lengths: Lengths,
        }

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Input {
            #[serde(with = "hex::serde")]
            key_hex: Vec<u8>,
            #[serde(default, with = "hex::serde")]
            tweak_hex: Vec<u8>,
            #[serde(with = "hex::serde")]
            message_hex: Vec<u8>,
            #[serde(default, with = "hex::serde")]
            nonce_hex: Vec<u8>,
        }

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Cipher {
            cipher: String,
            block_cipher: Option<BlockCipher>,
        }

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct TestVector {
            cipher: Cipher,
            description: String,
            input: Input,
            #[serde(default, with = "hex::serde")]
            plaintext_hex: Vec<u8>,
            #[serde(default, with = "hex::serde")]
            ciphertext_hex: Vec<u8>,
            #[serde(with = "hex::serde")]
            hash_hex: Vec<u8>,
        }

        const DATA: &str = include_str!("testdata/polyval.json");
        let tests: Vec<TestVector> = serde_json::from_str(DATA).expect("should be valid JSON");
        for (i, tc) in tests.iter().enumerate() {
            let b: [u8; BLOCK_SIZE] = (&*tc.input.key_hex).try_into().unwrap_or_else(|_| {
                panic!(
                    "#{i}: {} should be `BLOCK_SIZE` all non-zero bytes",
                    tc.description
                )
            });
            let key = Key::new_unchecked(&b);
            let mut p = Polyval::<Precomputed>::new(&key);
            let (blocks, []) = as_blocks(&tc.input.message_hex) else {
                panic!("#{i}: {} should block sized", tc.description);
            };
            p.update(blocks);
            let got: [u8; 16] = p.clone().tag().into();
            let want = &tc.hash_hex[..];
            assert_eq!(got, want, "#{i}: (precomp) {}", tc.description);

            let mut p = Polyval::<Lite>::new(&key);
            let (blocks, []) = as_blocks(&tc.input.message_hex) else {
                panic!("#{i}: {} should block sized", tc.description);
            };
            p.update(blocks);
            let got: [u8; 16] = p.clone().tag().into();
            let want = &tc.hash_hex[..];
            assert_eq!(got, want, "#{i}: (lite) {}", tc.description);
        }
    }
}
