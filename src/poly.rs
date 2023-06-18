use {
    cfg_if::cfg_if,
    core::{fmt, result::Result},
    subtle::{Choice, ConstantTimeEq},
};

cfg_if! {
    if #[cfg(feature = "soft")] {
        use crate::soft::*;
    } else if #[cfg(target_arch = "aarch64")] {
        use crate::aarch64::*;
    } else if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        use crate::x86::*;
    } else {
        use crate::soft::*;
    }
}

cfg_if! {
    if #[cfg(feature = "error_in_core")] {
        use core::error;
    } else if #[cfg(feature = "std")] {
        use std::error;
    }
}

/// The size in bytes of a POLYVAL key.
pub const KEY_SIZE: usize = 16;

/// The size in bytes of a POLYVAL digest.
pub const SIZE: usize = 16;

/// The size in bytes of a POLYVAL block.
pub const BLOCK_SIZE: usize = 16;

/// An error returned by this crate.
#[derive(Copy, Clone, Debug)]
pub enum Error {
    /// The key is all zero.
    AllZeroKey,
    /// The length of the input is not divisible by
    /// [`BLOCK_SIZE`].
    InvalidInputLength,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllZeroKey => write!(f, "all zero key"),
            Self::InvalidInputLength => write!(f, "invalid input length"),
        }
    }
}

#[cfg_attr(docs, doc(cfg(any(feature = "error_in_core", feature = "std"))))]
#[cfg(any(feature = "error_in_core", feature = "std"))]
impl error::Error for Error {}

/// A POLYVAL key.
///
/// The key cannot be all zero.
#[derive(Clone)]
#[cfg_attr(
    feature = "zeroize",
    derive(zeroize::Zeroize, zeroize::ZeroizeOnDrop)
)]
pub struct Key([u8; KEY_SIZE]);

impl Key {
    /// Creates a POLYVAL key.
    ///
    /// It is an error if the key is all zero.
    pub fn new(key: &[u8; KEY_SIZE]) -> Result<Self, Error> {
        const ZERO: &[u8; KEY_SIZE] = &[0u8; KEY_SIZE];

        if bool::from(ZERO.ct_eq(key)) {
            Err(Error::AllZeroKey)
        } else {
            Ok(Self(*key))
        }
    }
}

impl TryFrom<&[u8; KEY_SIZE]> for Key {
    type Error = Error;

    fn try_from(key: &[u8; KEY_SIZE]) -> Result<Self, Self::Error> {
        Self::new(key)
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
            pow[i] = if i < pow.len() - 1 { h * pow[i + 1] } else { h };
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
    pub fn update(&mut self, blocks: &[u8]) -> Result<(), Error> {
        if blocks.len() % BLOCK_SIZE != 0 {
            Err(Error::InvalidInputLength)
        } else {
            self.update_blocks(blocks);
            Ok(())
        }
    }

    /// Writes one or more blocks to the running hash.
    ///
    /// If the length of `blocks` is not a multiple of
    /// [`BLOCK_SIZE`], it's padded with zeros.
    pub fn update_padded(&mut self, mut blocks: &[u8]) {
        let n = (blocks.len() / BLOCK_SIZE) * BLOCK_SIZE;
        if n > 0 {
            self.update(&blocks[..n])
                .expect("should be a multiple of `BLOCK_SIZE`");
            blocks = &blocks[n..];
        }
        if !blocks.is_empty() {
            let mut block = [0u8; BLOCK_SIZE];
            block[..blocks.len()].copy_from_slice(blocks);
            self.update(&block)
                .expect("should be a multiple of `BLOCK_SIZE`");
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

/// An authentication tag.
#[derive(Copy, Clone, Debug)]
pub struct Tag([u8; 16]);

impl ConstantTimeEq for Tag {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use {
        super::*,
        rand::{rngs::StdRng, RngCore, SeedableRng},
        serde::Deserialize,
    };

    fn unhex(s: &str) -> Vec<u8> {
        hex::decode(s).expect("should be valid hex")
    }

    macro_rules! fe {
        ($s:expr) => {{
            FieldElement::from_le_bytes(unhex($s).as_slice())
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
    fn test_gf128_mul_commutative() {
        let mut rng = StdRng::from_entropy();
        for _ in 0..100_000 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let xy = gf128_mul(x, y);
            let yx = gf128_mul(y, x);
            assert_eq!(xy, yx, "{x}*{y}");
        }
    }

    #[test]
    fn test_rfc_vectors() {
        struct TestCase {
            h: Vec<u8>,
            x: Vec<Vec<u8>>,
            r: Vec<u8>,
        }
        let cases = [
            TestCase {
                h: unhex("25629347589242761d31f826ba4b757b"),
                x: vec![unhex("4f4f95668c83dfb6401762bb2d01a262")],
                r: unhex("cedac64537ff50989c16011551086d77"),
            },
            TestCase {
                h: unhex("25629347589242761d31f826ba4b757b"),
                x: vec![
                    unhex("4f4f95668c83dfb6401762bb2d01a262"),
                    unhex("d1a24ddd2721d006bbe45f20d3c9f362"),
                ],
                r: unhex("f7a3b47b846119fae5b7866cf5e5b77e"),
            },
        ];

        for (i, tc) in cases.iter().enumerate() {
            let k = Key::new(
                tc.h.as_slice()
                    .try_into()
                    .expect("should be `KEY_SIZE` bytes"),
            )
            .expect("should not be all zero");
            let mut p = Polyval::new(&k);
            for x in &tc.x {
                p.update(x).expect("should be `BLOCK_SIZE` bytes");
            }
            assert_eq!(&p.clone().tag().0, &tc.r[..], "#{i}");
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
        let tests: Vec<TestVector> =
            serde_json::from_str(DATA).expect("should be valid JSON");
        for (i, tc) in tests.iter().enumerate() {
            let b: [u8; BLOCK_SIZE] =
                (&*tc.input.key_hex).try_into().unwrap_or_else(|_| {
                    panic!(
                        "#{i}: {} should be `BLOCK_SIZE` all non-zero bytes",
                        tc.description
                    )
                });
            let key = Key::new(&b).unwrap_or_else(|_| {
                panic!("#{i}: {} should be a valid key", tc.description)
            });
            let mut p = Polyval::new(&key);
            p.update(&tc.input.message_hex[..]).unwrap_or_else(|_| {
                panic!("#{i}: {} should block sized", tc.description)
            });
            assert_eq!(
                &p.clone().tag().0,
                &tc.hash_hex[..],
                "#{i}: {}",
                tc.description
            );
        }
    }
}
