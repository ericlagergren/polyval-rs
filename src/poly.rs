use crate::{
    backend::{Big, Small, POLYVAL},
    impl_hash, impl_state,
};

impl_state!(PolyvalState, LE);

#[cfg(feature = "experimental")]
type State = PolyvalState;

impl_hash! {
    /// An implementation of POLYVAL.
    ///
    /// POLYVAL is similar to GHASH. It operates in `GF(2¹²⁸)`
    /// defined by the irreducible polynomial
    ///
    /// ```text
    /// x^128 + x^127 + x^126 + x^121 + 1
    /// ```
    ///
    /// The field has characteristic 2, so addition is performed
    /// with XOR. Multiplication is polynomial multiplication
    /// reduced modulo the polynomial.
    ///
    /// For more information on POLYVAL, see [RFC 8452].
    ///
    /// [RFC 8452]: https://datatracker.ietf.org/doc/html/rfc8452
    pub struct Polyval(Big<POLYVAL>);
}

impl_hash! {
    /// The same thing as [`Polyval`], except it only processes
    /// one block at a time.
    ///
    /// This saves space, but can be slower if the input is more
    /// than a couple blocks long.
    pub struct PolyvalLite(Small<POLYVAL>);
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;
    use crate::{as_blocks, KEY_SIZE};

    macro_rules! hex {
        ($($s:literal)*) => {{
            const LEN: usize = hex_literal::hex!($($s)*).len();
            const OUTPUT: [u8; LEN] = hex_literal::hex!($($s)*);
            &OUTPUT
        }};
    }

    #[test]
    fn test_rfc_vectors() {
        type TestCase = (&'static [u8; KEY_SIZE], &'static [u8], &'static [u8; 16]);
        let cases: &[TestCase] = &[
            (
                hex!("25629347589242761d31f826ba4b757b"),
                hex!("4f4f95668c83dfb6401762bb2d01a262"),
                hex!("cedac64537ff50989c16011551086d77"),
            ),
            (
                hex!("25629347589242761d31f826ba4b757b"),
                hex!(
                    "4f4f95668c83dfb6401762bb2d01a262"
                    "d1a24ddd2721d006bbe45f20d3c9f362"
                ),
                hex!("f7a3b47b846119fae5b7866cf5e5b77e"),
            ),
            (
                hex!("d9b360279694941ac5dbc6987ada7377"),
                hex!("00000000000000000000000000000000"),
                hex!("00000000000000000000000000000000"),
            ),
            (
                hex!("d9b360279694941ac5dbc6987ada7377"),
                hex!("01000000000000000000000000000000"
                    "000000000000000040"),
                hex!("eb93b7740962c5e49d2a90a7dc5cec74"),
            ),
            (
                hex!("d9b360279694941ac5dbc6987ada7377"),
                hex!("01000000000000000000000000000000"
                    "000000000000000060"),
                hex!("48eb6c6c5a2dbe4a1dde508fee06361b"),
            ),
            (
                hex!("d9b360279694941ac5dbc6987ada7377"),
                hex!("01000000000000000000000000000000"
                    "000000000000000080"),
                hex!("20806c26e3c1de019e111255708031d6"),
            ),
            (
                hex!("d9b360279694941ac5dbc6987ada7377"),
                hex!(
                    "01000000000000000000000000000000"
                    "02000000000000000000000000000000"
                    "00000000000000000001"
                ),
                hex!("ce6edc9a50b36d9a98986bbf6a261c3b"),
            ),
            (
                hex!("0533fd71f4119257361a3ff1469dd4e5"),
                hex!(
                    "489c8fde2be2cf97e74e932d4ed87d00"
                    "c9882e5386fd9f92ec00000000000000"
                    "780000000000000048"
                ),
                hex!("bf160bc9ded8c63057d2c38aae552fb4"),
            ),
            (
                hex!("64779ab10ee8a280272f14cc8851b727"),
                hex!(
                    "0da55210cc1c1b0abde3b2f204d1e9f8"
                    "b06bc47f000000000000000000000000"
                    "1db2316fd568378da107b52b00000000"
                    "a00000000000000060"
                ),
                hex!("cc86ee22c861e1fd474c84676b42739c"),
            ),
            (
                hex!("27c2959ed4daea3b1f52e849478de376"),
                hex!(
                    "f37de21c7ff901cfe8a69615a93fdf7a"
                    "98cad481796245709f00000000000000"
                    "21702de0de18baa9c9596291b0846600"
                    "c80000000000000078"
                ),
                hex!("c4fa5e5b713853703bcf8e6424505fa5"),
            ),
            (
                hex!("670b98154076ddb59b7a9137d0dcc0f0"),
                hex!(
                    "9c2159058b1f0fe91433a5bdc20e214e"
                    "ab7fecef4454a10ef0657df21ac70000"
                    "b202b370ef9768ec6561c4fe6b7e7296"
                    "fa850000000000000000000000000000"
                    "f00000000000000090"
                ),
                hex!("4e4108f09f41d797dc9256f8da8d58c7"),
            ),
            (
                hex!("cb8c3aa3f8dbaeb4b28a3e86ff6625f8"),
                hex!(
                    "734320ccc9d9bbbb19cb81b2af4ecbc3"
                    "e72834321f7aa0f70b7282b4f33df23f"
                    "16754100000000000000000000000000"
                    "ced532ce4159b035277d4dfbb7db6296"
                    "8b13cd4eec0000000000000000000000"
                    "1801000000000000a8"
                ),
                hex!("ffd503c7dd712eb3791b7114b17bb0cf"),
            ),
        ];

        for (i, &(h, x, r)) in cases.iter().enumerate() {
            let mut p = Polyval::new_unchecked(h);
            p.update_padded(x);
            assert_eq!(&p.tag().0, r, "#{i} (precomp)");

            let mut p = PolyvalLite::new_unchecked(&h);
            p.update_padded(&x);
            assert_eq!(&p.tag().0, r, "#{i} (lite)");
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
            let h: [u8; KEY_SIZE] = (&*tc.input.key_hex).try_into().unwrap_or_else(|_| {
                panic!(
                    "#{i}: {} should be `BLOCK_SIZE` all non-zero bytes",
                    tc.description
                )
            });
            let mut p = Polyval::new_unchecked(&h);
            let (blocks, []) = as_blocks(&tc.input.message_hex) else {
                panic!("#{i}: {} should block sized", tc.description);
            };
            p.update_blocks(blocks);
            let got: [u8; 16] = p.clone().tag().into();
            let want = &tc.hash_hex[..];
            assert_eq!(got, want, "#{i}: (precomp) {}", tc.description);

            let mut p = PolyvalLite::new_unchecked(&h);
            let (blocks, []) = as_blocks(&tc.input.message_hex) else {
                panic!("#{i}: {} should block sized", tc.description);
            };
            p.update_blocks(blocks);
            let got: [u8; 16] = p.clone().tag().into();
            let want = &tc.hash_hex[..];
            assert_eq!(got, want, "#{i}: (lite) {}", tc.description);
        }
    }
}
