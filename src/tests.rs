#![cfg(test)]

use serde::Deserialize;

use crate::poly::{as_blocks, Key, Polyval, BLOCK_SIZE};

fn unhex(s: &str) -> Vec<u8> {
    hex::decode(s).expect("should be valid hex")
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
        let mut p = Polyval::new(&k);
        p.update_padded(&x);
        let got: [u8; 16] = p.tag().into();
        let want = &r[..];
        assert_eq!(got, want, "#{i}");
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
        let mut p = Polyval::new(&key);
        let (blocks, []) = as_blocks(&tc.input.message_hex) else {
            panic!("#{i}: {} should block sized", tc.description);
        };
        p.update(blocks);
        let got: [u8; 16] = p.clone().tag().into();
        let want = &tc.hash_hex[..];
        assert_eq!(got, want, "#{i}: {}", tc.description);
    }
}
