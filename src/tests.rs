#![cfg(test)]

use serde::Deserialize;

use crate::*;

fn unhex(s: &str) -> Vec<u8> {
    hex::decode(s).expect("should be valid hex")
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
        let got: [u8; 16] = p.clone().tag().into();
        let want = &tc.r[..];
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
        let key = Key::new(b)
            .unwrap_or_else(|_| panic!("#{i}: {} should be a valid key", tc.description));
        let mut p = Polyval::new(&key);
        p.update(&tc.input.message_hex[..])
            .unwrap_or_else(|_| panic!("#{i}: {} should block sized", tc.description));
        let got: [u8; 16] = p.clone().tag().into();
        let want = &tc.hash_hex[..];
        assert_eq!(got, want, "#{i}: {}", tc.description);
    }
}
