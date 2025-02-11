//! GHASH.

use crate::{
    backend::{Big, Small, GHASH},
    impl_hash, impl_state,
};

impl_state!(GHashState, BE);

#[cfg(feature = "experimental")]
type State = GHashState;

impl_hash! {
    /// An implementation of GHASH.
    ///
    /// GHASH is similar to POLYVAL. It operates in `GF(2¹²⁸)`
    /// defined by the irreducible polynomial
    ///
    /// ```text
    /// x^128 + x^7 + x^2 + x + 1
    /// ```
    ///
    /// The field has characteristic 2, so addition is performed
    /// with XOR. Multiplication is polynomial multiplication
    /// reduced modulo the polynomial.
    ///
    /// For more information on GHASH, see [RFC 8452].
    ///
    /// [RFC 8452]: https://datatracker.ietf.org/doc/html/rfc8452
    pub struct GHash(Big<GHASH>);
}

impl_hash! {
    /// The same thing as [`GHash`], except it only processes
    /// one block at a time.
    ///
    /// This saves space, but can be slower if the input is more
    /// than a couple blocks long.
    pub struct GHashLite(Small<GHASH>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tag, BLOCK_SIZE, KEY_SIZE};

    macro_rules! hex {
        ($($s:literal)*) => {{
            const LEN: usize = hex_literal::hex!($($s)*).len();
            const OUTPUT: [u8; LEN] = hex_literal::hex!($($s)*);
            &OUTPUT
        }};
    }

    #[test]
    fn test_rfc() {
        #[track_caller]
        fn test<F>(f: F)
        where
            F: FnOnce(&[u8; KEY_SIZE], &[[u8; BLOCK_SIZE]]) -> Tag,
        {
            let h = hex!("25629347589242761d31f826ba4b757b");
            let x1 = hex!("4f4f95668c83dfb6401762bb2d01a262");
            let x2 = hex!("d1a24ddd2721d006bbe45f20d3c9f362");
            let want = hex!("bd9b3997046731fb96251b91f9c99d7a");

            let got = f(h, &[*x1, *x2]);
            assert_eq!(&got.0, want);
        }

        test(|h, blocks| {
            let mut ghash = GHash::new_unchecked(h);
            ghash.update_blocks(blocks);
            ghash.tag()
        });
        test(|h, blocks| {
            let mut ghash = GHashLite::new_unchecked(h);
            ghash.update_blocks(blocks);
            ghash.tag()
        });
    }
}
