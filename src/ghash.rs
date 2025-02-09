//! GHASH.

use crate::{
    backend::{self, BE},
    impl_hash, impl_state,
};

impl_state!(GHashState, BE);
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
    pub struct Polyval(backend::Precomputed<BE>);
}

impl_hash! {
    /// The same thing as [`GHash`], except it only processes
    /// one block at a time.
    ///
    /// This saves space, but can be slower if the input is more
    /// than a couple blocks long.
    pub struct GHashLite(backend::Precomputed<BE>);
}
