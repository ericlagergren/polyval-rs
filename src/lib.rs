//! POLYVAL per [RFC 8452].
//!
//! [RFC 8452]: https://datatracker.ietf.org/doc/html/rfc8452

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(any(test, doctest, feature = "std")), no_std)]
#![cfg_attr(not(any(feature = "std", test)), deny(clippy::std_instead_of_core))]

mod backend;
pub mod experimental;
mod lite;
mod poly;
mod precomp;

pub use lite::Lite;
pub use poly::{Backend, Key, Polyval, Tag, BLOCK_SIZE, KEY_SIZE};
pub use precomp::Precomputed;
