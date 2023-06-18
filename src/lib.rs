//! POLYVAL per [RFC 8452].
//!
//! [RFC 8452]: https://datatracker.ietf.org/doc/html/rfc8452

#![cfg_attr(docs, feature(doc_cfg))]
#![cfg_attr(feature = "error_in_core", feature(error_in_core))]
#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![deny(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::implicit_saturating_sub,
    clippy::panic,
    clippy::unwrap_used,
    missing_docs,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications
)]

mod aarch64;
mod generic;
mod poly;
mod soft;
mod x86;

pub use poly::*;
