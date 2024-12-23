//! The software implementation.

#![forbid(unsafe_code)]
#![cfg(any(
    feature = "soft",
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))
))]

pub(crate) use super::generic::*;
