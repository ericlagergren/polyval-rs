//! The software implementation.

#![forbid(unsafe_code)]
#![cfg(any(
    feature = "soft",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
    ))
))]

pub(crate) use crate::generic::*;
use crate::poly::Polyval;

impl Polyval {
    pub(crate) fn update_blocks(&mut self, blocks: &[u8]) {
        self.y = polymul_series(self.y, &self.pow, blocks)
    }
}
