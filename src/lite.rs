#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{
    backend::FieldElement,
    poly::{Backend, Key, Sealed, State, Tag, BLOCK_SIZE},
};

/// POLYVAL without precomputed tables.
pub struct Lite {
    /// The running state.
    y: FieldElement,
    /// The key.
    h: FieldElement,
}

impl Backend for Lite {}
impl Sealed for Lite {
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn new(key: &Key) -> Self {
        Self {
            y: FieldElement::default(),
            h: key.0,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        let fe = FieldElement::from_le_bytes(block);
        self.y = (self.y ^ fe) * self.h;
    }

    #[inline]
    fn update_blocks(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
        for block in blocks {
            self.update_block(block);
        }
    }

    #[inline]
    fn tag(&self) -> Tag {
        Tag(self.y.to_le_bytes())
    }

    #[inline]
    fn export(&self) -> State {
        State { y: self.y }
    }

    #[inline]
    fn reset(&mut self, state: &State) {
        self.y = state.y;
    }
}

impl Clone for Lite {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            y: self.y,
            h: self.h,
        }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.y = other.y;
        self.h = other.h;
    }
}

#[cfg(feature = "zeroize")]
#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl ZeroizeOnDrop for Lite {}

impl Drop for Lite {
    fn drop(&mut self) {
        #[cfg(feature = "zeroize")]
        {
            self.y.zeroize();
            self.h.zeroize();
        }
        #[cfg(not(feature = "zeroize"))]
        {
            self.y ^= self.y;
            self.h ^= self.h;
        }
    }
}
