#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{
    backend::FieldElement,
    poly::{Backend, Key, Sealed, State, Tag, BLOCK_SIZE},
};

/// POLYVAL with precomputed powers.
///
/// This is faster than [`Lite`][crate::Lite] when writing many
/// blocks of data, but is about five times larger.
pub struct Precomputed {
    /// The running state.
    y: FieldElement,
    /// Precomputed table of powers of `h` for batched
    /// computations.
    pow: [FieldElement; 8],
}

impl Backend for Precomputed {}
impl Sealed for Precomputed {
    #[allow(clippy::arithmetic_side_effects)]
    fn new(key: &Key) -> Self {
        let pow = {
            let h = key.0;
            let mut prev = h;
            let mut pow: [FieldElement; 8] = Default::default();
            for (i, v) in pow.iter_mut().rev().enumerate() {
                *v = h;
                if i > 0 {
                    *v *= prev;
                }
                prev = *v;
            }
            pow
        };
        Self {
            y: FieldElement::default(),
            pow,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        let fe = FieldElement::from_le_bytes(block);
        self.y = (self.y ^ fe) * self.pow[7];
    }

    fn update_blocks(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
        // TODO(eric): Should the backends use `&[[u8; 16]]`
        // instead of `&[u8]`?
        self.y = self.y.mul_series(&self.pow, blocks.as_flattened());
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

impl Clone for Precomputed {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            y: self.y,
            pow: self.pow,
        }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.y = other.y;
        self.pow = other.pow;
    }
}

#[cfg(feature = "zeroize")]
#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl ZeroizeOnDrop for Precomputed {}

impl Drop for Precomputed {
    fn drop(&mut self) {
        #[cfg(feature = "zeroize")]
        {
            self.y.zeroize();
            self.pow.zeroize();
        }
        #[cfg(not(feature = "zeroize"))]
        {
            self.y ^= self.y;
            for h in &mut self.pow {
                *h ^= *h;
            }
        }
    }
}
