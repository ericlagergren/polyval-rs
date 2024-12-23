//! Experimental features NOT covered by semver.

#![cfg(feature = "experimental")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental")))]

use core::fmt;

#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::backend::FieldElement;
pub use super::poly::{Polyval, Tag, BLOCK_SIZE, KEY_SIZE};

impl Polyval {
    /// Exports the current state.
    #[inline]
    pub fn export(&self) -> State {
        State { y: self.y }
    }

    /// Resets the hash to `state`.
    #[inline]
    pub fn reset(&mut self, state: &State) {
        self.y = state.y;
    }

    /// Returns the current authentication tag.
    #[inline]
    pub fn current_tag(&self) -> Tag {
        Tag(self.y.to_le_bytes())
    }
}

/// Saved [`Polyval`] state.
#[derive(Clone, Default)]
pub struct State {
    y: FieldElement,
}

#[cfg(feature = "zeroize")]
#[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
impl ZeroizeOnDrop for State {}

impl Drop for State {
    #[inline]
    fn drop(&mut self) {
        #[cfg(feature = "zeroize")]
        {
            self.y.zeroize();
        }
        #[cfg(not(feature = "zeroize"))]
        {
            self.y ^= self.y;
        }
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("State").finish_non_exhaustive()
    }
}
