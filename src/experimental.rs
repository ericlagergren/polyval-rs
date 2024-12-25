//! Experimental features NOT covered by semver.

#![cfg(feature = "experimental")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental")))]

pub use super::poly::State;
use super::poly::{Polyval, Sealed, Tag};

impl Polyval {
    /// Exports the current state.
    #[inline]
    pub fn export(&self) -> State {
        self.0.export()
    }

    /// Resets the hash to `state`.
    #[inline]
    pub fn reset(&mut self, state: &State) {
        self.0.reset(state)
    }

    /// Returns the current authentication tag without consuming
    /// `self`.
    #[inline]
    pub fn current_tag(&self) -> Tag {
        self.0.tag()
    }
}
