//! Experimental features NOT covered by semver.

#![cfg(feature = "experimental")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental")))]

pub use crate::poly::State;
use crate::poly::{Polyval, Sealed, Tag};

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

// /// TODO
// #[derive(Debug)]
// pub struct Ghash<B = Precomputed>(B);

/// TODO
#[derive(Clone)]
pub struct Precomputed {
    poly: crate::lite::Lite,
    pow: [crate::backend::FieldElement; 8],
}
