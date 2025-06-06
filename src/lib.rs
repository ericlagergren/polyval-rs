//! POLYVAL per [RFC 8452].
//!
//! [RFC 8452]: https://datatracker.ietf.org/doc/html/rfc8452

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(any(test, doctest, feature = "std")), no_std)]
#![cfg_attr(not(any(feature = "std", test)), deny(clippy::std_instead_of_core))]

mod backend;
pub mod ghash;
mod poly;

use core::slice;

pub use subtle::Choice;
use subtle::ConstantTimeEq;

pub use crate::poly::{Polyval, PolyvalLite};

/// The size in bytes of a POLYVAL (or GHASH) key.
pub const KEY_SIZE: usize = 16;

/// The size in bytes of a POLYVAL (or GHASH) block.
pub const BLOCK_SIZE: usize = 16;

/// An authentication tag.
#[derive(Copy, Clone, Debug)]
pub struct Tag(pub(crate) [u8; 16]);

impl ConstantTimeEq for Tag {
    #[inline]
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl From<Tag> for [u8; 16] {
    #[inline]
    fn from(tag: Tag) -> Self {
        tag.0
    }
}

// See https://doc.rust-lang.org/std/primitive.slice.html#method.as_chunks
#[inline(always)]
const fn as_blocks(blocks: &[u8]) -> (&[[u8; BLOCK_SIZE]], &[u8]) {
    #[allow(clippy::arithmetic_side_effects)]
    let len_rounded_down = (blocks.len() / BLOCK_SIZE) * BLOCK_SIZE;
    // SAFETY: The rounded-down value is always the same or
    // smaller than the original length, and thus must be
    // in-bounds of the slice.
    let (head, tail) = unsafe { blocks.split_at_unchecked(len_rounded_down) };
    let new_len = head.len() / BLOCK_SIZE;
    // SAFETY: We cast a slice of `new_len * N` elements into
    // a slice of `new_len` many `N` elements chunks.
    let head = unsafe { slice::from_raw_parts(head.as_ptr().cast(), new_len) };
    (head, tail)
}

macro_rules! impl_state {
    ($name:ident, $endian:ident) => {
        /// Saved hash state.
        #[derive(Clone, Default)]
        pub struct $name {
            pub(crate) y: $crate::backend::FieldElement,
        }

        #[cfg(feature = "zeroize")]
        #[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
        impl ::zeroize::ZeroizeOnDrop for $name {}

        impl Drop for $name {
            #[inline]
            fn drop(&mut self) {
                cfg_if::cfg_if! {
                    if #[cfg(feature = "zeroize")] {
                        ::zeroize::Zeroize::zeroize(&mut self.y);
                    } else {
                        self.y = ::core::hint::black_box(Default::default());
                    }
                }
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    };
}
pub(crate) use impl_state;

macro_rules! impl_hash {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident($inner:ty);
    ) => {
        $(#[$meta])*
        #[derive(Clone)]
        $vis struct $name($inner);

        impl $name {
            /// Creates a new hash instance.
            ///
            /// It returns `None` if `key` is all zeroes.
            #[inline]
            pub fn new(key: &[u8; $crate::KEY_SIZE]) -> Option<Self> {
                use ::subtle::ConstantTimeEq;

                if bool::from(key.ct_eq(&[0; $crate::KEY_SIZE])) {
                    None
                } else {
                    Some(Self::new_unchecked(key))
                }
            }

            /// Creates a hash instance from a known non-zero
            /// key.
            ///
            /// # Warning
            ///
            /// Only use this method if `key` is known to be
            /// non-zero. Using an all zero key fixes the output
            /// to zero, regardless of the input.
            #[inline]
            pub fn new_unchecked(key: &[u8; $crate::KEY_SIZE]) -> Self {
                Self(<$inner>::new(key))
            }

            /// Writes a single block to the running hash.
            #[inline]
            pub fn update_block(&mut self, block: &[u8; $crate::BLOCK_SIZE]) {
                self.0.update_block(block);
            }

            /// Writes one or more blocks to the running hash.
            #[inline]
            pub fn update_blocks(&mut self, blocks: &[[u8; $crate::BLOCK_SIZE]]) {
                self.0.update_blocks(blocks);
            }

            /// Writes one or more blocks to the running hash.
            ///
            /// If the length of `blocks` is non-zero, it's
            /// padded to the lowest multiple of
            /// [`BLOCK_SIZE`][crate::BLOCK_SIZE].
            //#[inline]
            pub fn update_padded(&mut self, blocks: &[u8]) {
                let (head, tail) = $crate::as_blocks(blocks);
                if !head.is_empty() {
                    self.update_blocks(head);
                }
                if !tail.is_empty() {
                    let mut block = [0u8; $crate::BLOCK_SIZE];
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "The compiler can prove the slice is in bounds."
                    )]
                    block[..tail.len()].copy_from_slice(tail);
                    self.update_block(&block);
                }
            }

            /// Returns the current authentication tag.
            #[inline]
            pub fn tag(self) -> $crate::Tag {
                $crate::Tag(self.0.tag())
            }

            /// Reports whether the current authentication tag matches
            /// `expected_tag`.
            #[inline]
            pub fn verify(self, expected_tag: &$crate::Tag) -> ::subtle::Choice {
                ::subtle::ConstantTimeEq::ct_eq(&self.tag(), expected_tag)
            }

            /// Exports the current state.
            #[inline]
            #[cfg(feature = "experimental")]
            #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
            pub fn export(&self) -> State {
                let y = self.0.export();
                State { y }
            }

            /// Resets the hash to `state`.
            #[inline]
            #[cfg(feature = "experimental")]
            #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
            pub fn reset(&mut self, state: &State) {
                self.0.reset(state.y)
            }

            /// Returns the current authentication tag without
            /// consuming the hash.
            #[inline]
            #[cfg(feature = "experimental")]
            #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
            pub fn current_tag(&self) -> $crate::Tag {
                $crate::Tag(self.0.tag())
            }
        }

        #[cfg(feature = "zeroize")]
        #[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
        impl ::zeroize::ZeroizeOnDrop for $name {}

        impl Drop for $name {
            #[inline]
            fn drop(&mut self) {
                #[cfg(feature = "zeroize")]
                // SAFETY: `self` is "flat" data and is not used
                // after this point.
                unsafe {
                    zeroize::zeroize_flat_type(self);
                }
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.debug_struct(stringify!($name)).finish_non_exhaustive()
            }
        }
    };
}
pub(crate) use impl_hash;
