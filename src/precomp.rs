use crate::{
    backend,
    poly::{Backend, Key, Sealed, State, Tag, BLOCK_SIZE},
};

/// POLYVAL with precomputed powers.
///
/// This backend can process multiple blocks at once, but is
/// larger than the [`Lite`][crate::Lite] backend.
#[derive(Clone)]
pub struct Precomputed(backend::Precomputed);

impl Backend for Precomputed {}
impl Sealed for Precomputed {
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    fn new(key: &Key) -> Self {
        Self(backend::Precomputed::new(key.0))
    }

    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        self.0.update_block(block);
    }

    #[inline]
    fn update_blocks(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
        self.0.update_blocks(blocks);
    }

    #[inline]
    fn tag(&self) -> Tag {
        Tag(self.0.tag())
    }

    #[inline]
    fn export(&self) -> State {
        State { y: self.0.export() }
    }

    #[inline]
    fn reset(&mut self, state: &State) {
        self.0.reset(state.y);
    }
}
