use crate::{
    backend,
    poly::{Backend, Key, Sealed, State, Tag, BLOCK_SIZE},
};

/// POLYVAL without precomputed powers.
///
/// This backend can only process one block at a time, but is
/// smaller than the [`Precomputed`][crate::Precomputed] backend.
#[derive(Clone)]
pub struct Lite(backend::Lite);

impl Backend for Lite {}
impl Sealed for Lite {
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    fn new(key: &Key) -> Self {
        Self(backend::Lite::new(key.0))
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
