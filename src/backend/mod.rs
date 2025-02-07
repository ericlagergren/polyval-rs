mod aarch64;
mod generic;
mod soft;
mod x86;

use core::ops::{BitXor, BitXorAssign, Mul, MulAssign};

#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

cfg_if::cfg_if! {
    if #[cfg(feature = "soft")] {
        use soft as imp;
    } else if #[cfg(target_arch = "aarch64")] {
        use aarch64 as imp;
    } else if #[cfg(any(target_arch = "x86", target_arch="x86_64"))] {
        use x86 as imp;
    } else {
        use soft as imp;
    }
}

#[cfg(all(
    test,
    not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))
))]
pub(crate) use imp::gf128_mul;

use crate::poly::BLOCK_SIZE;

/// An element in the field
///
/// ```text
/// x^128 + x^127 + x^126 + x^121 + 1
/// ```
#[derive(Copy, Clone, Default, Debug)]
#[repr(transparent)]
pub struct FieldElement<const LE: bool = true>(imp::FieldElement);

impl<const LE: bool> FieldElement<LE> {
    /// Creates a field element from bytes.
    #[inline]
    pub fn from_bytes(data: &[u8; BLOCK_SIZE]) -> Self {
        if LE {
            Self(imp::FieldElement::from_le_bytes(data))
        } else {
            Self(imp::FieldElement::from_be_bytes(data))
        }
    }

    /// Converts the field element to bytes.
    #[inline]
    pub fn to_bytes(self) -> [u8; BLOCK_SIZE] {
        if LE {
            self.0.to_le_bytes()
        } else {
            self.0.to_be_bytes()
        }
    }

    /// Multiplies `self` with the series of field elements in
    /// `blocks`.
    #[inline]
    pub fn polymul_series(self, pow: &[Self; 8], blocks: &[[u8; BLOCK_SIZE]]) -> Self {
        if imp::supported() {
            // SAFETY: `FieldElement` and `imp::FieldElement`
            // have the same layout in memory. The pointer came
            // from a reference, so it is safe to dereference.
            let pow = unsafe { &*(pow as *const [FieldElement<LE>; 8]).cast() };
            // SAFETY: `supported` is true, which means we can
            // call `polymul_series`.
            let fe = unsafe { self.0.polymul_series::<LE>(pow, blocks) };
            Self(fe)
        } else {
            let pow = pow.map(|x| x.0.into_generic());
            let fe = self.0.into_generic().polymul_series::<LE>(&pow, blocks);
            Self(imp::FieldElement::from_generic(fe))
        }
    }
}

impl<const LE: bool> BitXor for FieldElement<LE> {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl<const LE: bool> BitXorAssign for FieldElement<LE> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl<const LE: bool> Mul for FieldElement<LE> {
    type Output = Self;

    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    fn mul(self, rhs: Self) -> Self {
        if imp::supported() {
            // SAFETY: `supported` is true, which means we can
            // call `polymul`.
            let fe = unsafe { self.0.polymul(rhs.0) };
            Self(fe)
        } else {
            let fe = self.0.into_generic() * rhs.0.into_generic();
            Self(imp::FieldElement::from_generic(fe))
        }
    }
}

impl<const LE: bool> MulAssign for FieldElement<LE> {
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for FieldElement {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

/// POLYVAL without precomputed powers for shorter inputs.
pub struct Lite<const LE: bool = true> {
    /// The running state.
    y: FieldElement<LE>,
    /// The key.
    h: FieldElement<LE>,
}

impl<const LE: bool> Lite<LE> {
    #[inline]
    pub fn new(key: FieldElement<LE>) -> Self {
        Self {
            y: FieldElement::default(),
            h: key,
        }
    }

    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        let fe = FieldElement::from_bytes(block);
        self.y = (self.y ^ fe) * self.h;
    }

    #[inline]
    pub fn update_blocks(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
        for block in blocks {
            self.update_block(block);
        }
    }

    #[inline]
    pub fn tag(&self) -> [u8; 16] {
        self.y.to_bytes()
    }

    #[inline]
    pub fn export(&self) -> FieldElement<LE> {
        self.y
    }

    #[inline]
    pub fn reset(&mut self, y: FieldElement<LE>) {
        self.y = y;
    }
}

impl<const LE: bool> Clone for Lite<LE> {
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
impl<const LE: bool> ZeroizeOnDrop for Lite<LE> {}

impl<const LE: bool> Drop for Lite<LE> {
    #[inline]
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

/// POLYVAL with precomputed powers for longer inputs.
pub struct Precomputed<const LE: bool = true> {
    /// The running state.
    y: FieldElement<LE>,
    /// Precomputed table of powers of `h` for batched
    /// computations.
    pow: [FieldElement<LE>; 8],
}

impl<const LE: bool> Precomputed<LE> {
    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn new(key: FieldElement<LE>) -> Self {
        let pow = {
            let h = key;
            let mut prev = h;
            let mut pow: [FieldElement<LE>; 8] = Default::default();
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

    #[inline]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn update_block(&mut self, block: &[u8; BLOCK_SIZE]) {
        let fe = FieldElement::from_bytes(block);
        self.y = (self.y ^ fe) * self.pow[7];
    }

    #[inline]
    pub fn update_blocks(&mut self, blocks: &[[u8; BLOCK_SIZE]]) {
        self.y = self.y.polymul_series(&self.pow, blocks);
    }

    #[inline]
    pub fn tag(&self) -> [u8; 16] {
        self.y.to_bytes()
    }

    #[inline]
    pub fn export(&self) -> FieldElement<LE> {
        self.y
    }

    #[inline]
    pub fn reset(&mut self, y: FieldElement<LE>) {
        self.y = y;
    }
}

impl<const LE: bool> Clone for Precomputed<LE> {
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
impl<const LE: bool> ZeroizeOnDrop for Precomputed<LE> {}

impl<const LE: bool> Drop for Precomputed<LE> {
    #[inline]
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
