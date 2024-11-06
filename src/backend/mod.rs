mod aarch64;
pub mod generic;
mod soft;
mod x86;

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

#[cfg(all(test, not(target_arch = "aarch64")))]
pub(crate) use imp::gf128_mul;
/// An element in the field
///
/// ```text
/// x^128 + x^127 + x^126 + x^121 + 1
/// ```
pub(crate) use imp::FieldElement;
