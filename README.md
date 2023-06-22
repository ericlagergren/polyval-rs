# polyhash

[![Docs][docs-img]][docs-link]

This crate implements POLYVAL per [RFC 8452](https://datatracker.ietf.org/doc/html/rfc8452).

The universal hash function POLYVAL is the byte-wise reverse of
GHASH.

## Installation

```bash
[dependencies]
polyhash = "0.1"
```

## Performance

The ARMv8 and x86-64 assembly backends run at about 0.17 cycles
per byte. The x86-64 implementation requires SSE2 and PCLMULQDQ
instructions. The ARMv8 implementation requires NEON and PMULL.

The defualt Rust implementation will be selected if the CPU does
not support either assembly implementation. (This implementation
can also be selected with the `soft` feature.) It is much 
slower at around 7 cycles per byte.

## Security

### Disclosure

This project uses full disclosure. If you find a security bug in
an implementation, please e-mail me or create a GitHub issue.

### Disclaimer

You should only use cryptography libraries that have been
reviewed by cryptographers or cryptography engineers. While I am
a cryptography engineer, I'm not your cryptography engineer, and
I have not had this project reviewed by any other cryptographers.

[//]: # (badges)

[docs-img]: https://docs.rs/polyhash/badge.svg
[docs-link]: https://docs.rs/polyhash
