# Third-Party Notices

mu DNG Converter is released under the BSD 3-Clause License (see LICENSE).
The application bundles the components below, each under its own license.
The full license texts are in the `licenses` folder of the application.

## VNG demosaicing (muraw-demosaic)

- **Component**: the `muraw_demosaic._vng` extension, built from
  `muraw-demosaic/c-src/demosaic/vng.c`
- **License**: Common Development and Distribution License 1.0
  (`licenses/LICENSE-CDDL-1.0`)
- **Copyright**: (C) 2008-2024 LibRaw LLC; (C) 1997-2018 Dave Coffin
- **Origin**: [LibRaw](https://github.com/LibRaw/LibRaw), originally from
  dcraw.c by Dave Coffin, with modifications by mu-files
- **Source**: https://github.com/mu-files/mufusion/tree/main/muraw-demosaic/c-src/demosaic

## Arm Optimized Routines (mufusion engine, macOS arm64 and Linux aarch64)

- **Component**: math routines statically linked into the mufusion engine
- **License**: MIT OR Apache-2.0 WITH LLVM-exception, used under the MIT
  License (`licenses/LICENSE-ARM-OPTIMIZED-ROUTINES`)
- **Copyright**: (c) 1999-2022, Arm Limited
- **Source**: https://github.com/ARM-software/optimized-routines
