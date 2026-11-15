# muraw-demosaic

The `VNG` and `RCD` demosaic algorithms of [muraw](../muraw), as C
extensions that you build from source. muraw runs every other algorithm
(`BILINEAR`, `EA`, `EA_FAST`, `OPENCV_EA`) without this package, and its
defaults are among those. Install this package only to use `VNG` or `RCD`.

It is not published on PyPI. Install it from a checkout of the repository,
next to muraw:

```bash
git clone https://github.com/mu-files/mufusion.git
cd mufusion
pip install ./muraw-demosaic
```

or straight from GitHub:

```bash
pip install "muraw-demosaic @ git+https://github.com/mu-files/mufusion.git#subdirectory=muraw-demosaic"
```

Building needs a C compiler. macOS (Xcode command line tools) and Linux
(`gcc` and the Python headers, `python3-dev` on Debian) have one. Windows
needs [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

## Enabling RCD

RCD (Ratio Corrected Demosaicing) is licensed under **GPL v3**, which is
separate from muraw's BSD-3-Clause license, so it is not built by default.
To enable it, from a checkout:

1. Rename `muraw-demosaic/c-src/demosaic/rcd.txt` to `muraw-demosaic/c-src/demosaic/rcd.c`
2. Rebuild: `pip install ./muraw-demosaic`

By enabling RCD, you accept the GPL v3 license terms for that component. The
RCD source is based on
[Luis Sanz Rodríguez's implementation](https://github.com/LuisSR/RCD-Demosaicing).

## License

The Python package is released under the BSD 3-Clause License.

- **VNG Demosaicing**: CDDL 1.0, from
  [LibRaw](https://github.com/LibRaw/LibRaw), which offers it under LGPL v2.1
  or CDDL 1.0
- **RCD Demosaicing** (optional): GPL v3

See [LICENSE](LICENSE) and the full texts in [LICENSES](LICENSES).
