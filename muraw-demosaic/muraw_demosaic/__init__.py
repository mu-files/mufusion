# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""VNG and RCD demosaicing for muraw, as C extensions built from source.

muraw imports ``muraw_demosaic._vng`` for ``DemosaicAlgorithm.VNG`` and
``muraw_demosaic._rcd`` for ``DemosaicAlgorithm.RCD``. ``_rcd`` exists only
when the package was built with ``c-src/demosaic/rcd.c``; see README.md.
"""
