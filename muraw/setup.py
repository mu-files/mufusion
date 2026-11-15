# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

import os
import sys

import numpy as np
from setuptools import Extension, setup

if sys.platform == "win32":
    common_compile_args = [
        "/O2",
        "/fp:fast",
        "/GL",
    ]
    common_link_args = [
        "/LTCG",
    ]
else:
    is_ci = (
        os.environ.get("CI") == "true"
        or os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CIBUILDWHEEL") == "1"
    )

    common_compile_args = [
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-flto",
        "-fomit-frame-pointer",
        "-fno-strict-aliasing",
    ]

    if not is_ci:
        common_compile_args.extend(
            [
                "-march=native",
                "-mtune=native",
            ]
        )

    common_link_args = [
        "-flto",
    ]

# VNG demosaic extension (stays in public; LGPL/CDDL)
vng_extension = Extension(
    "muraw._vng",
    sources=["c-src/demosaic/vng.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=common_compile_args,
    extra_link_args=common_link_args,
)

# RCD extension - only built if user has renamed rcd.txt to rcd.c (GPL code)
rcd_source = "c-src/demosaic/rcd.c"
rcd_extension = None
if os.path.exists(rcd_source):
    rcd_extension = Extension(
        "muraw._rcd",
        sources=[rcd_source],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    )

ext_modules = [vng_extension]
if rcd_extension:
    ext_modules.append(rcd_extension)

setup(
    ext_modules=ext_modules,
    include_package_data=False,
)
