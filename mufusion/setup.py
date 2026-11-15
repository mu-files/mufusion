# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

import os
import platform
import sys

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel


def _is_android():
    if sys.platform == "android" or hasattr(sys, "getandroidapilevel"):
        return True
    # Termux reports Linux. The Android dynamic linker is on the device.
    return os.path.exists("/system/bin/linker64") or os.path.exists(
        "/system/bin/linker"
    )


def host_core_binaries():
    """Ship only the host abi3 CoreEngine extension in wheels; sdist keeps all via MANIFEST.in."""
    system = platform.system()
    machine = platform.machine().lower()
    if _is_android():
        raise RuntimeError("Android is not a published CoreEngine target")
    if system == "Windows":
        if machine in ("arm64", "aarch64"):
            raise RuntimeError("Windows on ARM is not a supported CoreEngine target")
        name = "_core_engine.windows-amd64.abi3.pyd"
    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            name = "_core_engine.macos-arm64.abi3.so"
        else:
            name = "_core_engine.macos-x86_64.abi3.so"
    else:
        if machine in ("arm64", "aarch64"):
            name = "_core_engine.linux-aarch64.abi3.so"
        else:
            name = "_core_engine.linux-x86_64.abi3.so"
    return [f"_binaries/{name}"]


class BinaryDistribution(Distribution):
    """A distribution that holds a native binary.

    The package compiles nothing, so setuptools would build a pure wheel,
    tagged ``py3-none-any``, but it holds the host's abi3 CoreEngine binary.
    """

    def has_ext_modules(self):
        return True


class HostWheel(bdist_wheel):
    """A wheel for the host platform and CPython 3.12 or later."""

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        # A universal2 interpreter would tag the wheel universal2; the
        # binary it ships is built for this machine only.
        if platform_tag.endswith("_universal2"):
            machine = "arm64" if platform.machine().lower() == "arm64" else "x86_64"
            platform_tag = platform_tag.removesuffix("universal2") + machine
        return "cp312", "abi3", platform_tag


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": HostWheel},
    include_package_data=False,
    # All package-data lives here. A [tool.setuptools.package-data] table
    # in pyproject.toml replaces this dict, which dropped the host binary
    # from wheels and left only ops.yaml.
    package_data={
        "mufusion._engines": ["catalog/ops.yaml"],
        "mufusion._engines.core": host_core_binaries(),
    },
)
