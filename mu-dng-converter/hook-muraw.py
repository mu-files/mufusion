# PyInstaller hook for muraw

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# muraw / mufusion plus deps (excluding those with dedicated hooks: astropy, imagecodecs)
DEPS = [
    "muraw",
    "mufusion",
    "cv2",
    "numpy",
    "tifffile",
    "defusedxml",
    "click",
    "setproctitle",
]

hiddenimports = []
binaries = []

for dep in DEPS:
    hiddenimports += collect_submodules(dep)
    binaries += collect_dynamic_libs(dep)

# mufusion loads its engine extension from _engines/core/_binaries by path and
# reads _engines/catalog/ops.yaml, so import analysis finds neither.
binaries += collect_dynamic_libs("mufusion", search_patterns=["*.so", "*.pyd"])
datas = collect_data_files("mufusion")
