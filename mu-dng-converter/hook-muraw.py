# PyInstaller hook for muraw

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

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
