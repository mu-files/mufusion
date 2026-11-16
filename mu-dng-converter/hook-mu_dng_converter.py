# PyInstaller hook for mu_dng_converter

from PyInstaller.utils.hooks import collect_data_files

# The UI (web/) and the app icon (assets/) are package data.
datas = collect_data_files("mu_dng_converter")
