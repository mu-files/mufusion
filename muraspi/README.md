# muraspi

Example scripts for capturing and converting raw camera data to DNG using [muraw](../muraw/README.md).

## Scripts

| Script | Description | Audience |
|--------|-------------|----------|
| [`zwo_capture.py`](zwo_capture.py) | Live capture from ZWO ASI cameras to DNG | Astrophotography |
| [`picamera2_capture.py`](picamera2_capture.py) | Picamera2 capture & DNG benchmark | Raspberry Pi |

## Documentation

- **[Astronomy / ZWO](docs/README_astro.md)** — ZWO ASI live capture; FITS→DNG conversion is in [mu-dng-converter](../mu-dng-converter/README.md)
- **[Raspberry Pi Camera](docs/README_picamera2.md)** — Picamera2 benchmark & DNG writing

## Installation

muraspi is in the [mufusion repository](https://github.com/mu-files/mufusion),
next to muraw. It is not published to PyPI.

```bash
git clone https://github.com/mu-files/mufusion.git
cd mufusion/muraspi
python3 -m venv venv
venv/bin/pip install -e ".[zwo]"     # ZWO ASI support (zwoasi)
```

For the Raspberry Pi camera, picamera2 comes with Raspberry Pi OS, so the venv
uses the system packages:

```bash
python3 -m venv --system-site-packages venv
venv/bin/pip install -e ".[pi]"      # Raspberry Pi support (picamera2, PiDNG)
```

Each guide below has the details for its camera.

## License

BSD 3-Clause License — see [LICENSE](LICENSE) for details.

## Credits

- **muraw**: https://github.com/mu-files/mufusion/tree/main/muraw
- **PiDNG**: https://github.com/schoolpost/PiDNG
- **picamera2**: https://github.com/raspberrypi/picamera2
- **zwoasi**: https://github.com/python-zwo/python-zwoasi
