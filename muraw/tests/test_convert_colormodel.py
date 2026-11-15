"""RGB and gray through ``convert_colormodel``.

The other model names are accepted and not computed yet.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi

_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _image(dtype: str, shape: tuple[int, ...] = (5, 9, 3)) -> np.ndarray:
    rng = np.random.default_rng(3)
    if dtype == "uint8":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if dtype == "uint16":
        return rng.integers(0, 65536, shape, dtype=np.uint16)
    return rng.uniform(0.0, 1.0, shape).astype(dtype)


def _store(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """``values``, a float32 sum, stored as ``dtype``."""
    if dtype == np.dtype(np.float32):
        return values.astype(np.float32)
    if dtype == np.dtype(np.float16):
        return values.astype(np.float16)
    stored = np.floor(np.abs(values) + np.float32(0.5))
    return np.clip(stored, 0, np.iinfo(dtype).max).astype(dtype)


def _gray(rgb: np.ndarray) -> np.ndarray:
    """``rgb`` as gray, channels last."""
    values = rgb.astype(np.float32)
    luma = values @ _WEIGHTS
    return _store(luma, rgb.dtype)


def _check(got: mi.Array, want: np.ndarray) -> None:
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    realized = got.realize()
    if want.dtype == np.dtype(np.float32):
        np.testing.assert_allclose(realized, want, rtol=1e-6, atol=1e-6)
    elif want.dtype == np.dtype(np.float16):
        np.testing.assert_allclose(
            realized.astype(np.float32), want.astype(np.float32), rtol=1e-3, atol=1e-3
        )
    else:
        np.testing.assert_array_equal(realized, want)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float16", "float32"])
def test_rgb_to_gray_matches_the_weights(dtype: str) -> None:
    rgb = _image(dtype)
    got = mi.convert_colormodel(rgb, "RGB", "GRAY")
    assert got._node is not None
    assert got._node.op == "convert_colormodel"
    _check(got, _gray(rgb))


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float16", "float32"])
def test_gray_to_rgb_copies_the_channel(dtype: str) -> None:
    gray = _image(dtype, (5, 9))
    got = mi.convert_colormodel(gray, "GRAY", "RGB")
    assert got.shape == (5, 9, 3)
    _check(got, np.stack([gray, gray, gray], axis=-1))


def test_channels_first_rgb_becomes_gray_without_a_channel_axis() -> None:
    rgb = _image("float32").transpose(2, 0, 1)
    got = mi.convert_colormodel(mi.Array(rgb, channel_axis=0), "RGB", "GRAY")
    assert got.shape == rgb.shape[1:]
    _check(got, _gray(np.moveaxis(rgb, 0, -1)))


def test_a_stack_drops_the_channel_axis() -> None:
    rgb = _image("uint8", (2, 4, 6, 3))
    got = mi.convert_colormodel(rgb, "RGB", "GRAY")
    assert got.shape == (2, 4, 6)
    assert got.meta.channel_axis == 0
    _check(got, _gray(rgb))


def test_one_channel_gray_is_the_same_input() -> None:
    gray = _image("float32", (4, 6))
    last = mi.convert_colormodel(gray[..., None], "GRAY", "RGB")
    first = mi.convert_colormodel(mi.Array(gray[None], channel_axis=0), "GRAY", "RGB")
    assert last.shape == (4, 6, 3)
    assert first.shape == (4, 6, 3)
    _check(last, np.stack([gray, gray, gray], axis=-1))
    _check(first, np.stack([gray, gray, gray], axis=-1))


def test_the_same_model_returns_the_array() -> None:
    rgb = mi.Array(_image("uint8"))
    assert mi.convert_colormodel(rgb, "RGB", "RGB") is rgb
    assert mi.convert_colormodel(rgb, "HSV", "HSV") is rgb
    assert mi.convert_colormodel(rgb, "RGB", "RGB")._node is rgb._node


def test_a_view_keeps_its_placement() -> None:
    image = mi.Array(_image("float32", (8, 10, 3)))
    window = image[2:6, 1:7]
    got = mi.convert_colormodel(window, "RGB", "GRAY")
    assert (got.meta.origin, got.meta.canvas, got.meta.oob_valid) == (
        window.meta.origin,
        window.meta.canvas,
        window.meta.oob_valid,
    )
    _check(got, _gray(np.asarray(window)))


def test_an_unknown_name_is_rejected() -> None:
    rgb = _image("uint8")
    with pytest.raises(ValueError, match="src_model"):
        mi.convert_colormodel(rgb, "rgb", "GRAY")
    with pytest.raises(ValueError, match="dst_model"):
        mi.convert_colormodel(rgb, "RGB", "Bgr")


def test_a_bool_is_not_supported() -> None:
    rgb = np.zeros((2, 3, 3), dtype=bool)
    with pytest.raises(NotImplementedError, match="bool"):
        mi.convert_colormodel(rgb, "RGB", "GRAY")


def test_a_fourth_channel_is_not_supported() -> None:
    rgb = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="3 channels"):
        mi.convert_colormodel(rgb, "RGB", "GRAY")


def test_channels_on_axis_1_names_moveaxis() -> None:
    rgb = np.zeros((4, 3, 5), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="moveaxis"):
        mi.convert_colormodel(mi.Array(rgb, channel_axis=1), "RGB", "GRAY")


def test_a_stack_that_is_not_nhwc_names_the_layout() -> None:
    rgb = np.zeros((2, 3, 4, 5), dtype=np.float32)
    with pytest.raises(NotImplementedError, match=r"\(N, H, W, C\)"):
        mi.convert_colormodel(mi.Array(rgb, channel_axis=1), "RGB", "GRAY")


def test_a_model_other_than_gray_is_not_supported_yet() -> None:
    rgb = _image("float32")
    with pytest.raises(NotImplementedError, match="YUV"):
        mi.convert_colormodel(rgb, "RGB", "YUV")


def test_a_1d_array_is_not_supported() -> None:
    gray = np.zeros(8, dtype=np.float32)
    with pytest.raises(NotImplementedError, match=r"\(8,\)"):
        mi.convert_colormodel(gray, "GRAY", "RGB")


def test_uint8_primaries_round_half_away_from_zero() -> None:
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 255
    rgb[0, 1, 1] = 255
    rgb[0, 2, 2] = 255
    got = mi.convert_colormodel(rgb, "RGB", "GRAY").realize()
    np.testing.assert_array_equal(got, _gray(rgb))
    assert got[0, 0] == np.uint8(76)
    assert got[0, 1] == np.uint8(150)
    assert got[0, 2] == np.uint8(29)
