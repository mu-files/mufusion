"""sin, cos, tan, arcsin/asin, arccos/acos, arctan/atan, and arctan2/atan2 against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import mufusion as mu


def _unary(lib, name, src):
    """``name(src)``.

    NumPy warns on an infinite angle and on ``arcsin`` or ``arccos``
    outside ``[-1, 1]``. The value is still NaN.
    """
    with np.errstate(invalid="ignore"):
        return getattr(lib, name)(src)


def _arctan2(lib, x1, x2):
    """``arctan2(x1, x2)``."""
    with np.errstate(invalid="ignore", over="ignore"):
        return lib.arctan2(x1, x2)


def _atan2(lib, x1, x2):
    """``atan2(x1, x2)``."""
    with np.errstate(invalid="ignore", over="ignore"):
        return lib.atan2(x1, x2)


_UNARY_NAMES = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "asin",
    "arccos",
    "acos",
    "arctan",
    "atan",
]


@pytest.mark.parametrize("name", _UNARY_NAMES)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_unary_trig_matches_numpy(name, dtype):
    src = np.linspace(-3.5, 3.5, 46, dtype=dtype).reshape(2, 23)
    got = _unary(mu, name, mu.Array(src))
    want = _unary(np, name, src)
    assert got.dtype == want.dtype
    rtol = 2e-3 if dtype == "float16" else 1e-6
    atol = 1e-3 if dtype == "float16" else 1e-7
    np.testing.assert_allclose(
        got.realize(), want, rtol=rtol, atol=atol, equal_nan=True
    )


@pytest.mark.parametrize("name", _UNARY_NAMES)
def test_unary_trig_specials_match_numpy(name):
    src = np.array(
        [
            -np.inf,
            -1.0e8,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            1.0e8,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    got = _unary(mu, name, mu.Array(src))
    want = _unary(np, name, src)
    assert got.dtype == want.dtype
    np.testing.assert_allclose(
        got.realize(), want, rtol=1e-6, atol=1e-7, equal_nan=True
    )


@pytest.mark.parametrize("name", _UNARY_NAMES)
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_unary_trig_of_integers_matches_numpy(name, dtype):
    """uint8 gives float16, and uint16 gives float32, as NumPy does."""
    src = np.arange(23, dtype=dtype)
    got = _unary(mu, name, mu.Array(src))
    want = _unary(np, name, src)
    assert got.dtype == want.dtype
    rtol = 2e-3 if dtype == "uint8" else 1e-6
    atol = 1e-3 if dtype == "uint8" else 1e-7
    np.testing.assert_allclose(
        got.realize(), want, rtol=rtol, atol=atol, equal_nan=True
    )


@pytest.mark.parametrize("name", _UNARY_NAMES)
def test_unary_trig_of_bool_matches_numpy(name):
    src = np.array([[False, True, False], [True, True, False]])
    got = _unary(mu, name, mu.Array(src))
    want = _unary(np, name, src)
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got.realize(), want, rtol=2e-3, atol=1e-3)


def test_lazy_bool_trig_reads_bool_directly():
    src = np.array([[0, 1, 2], [0, 3, 0]], dtype=np.uint8)
    mask = mu.logical_not(mu.Array(src))
    got = mu.sin(mask)
    assert got._node is not None
    assert got._node.inputs == (mask,)
    np.testing.assert_allclose(
        got.realize(), np.sin(np.logical_not(src)), rtol=2e-3, atol=1e-3
    )

    pair = mu.arctan2(mask, mask)
    assert pair._node is not None
    assert pair._node.inputs == (mask, mask)
    np.testing.assert_allclose(
        pair.realize(),
        np.arctan2(np.logical_not(src), np.logical_not(src)),
        rtol=2e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("dtype", ["float16", "float32", "uint8", "uint16", "bool"])
def test_arctan2_matches_numpy(dtype):
    if dtype == "bool":
        first = np.array([[False, True, False, True], [True, True, False, False]])
        second = np.array([[True, True, False, False], [False, True, True, False]])
    elif dtype in ("uint8", "uint16"):
        first = np.arange(23, dtype=dtype)
        second = np.arange(1, 24, dtype=dtype)
    else:
        first = np.linspace(-5.0, 5.0, 23, dtype=dtype)
        second = np.linspace(4.0, -4.0, 23, dtype=dtype)
    got = _arctan2(mu, mu.Array(first), mu.Array(second))
    want = _arctan2(np, first, second)
    assert got.dtype == want.dtype
    rtol = 2e-3 if want.dtype == np.float16 else 1e-6
    atol = 1e-3 if want.dtype == np.float16 else 1e-7
    np.testing.assert_allclose(got.realize(), want, rtol=rtol, atol=atol)


def test_atan2_is_arctan2():
    first = np.linspace(-3.0, 3.0, 12, dtype=np.float32).reshape(3, 4)
    second = np.linspace(2.0, -2.0, 12, dtype=np.float32).reshape(3, 4)
    got = _atan2(mu, mu.Array(first), mu.Array(second))
    want = _atan2(np, first, second)
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0)


def test_arctan2_specials_and_quadrants_match_numpy():
    first = np.array(
        [
            0.0,
            0.0,
            -0.0,
            -0.0,
            1.0,
            -1.0,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            1.0,
            -1.0,
            np.inf,
            -np.inf,
            np.nan,
            1.0,
        ],
        dtype=np.float32,
    )
    second = np.array(
        [
            0.0,
            -0.0,
            0.0,
            -0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            np.inf,
            np.inf,
            np.inf,
            -np.inf,
            1.0,
            np.nan,
        ],
        dtype=np.float32,
    )
    got = _arctan2(mu, mu.Array(first), mu.Array(second)).realize()
    want = _arctan2(np, first, second)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=0, equal_nan=True)
    known = ~np.isnan(want)
    np.testing.assert_array_equal(np.signbit(got[known]), np.signbit(want[known]))


def test_arctan2_broadcast_and_scalars_match_numpy():
    column = np.linspace(-2.0, 3.0, 4, dtype=np.float32).reshape(4, 1)
    row = np.linspace(-3.0, 4.0, 7, dtype=np.float32).reshape(1, 7)
    got = _arctan2(mu, mu.Array(column), mu.Array(row))
    want = _arctan2(np, column, row)
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0)

    image = np.linspace(-2.0, 2.0, 30, dtype=np.float32).reshape(2, 5, 3)
    per_channel = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    got_hwc = _arctan2(mu, mu.Array(image), mu.Array(per_channel))
    want_hwc = _arctan2(np, image, per_channel)
    np.testing.assert_allclose(got_hwc.realize(), want_hwc, rtol=1e-6, atol=0)

    for scalar in (2.5, -1, 3):
        got_right = _arctan2(mu, mu.Array(image), scalar)
        want_right = _arctan2(np, image, scalar)
        assert got_right.dtype == want_right.dtype
        np.testing.assert_allclose(got_right.realize(), want_right, rtol=1e-6, atol=0)

        got_left = _arctan2(mu, scalar, mu.Array(image))
        want_left = _arctan2(np, scalar, image)
        assert got_left.dtype == want_left.dtype
        np.testing.assert_allclose(got_left.realize(), want_left, rtol=1e-6, atol=0)

    u8 = np.arange(1, 9, dtype=np.uint8)
    for scalar in (-1, 2, 300):
        got_u8 = _arctan2(mu, mu.Array(u8), scalar)
        want_u8 = _arctan2(np, u8, scalar)
        assert got_u8.dtype == want_u8.dtype
        np.testing.assert_allclose(got_u8.realize(), want_u8, rtol=2e-3, atol=1e-3)


def test_arctan2_python_float_beside_uint8_is_float32():
    """``arctan2(uint8, 2.0)`` is float64 in NumPy and float32 here."""
    src = np.arange(1, 5, dtype=np.uint8)
    got = _arctan2(mu, mu.Array(src), 2.0)
    assert got.dtype == np.float32
    want = _arctan2(np, src.astype(np.float32), np.float32(2.0))
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0)


def test_arctan2_rejects_string_with_type_error():
    src = np.arange(4, dtype=np.uint8)
    with pytest.raises(TypeError):
        _arctan2(mu, mu.Array(src), "bad")
    with pytest.raises(TypeError):
        _arctan2(np, src, "bad")


def test_arctan2_int64_is_not_implemented():
    src = np.arange(4, dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        mu.arctan2(mu.Array(src), np.int64(2))
