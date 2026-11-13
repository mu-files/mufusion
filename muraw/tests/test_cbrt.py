"""cbrt against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _unary(lib, src):
    """``cbrt(src)``."""
    return lib.cbrt(src)


def _assert_close(got, want):
    assert got.dtype == want.dtype
    rtol = 2e-3 if want.dtype == np.float16 else 1e-6
    atol = 1e-3 if want.dtype == np.float16 else 1e-7
    np.testing.assert_allclose(
        got.realize(), want, rtol=rtol, atol=atol, equal_nan=True
    )


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_cbrt_matches_numpy(dtype):
    src = np.linspace(-8.0, 8.0, 46, dtype=dtype).reshape(2, 23)
    got = _unary(mi, mi.Array(src))
    want = _unary(np, src)
    _assert_close(got, want)


def test_cbrt_specials_match_numpy():
    src = np.array(
        [
            -np.inf,
            -27.0,
            -8.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            8.0,
            27.0,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    got = _unary(mi, mi.Array(src))
    want = _unary(np, src)
    _assert_close(got, want)
    np.testing.assert_equal(np.signbit(got.realize()), np.signbit(want))


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_cbrt_of_integers_matches_numpy(dtype):
    """uint8 gives float16, and uint16 gives float32, as NumPy does."""
    src = np.arange(23, dtype=dtype)
    got = _unary(mi, mi.Array(src))
    want = _unary(np, src)
    _assert_close(got, want)


def test_cbrt_of_bool_matches_numpy():
    src = np.array([[False, True, False], [True, True, False]])
    got = _unary(mi, mi.Array(src))
    want = _unary(np, src)
    _assert_close(got, want)


def test_lazy_bool_cbrt_reads_bool_directly():
    src = np.array([[0, 1, 2], [0, 3, 0]], dtype=np.uint8)
    mask = mi.logical_not(mi.Array(src))
    got = mi.cbrt(mask)
    assert got._node is not None
    assert got._node.inputs == (mask,)
    np.testing.assert_allclose(
        got.realize(),
        np.cbrt(np.logical_not(src)),
        rtol=2e-3,
        atol=1e-3,
    )
