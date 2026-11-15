"""exp, exp2, expm1, log, log2, log10, and log1p against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _unary(lib, name, src):
    """``name(src)``.

    NumPy warns on a log of zero or a negative, and on an overflowing exp.
    The value is still the IEEE result.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return getattr(lib, name)(src)


_UNARY_NAMES = ["exp", "exp2", "expm1", "log", "log2", "log10", "log1p"]


def _assert_close(got, want):
    assert got.dtype == want.dtype
    rtol = 2e-3 if want.dtype == np.float16 else 1e-6
    atol = 1e-3 if want.dtype == np.float16 else 1e-7
    np.testing.assert_allclose(
        got.realize(), want, rtol=rtol, atol=atol, equal_nan=True
    )


@pytest.mark.parametrize("name", _UNARY_NAMES)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_exp_log_matches_numpy(name, dtype):
    src = np.linspace(-2.0, 2.0, 46, dtype=dtype).reshape(2, 23)
    got = _unary(mi, name, mi.Array(src))
    want = _unary(np, name, src)
    _assert_close(got, want)


@pytest.mark.parametrize("name", _UNARY_NAMES)
def test_exp_log_specials_match_numpy(name):
    src = np.array(
        [
            -np.inf,
            -2.0,
            -1.0,
            -1.0e-8,
            -0.0,
            0.0,
            1.0e-8,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            10.0,
            12.0,
            16.0,
            88.0,
            100.0,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    got = _unary(mi, name, mi.Array(src))
    want = _unary(np, name, src)
    _assert_close(got, want)
    np.testing.assert_equal(np.signbit(got.realize()), np.signbit(want))


@pytest.mark.parametrize("name", _UNARY_NAMES)
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_exp_log_of_integers_matches_numpy(name, dtype):
    """uint8 gives float16, and uint16 gives float32, as NumPy does."""
    src = np.arange(23, dtype=dtype)
    got = _unary(mi, name, mi.Array(src))
    want = _unary(np, name, src)
    _assert_close(got, want)


@pytest.mark.parametrize("name", _UNARY_NAMES)
def test_exp_log_of_bool_matches_numpy(name):
    src = np.array([[False, True, False], [True, True, False]])
    got = _unary(mi, name, mi.Array(src))
    want = _unary(np, name, src)
    _assert_close(got, want)


def test_lazy_bool_exp_reads_bool_directly():
    src = np.array([[0, 1, 2], [0, 3, 0]], dtype=np.uint8)
    mask = mi.logical_not(mi.Array(src))
    got = mi.exp(mask)
    assert got._node is not None
    assert got._node.inputs == (mask,)
    np.testing.assert_allclose(
        got.realize(),
        np.exp(np.logical_not(src)),
        rtol=2e-3,
        atol=1e-3,
    )
