"""floor_divide and // against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import mufusion as mu


def _quot(lib, x1, x2):
    """``x1 // x2``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return lib.floor_divide(x1, x2)


def _by_operator(lib, x1, x2):
    """``x1 // x2`` through the operator."""
    del lib
    with np.errstate(divide="ignore", invalid="ignore"):
        return x1 // x2


def _check(got, want):
    assert got.dtype == want.dtype
    realized = got.realize()
    np.testing.assert_allclose(realized, want, rtol=0, atol=0, equal_nan=True)
    finite = np.isfinite(want)
    np.testing.assert_array_equal(
        np.signbit(realized[finite]), np.signbit(want[finite])
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_integer_floor_divide_matches_numpy(dtype):
    first = np.arange(46, dtype=dtype).reshape(2, 23)
    second = np.arange(46, dtype=dtype).reshape(2, 23)
    second = (second * np.array(3, dtype=dtype)) % np.array(7, dtype=dtype)
    got = _quot(mu, mu.Array(first), mu.Array(second))
    want = _quot(np, first, second)
    _check(got, want)
    assert got._node.op == "floor_divide"


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_integer_operator_and_reverse_match_numpy(dtype):
    values = np.arange(23, dtype=dtype).reshape(1, 23)
    array = mu.Array(values)
    _check(_by_operator(mu, array, 3), _by_operator(np, values, 3))
    _check(_by_operator(mu, 5, array), _by_operator(np, 5, values))
    assert (array // 3)._node.op == "floor_div_scalar"
    assert (5 // array)._node.attrs["reverse"] == 1


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_float_floor_divide_matches_numpy(dtype):
    first = np.array(
        [
            100.0,
            1.0,
            -1.0,
            0.0,
            -0.0,
            1.0,
            np.inf,
            -np.inf,
            1.0,
            np.nan,
            -8.0,
            7.5,
            -7.5,
            3.0,
            3.0,
            8.0,
            8.0,
            1.0,
            -1.0,
            4.0,
            -4.0,
            9.0,
            2.0,
        ],
        dtype=dtype,
    ).reshape(1, 23)
    second = np.array(
        [
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.0,
            2.0,
            2.0,
            np.inf,
            2.0,
            2.0,
            2.5,
            2.5,
            -2.0,
            2.0,
            -3.0,
            3.0,
            3.0,
            3.0,
            -0.5,
            0.5,
            np.nan,
            -0.0,
        ],
        dtype=dtype,
    ).reshape(1, 23)
    _check(_quot(mu, mu.Array(first), mu.Array(second)), _quot(np, first, second))
    _check(_by_operator(mu, mu.Array(first), 2), _by_operator(np, first, 2))
    _check(_by_operator(mu, 2, mu.Array(second)), _by_operator(np, 2, second))


def test_channel_broadcast_matches_numpy():
    full = np.arange(30, dtype=np.uint8).reshape(2, 5, 3)
    mono = np.arange(10, dtype=np.uint8).reshape(2, 5, 1) % np.uint8(4)
    _check(_quot(mu, mu.Array(full), mu.Array(mono)), _quot(np, full, mono))


def test_float_constant_beside_an_integer_is_float32():
    values = np.arange(8, dtype=np.uint8).reshape(2, 4)
    got = _by_operator(mu, mu.Array(values), 2.5)
    want = _quot(np, values.astype(np.float32), np.float32(2.5))
    assert got.dtype == np.float32
    _check(got, want)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_integer_out_of_range_raises(dtype):
    values = mu.Array(np.arange(4, dtype=dtype))
    with pytest.raises(OverflowError, match=f"out of bounds for {dtype}"):
        values // -1
    with pytest.raises(OverflowError, match=f"out of bounds for {dtype}"):
        -1 // values
    with pytest.raises(OverflowError, match=f"out of bounds for {dtype}"):
        mu.floor_divide(values, -1)


def test_bool_floor_divide_is_not_supported():
    values = mu.Array(np.array([True, False, True]))
    with pytest.raises(NotImplementedError):
        values // values
