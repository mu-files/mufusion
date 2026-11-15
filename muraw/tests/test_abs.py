"""abs and unary minus against NumPy.

A uint8 or uint16 ``-x`` saturates to 0. NumPy wraps.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _abs(lib, x):
    """``abs(x)``."""
    return lib.abs(x)


def _negative(lib, x):
    """Unary ``-x``."""
    return lib.negative(x)


def _by_operator(lib, x):
    """Unary ``-x`` through the operator."""
    del lib
    return -x


def _check(got, want):
    assert got.dtype == want.dtype
    realized = got.realize()
    np.testing.assert_allclose(realized, want, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(np.signbit(realized), np.signbit(want))


_EDGES = np.array(
    [
        -2.5,
        -0.0,
        0.0,
        1.5,
        np.inf,
        -np.inf,
        np.nan,
        -np.nan,
        -1.0,
        4.0,
        -4.0,
        0.5,
        -0.5,
        8.0,
        -8.0,
        3.0,
        -3.0,
        2.0,
        -2.0,
        9.0,
        -9.0,
        1.0,
        0.0,
    ],
    dtype=np.float32,
)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "bool"])
def test_abs_of_an_integer_or_bool_is_the_array(dtype):
    if dtype == "bool":
        values = np.array([False, True, False, True, True, False, True, False])
    else:
        values = np.arange(23, dtype=dtype).reshape(1, 23) % np.array(17, dtype=dtype)
    source = mi.Array(values)
    got = _abs(mi, source)
    assert got is source
    _check(got, _abs(np, values))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_abs_clears_the_sign(dtype):
    values = _EDGES.astype(dtype)
    got = _abs(mi, mi.Array(values))
    _check(got, _abs(np, values))
    assert got._node.op == "abs"


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_negative_of_an_unsigned_integer_saturates_to_zero(dtype):
    """Every sample saturates to 0. NumPy wraps."""
    values = np.arange(23, dtype=dtype).reshape(1, 23)
    got = _negative(mi, mi.Array(values))
    _check(got, np.zeros_like(values))
    assert got._node.op == "negative"
    by_operator = _by_operator(mi, mi.Array(values))
    _check(by_operator, np.zeros_like(values))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_negative_flips_the_sign(dtype):
    values = _EDGES.astype(dtype)
    array = mi.Array(values)
    got = _negative(mi, array)
    _check(got, _negative(np, values))
    _check(_by_operator(mi, array), _by_operator(np, values))
    assert got._node.op == "negative"


def test_negative_of_a_bool_raises():
    values = np.array([True, False])
    with pytest.raises(TypeError, match="boolean negative"):
        _negative(mi, mi.Array(values))
    with pytest.raises(TypeError, match="boolean negative"):
        _negative(np, values)
    with pytest.raises(TypeError, match="boolean negative"):
        _by_operator(mi, mi.Array(values))
