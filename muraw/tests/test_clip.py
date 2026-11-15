"""minimum, maximum, and clip against NumPy.

A Python float beside an integer array is float32, where NumPy gives
float64. ``clip`` of an unsigned array treats a Python integer outside
that dtype's range as no bound on that edge.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _minimum(lib, x1, x2):
    """``minimum(x1, x2)``."""
    return lib.minimum(x1, x2)


def _maximum(lib, x1, x2):
    """``maximum(x1, x2)``."""
    return lib.maximum(x1, x2)


def _clip(lib, x, a_min, a_max):
    """``clip(x, a_min, a_max)``."""
    return lib.clip(x, a_min, a_max)


def _check(got, want):
    assert got.dtype == want.dtype
    realized = got.realize()
    np.testing.assert_allclose(realized, want, rtol=0, atol=0, equal_nan=True)
    finite = np.isfinite(want)
    both_nan = np.isnan(realized) & np.isnan(want)
    compared = finite | both_nan
    np.testing.assert_array_equal(
        np.signbit(realized[compared]), np.signbit(want[compared])
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_integer_minimum_and_maximum_match_numpy(dtype):
    first = np.arange(46, dtype=dtype).reshape(2, 23)
    second = (first * np.array(3, dtype=dtype)) % np.array(7, dtype=dtype)
    got = _minimum(mi, mi.Array(first), mi.Array(second))
    _check(got, _minimum(np, first, second))
    assert got._node.op == "minimum"
    got = _maximum(mi, mi.Array(first), mi.Array(second))
    _check(got, _maximum(np, first, second))
    assert got._node.op == "maximum"


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_scalar_minimum_matches_numpy_from_either_side(dtype):
    values = np.arange(23, dtype=dtype).reshape(1, 23)
    array = mi.Array(values)
    _check(_minimum(mi, array, 3), _minimum(np, values, 3))
    _check(_minimum(mi, 3, array), _minimum(np, 3, values))
    assert _minimum(mi, array, 3)._node.op == "minimum_scalar"


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_float_minimum_and_maximum_keep_nan_and_signed_zero(dtype):
    first = np.array(
        [
            1.0,
            -1.0,
            0.0,
            -0.0,
            np.nan,
            2.0,
            np.inf,
            -np.inf,
            3.0,
            -4.0,
            0.5,
            -0.5,
            8.0,
            8.0,
            -2.0,
            0.0,
            -0.0,
            5.0,
            np.nan,
            1.5,
            -1.5,
            9.0,
            0.0,
        ],
        dtype=dtype,
    )
    second = np.array(
        [
            -2.0,
            4.0,
            -0.0,
            0.0,
            3.0,
            np.nan,
            1.0,
            1.0,
            -np.inf,
            np.inf,
            0.5,
            1.5,
            -3.0,
            3.0,
            -2.0,
            0.0,
            -0.0,
            -5.0,
            np.nan,
            -0.0,
            0.0,
            2.0,
            1.0,
        ],
        dtype=dtype,
    )
    with np.errstate(invalid="ignore"):
        for function in (_minimum, _maximum):
            got = function(mi, mi.Array(first), mi.Array(second))
            # NumPy's float16 minimum of +0 and -0 returns the first
            # zero. The op returns -0, NumPy's float32 result narrowed.
            precise = function(np, first.astype(np.float32), second.astype(np.float32))
            _check(got, precise.astype(dtype))


def test_minimum_broadcasts_a_channel():
    image = np.arange(30, dtype=np.uint8).reshape(2, 5, 3)
    channel = np.arange(10, dtype=np.uint8).reshape(2, 5, 1)
    _check(
        _minimum(mi, mi.Array(image), mi.Array(channel)),
        _minimum(np, image, channel),
    )


@pytest.mark.parametrize("name", ["minimum", "maximum"])
def test_a_python_int_outside_an_unsigned_dtype_raises(name):
    values = np.arange(4, dtype=np.uint8)
    function = getattr(mi, name)
    with pytest.raises(OverflowError, match="out of bounds for uint8"):
        function(mi.Array(values), -1)
    with pytest.raises(OverflowError, match="out of bounds for uint16"):
        function(mi.Array(values.astype(np.uint16)), 70000)


def test_a_float_constant_beside_an_integer_is_float32():
    values = np.arange(8, dtype=np.uint8)
    got = _minimum(mi, mi.Array(values), 2.5)
    want = _minimum(np, values.astype(np.float32), np.float32(2.5))
    _check(got, want)


def test_a_wider_integer_constant_is_not_cast_to_the_array():
    values = np.arange(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="int16"):
        _minimum(mi, mi.Array(values), np.int16(-1))


def test_minimum_and_maximum_of_two_bools_are_logical():
    first = np.array([True, True, False, False])
    second = np.array([True, False, True, False])
    _check(_minimum(mi, mi.Array(first), mi.Array(second)), _minimum(np, first, second))
    _check(_maximum(mi, mi.Array(first), mi.Array(second)), _maximum(np, first, second))
    assert _minimum(mi, mi.Array(first), mi.Array(second))._node.op == "logical_and"
    assert _maximum(mi, mi.Array(first), mi.Array(second))._node.op == "logical_or"


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
def test_clip_matches_numpy(dtype):
    values = np.array([0, 1, 5, 10, 40, 200], dtype=dtype)
    _check(_clip(mi, mi.Array(values), 2, 20), _clip(np, values, 2, 20))


def test_clip_of_an_unsigned_array_ignores_a_python_int_outside_the_dtype():
    values = np.array([0, 1, 5, 255], dtype=np.uint8)
    array = mi.Array(values)
    got = _clip(mi, array, -1, 10)
    _check(got, np.minimum(values, np.uint8(10)))
    assert mi.clip(array, -1, 300) is array
    with pytest.raises(OverflowError, match="out of bounds for uint8"):
        mi.clip(array, -10, -1)


def test_clip_with_min_above_max_is_the_upper_bound():
    values = np.array([0, 1, 5, 255], dtype=np.uint8)
    _check(_clip(mi, mi.Array(values), 10, 1), _clip(np, values, 10, 1))


def test_clip_none_bounds_and_identity():
    values = np.array([-1.0, -0.0, 0.0, 2.0], dtype=np.float32)
    array = mi.Array(values)
    assert mi.clip(array) is array
    _check(mi.clip(array, None, 0), np.clip(values, None, 0))
    # A lower bound of +0 turns -0 into +0. NumPy's clip keeps -0.
    bounded = np.minimum(np.maximum(values, np.float32(0)), np.float32(1))
    _check(mi.clip(array, min=0, max=1), bounded)


def test_clip_array_bounds_broadcast():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    low = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float32)
    high = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    _check(
        _clip(mi, mi.Array(values), mi.Array(low), mi.Array(high)),
        _clip(np, values, low, high),
    )


def test_clip_a_float_constant_beside_an_integer_is_float32():
    values = np.arange(8, dtype=np.uint8)
    got = mi.clip(mi.Array(values), 0.5, 2.5)
    want = np.clip(values.astype(np.float32), np.float32(0.5), np.float32(2.5))
    _check(got, want)


def test_clip_of_bool_stays_bool_for_bool_bounds():
    values = np.array([True, False, True])
    _check(_clip(mi, mi.Array(values), False, True), _clip(np, values, False, True))
    with pytest.raises(NotImplementedError):
        mi.clip(mi.Array(values), 0, 1)


def test_clip_argument_errors_match_numpy():
    values = np.array([1.0, 2.0], dtype=np.float32)
    array = mi.Array(values)
    with pytest.raises(TypeError, match="a_max"):
        mi.clip(array, 1)
    with pytest.raises(TypeError, match="a_max"):
        np.clip(values, 1)
    with pytest.raises(ValueError, match="forbidden"):
        mi.clip(array, 0, 1, min=0)
    with pytest.raises(ValueError, match="forbidden"):
        np.clip(values, 0, 1, min=0)
    with pytest.raises(NotImplementedError, match="realize"):
        mi.clip(array, 0, 1, out=np.zeros_like(values))
