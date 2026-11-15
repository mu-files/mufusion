# mufusion

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

A lazy image compute graph for 2D images, with a NumPy interface. Array
code builds a graph; `.realize()` runs it in native code, fusing the ops
so a pipeline makes one pass over the image where it can, instead of one
pass and one temporary array per op.

```python
import numpy as np
import mufusion as mu

x = mu.Array(np.random.rand(3000, 4000, 3).astype(np.float32))
y = mu.clip((x - 0.1) * 1.2, 0.0, 1.0) @ np.eye(3, dtype=np.float32)
out = y.realize()  # a NumPy array
```

- Functions and methods follow NumPy: slicing, `astype`, arithmetic,
  `mu.pad`, `mu.take`, `mu.concatenate`, and the rest. A call NumPy
  accepts gives NumPy's result or raises `NotImplementedError`.
- `mu.Array` is a handle, not a NumPy array. `.realize()` and
  `np.asarray(x)` return NumPy arrays, so NumPy and OpenCV stay one call
  away.
- uint8, uint16, float16 and float32. Integer arithmetic saturates where
  NumPy wraps.
- The only dependency is NumPy. The native engine ships in the wheel.

## Installation

```bash
pip install mufusion
```

[muraw](https://pypi.org/project/muraw/), the DNG and raw image library,
is built on mufusion.
