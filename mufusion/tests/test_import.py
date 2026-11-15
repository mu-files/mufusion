"""What ``import mufusion`` loads, checked in a fresh interpreter."""

from __future__ import annotations

import json
import subprocess
import sys

_LOADED = """
import json, sys
import mufusion
print(json.dumps(sorted(sys.modules)))
"""


def test_import_loads_neither_muraw_nor_cv2(tmp_path) -> None:
    # Run outside the checkout, where a mufusion/ directory would be
    # imported in place of the installed package.
    result = subprocess.run(
        [sys.executable, "-c", _LOADED],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    loaded = json.loads(result.stdout)
    assert not [name for name in loaded if name == "muraw" or name.startswith("muraw.")]
    assert "cv2" not in loaded


def test_public_classes_name_mufusion() -> None:
    import mufusion as mu

    assert mu.Array.__module__ == "mufusion"
    assert mu.ArrayMeta.__module__ == "mufusion"
    assert mu.EngineTiming.__module__ == "mufusion"
    assert mu.PerfTimer.__module__ == "mufusion"
