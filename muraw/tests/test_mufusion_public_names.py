"""muraw's package code uses only mufusion's public names."""

from __future__ import annotations

import ast
from pathlib import Path

import muraw

_PUBLIC_MODULES = {"mufusion"}


def _mufusion_imports(path: Path) -> list[tuple[int, str]]:
    """Each module of mufusion that ``path`` imports, with its line."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found += [
            (node.lineno, name) for name in names if name.split(".")[0] == "mufusion"
        ]
    return found


def test_muraw_imports_only_public_mufusion_modules() -> None:
    package = Path(muraw.__file__).parent
    private = [
        f"{path.relative_to(package)}:{line}: {name}"
        for path in sorted(package.rglob("*.py"))
        for line, name in _mufusion_imports(path)
        if name not in _PUBLIC_MODULES
    ]
    message = "muraw imports a private mufusion module:\n" + "\n".join(private)
    assert not private, message
