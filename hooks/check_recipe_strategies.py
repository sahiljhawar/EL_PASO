# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences  # noqa: INP001
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""Guards keeping `el_paso/recipes/strategies.py` and its tests honest.

Two independent checks:

1. `find_violations` — every recipe under `el_paso/recipes/` should save its output through a
   named factory function from `el_paso/recipes/strategies.py` (e.g.
   `arase_xep_strategy(path, mag_field)`) rather than constructing an
   `ep.saving_strategies.*Strategy(...)` directly inline. That keeps the
   mission/satellite/instrument/data-standard literals for every recipe in one place,
   reviewable and reusable, instead of scattered and re-typed across `process_*.py` files.
   Flags any call shaped like `<...>.saving_strategies.<ClassName>(...)`, and any
   `from el_paso.saving_strategies import ...`, in any recipe file except `strategies.py`
   itself.

2. `find_untested_strategy_functions` — every public function in `strategies.py` must be
   exercised by `tests/unittests/test_recipes_strategies.py`: either as an entry in that
   file's `CASES` table (for the common `(path, mag_field[, satellite])` shape), or by a
   dedicated test function for anything that doesn't fit that shape (e.g.
   `rbsp_emfisis_waves_strategy`, which takes no `mag_field`).

New recipes should add (or reuse) a function in `strategies.py`, and add test coverage for it,
instead of triggering either guard. Run directly as a script (also wired up as a pre-commit
hook), or import `find_violations` / `find_untested_strategy_functions` from a test.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "el_paso" / "recipes"
STRATEGIES_MODULE = RECIPES_DIR / "strategies.py"
STRATEGIES_TEST_MODULE = REPO_ROOT / "tests" / "unittests" / "test_recipes_strategies.py"


ALLOWED_VIOLATIONS: dict[Path, set[int]] = {
    RECIPES_DIR / "rbsp" / "process_rbsp_efw_emfisis_density_combined.py": {
        181
    },  # https://github.com/GFZ/EL_PASO/issues/139
}


class Violation(NamedTuple):
    """A disallowed direct `ep.saving_strategies.*` construction found in a recipe file."""

    path: Path
    lineno: int
    detail: str


def _iter_recipe_files() -> list[Path]:
    return sorted(p for p in RECIPES_DIR.rglob("*.py") if p != STRATEGIES_MODULE and "__pycache__" not in p.parts)


def _is_saving_strategies_construction(func: ast.expr) -> bool:
    """True for `<...>.saving_strategies.<Name>(...)`, however the module was imported."""
    if not isinstance(func, ast.Attribute):
        return False
    value = func.value
    if isinstance(value, ast.Attribute):
        return value.attr == "saving_strategies"
    if isinstance(value, ast.Name):
        return value.id == "saving_strategies"
    return False


def _find_violations_in_file(path: Path) -> list[Violation]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_saving_strategies_construction(node.func):
            violations.append(Violation(path, node.lineno, ast.unparse(node.func) + "(...)"))
        elif isinstance(node, ast.ImportFrom) and node.module == "el_paso.saving_strategies":
            names = ", ".join(alias.name for alias in node.names)
            violations.append(Violation(path, node.lineno, f"from el_paso.saving_strategies import {names}"))

    return violations


def find_violations() -> list[Violation]:
    """Return every disallowed direct saving-strategy construction under el_paso/recipes/."""
    violations: list[Violation] = []
    for path in _iter_recipe_files():
        allowed_lines = ALLOWED_VIOLATIONS.get(path, set())
        violations.extend(v for v in _find_violations_in_file(path) if v.lineno not in allowed_lines)
    return violations


def _public_strategy_function_names() -> set[str]:
    """Names of every public (non-underscore) top-level function defined in strategies.py."""
    tree = ast.parse(STRATEGIES_MODULE.read_text(), filename=str(STRATEGIES_MODULE))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")}


def _strategy_function_names_referenced_in_tests() -> set[str]:
    """Names accessed as `rs.<name>` anywhere in the strategies test module."""
    tree = ast.parse(STRATEGIES_TEST_MODULE.read_text(), filename=str(STRATEGIES_TEST_MODULE))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "rs"
    }


def find_untested_strategy_functions() -> set[str]:
    """Public functions in strategies.py with no reference anywhere in its test module."""
    return _public_strategy_function_names() - _strategy_function_names_referenced_in_tests()


def main() -> int:
    """Print every violation/gap found and return a nonzero exit code if there were any."""
    violations = find_violations()
    untested = find_untested_strategy_functions()
    exit_code = 0

    if violations:
        exit_code = 1
        print(  # noqa: T201
            "Recipe files must build saving strategies through el_paso.recipes.strategies, not\n"
            "by constructing ep.saving_strategies.* directly. Add or reuse a named function in\n"
            "el_paso/recipes/strategies.py instead. Violations:\n"
        )
        for violation in violations:
            rel = violation.path.relative_to(REPO_ROOT)
            print(f"  {rel}:{violation.lineno}: {violation.detail}")  # noqa: T201

    if untested:
        exit_code = 1
        strategies_rel = STRATEGIES_MODULE.relative_to(REPO_ROOT)
        tests_rel = STRATEGIES_TEST_MODULE.relative_to(REPO_ROOT)
        print(  # noqa: T201
            f"\nThese public functions in {strategies_rel} have no test coverage in\n"
            f"{tests_rel}. Add a CASES entry (for the common (path, mag_field[, satellite])\n"
            "shape) or a dedicated test function otherwise:\n"
        )
        for name in sorted(untested):
            print(f"  {name}")  # noqa: T201

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
