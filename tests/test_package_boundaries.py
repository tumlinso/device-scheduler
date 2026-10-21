"""Keep the scheduling authority usable without an environment or transport."""

import ast
import sys
from pathlib import Path


def test_core_imports_only_itself_and_the_standard_library():
    source = Path(__file__).resolve().parents[1] / "packages/device-scheduler-core/src/device_scheduler_core"
    violations = []
    for path in source.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".", 1)[0]]
            else:
                continue
            for root in roots:
                if root not in sys.stdlib_module_names and root != "device_scheduler_core":
                    violations.append(f"{path.name}:{node.lineno}: {root}")
    assert not violations, "Core acquired an environment or transport dependency: " + ", ".join(violations)
