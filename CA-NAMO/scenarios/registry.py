"""场景注册与加载：自动发现场景模块并统一校验。"""

from __future__ import annotations
import pkgutil
from importlib import import_module
from pathlib import Path
from typing import Any

# ---- 默认地图 ---- #
DEFAULT_SCENARIO = "two_doors"
REQUIRED_FIELDS = {
    "workspace",
    "static",
    "movable",
    "start",
    "goal",
    "cfg",
}

_PKG_DIR = Path(__file__).resolve().parent
_PACKAGE = __name__.rpartition(".")[0] or "scenarios"

# -------- 地图加载 -------- #

def names() -> tuple[str, ...]:
    """列出所有可用场景名。"""
    return tuple(sorted(
        m.name for m in pkgutil.iter_modules([str(_PKG_DIR)])
        if not m.name.startswith("_") and m.name != "registry"
    ))
def load(name: str | None = None) -> dict[str, Any]:
    """加载指定场景（缺省用默认地图），校验字段并归一化。"""
    selected = name or DEFAULT_SCENARIO
    available = names()
    if selected not in available:
        raise ValueError(f"Unknown map {selected!r}; available: {', '.join(available)}")

    module = import_module(f".{selected}", package=_PACKAGE)
    create = getattr(module, "create", None)
    if not callable(create):
        raise TypeError(f"Map module {selected!r} must provide a parameterless create() function")

    scenario = create()
    if not isinstance(scenario, dict):
        raise TypeError(f"{selected}.create() must return a dict")

    # 校验必需字段
    missing = REQUIRED_FIELDS.difference(scenario)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Map {selected!r} is missing fields: {fields}")

    # 起终点统一为 float 坐标
    for field in ("start", "goal"):
        point = scenario[field]
        if len(point) != 2:
            raise ValueError(f"Map {selected!r}: {field} must be a single (x, y) point")
        scenario[field] = (float(point[0]), float(point[1]))

    scenario["name"] = selected
    return scenario
