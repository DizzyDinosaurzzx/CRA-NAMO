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

# -------- 加载地图 -------- #

def names() -> tuple[str, ...]:
    return tuple(sorted(
        m.name for m in pkgutil.iter_modules([str(_PKG_DIR)])
        if not m.name.startswith("_") and m.name != "registry"
    ))
def load(name: str | None = None) -> dict[str, Any]:
    selected = name or DEFAULT_SCENARIO
    available = names()
    if selected not in available:
        raise ValueError(f"未知地图 {selected!r}；可选地图：{', '.join(available)}")

    module = import_module(f".{selected}", package=_PACKAGE)
    create = getattr(module, "create", None)
    if not callable(create):
        raise TypeError(f"地图模块 {selected!r} 必须提供无参数的 create() 函数")

    scenario = create()
    if not isinstance(scenario, dict):
        raise TypeError(f"{selected}.create() 必须返回 dict")

    # 检查必填字段
    missing = REQUIRED_FIELDS.difference(scenario)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"地图 {selected!r} 缺少字段：{fields}")

    # 起点/终点标准化为浮点坐标
    for field in ("start", "goal"):
        point = scenario[field]
        if len(point) != 2:
            raise ValueError(f"地图 {selected!r} 的 {field} 必须是 (x, y) 单点")
        scenario[field] = (float(point[0]), float(point[1]))

    scenario["name"] = selected
    return scenario
