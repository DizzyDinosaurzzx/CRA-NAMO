"""地图注册与加载逻辑"""

from __future__ import annotations
from importlib import import_module
from typing import Any

# ---- 注册表 ---- #
DEFAULT_SCENARIO = "two_doors"
SCENARIOS: dict[str, str] = {
    "corridor": "scenario_corridor",
    "two_doors": "scenario_two_doors",
    "two_doors_hidden_c": "scenario_two_doors_hidden_c",
    "maze_three_movable": "scenario_maze_three_movable",
    "maze_two_movable": "scenario_maze_two_movable",
    "maze_three_a": "scenario_maze_three_a",
    "maze_three_b": "scenario_maze_three_b",
    "maze_four": "scenario_maze_four",
    "maze_to_house": "scenario_maze_to_house",
}
REQUIRED_FIELDS = {
    "name",
    "workspace",
    "static",
    "movable",
    "start",
    "goal",
    "cfg",
}

# -------- 加载地图 -------- #

def names() -> tuple[str, ...]:
    return tuple(SCENARIOS)

def load(name: str | None = None) -> dict[str, Any]:
    selected = name or DEFAULT_SCENARIO
    module_name = SCENARIOS.get(selected)
    if module_name is None:
        available = ", ".join(names())
        raise ValueError(f"未知地图 {selected!r}；可选地图：{available}")

    # 包内相对导入：注册表只存模块名，运行时按 scenarios 包子模块解析
    module = import_module(f".{module_name}", package="scenarios")
    create = getattr(module, "create", None)
    if not callable(create):
        raise TypeError(f"地图模块 {module_name!r} 必须提供无参数的 create() 函数")

    scenario = create()
    if not isinstance(scenario, dict):
        raise TypeError(f"{module_name}.create() 必须返回 dict")

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

    # 防止注册名与内部名不一致
    if scenario["name"] != selected:
        raise ValueError(
            f"地图注册名 {selected!r} 与 create() 返回的名称 "
            f"{scenario['name']!r} 不一致"
        )

    return scenario
