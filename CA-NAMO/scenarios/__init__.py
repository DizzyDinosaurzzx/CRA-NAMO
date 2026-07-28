"""
地图加载接口（scenarios/ 包）
所有地图文件都放在本目录下，与规划器代码分开。新增地图：
1. 在 ``scenarios/`` 下新建 ``scenario_<地图名>.py``，提供无参数的 ``create()`` 函数
2. 在 ``SCENARIOS`` 中添加一行：``"<地图名>": "scenario_<地图名>"``
   （只写模块名，不带 ``scenarios.`` 前缀，加载时按包内相对导入解析）
切换 ``python main.py`` 默认运行的地图时，只需修改 ``DEFAULT_SCENARIO``
命令行仍可用 ``--scenario <地图名>`` 临时选择其他已注册地图。
"""

from __future__ import annotations
from importlib import import_module
from typing import Any

DEFAULT_SCENARIO = "corridor"
SCENARIOS: dict[str, str] = {
    "corridor": "scenario_corridor",
    "two_doors": "scenario_two_doors",
    "two_doors_hidden_c": "scenario_two_doors_hidden_c",
    "maze_three_movable": "scenario_maze_three_movable",
    "maze_two_movable": "scenario_maze_two_movable",
    "maze_three_a": "scenario_maze_three_a",
    "maze_three_b": "scenario_maze_three_b",
    "maze_four": "scenario_maze_four",
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

def names() -> tuple[str, ...]:
    return tuple(SCENARIOS)

def load(name: str | None = None) -> dict[str, Any]:
    """按注册名创建一份全新的地图数据。"""
    selected = name or DEFAULT_SCENARIO
    module_name = SCENARIOS.get(selected)
    if module_name is None:
        available = ", ".join(names())
        raise ValueError(f"未知地图 {selected!r}；可选地图：{available}")

    # 包内相对导入：注册表里只写模块名，文件实际位于 scenarios/ 下
    module = import_module(f".{module_name}", package=__name__)
    create = getattr(module, "create", None)
    if not callable(create):
        raise TypeError(f"地图模块 {module_name!r} 必须提供无参数的 create() 函数")

    scenario = create()
    if not isinstance(scenario, dict):
        raise TypeError(f"{module_name}.create() 必须返回 dict")

    missing = REQUIRED_FIELDS.difference(scenario)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"地图 {selected!r} 缺少字段：{fields}")

    for field in ("start", "goal"):
        point = scenario[field]
        if len(point) != 2:
            raise ValueError(f"地图 {selected!r} 的 {field} 必须是 (x, y) 单点")
        scenario[field] = (float(point[0]), float(point[1]))

    if scenario["name"] != selected:
        raise ValueError(
            f"地图注册名 {selected!r} 与 create() 返回的名称 "
            f"{scenario['name']!r} 不一致"
        )

    return scenario
