"""
地图注册与加载接口。

新增地图：
1. 新建一个 ``scenario_<地图名>.py`` 文件，并提供无参数的 ``create()`` 函数。
2. 在 ``SCENARIOS`` 中添加一行：``"<地图名>": "scenario_<地图名>"``。

切换 ``python main.py`` 默认运行的地图时，只需修改 ``DEFAULT_SCENARIO``。
命令行仍可用 ``--scenario <地图名>`` 临时选择其他已注册地图。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


# 默认测试地图：平时切换地图只需要修改这一行。
DEFAULT_SCENARIO = "two_doors"

# 地图名 -> 包含 create() 函数的 Python 模块。
# 新增地图时只需要在这里增加一行注册。
SCENARIOS: dict[str, str] = {
    "two_doors": "scenario_two_doors",
    "two_doors_hidden_c": "scenario_two_doors_hidden_c",
    "maze_three_movable": "scenario_maze_three_movable",
    "maze_two_movable": "scenario_maze_two_movable",
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
    """返回所有已注册地图名，供命令行参数和界面展示使用。"""
    return tuple(SCENARIOS)


def load(name: str | None = None) -> dict[str, Any]:
    """按注册名创建一份全新的地图数据。"""
    selected = name or DEFAULT_SCENARIO
    module_name = SCENARIOS.get(selected)
    if module_name is None:
        available = ", ".join(names())
        raise ValueError(f"未知地图 {selected!r}；可选地图：{available}")

    module = import_module(module_name)
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

    # 起点与终点都是 (x, y) 单点，这里统一校验并归一化成 float 元组。
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
