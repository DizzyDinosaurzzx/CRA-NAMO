"""大型仓储场景：60x40 m 货架仓库，横向通道被杂物与门板阻断。"""

from __future__ import annotations

from shapely.geometry import box

from config import Config
from obstacle import MovableObstacle, StaticObstacle


def create():
    """按参考图构建 60 m x 40 m 仓储地图。"""

    workspace = box(0.0, 0.0, 60.0, 40.0)
    wall_t = 0.75

    static = [
        # 仓库外墙
        StaticObstacle(box(0.0, 0.0, 60.0, wall_t), "outer_bottom"),
        StaticObstacle(box(0.0, 40.0 - wall_t, 60.0, 40.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, wall_t, 40.0), "outer_left"),
        StaticObstacle(box(60.0 - wall_t, 0.0, 60.0, 40.0), "outer_right"),
    ]

    # 货架列 1--7；列间纵向空隙构成横向通道 A/B/C/D，与地图一致
    shelf_columns = (
        (7.0, 10.0),
        (13.5, 17.0),
        (21.0, 26.0),
        (29.5, 34.0),
        (38.0, 42.5),
        (46.0, 50.0),
        (54.0, 58.0),
    )

    # 顶部货架排仅存在于第 1--6 列；第 7 列为发货区
    for column, (x0, x1) in enumerate(shelf_columns[:6], start=1):
        static.append(
            StaticObstacle(box(x0, 32.0, x1, 38.5), f"shelf_{column}_top")
        )

    # 通道 B 左侧为上下分开的两组货架
    for column, (x0, x1) in enumerate(shelf_columns[:4], start=1):
        static.extend(
            (
                StaticObstacle(
                    box(x0, 24.5, x1, 29.0), f"shelf_{column}_upper"
                ),
                StaticObstacle(
                    box(x0, 16.0, x1, 21.5), f"shelf_{column}_middle"
                ),
            )
        )

    # 第 5--7 列按参考图为横跨通道 B 的整排货架
    for column, (x0, x1) in enumerate(shelf_columns[4:], start=5):
        static.append(
            StaticObstacle(box(x0, 16.0, x1, 29.0),
                           f"shelf_{column}_middle_tall")
        )

    # 通道 D 两侧的货架排
    for column, (x0, x1) in enumerate(shelf_columns, start=1):
        static.extend(
            (
                StaticObstacle(
                    box(x0, 9.0, x1, 13.0), f"shelf_{column}_lower"
                ),
                StaticObstacle(
                    box(x0, 2.5, x1, 6.5), f"shelf_{column}_bottom"
                ),
            )
        )

    # 据标注参考图添加的货架连接；个别区块横跨两条相邻通道
    shelf_bridges = (
        # 横向通道 A（y=29.0--32.0）
        (1, 29.0, 32.0, "A"),
        (3, 29.0, 32.0, "A"),
        (6, 29.0, 32.0, "A"),

        # 横向通道 B（y=21.5--24.5）
        (2, 21.5, 24.5, "B"),
        (4, 21.5, 24.5, "B"),

        # 横向通道 C（y=13.0--16.0）
        (1, 13.0, 16.0, "C"),
        (2, 13.0, 16.0, "C"),
        (3, 13.0, 16.0, "C"),

        # 横向通道 D（y=6.5--9.0）
        (1, 6.5, 9.0, "D"),
        (3, 6.5, 9.0, "D"),
        (4, 6.5, 9.0, "D"),
        (5, 6.5, 9.0, "D"),
        (7, 6.5, 9.0, "D"),

        # 将标注的底部货架与内墙边齐平封住
        (2, wall_t, 2.5, "floor"),
        (3, wall_t, 2.5, "floor"),
        (5, wall_t, 2.5, "floor"),
        (6, wall_t, 2.5, "floor"),
        (7, wall_t, 2.5, "floor"),
    )
    for column, y0, y1, region in shelf_bridges:
        x0, x1 = shelf_columns[column - 1]
        static.append(
            StaticObstacle(box(x0, y0, x1, y1),
                           f"shelf_{column}_bridge_{region}")
        )

    # 通道 A 最右端的小型固定货架
    static.append(
        StaticObstacle(box(58.0, 29.0, 59.25, 32.0), "shelf_shipping_corner")
    )

    # 以下 difficulty 均按 material_mu_rho(material)*l*d*h*G 与所在行的 material/l/d/h
    # 对齐重算（原硬编码值与该公式偏差 1.4×~8.2×，不在 [0.8, 1.2] 容忍带内）；
    # l/d/h 本身未动，即便个别高度（BX_A_2 的 250、PL(H) 的 10）看着奇怪也保留原样。
    movable = [
        # 左侧物体
        MovableObstacle(
            x=19.2,
            y=10.5,
            l=3.9,
            d=0.1,
            h=0.8,
            theta=0.5,
            material="box",
            difficulty=42.85,
            oid="BX-A",
        ),
        MovableObstacle(
            x=19.2,
            y=25.0,
            l=3.9,
            d=0.2,
            h=0.8,
            theta=1.0,
            material="box",
            difficulty=85.7,
            oid="BX_A_1",
        ),
        MovableObstacle(
            x=19.2,
            y=21.0,
            l=3.5,
            d=0.9,
            h=250,
            theta=0.5,
            material="cart",
            difficulty=34764.188,
            oid="BX_A_2",
        ),
        MovableObstacle(
            x=11.0,
            y=23.0,
            l=2.7,
            d=2.7,
            h=10,
            theta=0.8,
            material="box",
            difficulty=10012.086,
            oid="PL(H)",
        ),

        # 中部物体
        MovableObstacle(
            x=36.0,
            y=23.0,
            l=3.0,
            d=0.5,
            h=1.1,
            theta=0.23,
            material="cart",
            difficulty=72.839,
            oid="CT",
        ),
        MovableObstacle(
            x=32.5,
            y=14.5,
            l=4.0,
            d=2.3,
            h=0.15,
            theta=0.0,
            material="pallet",
            difficulty=942.231,
            oid="PL(L)",
        ),
        MovableObstacle(
            x=40.0,
            y=14.3,
            l=3.0,
            d=0.5,
            h=2.0,
            theta=0.8,
            material="loaded_pallet",
            difficulty=5109.048,
            oid="SC",
        ),

        # 右侧物体
        MovableObstacle(
            x=52.0,
            y=16.0,
            l=3.4,
            d=1.9,
            h=0.8,
            theta=0.0,
            material="box",
            difficulty=709.773,
            oid="BX_D",
        ),
        MovableObstacle(
            x=52.0,
            y=18.7,
            l=3.4,
            d=0.2,
            h=0.8,
            theta=0.0,
            material="box",
            difficulty=74.713,
            oid="BX_D_1",
        ),
        MovableObstacle(
            x=52.0,
            y=25.0,
            l=3.2,
            d=1.3,
            h=0.8,
            theta=0.3,
            material="box",
            difficulty=457.068,
            oid="BX_D_2",
        ),

        # 其余编号物体
        MovableObstacle(
            x=16.0,
            y=7.35,
            l=0.5,
            d=0.5,
            h=0.3,
            theta=0.0,
            material="wooden_crate",
            difficulty=19.865,
            oid="33",
        ),
        MovableObstacle(
            x=16.0,
            y=8.05,
            l=0.5,
            d=0.5,
            h=0.3,
            theta=0.0,
            material="wooden_crate",
            difficulty=19.865,
            oid="33_1",
        ),

        # 以下六个此前 h 是复制拖填时留下的离谱占位值（250~5000 米），
        # difficulty 也照抄了上面 33 号 wooden_crate 的 66.218，跟 material="cart"
        # 对不上；现按 material_mu_rho("cart")*l*d*h*G 重新算，h 统一取
        # MATERIAL_HEIGHT["cart"]=1.1（仓储小车典型高度）。
        MovableObstacle(
            x=33.5,
            y=30.5,
            l=1.5,
            d=1.5,
            h=1.1,
            theta=0.70,
            material="cart",
            difficulty=109.259,
            oid="35",
        ),

        MovableObstacle(
            x=29.5,
            y=1.8,
            l=0.5,
            d=0.5,
            h=1.1,
            theta=0.70,
            material="cart",
            difficulty=12.14,
            oid="34_1",
        ),
        MovableObstacle(
            x=33.5,
            y=1.8,
            l=0.5,
            d=0.5,
            h=1.1,
            theta=0.5,
            material="cart",
            difficulty=12.14,
            oid="36_1",
        ),
        MovableObstacle(
            x=46.8,
            y=14.6,
            l=0.1,
            d=1.8,
            h=1.1,
            theta=0.3,
            material="cart",
            difficulty=8.741,
            oid="38",
        ),
        MovableObstacle(
            x=48.8,
            y=14.6,
            l=0.1,
            d=1.8,
            h=1.1,
            theta=0.0,
            material="cart",
            difficulty=8.741,
            oid="38_2",
        ),
        MovableObstacle(
            x=48.0,
            y=7.8,
            l=0.1,
            d=1.6,
            h=1.1,
            theta=0.4,
            material="cart",
            difficulty=7.77,
            oid="39",
        ),
    ]

    # 图中所示的收货区起点与发货区终点
    start = (5.0, 1.25)
    goal = (56.0, 33.0)

    return {
        "workspace": workspace,
        "static": static,
        "movable": movable,
        "start": start,
        "goal": goal,
        "cfg": Config(
            grid_step=0.5,
            conn_radius=0.75,
            se2_cell=0.5,
        ),
    }
