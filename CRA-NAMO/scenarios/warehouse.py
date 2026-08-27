from __future__ import annotations

import math

from shapely.geometry import box

from config import Config
from dynamics import Event, MoveTo, at_time
from llm_difficulty import friction_force, material_mu_rho
from obstacle import MovableObstacle, StaticObstacle


def create():
    """60 m x 40 m hard warehouse map shown in the reference diagram."""

    # 坐标约定：原点在仓库左下角，x 轴向右，y 轴向上；单位均为米。
    workspace = box(0.0, 0.0, 60.0, 40.0)
    wall_t = 0.75

    # ======================================================================
    # 固定墙体与货架
    # StaticObstacle.rect 的参数依次表示：中心 (x, y)、长边 l、厚度 d、
    # 长边相对 x 轴的逆时针角度 theta。水平墙为 0°，竖直墙为 90°。
    # ======================================================================
    walls = [
        # 仓库外框：下、上、左、右四面边界墙。
        StaticObstacle.rect(
            x=30.0,
            y=wall_t / 2.0,
            l=60.0,
            d=wall_t,
            theta=math.radians(0.0),
            name="outer_bottom",
        ),
        StaticObstacle.rect(
            x=30.0,
            y=40.0 - wall_t / 2.0,
            l=60.0,
            d=wall_t,
            theta=math.radians(0.0),
            name="outer_top",
        ),
        StaticObstacle.rect(
            x=wall_t / 2.0,
            y=20.0,
            l=40.0,
            d=wall_t,
            theta=math.radians(90.0),
            name="outer_left",
        ),
        StaticObstacle.rect(
            x=60.0 - wall_t / 2.0,
            y=20.0,
            l=40.0,
            d=wall_t,
            theta=math.radians(90.0),
            name="outer_right",
        ),
    ]

    # 七列货架的 x 范围，从左到右编号 1--7。
    # 货架在 y 方向留出四条横向通道：
    # A: 29.0--32.0，B: 21.5--24.5，C: 13.0--16.0，D: 6.5--9.0。
    shelf_columns = (
        (8.5, 10.0),    # 第 1 列
        (15, 16.50),     # 第 2 列
        (21.0, 22.5),   # 第 3 列
        (32, 33.50),     # 第 4 列
        (38.5, 40),     # 第 5 列
        (47.5,49.0),    # 第 6 列
        (54.5, 56.0),   # 第 7 列（上部是发货区）
    )

    # 顶部货架段（y=32--40）：只生成第 1--6 列，第 7 列留作发货区。
    for column, (x0, x1) in enumerate(shelf_columns[:6], start=1):
        walls.append(
            StaticObstacle.rect(
                x=(x0 + x1) / 2.0,
                y=(32.0 + 40) / 2.0,
                l=40 - 32.0,
                d=x1 - x0,
                theta=math.radians(90.0),
                name=f"shelf_{column}_top",
            )
        )

    # 左半区第 1--4 列：通道 B 上下各是一段独立货架。
    # upper 位于 y=24.5--29.0，middle 位于 y=16.0--21.5。
    for column, (x0, x1) in enumerate(shelf_columns[:4], start=1):
        walls.extend(
            (
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(24.5 + 29.0) / 2.0,
                    l=29.0 - 24.5,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_upper",
                ),
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(16.0 + 21.5) / 2.0,
                    l=21.5 - 16.0,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_middle",
                ),
            )
        )

    # 右半区第 5--7 列：货架从 y=16.0 连续延伸至 29.0，跨过通道 B。
    for column, (x0, x1) in enumerate(shelf_columns[4:], start=5):
        walls.append(
            StaticObstacle.rect(
                x=(x0 + x1) / 2.0,
                y=(16.0 + 29.0) / 2.0,
                l=29.0 - 16.0,
                d=x1 - x0,
                theta=math.radians(90.0),
                name=f"shelf_{column}_middle_tall",
            )
        )

    # 所有七列在通道 D 两侧的货架：
    # lower 位于 y=9.0--13.0，bottom 位于 y=2.5--6.5。
    for column, (x0, x1) in enumerate(shelf_columns, start=1):
        walls.extend(
            (
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(9.0 + 13.0) / 2.0,
                    l=13.0 - 9.0,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_lower",
                ),
                StaticObstacle.rect(
                    x=(x0 + x1) / 2.0,
                    y=(2.5 + 6.5) / 2.0,
                    l=6.5 - 2.5,
                    d=x1 - x0,
                    theta=math.radians(90.0),
                    name=f"shelf_{column}_bottom",
                ),
            )
        )

    # 参考图中被货架封住的通道段。
    # 每个条目格式为：(货架列编号, y 下界, y 上界, 区域标签)。
    # 后续循环会用对应货架列的 x 范围生成一段竖直连接墙。
    shelf_bridges = (
        # 通道 A（y=29.0--32.0）中被封住的位置。
        (1, 29.0, 32.0, "A"),
        (3, 29.0, 32.0, "A"),
        (6, 29.0, 32.0, "A"),

        # 通道 B（y=21.5--24.5）中被封住的位置。
        (2, 21.5, 24.5, "B"),
        (4, 21.5, 24.5, "B"),

        # 通道 C（y=13.0--16.0）中被封住的位置。
        (1, 13.0, 16.0, "C"),
        (2, 13.0, 16.0, "C"),
        (3, 13.0, 16.0, "C"),

        # 通道 D（y=6.5--9.0）中被封住的位置。
        (1, 6.5, 9.0, "D"),
        (3, 6.5, 9.0, "D"),
        (4, 6.5, 9.0, "D"),
        (5, 6.5, 9.0, "D"),
        (7, 6.5, 9.0, "D"),

        # 从仓库下边界内沿延伸到 y=2.5，使指定底部货架贴住外墙。
        (2, wall_t, 2.5, "floor"),
        (3, wall_t, 2.5, "floor"),
        (5, wall_t, 2.5, "floor"),
        (6, wall_t, 2.5, "floor"),
        (7, wall_t, 2.5, "floor"),
    )
    for column, y0, y1, region in shelf_bridges:
        x0, x1 = shelf_columns[column - 1]
        walls.append(
            StaticObstacle.rect(
                x=(x0 + x1) / 2.0,
                y=(y0 + y1) / 2.0,
                l=y1 - y0,
                d=x1 - x0,
                theta=math.radians(90.0),
                name=f"shelf_{column}_bridge_{region}",
            )
        )

    # 发货区右侧的四个小型固定货架，按从上到下的顺序排列。
    # 右上角小货架：x=58.7--59.25，y=28.5--29.0。
    walls.append(
        StaticObstacle.rect(
            x=(58 + 59.25) / 2.0,
            y=(28.5 + 29.0) / 2.0,
            l=59.25 - 58,
            d=29.0 - 28.5,
            theta=math.radians(0.0),
            name="shelf_shipping_corner",
        )
    )
    # 左侧小货架：x=56.0--57.0，y=27.0--27.5。
    walls.append(
        StaticObstacle.rect(
            x=(56.0 + 58.0) / 2.0,
            y=(27.0 + 27.5) / 2.0,
            l=58.0 - 56.0,
            d=27.5 - 27.0,
            theta=math.radians(0.0),
            name="shelf_shipping_corner",
        )
    )
    # 右侧小货架：x=58.7--59.25，y=25.5--26.0。
    walls.append(
        StaticObstacle.rect(
            x=(58 + 59.25) / 2.0,
            y=(25.5 + 26.0) / 2.0,
            l=59.25 - 58,
            d=26.0 - 25.5,
            theta=math.radians(0.0),
            name="shelf_shipping_corner",
        )
    )
    # 左下角小货架：x=56.0--57.0，y=24.0--24.5。
    walls.append(
        StaticObstacle.rect(
            x=(56.0 + 58.0) / 2.0,
            y=(24.0 + 24.5) / 2.0,
            l=58.0 - 56.0,
            d=24.5 - 24.0,
            theta=math.radians(0.0),
            name="shelf_shipping_corner",
        )
    )
    walls.append(
        StaticObstacle.rect(
            x=(58 + 59.25) / 2.0,
            y=(22 + 22.5) / 2.0,
            l=59.25 - 58,
            d=26.0 - 25.5,
            theta=math.radians(0.0),
            name="shelf_shipping_corner",
        )
    )
    
    walls.append(
        StaticObstacle.rect(
            x=(56.0 + 58.0) / 2.0,
            y=(20.0 + 20.5) / 2.0,
            l=58.0 - 56.0,
            d=24.5 - 24.0,
            theta=math.radians(0.0),
            name="shelf_shipping_corner",
        )
    )
    # ======================================================================
    # 可移动障碍物
    # x/y 是中心位置，l/d/h 是长、宽、高，theta 使用弧度；difficulty
    # 是机器人搬移该物体的真实阻力，oid 是规划和动态事件使用的唯一标识。
    # 下方只按地图区域分组，不改变参考图中的任何参数。
    # ======================================================================
    movable = [
        # ------------------------------------------------------------------
        # 左侧区域（x < 30）
        # ------------------------------------------------------------------
        # BX-A：左侧下部的长条薄箱；动态事件会让这个物体自主移动。
        MovableObstacle(
            x=18.7,
            y=10.5,
            l=3.8,
            d=0.2,
            h=0.8,
            theta=0.5,
            material="cardboard_box",
            oid="box101",
        ),
        # BX_A_1：左侧上部、靠近通道 B 的长条薄箱。
        MovableObstacle(
            x=18,
            y=25.0,
            l=5.2,
            d=0.3,
            h=0.8,
            theta=1.2,
            material="cardboard_box",
            oid="box102",
        ),
        # BX_A_2：左侧中部手推车。
        MovableObstacle(
            x=19.2,
            y=21.0,
            l=4.8,
            d=0.8,
            h=250,
            theta=0.5,
            material="cart",
            oid="cart101",
        ),
        # PL(H)：左侧中部的方形大件。
        MovableObstacle(
            x=12,
            y=23.0,
            l=3.5,
            d=3.5,
            h=10,
            theta=0.8,
            material="cardboard_box",
            oid="box103",
        ),

        # ------------------------------------------------------------------
        # 中央区域（约 x=30--45）
        # ------------------------------------------------------------------
        # CT：中央通道内的长条手推车。
        MovableObstacle(
            x=36.0,
            y=23.0,
            l=3.7,
            d=0.8,
            h=1.1,
            theta=0.23,
            material="cart",
            oid="cart102",
        ),
        # PL(L)：中央下部的低矮托盘。
        MovableObstacle(
            x=32.5,
            y=14.5,
            l=4.4,
            d=2.5,
            h=0.15,
            theta=0.0,
            material="pallet",
            oid="pallet101",
        ),
        # SC：中央偏右的重载托盘。
        MovableObstacle(
            x=39,
            y=14.3,
            l=3.3,
            d=1,
            h=2.0,
            theta=0.8,
            material="loaded_pallet",
            oid="loadedPallet101",
        ),

        # ------------------------------------------------------------------
        # 右侧区域（x > 45）
        # ------------------------------------------------------------------
        # BX_D：右侧下部的大箱体。
        MovableObstacle(
            x=52.0,
            y=16.0,
            l=4.5,
            d=2.4,
            h=0.8,
            theta=0.0,
            material="cardboard_box",
            oid="box201",
        ),
        # BX_D_1：BX_D 上方的长条薄箱。
        MovableObstacle(
            x=52.0,
            y=21,
            l=4,
            d=0.5,
            h=0.8,
            theta=0.0,
            material="cardboard_box",
            oid="box202",
        ),
        # BX_D_2：右侧上部、靠近通道 B 的箱体。
        MovableObstacle(
            x=52.0,
            y=25.0,
            l=4.5,
            d=1.8,
            h=0.8,
            theta=0.3,
            material="cardboard_box",
            oid="box203",
        ),

        # ------------------------------------------------------------------
        # 参考图中的附加编号物体
        # ------------------------------------------------------------------
        # 33 / 33_1：第 2 列附近、通道 D 内上下相邻的两个小木箱。
        MovableObstacle(
            x=16.0,
            y=7.35,
            l=0.5,
            d=0.5,
            h=0.3,
            theta=0.0,
            material="wooden_crate",
            oid="crate101",
        ),
        MovableObstacle(
            x=16.0,
            y=8.05,
            l=0.5,
            d=0.5,
            h=0.3,
            theta=0.0,
            material="wooden_crate",
            oid="crate102",
        ),

        # 35：中央上部、通道 A 内的大型手推车。
        MovableObstacle(
            x=31.5,
            y=30.5,
            l=1.8,
            d=1.8,
            h=1000,
            theta=0.70,
            material="cart",
            oid="cart103",
        ),

        # 34_1 / 36_1：仓库底部中央区域的两个小型手推车。
        MovableObstacle(
            x=31,
            y=1.8,
            l=0.9,
            d=1,
            h=300,
            theta=0.70,
            material="cart",
            oid="cart104",
        ),
        MovableObstacle(
            x=33.5,
            y=1.8,
            l=1,
            d=1,
            h=10.0,
            theta=0.5,
            material="cart",
            oid="cart105",
        ),
        # 38 / 38_2：右侧中部并排的两条窄型手推车。
        MovableObstacle(
            x=47.5,
            y=14.6,
            l=0.3,
            d=2.3,
            h=1000.0,
            theta=0.3,
            material="cart",
            oid="cart106",
        ),
        MovableObstacle(
            x=48.8,
            y=14.6,
            l=0.3,
            d=2.3,
            h=5000.0,
            theta=0.0,
            material="cart",
            oid="cart107",
        ),
        # 39：右侧下部、通道 D 附近的窄型手推车。
        MovableObstacle(
            x=48.0,
            y=8,
            l=0.2,
            d=1.7,
            h=100.0,
            theta=0.4,
            material="cart",
            oid="cart108",
        ),
    
        # ------------------------------------------------------------------
        # 依次从三条不同竖向入口进入上方送货通道 A 的动态障碍物。
        # 三者尺寸均为 1 m × 1 m × 1 m，初始位置位于通道 A 南侧。
        # --------------------------------------------------------
    
        MovableObstacle(
            x=13.5,
            y=28.0,
            l=2.0,
            d=2.0,
            h=2.0,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox101",
        ),

     MovableObstacle(
            x=24.5,
            y=27.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox102",
        ),
     MovableObstacle(
            x=24.5,
            y=29.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox103",
        ),
     MovableObstacle(
            x=24.5,
            y=31.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox104",
        ),
     MovableObstacle(
            x=24.5,
            y=33.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox105",
        ),

        MovableObstacle(
            x=28.0,
            y=27.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox106",
        ),
    
        MovableObstacle(
            x=26.0,
            y=27.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox107",
        ),
        MovableObstacle(
            x=28.0,
            y=29.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox108",
        ),
        MovableObstacle(
            x=26.0,
            y=29.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox109",
        ),
        MovableObstacle(
            x=28.0,
            y=31.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox110",
        ),
        MovableObstacle(
            x=26.0,
            y=31.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox111",
        ),
        MovableObstacle(
            x=28.0,
            y=33.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox112",
        ),
        MovableObstacle(
            x=26.0,
            y=33.0,
            l=1.2,
            d=1.2,
            h=1.2,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox113",
        ),
  
        MovableObstacle(
            x=57,
            y=3.0,
            l=1.5,
            d=1.5,
            h=1.5,
            theta=0.0,
            material="cardboard_box",
            oid="deliveryBox114",
        ),
    ]

    # 使用项目预设的材料系数和统一物理公式计算真实搬移难度：
    # difficulty = (mu * rho) * (l * d * h) * g。
    for obstacle in movable:
        obstacle.difficulty = round(
            friction_force(material_mu_rho(obstacle.material), obstacle.volume),
            3,
        )

    # 三个新障碍物按 post-1 -> post-2 -> post-3 的顺序，从不同入口
    # 向北进入送货通道 A（通道中心线 y=30.5）。
    events = [
        Event(
            name="post-0 enters the upper delivery aisle",
            trigger=at_time(20.0),
            effect=MoveTo(
                oid="deliveryBox101",
                goal=(18, 35, 0.0),
                speed=0.1,
            ),
        ),
        Event(
            name="post-1 enters the upper delivery aisle",
            trigger=at_time(40.0),
            effect=MoveTo(
                oid="deliveryBox102",
                goal=(24.5, 5, 0.0),
                speed=0.2,
            ),
        ),

        Event(
            name="post-2 enters the upper delivery aisle",
            trigger=at_time(50.0),
            effect=MoveTo(
                oid="deliveryBox106",
                goal=(28, 5, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3 enters the upper delivery aisle",
            trigger=at_time(60.0),
            effect=MoveTo(
                oid="deliveryBox107",
                goal=(26, 5, 0.0),
                speed=0.3,
            ),
        ),

        Event(
            name="post-1-2 enters the upper delivery aisle",
            trigger=at_time(45.0),
            effect=MoveTo(
                oid="deliveryBox103",
                goal=(24.5, 7, 0.0),
                speed=0.3,
            ),
        ),

        Event(
            name="post-2-2 enters the upper delivery aisle",
            trigger=at_time(55.0),
            effect=MoveTo(
                oid="deliveryBox108",
                goal=(28, 7, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3-2 enters the upper delivery aisle",
            trigger=at_time(65.0),
            effect=MoveTo(
                oid="deliveryBox109",
                goal=(26, 7, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-1-3 enters the upper delivery aisle",
            trigger=at_time(50.0),
            effect=MoveTo(
                oid="deliveryBox104",
                goal=(24.5, 9, 0.0),
                speed=0.3,
            ),
        ),


        Event(
            name="post-2-3 enters the upper delivery aisle",
            trigger=at_time(60.0),
            effect=MoveTo(
                oid="deliveryBox110",
                goal=(28, 9, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3-3 enters the upper delivery aisle",
            trigger=at_time(70.0),
            effect=MoveTo(
                oid="deliveryBox111",
                goal=(26, 9, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-1-4 enters the upper delivery aisle",
            trigger=at_time(55.0),
            effect=MoveTo(
                oid="deliveryBox105",
                goal=(24.5, 11, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-2-4 enters the upper delivery aisle",
            trigger=at_time(65.0),
            effect=MoveTo(
                oid="deliveryBox112",
                goal=(28, 11, 0.0),
                speed=0.3,
            ),
        ),
        Event(
            name="post-3-4 enters the upper delivery aisle",
            trigger=at_time(75.0),
            effect=MoveTo(
                oid="deliveryBox113",
                goal=(26, 11, 0.0),
                speed=0.3,
            ),
        ),

        
        Event(
            name="post-4 enters the upper delivery aisle",
            trigger=at_time(110.0),
            effect=MoveTo(
                oid="deliveryBox114",
                goal=(57, 15, 0.5),
                speed=0.3,
            ),
        ),
    ]

   


    # Receiving-area start and shipping-area goal shown in the diagram.
    start = (5.0, 1.25)
    goal = (56.0, 33.0)

    return {
        "workspace": workspace,
        "static": walls,
        "movable": movable,
        "start": start,
        "goal": goal,
        "dynamics": events,
        "cfg": Config(
            grid_step=0.5,
            conn_radius=0.75,
            se2_cell=0.5,
        ),
    }
