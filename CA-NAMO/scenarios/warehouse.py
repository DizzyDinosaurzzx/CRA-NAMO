from __future__ import annotations

from shapely.geometry import box

from config import Config
from obstacle import MovableObstacle, StaticObstacle


def create():
    """60 m x 40 m hard warehouse map shown in the reference diagram."""

    workspace = box(0.0, 0.0, 60.0, 40.0)
    wall_t = 0.75

    static = [
        # Outer warehouse walls.
        StaticObstacle(box(0.0, 0.0, 60.0, wall_t), "outer_bottom"),
        StaticObstacle(box(0.0, 40.0 - wall_t, 60.0, 40.0), "outer_top"),
        StaticObstacle(box(0.0, 0.0, wall_t, 40.0), "outer_left"),
        StaticObstacle(box(60.0 - wall_t, 0.0, 60.0, 40.0), "outer_right"),
    ]

    # Shelf columns 1--7. Their vertical gaps form cross-aisles
    # A (y=29.0--32.0), B (21.5--24.5), C (13.0--16.0), and
    # D (6.5--9.0), matching the map.
    shelf_columns = (
        (7.0, 10.0),
        (13.5, 17.0),
        (21.0, 26.0),
        (29.5, 34.0),
        (38.0, 42.5),
        (46.0, 50.0),
        (54.0, 58.0),
    )

    # Top shelf row exists only in columns 1--6; column 7 is the shipping area.
    for column, (x0, x1) in enumerate(shelf_columns[:6], start=1):
        static.append(
            StaticObstacle(box(x0, 32.0, x1, 38.5), f"shelf_{column}_top")
        )

    # Separate shelf banks on the left side of cross-aisle B.
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

    # In columns 5--7 the diagram has one continuous bank across aisle B.
    for column, (x0, x1) in enumerate(shelf_columns[4:], start=5):
        static.append(
            StaticObstacle(box(x0, 16.0, x1, 29.0),
                           f"shelf_{column}_middle_tall")
        )

    # Shelf rows on either side of cross-aisle D.
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

    # Shelf connections added from the annotated reference images. Some marked
    # regions span two adjacent cross-aisle breaks.
    shelf_bridges = (
        # Cross-aisle A (y=29.0--32.0).
        (1, 29.0, 32.0, "A"),
        (3, 29.0, 32.0, "A"),
        (6, 29.0, 32.0, "A"),

        # Cross-aisle B (y=21.5--24.5).
        (2, 21.5, 24.5, "B"),
        (4, 21.5, 24.5, "B"),

        # Cross-aisle C (y=13.0--16.0).
        (1, 13.0, 16.0, "C"),
        (2, 13.0, 16.0, "C"),
        (3, 13.0, 16.0, "C"),

        # Cross-aisle D (y=6.5--9.0).
        (1, 6.5, 9.0, "D"),
        (3, 6.5, 9.0, "D"),
        (4, 6.5, 9.0, "D"),
        (5, 6.5, 9.0, "D"),
        (7, 6.5, 9.0, "D"),

        # Seal the marked bottom shelves flush against the inner wall edge.
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

    # Small fixed shelf at the far-right end of cross-aisle A.
    static.append(
        StaticObstacle(box(58.0, 29.0, 59.25, 32.0), "shelf_shipping_corner")
    )

    movable = [
        # Left-side objects.
        MovableObstacle(
            x=19.2,
            y=10.5,
            l=3.9,
            d=0.1,
            h=0.8,
            theta=0.5,
            material="box",
            difficulty=120.0,
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
            difficulty=120.0,
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
            difficulty=3000.0,
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
            difficulty=1500.0,
            oid="PL(H)",
        ),

        # Central objects.
        MovableObstacle(
            x=36.0,
            y=23.0,
            l=3.0,
            d=0.5,
            h=1.1,
            theta=0.23,
            material="cart",
            difficulty=600.0,
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
            difficulty=300.0,
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
            difficulty=8000.0,
            oid="SC",
        ),

        # Right-side objects.
        MovableObstacle(
            x=52.0,
            y=16.0,
            l=3.4,
            d=1.9,
            h=0.8,
            theta=0.0,
            material="box",
            difficulty=120.0,
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
            difficulty=120.0,
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
            difficulty=120.0,
            oid="BX_D_2",
        ),

        # Additional numbered objects.
        MovableObstacle(
            x=16.0,
            y=7.35,
            l=0.5,
            d=0.5,
            h=0.3,
            theta=0.0,
            material="wooden_crate",
            difficulty=66.218,
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
            difficulty=66.218,
            oid="33_1",
        ),

        MovableObstacle(
            x=33.5,
            y=30.5,
            l=1.5,
            d=1.5,
            h=1000,
            theta=0.70,
            material="cart",
            difficulty=66.218,
            oid="35",
        ),

        MovableObstacle(
            x=29.5,
            y=1.8,
            l=0.5,
            d=0.5,
            h=300,
            theta=0.70,
            material="cart",
            difficulty=66.218,
            oid="34_1",
        ),
        MovableObstacle(
            x=33.5,
            y=1.8,
            l=0.5,
            d=0.5,
            h=10.0,
            theta=0.5,
            material="cart",
            difficulty=66.218,
            oid="36_1",
        ),
        MovableObstacle(
            x=46.8,
            y=14.6,
            l=0.1,
            d=1.8,
            h=1000.0,
            theta=0.3,
            material="cart",
            difficulty=66.218,
            oid="38",
        ),
        MovableObstacle(
            x=48.8,
            y=14.6,
            l=0.1,
            d=1.8,
            h=5000.0,
            theta=0.0,
            material="cart",
            difficulty=66.218,
            oid="38_2",
        ),
        MovableObstacle(
            x=48.0,
            y=7.8,
            l=0.1,
            d=1.6,
            h=100.0,
            theta=0.4,
            material="cart",
            difficulty=66.218,
            oid="39",
        ),
    ]

    # Receiving-area start and shipping-area goal shown in the diagram.
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
