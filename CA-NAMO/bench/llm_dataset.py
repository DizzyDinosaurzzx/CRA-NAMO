"""LLM mu*rho 估计器的基准真值数据集：所有标签都刻意避开锚点短路，迫使模型真的估计。"""

from __future__ import annotations

# 任意目录均可运行：库位于上一级目录。
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, List, Optional

from llm_difficulty import (
    MATERIAL_MU,
    MATERIAL_RHO,
    MATERIAL_MU_RHO,
    _canonical_anchor,
)


@dataclass(frozen=True)
class Item:
    """一个基准物体：待询问的标签与应有的答案。"""

    label: str                  # 原样进入 prompt 的 material 字段
    group: str                  # paraphrase | novel | state | brand
    mu: float                   # 参考摩擦/滚动阻力系数
    rho: float                  # 参考堆积密度 [kg/m^3] = 质量 / 包围盒体积
    l: float                    # 代表性包围盒尺寸 [m]
    d: float
    h: float
    note: str                   # mu 与 rho 的出处
    anchor: Optional[str] = None    # paraphrase 组：所复述的锚点

    @property
    def mu_rho(self) -> float:
        return self.mu * self.rho

    @property
    def volume(self) -> float:
        return self.l * self.d * self.h

    def observation(self, oid: int = 0, scale: float = 1.0) -> dict:
        """构造 `DifficultyEstimator.estimate` 消费的字典；scale 等比缩放线尺寸，用于尺寸无关性检验（mu*rho 不应随之变化）。"""
        return {
            "oid": oid,
            "material": self.label,
            "l": round(self.l * scale, 4),
            "d": round(self.d * scale, 4),
            "h": round(self.h * scale, 4),
        }


def _anchor_item(label: str, anchor: str, l: float, d: float, h: float) -> Item:
    """paraphrase 条目：mu 与 rho 直接取自锚点表。"""
    return Item(
        label=label,
        group="paraphrase",
        mu=MATERIAL_MU[anchor],
        rho=MATERIAL_RHO[anchor],
        l=l, d=d, h=h,
        note=f"restatement of anchor '{anchor}' (mu*rho={MATERIAL_MU_RHO[anchor]:g})",
        anchor=anchor,
    )


def _derived(label: str, group: str, mass_kg: float, mu: float,
             l: float, d: float, h: float, note: str) -> Item:
    """表外物体：rho 由质量除以包围盒体积推得。"""
    return Item(
        label=label, group=group, mu=mu, rho=mass_kg / (l * d * h),
        l=l, d=d, h=h, note=f"{mass_kg:g} kg / {l*d*h:.4g} m^3, mu={mu:g}; {note}",
    )


# --------------------------------------------------------------------------- #
# 第 1 组：项目自身锚点的改述；真值 = 锚点标定的 mu*rho，跨度 1.0 -> 1440 kg/m^3。
# --------------------------------------------------------------------------- #
PARAPHRASE: List[Item] = [
    _anchor_item("unloaded push trolley",            "empty_cart",         1.0, 0.7, 1.0),
    _anchor_item("loaded utility trolley",           "cart",               1.0, 0.7, 1.1),
    _anchor_item("expanded polystyrene packing box", "styrofoam_box",      0.8, 0.6, 1.0),
    _anchor_item("moulded plastic stacking chair",   "plastic_chair",      0.55, 0.55, 0.9),
    _anchor_item("four-legged wooden dining table",  "wooden_table",       1.6, 0.9, 0.75),
    _anchor_item("upholstered armchair",             "chair",              0.7, 0.7, 0.9),
    _anchor_item("corrugated fibreboard carton",     "cardboard_box",      0.6, 0.45, 0.8),
    _anchor_item("gym exercise floor mat",           "foam_mat",           2.0, 1.0, 0.1),
    _anchor_item("empty bookshelf unit",             "empty_shelf",        0.9, 0.35, 1.8),
    _anchor_item("waste receptacle",                 "trash_bin",          0.6, 0.6, 1.0),
    _anchor_item("four-legged wooden stool",         "stool",              0.4, 0.4, 0.5),
    _anchor_item("three-seat couch",                 "sofa",               2.1, 0.9, 0.85),
    _anchor_item("wooden shipping crate",            "wooden_crate",       1.0, 1.0, 1.0),
    _anchor_item("storage cupboard",                 "cabinet",            0.9, 0.5, 1.8),
    _anchor_item("empty wooden shipping pallet",     "pallet",             1.2, 1.0, 0.15),
    _anchor_item("stocked warehouse racking bay",    "shelf",              2.0, 0.9, 1.8),
    _anchor_item("document filing cabinet",          "filing_cabinet",     0.5, 0.6, 1.3),
    _anchor_item("steel storage rack with stock",    "steel_shelf",        2.0, 0.9, 2.0),
    _anchor_item("pallet stacked with goods",        "loaded_pallet",      1.2, 1.0, 1.2),
    _anchor_item("heavy factory machine",            "industrial_machine", 1.8, 1.2, 1.6),
    _anchor_item("steel security safe",              "steel_safe",         0.8, 0.7, 1.5),
    _anchor_item("solid concrete cube",              "concrete_block",     1.0, 1.0, 1.0),
]

# --------------------------------------------------------------------------- #
# 第 2 组：完全不在锚点表内的物体。
# --------------------------------------------------------------------------- #
NOVEL: List[Item] = [
    _derived("office water cooler with full 19 litre bottle", "novel",
             34, 0.40, 0.35, 0.35, 1.10, "cooler 15 kg + 19 kg water; plastic feet on hard floor"),
    _derived("upright piano on castors", "novel",
             220, 0.05, 1.50, 0.60, 1.25, "small hard castors, rolling resistance not sliding"),
    _derived("commercial chest freezer", "novel",
             90, 0.40, 1.40, 0.70, 0.85, "empty; sheet-steel cabinet on plastic feet"),
    _derived("stocked drinks vending machine", "novel",
             380, 0.45, 1.00, 0.90, 1.90, "machine ~250 kg + ~130 kg stock; steel base"),
    _derived("steel drum full of oil", "novel",
             190, 0.40, 0.60, 0.60, 0.90, "208 L drum, 20 kg shell + 170 kg oil; steel on concrete"),
    _derived("empty steel drum", "novel",
             20, 0.40, 0.60, 0.60, 0.90, "same 208 L shell, nothing in it"),
    _derived("fully populated wheeled server rack", "novel",
             700, 0.04, 0.60, 1.00, 2.00, "42U rack ~700 kg loaded, rolls on heavy castors"),
    _derived("unoccupied hospital bed", "novel",
             140, 0.04, 2.20, 1.00, 0.90, "electric bed frame on four braked castors, brakes off"),
    _derived("office task chair on castors", "novel",
             15, 0.03, 0.65, 0.65, 1.10, "five-star base, soft castors on hard floor"),
    _derived("unloaded hand pallet jack", "novel",
             75, 0.03, 1.55, 0.55, 1.20, "steel frame on polyurethane load rollers"),
    _derived("large potted ficus in ceramic planter", "novel",
             45, 0.50, 0.60, 0.60, 1.80, "planter + wet soil ~40 kg; unglazed ceramic on concrete"),
    _derived("granite countertop slab", "novel",
             126, 0.60, 2.40, 0.65, 0.03, "granite 2700 kg/m^3, slab is solid so bulk == material density"),
    _derived("full double-door refrigerator", "novel",
             160, 0.45, 0.90, 0.75, 1.80, "appliance 120 kg + 40 kg contents"),
    _derived("gym rack loaded with dumbbells", "novel",
             400, 0.50, 2.00, 0.60, 1.20, "steel rack + cast iron; rubber feet grip hard"),
    _derived("cast iron bathtub", "novel",
             140, 0.50, 1.70, 0.75, 0.60, "empty tub, enamelled cast iron on four feet"),
    _derived("stack of five empty plastic milk crates", "novel",
             7.5, 0.40, 0.50, 0.35, 1.50, "1.5 kg each; bbox is the whole stack, mostly air"),
    _derived("skid-mounted industrial air compressor", "novel",
             250, 0.50, 1.20, 0.60, 1.10, "tank + motor bolted to a steel skid dragging on concrete"),
    _derived("wooden church pew", "novel",
             60, 0.45, 2.50, 0.50, 1.00, "long but light; bbox is mostly the empty seat volume"),
    _derived("200 litre aquarium filled with water", "novel",
             240, 0.50, 1.20, 0.50, 0.60, "200 kg water + 40 kg glass; glass/silicone on floor"),
    _derived("empty 200 litre aquarium", "novel",
             40, 0.50, 1.20, 0.50, 0.60, "same tank drained"),
    _derived("office multifunction photocopier", "novel",
             120, 0.04, 0.70, 0.70, 1.20, "floor-standing unit delivered on castors"),
    _derived("rolled up broadloom carpet four metres wide", "novel",
             80, 0.60, 4.00, 0.50, 0.50, "carpet backing on concrete grips hard"),
    _derived("three metre steel I-beam", "novel",
             150, 0.50, 3.00, 0.15, 0.30, "~50 kg/m section; bbox close to solid steel"),
    _derived("25 kg sack of ready-mix concrete", "novel",
             25, 0.50, 0.50, 0.35, 0.12, "dense powder in a paper sack, slumps flat on the floor"),
    _derived("undercounter commercial dishwasher", "novel",
             65, 0.45, 0.60, 0.60, 0.85, "stainless cabinet on adjustable feet"),
]

# --------------------------------------------------------------------------- #
# 第 3 组：满/空成对出现，同一物体的 rho 相差一个数量级。
# --------------------------------------------------------------------------- #
STATE: List[Item] = [
    _derived("cardboard box packed with hardcover books", "state",
             35, 0.35, 0.50, 0.40, 0.40, "books ~440 kg/m^3 packed solid"),
    _derived("cardboard box of packing peanuts", "state",
             0.7, 0.35, 0.50, 0.40, 0.40, "loose fill ~5 kg/m^3 plus the carton itself"),
    _derived("empty filing cabinet", "state",
             35, 0.45, 0.47, 0.62, 1.30, "four-drawer steel shell with no paper in it"),
    _derived("bookshelf packed full of books", "state",
             95, 0.45, 0.90, 0.35, 1.80, "frame 25 kg + ~70 kg of books"),
    _derived("wheelie bin filled with waste", "state",
             75, 0.05, 0.58, 0.74, 1.07, "240 L bin 15 kg + 60 kg refuse; two wheels"),
    _derived("empty 240 litre wheelie bin", "state",
             15, 0.05, 0.58, 0.74, 1.07, "same bin, nothing in it"),
    _derived("shopping trolley full of groceries", "state",
             65, 0.03, 1.00, 0.60, 1.00, "trolley 25 kg + 40 kg goods; four swivel castors"),
    _derived("tool chest filled with hand tools", "state",
             90, 0.05, 0.68, 0.46, 0.90, "steel roller cabinet packed with steel tools"),
]

# --------------------------------------------------------------------------- #
# 第 4 组：品牌/产品名；prompt 要求模型还原成真实物体，本组检验是否照做。
# --------------------------------------------------------------------------- #
BRAND: List[Item] = [
    _derived("IKEA BILLY bookcase, empty", "brand",
             41, 0.45, 0.80, 0.28, 2.02, "particleboard carcass, listed shipping weight"),
    _derived("Rubbermaid Brute 121 litre container, empty", "brand",
             5, 0.40, 0.60, 0.60, 0.83, "moulded polyethylene bin"),
    _derived("Herman Miller Aeron chair", "brand",
             19, 0.03, 0.68, 0.68, 1.05, "mesh task chair on castors"),
    _derived("loaded Pelican 1650 case", "brand",
             30, 0.40, 0.79, 0.62, 0.34, "case 12 kg + ~18 kg kit; polymer shell on concrete"),
    _derived("empty EPAL Euro pallet", "brand",
             25, 0.40, 1.20, 0.80, 0.14, "standard 1200x800 four-way pallet"),
    _derived("Gaylord box full of scrap plastic", "brand",
             250, 0.35, 1.20, 1.00, 1.20, "bulk fibreboard container on the floor, no pallet"),
]

DATASET: List[Item] = PARAPHRASE + NOVEL + STATE + BRAND

# 锚点 -> 改述标签：导航实验用它重贴地图标签而不改变物理——真值相同，
# 各 arm 之间只有估计器看到的字符串不同。
PARAPHRASE_OF_ANCHOR: Dict[str, str] = {
    item.anchor: item.label for item in PARAPHRASE if item.anchor
}


def assert_all_off_anchor() -> None:
    """校验没有标签会命中锚点短路：命中的条目无需估计即可满分，会悄悄虚增所有准确率数字。"""
    leaked = [it.label for it in DATASET if _canonical_anchor(it.label) is not None]
    if leaked:
        raise AssertionError(
            "these labels hit the anchor short-circuit and would not test the LLM: "
            + ", ".join(leaked))
    missing = set(MATERIAL_MU_RHO) - {"unknown"} - set(PARAPHRASE_OF_ANCHOR)
    if missing:
        raise AssertionError(
            "anchors without a paraphrase, cannot relabel maps using them: "
            + ", ".join(sorted(missing)))


if __name__ == "__main__":
    assert_all_off_anchor()
    print(f"{len(DATASET)} items")
    for group in ("paraphrase", "novel", "state", "brand"):
        items = [it for it in DATASET if it.group == group]
        lo = min(it.mu_rho for it in items)
        hi = max(it.mu_rho for it in items)
        print(f"  {group:<11s} n={len(items):<3d} mu*rho {lo:8.2f} .. {hi:9.2f}")
