"""Ground-truth dataset for benchmarking the LLM mu*rho estimator.

The estimator in `llm_difficulty.py` short-circuits any label that is an exact
PROMPT_ANCHORS entry (or a MATERIAL_ALIASES synonym): the prompt then orders the
model to echo the calibrated number back. Testing those labels would only measure
instruction following, not estimation. Every label below is therefore chosen to
*miss* that short-circuit, so `DifficultyEstimator._deepseek` has to reason its
way to a number. `assert_all_off_anchor()` enforces this.

Where the ground truth comes from
---------------------------------
Two different, independently defensible sources — kept in separate groups
because they answer different questions:

`paraphrase`
    The label is a plain-language restatement of one of the project's own
    anchors ("expanded polystyrene packing box" == `styrofoam_box`). Ground
    truth is that anchor's calibrated mu*rho, so the reference number is exactly
    the one CA-NAMO already commits to elsewhere — nothing new is asserted. This
    group measures wording robustness: does dropping the magic string change the
    number the planner sees?

`novel` / `state` / `brand`
    The object is not in the anchor table at all. Ground truth is *derived*, not
    measured: mu * (mass_kg / (l*d*h)), with mass, bounding box and mu all
    recorded per item so the number can be audited and argued with. These are
    reference values from published spec sheets and standard friction
    coefficients, not laboratory measurements — treat a 1.5x disagreement on a
    single item as "the reference is arguable", and only the distribution over
    the whole group as evidence.

    `state` items exist in filled/empty pairs on purpose: bulk density is the
    term that moves by an order of magnitude in practice, and a pair isolates
    whether the model tracks it.
"""

from __future__ import annotations

# Run from anywhere: the library lives one directory up.
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
    """One benchmark object: a label to ask about, and what the answer should be."""

    label: str                  # exactly what goes into the prompt as `material`
    group: str                  # paraphrase | novel | state | brand
    mu: float                   # reference friction / rolling-resistance coefficient
    rho: float                  # reference BULK density [kg/m^3] = mass / bbox volume
    l: float                    # representative bounding box [m]
    d: float
    h: float
    note: str                   # where mu and rho come from
    anchor: Optional[str] = None    # paraphrase group: the anchor it restates

    @property
    def mu_rho(self) -> float:
        return self.mu * self.rho

    @property
    def volume(self) -> float:
        return self.l * self.d * self.h

    def observation(self, oid: int = 0, scale: float = 1.0) -> dict:
        """The dict shape `DifficultyEstimator.estimate` consumes.

        `scale` multiplies every linear dimension, for the size-independence
        probe: mu*rho is defined to be size-free, so the answer must not move.
        """
        return {
            "oid": oid,
            "material": self.label,
            "l": round(self.l * scale, 4),
            "d": round(self.d * scale, 4),
            "h": round(self.h * scale, 4),
        }


def _anchor_item(label: str, anchor: str, l: float, d: float, h: float) -> Item:
    """A paraphrase item, taking mu and rho straight from the anchor tables."""
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
    """A novel object: rho is computed from mass over the bounding box."""
    return Item(
        label=label, group=group, mu=mu, rho=mass_kg / (l * d * h),
        l=l, d=d, h=h, note=f"{mass_kg:g} kg / {l*d*h:.4g} m^3, mu={mu:g}; {note}",
    )


# Group 1 - paraphrases of the project's own anchors.
# Ground truth = the anchor's calibrated mu*rho. Spans 1.0 -> 1440 kg/m^3.
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

# Group 2 - objects outside the anchor table entirely.
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

# Group 3 - filled vs empty. Same object, an order of magnitude apart in rho.
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

# Group 4 - brand and product names. The prompt tells the model to resolve these
# to the real object; this group checks whether it does.
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

# Anchor -> paraphrase label, used by the navigation experiment to relabel a map
# without changing its physics: the paraphrase carries the same ground truth, so
# only what the *estimator* sees differs between arms.
PARAPHRASE_OF_ANCHOR: Dict[str, str] = {
    item.anchor: item.label for item in PARAPHRASE if item.anchor
}


def assert_all_off_anchor() -> None:
    """Fail loudly if a label would take the echo-the-anchor short-circuit.

    Such an item would score a perfect match without the model estimating
    anything, quietly inflating every accuracy number in the report.
    """
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
