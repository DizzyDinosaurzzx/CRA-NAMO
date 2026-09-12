"""Themed obstacle vocabularies for generated maps.

The handwritten scenarios each speak a domain: a depot moves pallets and
trolleys, a home moves sofas, a hospital moves beds, a collapsed building moves
rubble.  Generated maps borrow those vocabularies so a random experiment reads
like the authored ones instead of like an abstract obstacle soup.

The second job of this module is to make the LLM arms measurable.  Both offline
estimators are lexical:

* ``risk.keyword_level`` returns ``low`` for any label it cannot match, and
* ``llm_difficulty._lookup`` falls back to ``unknown`` (mu*rho = 40) or, worse,
  to whichever table entry happens to share a word.

So a label whose danger or whose weight is *semantic* rather than lexical is
invisible to the heuristic while remaining obvious to anyone who knows what the
object is.  ``crash_cart`` is the sharpest example: the keyword table reads
"cart", concludes mu*rho = 4.5, and plans a free push; the object is a loaded
resuscitation trolley that must never be moved out of a corridor.

Two trap families follow from that:

``risk_traps``
    Cheap to push and read as ``low`` risk, but the label names something that
    must be left alone.  Each one carries ``reveals``: a label the keyword table
    *can* read.  The world hands that label over on contact, so an arm that
    pushed blind is charged the surcharge it failed to anticipate, while an arm
    that understood the visible label never went near it.

``weight_traps``
    Read as light, actually heavy.  Here the ground truth is charged directly:
    the executor bills the real newtons regardless of what anyone believed.

``false_alarms``
    The mirror image, and the reason a corpus of traps alone would be useless:
    a label the keyword table over-reads.  ``beam_offcut_carton`` is a box of
    timber offcuts, but "beam" is an extreme keyword, so the offline arm pays a
    detour to avoid a cardboard box.  Without these, "never push anything" is
    the winning strategy and the corpus stops measuring understanding.

Every claim above is checked against the live tables by ``validate_theme``;
``tests/scenario_generation/test_materials.py`` fails if an edit to either
heuristic table quietly makes a trap visible again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import risk as risk_model
from llm_difficulty import MATERIAL_MU_RHO, _canonical_anchor, material_mu_rho

# A range is expressed as (low, high) and sampled uniformly.
Range = tuple[float, float]


@dataclass(frozen=True)
class Material:
    """One obstacle vocabulary entry with its ground-truth physics."""

    label: str
    mu: float                  # true sliding friction against the floor
    density: Range             # true bulk density [kg/m^3]
    risk: str                  # what the label means to a reader of the domain
    reveals: str = ""          # contact label the offline keyword table can read
    length: Range = (0.6, 1.1)
    depth: Range = (0.5, 0.9)
    height: Range = (0.8, 1.2)

    def sample_density(self, rng) -> float:
        return rng.uniform(*self.density)

    def sample_size(self, rng) -> tuple[float, float, float]:
        return (rng.uniform(*self.length), rng.uniform(*self.depth),
                rng.uniform(*self.height))


@dataclass(frozen=True)
class Theme:
    """The vocabulary one profile draws from."""

    name: str
    note: str
    plain: tuple[Material, ...]          # the heuristic reads these correctly
    risk_traps: tuple[Material, ...]     # semantically dangerous, lexically dull
    weight_traps: tuple[Material, ...]   # semantically heavy, lexically light
    false_alarms: tuple[Material, ...]   # harmless, but lexically alarming

    def all_materials(self) -> tuple[Material, ...]:
        return (*self.plain, *self.risk_traps, *self.weight_traps,
                *self.false_alarms)


# --------------------------------------------------------------------------
# depot: moving_depot and warehouse speak this one.
# --------------------------------------------------------------------------
_DEPOT = Theme(
    name="depot",
    note="pallets, trolleys and the chemistry nobody labels clearly",
    plain=(
        Material("empty_cart", 0.02, (35.0, 65.0), risk_model.LOW,
                 length=(0.7, 1.2), depth=(0.45, 0.8)),
        Material("wooden_crate", 0.45, (55.0, 190.0), risk_model.LOW),
        Material("cardboard_box", 0.35, (30.0, 60.0), risk_model.LOW,
                 length=(0.45, 0.9), depth=(0.4, 0.8), height=(0.5, 1.1)),
        Material("loaded_pallet", 0.40, (260.0, 480.0), risk_model.LOW,
                 length=(0.8, 1.4), depth=(0.7, 1.2), height=(0.7, 1.3)),
        Material("steel_shelf", 0.50, (240.0, 330.0), risk_model.LOW,
                 height=(1.6, 2.0)),
        Material("service_trolley", 0.03, (60.0, 120.0), risk_model.LOW),
    ),
    risk_traps=(
        # A forklift traction battery: 1 t of lead and sulphuric acid in a
        # crate-sized box.  "battery" matches nothing in either table.
        Material("forklift_traction_battery", 0.35, (820.0, 1150.0),
                 risk_model.HIGH, reveals="chemical_drum",
                 length=(0.8, 1.0), depth=(0.6, 0.8), height=(0.6, 0.8)),
        # A cage of LPG bottles waiting for exchange.  "lpg" is not "gas".
        Material("lpg_exchange_cage", 0.30, (140.0, 220.0),
                 risk_model.EXTREME, reveals="gas_cylinder",
                 height=(1.1, 1.5)),
        # Aerosols repacked onto a pallet: the keyword table sees "pallet".
        Material("aerosol_repack_pallet", 0.40, (150.0, 240.0),
                 risk_model.MEDIUM_HIGH, reveals="chemical_drum",
                 length=(0.8, 1.2), depth=(0.7, 1.0)),
    ),
    weight_traps=(
        # A full cable drum reads as an unknown 40; it is closer to 900.
        Material("spool_of_cable", 0.45, (780.0, 1050.0), risk_model.LOW,
                 length=(0.9, 1.2), depth=(0.9, 1.2), height=(0.9, 1.2)),
        Material("granite_offcut_stack", 0.60, (2300.0, 2700.0),
                 risk_model.LOW, length=(0.7, 1.0), depth=(0.6, 0.9),
                 height=(0.5, 0.8)),
    ),
    false_alarms=(
        # A carton of timber offcuts.  "beam" is an extreme keyword, so the
        # offline arm pays a detour rather than nudge a cardboard box.
        Material("beam_offcut_carton", 0.35, (40.0, 70.0), risk_model.LOW,
                 length=(0.5, 0.9), depth=(0.4, 0.7), height=(0.4, 0.7)),
    ),
)


# --------------------------------------------------------------------------
# home: the vocabulary of home.py and maze.py.
# --------------------------------------------------------------------------
_HOME = Theme(
    name="home",
    note="furniture, plus the two objects in a house that must not be shoved",
    plain=(
        Material("sofa", 0.50, (45.0, 60.0), risk_model.LOW,
                 length=(1.4, 2.0), depth=(0.8, 1.0), height=(0.8, 0.95)),
        Material("wooden_table", 0.40, (22.0, 32.0), risk_model.LOW,
                 length=(0.9, 1.6), height=(0.7, 0.8)),
        Material("chair", 0.45, (25.0, 38.0), risk_model.LOW,
                 length=(0.45, 0.7), depth=(0.45, 0.7), height=(0.8, 1.0)),
        Material("cardboard_box", 0.35, (30.0, 60.0), risk_model.LOW,
                 length=(0.45, 0.9), depth=(0.4, 0.8), height=(0.5, 1.1)),
        Material("cabinet", 0.45, (85.0, 130.0), risk_model.LOW,
                 height=(1.4, 1.9)),
    ),
    risk_traps=(
        # A playpen: the keyword table knows "child" but not "toddler".
        Material("toddler_playpen", 0.35, (18.0, 30.0), risk_model.HIGH,
                 reveals="occupied_playpen", length=(0.9, 1.3),
                 depth=(0.8, 1.1), height=(0.7, 0.9)),
        # Home oxygen therapy: life support plus an oxidiser, reading as "low".
        Material("home_oxygen_concentrator", 0.30, (90.0, 150.0),
                 risk_model.HIGH, reveals="ventilator",
                 length=(0.4, 0.6), depth=(0.4, 0.6), height=(0.6, 0.9)),
        Material("space_heater", 0.35, (40.0, 70.0), risk_model.MEDIUM_HIGH,
                 reveals="electrical_cabinet", length=(0.4, 0.7),
                 depth=(0.35, 0.6), height=(0.5, 0.8)),
    ),
    weight_traps=(
        Material("upright_piano", 0.40, (330.0, 430.0), risk_model.LOW,
                 length=(1.4, 1.6), depth=(0.6, 0.75), height=(1.2, 1.4)),
        Material("cast_iron_radiator", 0.55, (600.0, 820.0), risk_model.LOW,
                 length=(0.9, 1.4), depth=(0.25, 0.4), height=(0.6, 0.9)),
    ),
    false_alarms=(
        # A floor-standing speaker.  The keyword table reads "column".
        Material("column_speaker", 0.40, (55.0, 95.0), risk_model.LOW,
                 length=(0.3, 0.45), depth=(0.3, 0.45), height=(1.0, 1.3)),
    ),
)


# --------------------------------------------------------------------------
# hospital: the vocabulary of hospital.py.
# --------------------------------------------------------------------------
_HOSPITAL = Theme(
    name="hospital",
    note="ward furniture whose danger is procedural, never lexical",
    plain=(
        Material("empty_hospital_bed", 0.06, (55.0, 85.0), risk_model.LOW,
                 length=(1.9, 2.1), depth=(0.85, 1.0), height=(0.7, 0.9)),
        Material("linen_cart", 0.03, (45.0, 80.0), risk_model.LOW),
        Material("cardboard_box", 0.35, (30.0, 60.0), risk_model.LOW,
                 length=(0.45, 0.9), depth=(0.4, 0.8), height=(0.5, 1.1)),
        Material("filing_cabinet", 0.45, (280.0, 330.0), risk_model.LOW,
                 height=(1.1, 1.5)),
        Material("empty_cart", 0.02, (35.0, 65.0), risk_model.LOW),
    ),
    risk_traps=(
        # Sharps: a biohazard that the keyword table reads as a container.
        Material("sharps_container", 0.40, (60.0, 110.0), risk_model.HIGH,
                 reveals="medical_waste_spill", length=(0.4, 0.6),
                 depth=(0.35, 0.55), height=(0.5, 0.8)),
        # A resuscitation trolley.  The table sees "cart" and prices it at 4.5.
        Material("crash_cart", 0.04, (130.0, 190.0), risk_model.HIGH,
                 reveals="medical_equipment", length=(0.6, 0.85),
                 depth=(0.45, 0.65), height=(0.9, 1.2)),
        # Moving the screen breaks an isolation barrier.
        Material("isolation_screen", 0.30, (25.0, 45.0), risk_model.HIGH,
                 reveals="patient_isolation", length=(1.0, 1.5),
                 depth=(0.25, 0.4), height=(1.5, 1.9)),
    ),
    weight_traps=(
        # Half a tonne of generator and column, parked with its castor brakes
        # set.  The table reads "unit" and prices it as an unknown 40.
        Material("mobile_xray_unit", 0.55, (420.0, 560.0), risk_model.MEDIUM,
                 reveals="medical_equipment", length=(0.8, 1.1),
                 depth=(0.6, 0.85), height=(1.4, 1.8)),
        Material("lead_shielding_screen", 0.50, (1500.0, 2100.0),
                 risk_model.LOW, length=(1.0, 1.4), depth=(0.2, 0.35),
                 height=(1.6, 2.0)),
    ),
    false_alarms=(
        # Paperwork on castors.  "patient" is a high keyword; the trolley is
        # the easiest thing on the ward to move.
        Material("patient_chart_trolley", 0.03, (35.0, 60.0), risk_model.LOW,
                 length=(0.4, 0.6), depth=(0.35, 0.5), height=(0.9, 1.1)),
    ),
)


# --------------------------------------------------------------------------
# earthquake: the vocabulary of earthquake.py.
# --------------------------------------------------------------------------
_EARTHQUAKE = Theme(
    name="earthquake",
    note="a damaged building, where the cheapest push is the structural one",
    plain=(
        Material("cardboard_box", 0.35, (30.0, 60.0), risk_model.LOW,
                 length=(0.45, 0.9), depth=(0.4, 0.8), height=(0.5, 1.1)),
        Material("wooden_crate", 0.45, (55.0, 190.0), risk_model.LOW),
        Material("empty_cart", 0.02, (35.0, 65.0), risk_model.LOW),
        Material("filing_cabinet", 0.45, (280.0, 330.0), risk_model.LOW,
                 height=(1.1, 1.5)),
    ),
    risk_traps=(
        # The prop under a cracked slab.  "shoring" is an extreme keyword;
        # "temporary_prop" is not, and it is light enough to look attractive.
        Material("temporary_prop", 0.35, (35.0, 60.0), risk_model.EXTREME,
                 reveals="shoring_prop", length=(0.25, 0.4),
                 depth=(0.25, 0.4), height=(1.6, 2.2)),
        # Panels stacked against the rubble behind them.
        Material("stacked_ceiling_panels", 0.40, (110.0, 180.0),
                 risk_model.MEDIUM_HIGH, reveals="debris_pile",
                 length=(0.9, 1.4), depth=(0.3, 0.5), height=(1.0, 1.5)),
        # Cutting gas left on site.
        Material("acetylene_bank", 0.30, (160.0, 240.0), risk_model.EXTREME,
                 reveals="gas_cylinder", length=(0.5, 0.8), depth=(0.4, 0.7),
                 height=(1.1, 1.5)),
    ),
    weight_traps=(
        Material("slab_fragment", 0.60, (2200.0, 2600.0), risk_model.LOW,
                 length=(0.7, 1.1), depth=(0.6, 0.9), height=(0.3, 0.6)),
        Material("machinery_skid", 0.50, (600.0, 900.0), risk_model.LOW,
                 length=(0.9, 1.3), depth=(0.7, 1.0), height=(0.8, 1.2)),
    ),
    false_alarms=(
        # A case of survey instruments, not a structural member.
        Material("structural_survey_kit", 0.35, (45.0, 80.0), risk_model.LOW,
                 length=(0.5, 0.8), depth=(0.4, 0.6), height=(0.4, 0.7)),
    ),
)


THEMES: dict[str, Theme] = {
    theme.name: theme
    for theme in (_DEPOT, _HOME, _HOSPITAL, _EARTHQUAKE)
}


def get_theme(name: str) -> Theme:
    key = str(name).strip().lower()
    if key not in THEMES:
        raise ValueError(f"unknown material theme {name!r}; available: "
                         + ", ".join(sorted(THEMES)))
    return THEMES[key]


def heuristic_risk(label: str) -> str:
    """What the offline keyword table makes of a label."""
    return risk_model.keyword_level(label)


def heuristic_mu_rho(label: str) -> float:
    """What the offline material table makes of a label."""
    return material_mu_rho(label)


def is_anchored(label: str) -> bool:
    """True when the label is pinned to a table value even for the LLM."""
    return _canonical_anchor(label) is not None


def validate_theme(theme: Theme) -> list[str]:
    """Return the ways a theme fails to separate the LLM arms from the heuristic.

    An empty list means every trap still works: the risk traps read as ``low``
    and hand over a label the keyword table can read, and the weight traps are
    priced by the table at a small fraction of their real resistance.
    """
    problems: list[str] = []
    unknown = MATERIAL_MU_RHO["unknown"]

    for material in theme.risk_traps:
        if material.risk == risk_model.LOW:
            problems.append(
                f"{theme.name}/{material.label}: a risk trap must carry a risk "
                "above low, otherwise there is nothing for the LLM to find")
        if heuristic_risk(material.label) != risk_model.LOW:
            problems.append(
                f"{theme.name}/{material.label}: the keyword table already "
                f"reads it as {heuristic_risk(material.label)}, so the "
                "heuristic arm is not blind to it")
        if not material.reveals:
            problems.append(
                f"{theme.name}/{material.label}: a risk trap needs a "
                "contact_reveals label, or a blind arm is never charged")
        elif heuristic_risk(material.reveals) == risk_model.LOW:
            problems.append(
                f"{theme.name}/{material.label}: contact reveals "
                f"{material.reveals!r}, which the keyword table also reads as "
                "low, so contact teaches the heuristic arm nothing")

    for material in theme.weight_traps:
        true_mu_rho = material.mu * sum(material.density) / 2.0
        guessed = heuristic_mu_rho(material.label)
        if guessed > true_mu_rho / 3.0:
            problems.append(
                f"{theme.name}/{material.label}: the table guesses "
                f"mu*rho={guessed:g} against a true {true_mu_rho:.0f}, which is "
                "too close to make the estimate matter")
        if guessed != unknown and guessed > true_mu_rho:
            problems.append(
                f"{theme.name}/{material.label}: the table overestimates it, "
                "so a blind arm detours instead of over-committing")

    for material in theme.false_alarms:
        if material.risk != risk_model.LOW:
            problems.append(
                f"{theme.name}/{material.label}: a false alarm must really be "
                "harmless, or avoiding it was the right call after all")
        if heuristic_risk(material.label) == risk_model.LOW:
            problems.append(
                f"{theme.name}/{material.label}: the keyword table reads it as "
                "low too, so there is no false alarm to see through")
        if material.reveals:
            problems.append(
                f"{theme.name}/{material.label}: a false alarm has nothing to "
                "reveal on contact")

    for material in theme.all_materials():
        if is_anchored(material.label) and material not in theme.plain:
            problems.append(
                f"{theme.name}/{material.label}: an anchored label pins the "
                "LLM to the table value, leaving both arms identical")
    return problems


def validate_all() -> list[str]:
    problems: list[str] = []
    for theme in THEMES.values():
        problems.extend(validate_theme(theme))
    return problems


def describe(materials: Iterable[Material]) -> str:
    """One line per material, for debugging a theme."""
    return "\n".join(
        f"{m.label:28} true mu*rho={m.mu * sum(m.density) / 2.0:8.1f}  "
        f"risk={m.risk:12} heuristic: risk={heuristic_risk(m.label):12} "
        f"mu*rho={heuristic_mu_rho(m.label):7.1f}"
        for m in materials)
