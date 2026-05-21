from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACSMKappaRecipe:
    """Two-component ACSM kappa recipe.

    Defaults use the globally representative organic/inorganic values from
    Poehlker et al. (2023). ACSM species are treated as bulk organic and bulk
    inorganic mass. The default `mass` basis follows the ACSM/AMS shortcut in
    Poehlker et al. (2023); `volume` is available for the original component
    volume-fraction basis from Petters and Kreidenweis (2007). The default
    row-level bulk mixing rule is geometric so ambient bulk kappa is handled in
    log space.
    """

    kappa_organic: float = 0.12
    kappa_inorganic: float = 0.63
    fraction_basis: str = "mass"
    mixing_rule: str = "geometric"
    organic_density_g_cm3: float = 1.4
    inorganic_density_g_cm3: float = 1.75
    min_total_amount: float = 1.0e-12
    min_kappa: float = 1.0e-4
    max_kappa: float = 1.5


def _nonnegative(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return max(float(value), 0.0)


def _mixed_kappa(
    organic_fraction: float,
    inorganic_fraction: float,
    recipe: ACSMKappaRecipe,
) -> float:
    if recipe.mixing_rule == "linear":
        return (
            organic_fraction * recipe.kappa_organic
            + inorganic_fraction * recipe.kappa_inorganic
        )
    if recipe.mixing_rule == "geometric":
        if recipe.kappa_organic <= 0.0 or recipe.kappa_inorganic <= 0.0:
            raise ValueError("Geometric kappa mixing requires positive component kappas.")
        log_kappa = (
            organic_fraction * np.log(recipe.kappa_organic)
            + inorganic_fraction * np.log(recipe.kappa_inorganic)
        )
        return float(np.exp(log_kappa))
    raise ValueError("mixing_rule must be 'geometric' or 'linear'")


def kappa_from_acsm_masses(
    organic_mass: float,
    sulfate_mass: float,
    ammonium_mass: float,
    nitrate_mass: float,
    chloride_mass: float,
    recipe: ACSMKappaRecipe = ACSMKappaRecipe(),
) -> float:
    """Estimate bulk aerosol kappa from ACSM mass concentrations.

    The expected units are any internally consistent mass concentration units.
    """

    organic = _nonnegative(organic_mass)
    inorganic_terms = [
        _nonnegative(sulfate_mass),
        _nonnegative(ammonium_mass),
        _nonnegative(nitrate_mass),
        _nonnegative(chloride_mass),
    ]
    if not np.isfinite(organic) or not all(np.isfinite(value) for value in inorganic_terms):
        return float("nan")

    organic_fraction, inorganic_fraction = acsm_component_fractions(
        organic,
        sulfate_mass,
        ammonium_mass,
        nitrate_mass,
        chloride_mass,
        recipe=recipe,
    )
    if not np.isfinite(organic_fraction) or not np.isfinite(inorganic_fraction):
        return float("nan")

    kappa = _mixed_kappa(organic_fraction, inorganic_fraction, recipe)
    return float(np.clip(kappa, recipe.min_kappa, recipe.max_kappa))


def acsm_component_fractions(
    organic_mass: float,
    sulfate_mass: float,
    ammonium_mass: float,
    nitrate_mass: float,
    chloride_mass: float,
    recipe: ACSMKappaRecipe = ACSMKappaRecipe(),
) -> tuple[float, float]:
    """Return organic and inorganic fractions on the configured basis."""

    organic = _nonnegative(organic_mass)
    inorganic = float(
        np.sum(
            [
                _nonnegative(sulfate_mass),
                _nonnegative(ammonium_mass),
                _nonnegative(nitrate_mass),
                _nonnegative(chloride_mass),
            ]
        )
    )
    if not np.isfinite(organic) or not np.isfinite(inorganic):
        return float("nan"), float("nan")
    if recipe.fraction_basis == "mass":
        total_mass = organic + inorganic
        if total_mass <= recipe.min_total_amount:
            return float("nan"), float("nan")
        return organic / total_mass, inorganic / total_mass
    if recipe.fraction_basis == "volume":
        organic_volume = organic / recipe.organic_density_g_cm3
        inorganic_volume = inorganic / recipe.inorganic_density_g_cm3
        total_volume = organic_volume + inorganic_volume
        if total_volume <= recipe.min_total_amount:
            return float("nan"), float("nan")
        return organic_volume / total_volume, inorganic_volume / total_volume
    raise ValueError("fraction_basis must be 'mass' or 'volume'")


def acsm_volume_fractions(
    organic_mass: float,
    sulfate_mass: float,
    ammonium_mass: float,
    nitrate_mass: float,
    chloride_mass: float,
    recipe: ACSMKappaRecipe = ACSMKappaRecipe(),
) -> tuple[float, float]:
    """Return organic and inorganic dry-volume fractions."""

    volume_recipe = ACSMKappaRecipe(
        kappa_organic=recipe.kappa_organic,
        kappa_inorganic=recipe.kappa_inorganic,
        fraction_basis="volume",
        mixing_rule=recipe.mixing_rule,
        organic_density_g_cm3=recipe.organic_density_g_cm3,
        inorganic_density_g_cm3=recipe.inorganic_density_g_cm3,
        min_total_amount=recipe.min_total_amount,
        min_kappa=recipe.min_kappa,
        max_kappa=recipe.max_kappa,
    )
    return acsm_component_fractions(
        organic_mass,
        sulfate_mass,
        ammonium_mass,
        nitrate_mass,
        chloride_mass,
        recipe=volume_recipe,
    )
