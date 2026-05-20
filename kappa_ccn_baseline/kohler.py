from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KohlerConstants:
    """Thermodynamic constants matching the local kappa_kohler_theory.py script."""

    surface_tension_j_m2: float = 0.072
    water_molar_mass_g_mol: float = 18.01528
    temperature_k: float = 298.15
    water_density_g_m3: float = 997048.0
    gas_constant_j_mol_k: float = 8.3145

    @property
    def kelvin_a_m(self) -> float:
        return (
            4.0
            * self.surface_tension_j_m2
            * self.water_molar_mass_g_mol
            / (
                self.gas_constant_j_mol_k
                * self.temperature_k
                * self.water_density_g_m3
            )
        )


def _as_float_array(values: float | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def critical_diameter_nm(
    kappa: float | np.ndarray,
    supersaturation_percent: float | np.ndarray,
    constants: KohlerConstants = KohlerConstants(),
) -> np.ndarray:
    """Return dry critical diameter from kappa and supersaturation.

    Supersaturation is in percent, so 0.4 means S_c = 1.004.
    The implementation is Petters and Kreidenweis Eq. 10, also used in the
    supplied kappa_kohler_theory.py helper.
    """

    kappa_array = _as_float_array(kappa)
    ss_percent = _as_float_array(supersaturation_percent)
    saturation_ratio = 1.0 + ss_percent / 100.0
    log_s = np.log(saturation_ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        diameter_m = (
            (4.0 * constants.kelvin_a_m**3)
            / (27.0 * kappa_array * log_s**2)
        ) ** (1.0 / 3.0)
    invalid = (kappa_array <= 0.0) | (ss_percent <= 0.0) | ~np.isfinite(diameter_m)
    return np.where(invalid, np.nan, diameter_m * 1.0e9)


def critical_supersaturation_percent(
    dry_diameter_nm: float | np.ndarray,
    kappa: float | np.ndarray,
    constants: KohlerConstants = KohlerConstants(),
) -> np.ndarray:
    """Return critical supersaturation percent from dry diameter and kappa."""

    diameter_m = _as_float_array(dry_diameter_nm) * 1.0e-9
    kappa_array = _as_float_array(kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturation_ratio = np.exp(
            np.sqrt(
                (4.0 * constants.kelvin_a_m**3)
                / (27.0 * diameter_m**3 * kappa_array)
            )
        )
    ss_percent = (saturation_ratio - 1.0) * 100.0
    invalid = (diameter_m <= 0.0) | (kappa_array <= 0.0) | ~np.isfinite(ss_percent)
    return np.where(invalid, np.nan, ss_percent)


def kappa_from_critical_diameter(
    critical_diameter_nm: float | np.ndarray,
    supersaturation_percent: float | np.ndarray,
    constants: KohlerConstants = KohlerConstants(),
) -> np.ndarray:
    """Return kappa from dry critical diameter and supersaturation."""

    diameter_m = _as_float_array(critical_diameter_nm) * 1.0e-9
    ss_percent = _as_float_array(supersaturation_percent)
    saturation_ratio = 1.0 + ss_percent / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (
            4.0
            * constants.kelvin_a_m**3
            / (27.0 * diameter_m**3 * np.log(saturation_ratio) ** 2)
        )
    invalid = (diameter_m <= 0.0) | (ss_percent <= 0.0) | ~np.isfinite(kappa)
    return np.where(invalid, np.nan, kappa)


def geometric_mean(values: np.ndarray) -> float:
    finite_positive = np.asarray(values, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(finite_positive))))


def geometric_std(values: np.ndarray) -> float:
    finite_positive = np.asarray(values, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size < 2:
        return float("nan")
    return float(np.exp(np.std(np.log(finite_positive), ddof=1)))


def _interp_at_log_diameter(
    log_diameter: np.ndarray,
    values: np.ndarray,
    log_critical: float,
) -> float:
    finite = np.isfinite(log_diameter) & np.isfinite(values)
    if finite.sum() < 2:
        return float("nan")
    return float(np.interp(log_critical, log_diameter[finite], values[finite]))


def _trapezoid(y_values: np.ndarray, x_values: np.ndarray) -> float:
    if y_values.size < 2 or x_values.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y_values[1:] + y_values[:-1]) * np.diff(x_values)))


def integrate_activated_number(
    diameter_nm: np.ndarray,
    dndlogdp: np.ndarray,
    critical_diameter_nm_value: float,
) -> float:
    """Integrate dN/dlog10Dp above the critical dry diameter."""

    diameter = np.asarray(diameter_nm, dtype=np.float64)
    spectrum = np.asarray(dndlogdp, dtype=np.float64)
    finite = np.isfinite(diameter) & np.isfinite(spectrum) & (diameter > 0.0)
    finite &= spectrum >= 0.0
    if finite.sum() < 2 or not np.isfinite(critical_diameter_nm_value):
        return float("nan")

    diameter = diameter[finite]
    spectrum = spectrum[finite]
    order = np.argsort(diameter)
    diameter = diameter[order]
    spectrum = spectrum[order]
    log_diameter = np.log10(diameter)
    log_critical = np.log10(critical_diameter_nm_value)

    if log_critical <= log_diameter[0]:
        return _trapezoid(spectrum, log_diameter)
    if log_critical >= log_diameter[-1]:
        return 0.0

    keep = log_diameter > log_critical
    critical_value = _interp_at_log_diameter(log_diameter, spectrum, log_critical)
    integration_x = np.concatenate([[log_critical], log_diameter[keep]])
    integration_y = np.concatenate([[critical_value], spectrum[keep]])
    return _trapezoid(integration_y, integration_x)


def predict_ccn_concentration(
    diameter_nm: np.ndarray,
    dndlogdp: np.ndarray,
    kappa: float,
    supersaturation_percent: float,
    constants: KohlerConstants = KohlerConstants(),
) -> tuple[float, float]:
    """Predict CCN number concentration and return (N_CCN, Dcrit_nm)."""

    dcrit_nm = float(
        critical_diameter_nm(
            kappa,
            supersaturation_percent,
            constants=constants,
        )
    )
    return integrate_activated_number(diameter_nm, dndlogdp, dcrit_nm), dcrit_nm
