from __future__ import annotations

from collections.abc import Mapping
import re

import torch
import torch.nn.functional as F


SIZE_MODALITIES = ("size_smps", "size_aps", "size_uhsas", "size_opc")
DIAMETER_RE = re.compile(
    r"__dN_dlogDp(?:__stat_[^_]+)?__diameter_common_nm_(?P<diameter>[0-9.eE+-]+)"
)
TIME_BIN_RE = re.compile(r"__time_bin_(?P<time_bin>\d+)")


def _indices_containing(feature_names: tuple[str, ...], needles: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(
        index
        for index, feature_name in enumerate(feature_names)
        if any(needle in feature_name for needle in needles)
    )


def _mean_indices_containing(
    feature_names: tuple[str, ...],
    needles: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, feature_name in enumerate(feature_names)
        if "__stat_" not in feature_name
        and any(needle in feature_name for needle in needles)
    )


def _element_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_kind: str,
    smooth_l1_beta: float,
    huber_delta: float,
) -> torch.Tensor:
    if loss_kind == "mse":
        return (prediction - target) ** 2
    if loss_kind == "smooth_l1":
        return F.smooth_l1_loss(
            prediction,
            target,
            beta=smooth_l1_beta,
            reduction="none",
        )
    if loss_kind == "huber":
        return F.huber_loss(
            prediction,
            target,
            delta=huber_delta,
            reduction="none",
        )
    raise ValueError(f"Unknown regression loss kind: {loss_kind}")


def _paired_neph_indices(feature_names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    dry_lookup = {}
    wet_lookup = {}
    for index, feature_name in enumerate(feature_names):
        if "__Bs_" not in feature_name and "__Bbs_" not in feature_name:
            continue
        key = (
            feature_name
            .replace("_Dry_Neph3W", "_Neph3W")
            .replace("_Wet_Neph3W", "_Neph3W")
            .replace("__neph_dry_", "__neph_")
            .replace("__neph_wet_", "__neph_")
        )
        if "_Dry_Neph3W" in feature_name:
            dry_lookup[key] = index
        elif "_Wet_Neph3W" in feature_name:
            wet_lookup[key] = index
    return tuple(
        (dry_lookup[key], wet_lookup[key])
        for key in sorted(dry_lookup.keys() & wet_lookup.keys())
    )


class AerosolClosureLosses:
    def __init__(
        self,
        feature_names_by_modality: Mapping[str, tuple[str, ...]],
        mean_by_modality: Mapping[str, torch.Tensor],
        std_by_modality: Mapping[str, torch.Tensor],
        weights: Mapping[str, float],
        loss_kind: str = "mse",
        smooth_l1_beta: float = 0.5,
        huber_delta: float = 1.0,
    ) -> None:
        self.weights = {name: float(weight) for name, weight in weights.items() if weight > 0}
        if loss_kind not in {"mse", "smooth_l1", "huber"}:
            raise ValueError(f"Unknown closure loss kind: {loss_kind}")
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.loss_kind = loss_kind
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.huber_delta = float(huber_delta)
        self.mean_by_modality = dict(mean_by_modality)
        self.std_by_modality = dict(std_by_modality)
        self.dry_neph_indices = _indices_containing(
            feature_names_by_modality.get("optical_neph", ()),
            ("_Dry_Neph3W",),
        )
        self.neph_humidification_pairs = _paired_neph_indices(
            feature_names_by_modality.get("optical_neph", ())
        )
        self.ccn_number_indices = _mean_indices_containing(
            feature_names_by_modality.get("ccn_activation", ()),
            ("__N_CCN",),
        )
        self.cpc_number_indices = _indices_containing(
            feature_names_by_modality.get("cpc_number", ()),
            ("__concentration",),
        )
        self._validate_requested_losses(feature_names_by_modality)

    @property
    def enabled(self) -> bool:
        return bool(self.weights)

    def _validate_requested_losses(
        self,
        feature_names_by_modality: Mapping[str, tuple[str, ...]],
    ) -> None:
        if "dry_scattering_from_size_composition" in self.weights:
            missing = [
                modality
                for modality in ("chemistry_acsm", "optical_neph")
                if modality not in feature_names_by_modality
            ]
            if missing:
                raise ValueError(
                    "dry_scattering_from_size_composition closure is missing modalities: "
                    + ", ".join(missing)
                )
            if not self.dry_neph_indices:
                raise ValueError("dry_scattering_from_size_composition found no dry neph features")
        if "humidification_response" in self.weights and not self.neph_humidification_pairs:
            raise ValueError("humidification_response closure found no paired dry/wet neph features")
        if "ccn_activation_ratio" in self.weights:
            if not self.ccn_number_indices:
                raise ValueError("ccn_activation_ratio closure found no N_CCN feature")
            if not self.cpc_number_indices:
                raise ValueError("ccn_activation_ratio closure found no CPC concentration feature")

    def to(self, device: torch.device) -> AerosolClosureLosses:
        self.mean_by_modality = {
            modality: values.to(device)
            for modality, values in self.mean_by_modality.items()
        }
        self.std_by_modality = {
            modality: values.to(device)
            for modality, values in self.std_by_modality.items()
        }
        return self

    def _transformed_values(self, modality: str, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.std_by_modality[modality] + self.mean_by_modality[modality]

    def _masked_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        valid = mask.sum()
        if valid <= 0:
            return None
        element_loss = _element_regression_loss(
            prediction,
            target,
            self.loss_kind,
            self.smooth_l1_beta,
            self.huber_delta,
        )
        return (element_loss * mask).sum() / valid

    def __call__(
        self,
        decoded: Mapping[str, torch.Tensor],
        x_by_modality: Mapping[str, torch.Tensor],
        mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        if "dry_scattering_from_size_composition" in self.weights:
            loss = self._dry_scattering_loss(
                decoded,
                x_by_modality,
                mask_by_modality,
                input_modality_mask,
            )
            if loss is not None:
                losses["closure_dry_scattering_from_size_composition"] = loss
        if "humidification_response" in self.weights:
            loss = self._humidification_response_loss(
                decoded,
                x_by_modality,
                mask_by_modality,
            )
            if loss is not None:
                losses["closure_humidification_response"] = loss
        if "ccn_activation_ratio" in self.weights:
            loss = self._ccn_activation_ratio_loss(
                decoded,
                x_by_modality,
                mask_by_modality,
            )
            if loss is not None:
                losses["closure_ccn_activation_ratio"] = loss
        return losses

    def _dry_scattering_loss(
        self,
        decoded: Mapping[str, torch.Tensor],
        x_by_modality: Mapping[str, torch.Tensor],
        mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        size_visible = None
        for modality in SIZE_MODALITIES:
            if modality not in input_modality_mask:
                continue
            size_visible = (
                input_modality_mask[modality]
                if size_visible is None
                else size_visible | input_modality_mask[modality]
            )
        if size_visible is None or "chemistry_acsm" not in input_modality_mask:
            return None
        source_visible = size_visible & input_modality_mask["chemistry_acsm"]
        indices = torch.as_tensor(
            self.dry_neph_indices,
            dtype=torch.long,
            device=decoded["optical_neph"].device,
        )
        prediction = self._transformed_values(
            "optical_neph",
            decoded["optical_neph"],
        ).index_select(1, indices)
        target = self._transformed_values(
            "optical_neph",
            x_by_modality["optical_neph"],
        ).index_select(1, indices)
        mask = mask_by_modality["optical_neph"].index_select(1, indices)
        mask = mask * source_visible.to(dtype=mask.dtype).unsqueeze(-1)
        return self._masked_loss(prediction, target, mask)

    def _humidification_response_loss(
        self,
        decoded: Mapping[str, torch.Tensor],
        x_by_modality: Mapping[str, torch.Tensor],
        mask_by_modality: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        device = decoded["optical_neph"].device
        dry_indices = torch.as_tensor(
            [dry for dry, _ in self.neph_humidification_pairs],
            dtype=torch.long,
            device=device,
        )
        wet_indices = torch.as_tensor(
            [wet for _, wet in self.neph_humidification_pairs],
            dtype=torch.long,
            device=device,
        )
        prediction_transformed = self._transformed_values("optical_neph", decoded["optical_neph"])
        target_transformed = self._transformed_values("optical_neph", x_by_modality["optical_neph"])
        prediction = (
            prediction_transformed.index_select(1, wet_indices)
            - prediction_transformed.index_select(1, dry_indices)
        )
        target = (
            target_transformed.index_select(1, wet_indices)
            - target_transformed.index_select(1, dry_indices)
        )
        mask = (
            mask_by_modality["optical_neph"].index_select(1, wet_indices)
            * mask_by_modality["optical_neph"].index_select(1, dry_indices)
        )
        return self._masked_loss(prediction, target, mask)

    def _ccn_activation_ratio_loss(
        self,
        decoded: Mapping[str, torch.Tensor],
        x_by_modality: Mapping[str, torch.Tensor],
        mask_by_modality: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        device = decoded["ccn_activation"].device
        ccn_indices = torch.as_tensor(
            self.ccn_number_indices,
            dtype=torch.long,
            device=device,
        )
        cpc_indices = torch.as_tensor(
            self.cpc_number_indices,
            dtype=torch.long,
            device=device,
        )
        prediction_ccn = self._transformed_values(
            "ccn_activation",
            decoded["ccn_activation"],
        ).index_select(1, ccn_indices).mean(dim=1, keepdim=True)
        prediction_cpc = self._transformed_values(
            "cpc_number",
            decoded["cpc_number"],
        ).index_select(1, cpc_indices).mean(dim=1, keepdim=True)
        target_ccn = self._transformed_values(
            "ccn_activation",
            x_by_modality["ccn_activation"],
        ).index_select(1, ccn_indices).mean(dim=1, keepdim=True)
        target_cpc = self._transformed_values(
            "cpc_number",
            x_by_modality["cpc_number"],
        ).index_select(1, cpc_indices).mean(dim=1, keepdim=True)
        prediction = prediction_ccn - prediction_cpc
        target = target_ccn - target_cpc
        mask = (
            mask_by_modality["ccn_activation"].index_select(1, ccn_indices).max(dim=1, keepdim=True).values
            * mask_by_modality["cpc_number"].index_select(1, cpc_indices).max(dim=1, keepdim=True).values
        )
        return self._masked_loss(prediction, target, mask)


def _size_spectrum_groups(feature_names: tuple[str, ...]) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
    grouped: dict[int, list[tuple[int, float]]] = {}
    for index, feature_name in enumerate(feature_names):
        diameter_match = DIAMETER_RE.search(feature_name)
        if diameter_match is None:
            continue
        time_match = TIME_BIN_RE.search(feature_name)
        time_bin = int(time_match.group("time_bin")) if time_match is not None else 0
        grouped.setdefault(time_bin, []).append(
            (index, float(diameter_match.group("diameter")))
        )

    output = []
    for time_bin in sorted(grouped):
        ordered = sorted(grouped[time_bin], key=lambda item: item[1])
        if len(ordered) >= 2:
            output.append(
                (
                    tuple(index for index, _ in ordered),
                    tuple(diameter for _, diameter in ordered),
                )
            )
    return tuple(output)


class AerosolSizeSpectralLosses:
    def __init__(
        self,
        feature_names_by_modality: Mapping[str, tuple[str, ...]],
        mean_by_modality: Mapping[str, torch.Tensor],
        std_by_modality: Mapping[str, torch.Tensor],
        weights: Mapping[str, float],
        loss_kind: str = "mse",
        smooth_l1_beta: float = 0.5,
        huber_delta: float = 1.0,
    ) -> None:
        self.weights = {name: float(weight) for name, weight in weights.items() if weight > 0}
        if loss_kind not in {"mse", "smooth_l1", "huber"}:
            raise ValueError(f"Unknown size spectral loss kind: {loss_kind}")
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.loss_kind = loss_kind
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.huber_delta = float(huber_delta)
        self.mean_by_modality = dict(mean_by_modality)
        self.std_by_modality = dict(std_by_modality)
        self.groups = {
            modality: _size_spectrum_groups(feature_names_by_modality.get(modality, ()))
            for modality in SIZE_MODALITIES
        }
        if self.enabled and not any(self.groups.values()):
            raise ValueError("size spectral losses requested, but no dN_dlogDp groups were found")

    @property
    def enabled(self) -> bool:
        return bool(self.weights)

    def to(self, device: torch.device) -> AerosolSizeSpectralLosses:
        self.mean_by_modality = {
            modality: values.to(device)
            for modality, values in self.mean_by_modality.items()
        }
        self.std_by_modality = {
            modality: values.to(device)
            for modality, values in self.std_by_modality.items()
        }
        return self

    def _log_values(
        self,
        modality: str,
        normalized: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.mean_by_modality[modality].index_select(0, indices)
        std = self.std_by_modality[modality].index_select(0, indices)
        return normalized.index_select(1, indices) * std.unsqueeze(0) + mean.unsqueeze(0)

    @staticmethod
    def _diameter_weights(diameters_nm: tuple[float, ...], device: torch.device) -> torch.Tensor:
        log_diameter = torch.log10(
            torch.as_tensor(diameters_nm, dtype=torch.float32, device=device)
        )
        if log_diameter.numel() == 2:
            spacing = torch.full_like(log_diameter, torch.diff(log_diameter).abs().mean())
        else:
            spacing = torch.empty_like(log_diameter)
            spacing[1:-1] = (log_diameter[2:] - log_diameter[:-2]).abs() * 0.5
            spacing[0] = (log_diameter[1] - log_diameter[0]).abs()
            spacing[-1] = (log_diameter[-1] - log_diameter[-2]).abs()
        return spacing.clamp_min(1e-6)

    @staticmethod
    def _moments(
        concentration: torch.Tensor,
        mask: torch.Tensor,
        diameters_nm: tuple[float, ...],
    ) -> torch.Tensor:
        device = concentration.device
        diameter = torch.as_tensor(diameters_nm, dtype=concentration.dtype, device=device)
        dlog = AerosolSizeSpectralLosses._diameter_weights(diameters_nm, device).to(
            dtype=concentration.dtype
        )
        weighted = concentration * mask
        number = (weighted * dlog.unsqueeze(0)).sum(dim=1)
        surface = (weighted * (diameter ** 2).unsqueeze(0) * dlog.unsqueeze(0)).sum(dim=1)
        volume = (weighted * (diameter ** 3).unsqueeze(0) * dlog.unsqueeze(0)).sum(dim=1)
        return torch.stack([number, surface, volume], dim=1)

    def __call__(
        self,
        decoded: Mapping[str, torch.Tensor],
        x_by_modality: Mapping[str, torch.Tensor],
        mask_by_modality: Mapping[str, torch.Tensor],
        loss_row_masks: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, list[torch.Tensor]] = {name: [] for name in self.weights}
        for modality, groups in self.groups.items():
            if not groups or modality not in decoded:
                continue
            row_mask = None if loss_row_masks is None else loss_row_masks.get(modality)
            for local_indices, diameters_nm in groups:
                indices = torch.as_tensor(
                    local_indices,
                    dtype=torch.long,
                    device=decoded[modality].device,
                )
                target_mask = mask_by_modality[modality].index_select(1, indices).to(
                    dtype=decoded[modality].dtype
                )
                if row_mask is not None:
                    target_mask = target_mask * row_mask.to(dtype=target_mask.dtype).unsqueeze(-1)
                if target_mask.sum() <= 0:
                    continue

                pred_log = self._log_values(modality, decoded[modality], indices)
                target_log = self._log_values(modality, x_by_modality[modality], indices)
                if "log_spectrum" in losses:
                    element_loss = _element_regression_loss(
                        pred_log,
                        target_log,
                        self.loss_kind,
                        self.smooth_l1_beta,
                        self.huber_delta,
                    )
                    losses["log_spectrum"].append(
                        (element_loss * target_mask).sum()
                        / target_mask.sum().clamp_min(1.0)
                    )

                pred_conc = torch.expm1(pred_log).clamp_min(0.0)
                target_conc = torch.expm1(target_log).clamp_min(0.0)
                if "moment" in losses:
                    pred_moments = self._moments(pred_conc, target_mask, diameters_nm)
                    target_moments = self._moments(target_conc, target_mask, diameters_nm)
                    losses["moment"].append(
                        _element_regression_loss(
                            torch.log1p(pred_moments),
                            torch.log1p(target_moments),
                            self.loss_kind,
                            self.smooth_l1_beta,
                            self.huber_delta,
                        ).mean()
                    )
                if "shape" in losses:
                    dlog = self._diameter_weights(diameters_nm, decoded[modality].device).to(
                        dtype=decoded[modality].dtype
                    )
                    pred_weighted = pred_conc * target_mask * dlog.unsqueeze(0)
                    target_weighted = target_conc * target_mask * dlog.unsqueeze(0)
                    pred_shape = pred_weighted / pred_weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    target_shape = target_weighted / target_weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    valid_rows = target_mask.sum(dim=1) > 0
                    if torch.any(valid_rows):
                        losses["shape"].append(
                            _element_regression_loss(
                                pred_shape[valid_rows],
                                target_shape[valid_rows],
                                self.loss_kind,
                                self.smooth_l1_beta,
                                self.huber_delta,
                            ).mean()
                        )

        return {
            f"size_spectral_{name}": torch.stack(values).mean()
            for name, values in losses.items()
            if values
        }
