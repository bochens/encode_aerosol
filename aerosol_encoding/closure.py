from __future__ import annotations

from collections.abc import Mapping

import torch


SIZE_MODALITIES = ("size_smps", "size_aps", "size_uhsas", "size_opc")


def _indices_containing(feature_names: tuple[str, ...], needles: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(
        index
        for index, feature_name in enumerate(feature_names)
        if any(needle in feature_name for needle in needles)
    )


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
    ) -> None:
        self.weights = {name: float(weight) for name, weight in weights.items() if weight > 0}
        self.mean_by_modality = dict(mean_by_modality)
        self.std_by_modality = dict(std_by_modality)
        self.dry_neph_indices = _indices_containing(
            feature_names_by_modality.get("optical_neph", ()),
            ("_Dry_Neph3W",),
        )
        self.neph_humidification_pairs = _paired_neph_indices(
            feature_names_by_modality.get("optical_neph", ())
        )
        self.ccn_number_indices = _indices_containing(
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

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        valid = mask.sum()
        if valid <= 0:
            return None
        return (((prediction - target) ** 2) * mask).sum() / valid

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
        return self._masked_mse(prediction, target, mask)

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
        return self._masked_mse(prediction, target, mask)

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
        return self._masked_mse(prediction, target, mask)
