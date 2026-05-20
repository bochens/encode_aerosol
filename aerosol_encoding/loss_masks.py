from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def feature_role(modality: str, feature_name: str) -> str:
    """Classify a selected model feature by its scientific role.

    Roles are intentionally conservative:

    * target_response: measured aerosol response that can be reconstructed.
    * conditioning_coordinate: coordinate or operating condition needed to query
      a response, but not itself a predicted aerosol property.
    * context: environmental or meteorological state used as input context.
    * diagnostic: correction or summary variable that is useful metadata but not
      a primary target in the current representation.
    """

    if modality == "met_context":
        return "context"

    if modality == "ccn_activation":
        if "__N_CCN" in feature_name and "__stat_" not in feature_name:
            return "target_response"
        if (
            "__supersaturation_calculated" in feature_name
            or "__supersaturation_set_point" in feature_name
        ):
            return "conditioning_coordinate"
        return "diagnostic"

    if modality == "optical_neph":
        if "__Bs_" in feature_name or "__Bbs_" in feature_name:
            return "target_response"
        if (
            "__RH_Neph_" in feature_name
            or "__T_Neph_" in feature_name
            or "__P_Neph_" in feature_name
        ):
            return "conditioning_coordinate"
        return "diagnostic"

    if modality.startswith("size_"):
        if "__dN_dlogDp" in feature_name:
            return "target_response"
        return "diagnostic"

    if modality == "cpc_number":
        if "__concentration" in feature_name:
            return "target_response"
        return "diagnostic"

    if modality == "chemistry_acsm":
        if "__CDCE" in feature_name:
            return "diagnostic"
        return "target_response"

    return "target_response"


def is_target_response_feature(modality: str, feature_name: str) -> bool:
    return feature_role(modality, feature_name) == "target_response"


def make_feature_loss_masks(
    feature_names: Sequence[str],
    modality_indices: Mapping[str, Sequence[int]],
    modalities: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return per-modality feature masks for quantities that are true targets.

    Some modalities carry both measured responses and coordinates/operating
    conditions. Coordinates condition the decoder, but they are not target
    variables to reconstruct.
    """

    masks: dict[str, np.ndarray] = {}
    for modality in modalities:
        indices = modality_indices.get(modality)
        if indices is None:
            continue
        local_names = [feature_names[int(index)] for index in indices]
        mask = np.asarray(
            [
                is_target_response_feature(modality, feature_name)
                for feature_name in local_names
            ],
            dtype=np.float32,
        )
        if not np.any(mask):
            raise ValueError(
                f"{modality} loss mask found no target_response features. "
                "Check feature role classification before training."
            )
        masks[modality] = mask
    return masks
