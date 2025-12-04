"""Module for computing losses."""

from typing import Any, Dict, List

import torch
from torch import nn

from src.losses.auxiliary_losses import AuxiliaryLosses
from src.losses.matcher import HungarianMatcher
from src.losses.mom2txt_losses import Moment2TextLosses
from src.losses.regression_losses.atss_losses import ATSSRetrievalLoss
from src.losses.regression_losses.aux_references_losses import AuxRefLosses
from src.losses.regression_losses.denoise_losses import DenoiseLosses
from src.losses.regression_losses.retrieval_losses import MainRegressionLosses
from src.losses.saliency_losses import SaliencyLosses

EPS: float = 1e-7


def create_targets_k_repeats(targets: Dict[str, Any], times: int) -> Dict[str, Any]:
    """
    Duplicate target k times.

    Args:
        targets (Dict[str, Any]): target spans data
        times (int): number of replications

    Returns:
        Dict[str, Any]: Duplicated target data
    """
    targets_spans_k = []
    for item_targets in targets["span_labels"]:
        spans = item_targets["spans"]
        x_repeated = [spans] * times  # noqa: WPS435
        result = torch.cat(x_repeated, dim=0)
        targets_spans_k.append({"spans": result})
    return {"span_labels": targets_spans_k}


class SetCriterion(nn.Module):  # noqa: WPS230, WPS211
    """Compute the loss for DETR."""

    # pylint: disable=too-many-locals, too-many-arguments
    def __init__(  # noqa: WPS211
        self,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, int],
        main_reg_losses: MainRegressionLosses,
        top_k_positive_anchors: int = 9,
        saliency_margin: float = 0.15,
        contrastive_reducer: float = 0.25,
        denoise_reducer: float = 0.5,
        colab_ref_reducer: float = 0.5,
        target_repeat: int = 3,
        one2one: bool = True,
        use_focal: bool = True,
        gamma: float = 2,
        local_saliency_loss_scale: float = 1.0,
        use_negative_losses: bool = True,
    ) -> None:
        """
        Create the criterion.

        Args:
            matcher (HungarianMatcher): instance of the Matcher class used to match predictions and targets
            weight_dict (Dict[str, int]): Key is the name of the loss and Value its relative weight.
            main_reg_losses (MainRegressionLosses): main detr losses.
            top_k_positive_anchors (int): top k samples to select for ATSS loss.
            saliency_margin (float): margin for saliency loss
            contrastive_reducer (float): weight reducer for constrastive loss
            denoise_reducer (float): weight reducer for denoise loss
            colab_ref_reducer (float): weight reducer for colab ref loss
            target_repeat (int): number of times to repeat the target.
            one2one (bool): matching type
            use_focal: whether to use focal loss or not.
            gamma (float): Gamma factor for focal loss calculation. Defaults to 2.0.
            local_saliency_loss_scale (float): scale for local saliency loss.
            use_negative_losses (bool): whether to use negative losses or not.
        """
        super().__init__()
        self.matcher = matcher
        self.target_repeat = target_repeat
        self.one2one = one2one
        self.weight_dict = weight_dict

        # span losses
        self.retrieval_losses = main_reg_losses
        self.denoise_losses = DenoiseLosses(use_focal=use_focal, gamma=gamma, denoise_reducer=denoise_reducer)
        self.aux_ref_losses = AuxRefLosses(use_focal=use_focal, gamma=gamma, colab_ref_reducer=colab_ref_reducer)
        self.aux_head_losses = ATSSRetrievalLoss(top_k_positive_anchors=top_k_positive_anchors)

        # saliency losses
        self.saliency_losses = SaliencyLosses(
            saliency_margin=saliency_margin,
            contrastive_reducer=contrastive_reducer,
            local_saliency_loss_scale=local_saliency_loss_scale,
            use_negative_losses=use_negative_losses,
        )
        #  other losses
        self.auxiliary_losses = AuxiliaryLosses()
        self.moment2text_losses = Moment2TextLosses()

    def compute_matches(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        pos_ref_points: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute matching for interim encoder, detector heads including auxiliary heads.

        Args:
            outputs (Dict[str, Any]): model outputs
            targets (Dict[str, Any]): targets
            pos_ref_points (torch.Tensor): reference points

        Returns:
            Dict[str, Any]: dict with matching indexes and corresponding costs.
        """
        matching = {}
        # Retrieve the matching
        if self.one2one:
            retrieval_targets = targets
        else:
            retrieval_targets = create_targets_k_repeats(targets, self.target_repeat)

        if "encoder_outputs" in outputs:
            ecnoder_outputs = outputs["encoder_outputs"]
            indices, enc_matcher_costs = self.matcher(ecnoder_outputs, retrieval_targets, pos_ref_points)
            matching["encoder"] = {"indices": indices, "costs": enc_matcher_costs}

        # outputs without aux
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}  # noqa: WPS204
        indices, pos_matcher_costs = self.matcher(outputs_without_aux, retrieval_targets, pos_ref_points)
        matching["positive"] = {"indices": indices, "costs": pos_matcher_costs}

        # compute auxiliary head matching
        matching["positive_aux"] = {"indices": [], "costs": []}
        for aux_outputs in outputs.get("aux_outputs"):  # type: ignore
            indices, costs = self.matcher(aux_outputs, retrieval_targets, pos_ref_points)
            matching["positive_aux"]["indices"].append(indices)
            matching["positive_aux"]["costs"].append(costs)
        return matching

    def forward(  # noqa: WPS213,C901
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
        matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the losses for the model during training.

        Args:
            outputs (Dict[str, Any]): dict of tensors, see the output specification of the model for the format
            targets (Dict[str, Any]): Targets to use.
            meta (List[Dict[str, Any]]): Meta information.
            matching (Dict[str, Any]): Matching results.

        Returns:
            Dict[str, Any]: dict of tensors, with the loss values.
        """
        # Compute main losses
        losses = {}

        # Retrieve the matching
        if self.one2one:
            retrieval_targets = targets
        else:
            retrieval_targets = create_targets_k_repeats(targets, self.target_repeat)

        # compute losses for main head
        outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}  # noqa: WPS204
        indices = matching["positive"]["indices"]
        enc_indices = matching["encoder"]["indices"] if "encoder" in matching else None
        losses.update(self.saliency_losses(outputs_without_aux, targets))
        losses.update(self.retrieval_losses(outputs_without_aux, retrieval_targets, indices, enc_indices))
        losses.update(self.auxiliary_losses(outputs_without_aux))
        losses.update(self.moment2text_losses(outputs_without_aux, targets))
        losses.update(self.aux_head_losses(outputs_without_aux, targets, meta))

        if outputs["collab_ref_dict"] is not None:
            losses.update(self.aux_ref_losses(outputs["collab_ref_dict"], aux_num=-1))

        if outputs["denoise_ref_dict"] is not None:
            losses.update(self.denoise_losses(outputs["denoise_ref_dict"], targets, aux_num=-1))

        if "aux_outputs" not in outputs:
            return losses

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        for idx, aux_outputs in enumerate(outputs.get("aux_outputs")):  # type: ignore
            indices = matching["positive_aux"]["indices"][idx]
            loss_dict = self.retrieval_losses(aux_outputs, retrieval_targets, indices)

            # update weights dict
            weight_dict = {f"{key}_{idx}": self.weight_dict.get(key, 0) for key, _ in loss_dict.items()}  # noqa: WPS221
            self.weight_dict.update(weight_dict)

            # update loss dict
            loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}  # noqa: WPS221
            losses.update(loss_dict)

            # compute aux references losses if it is possible
            if outputs["collab_ref_dict"] is not None:
                loss_dict_aux_ref = self.aux_ref_losses(outputs["collab_ref_dict"], aux_num=idx)
                loss_dict_aux_ref = {f"{key}_{idx}": value for key, value in loss_dict_aux_ref.items()}  # noqa: WPS221
                losses.update(loss_dict_aux_ref)

            # compute denoise losses if it is possible
            if outputs["denoise_ref_dict"] is not None:
                loss_dict_denoise = self.denoise_losses(outputs["denoise_ref_dict"], targets, aux_num=idx)
                loss_dict_denoise = {f"{key}_{idx}": value for key, value in loss_dict_denoise.items()}  # noqa: WPS221
                losses.update(loss_dict_denoise)
        return losses
