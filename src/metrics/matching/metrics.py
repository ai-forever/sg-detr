"""Matching metrics."""

from typing import Any, Dict

from torchmetrics import Metric


class MatchingMetric(Metric):
    """Matching costs statistics implemented based on torchmetrics.Metric."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MatchingMetric.

        Args:
            kwargs: some kwargs.
        """
        super().__init__(**kwargs)
        self.storage: Dict[str, Any] = {"positive": None, "positive_aux": None}

    # pylint: disable=arguments-differ
    def update(self, matching: Dict[str, Any]) -> None:  # noqa: WPS221,C901,WPS231
        """Update the statistics with new statistics values.

        Args:
            matching (Dict[str, Any]): Batch matching statistics.
        """
        if self.storage["positive"] is None:
            self.storage["positive"] = matching["positive"]["costs"]
            self.storage["positive_aux"] = matching["positive_aux"]["costs"]
            return None
        for stat_key, _ in self.storage["positive"].items():  # noqa: WPS204
            self.storage["positive"][stat_key] += matching["positive"]["costs"][stat_key]

        for storage_idx, storege_items in enumerate(self.storage["positive_aux"]):
            for stat_key, _ in storege_items.items():
                stat_value = matching["positive_aux"]["costs"][storage_idx][stat_key]  # noqa: WPS220
                self.storage["positive_aux"][storage_idx][stat_key] += stat_value

    def reset(self) -> None:
        """Reset statistics storate."""
        self.storage = {"positive": None, "positive_aux": None}

    def compute(self) -> Dict[str, float]:  # noqa: C901 WPS231
        """
        Compute matching statistics costs.

        Returns:
            Dict[str, float]: cumulative statistics dict
        """
        metrics: Dict[str, float] = {}

        if self.storage["positive"] is not None:  # noqa: WPS204
            n_gt_spans = self.storage["positive"]["n_gt_spans"]  # type: ignore
            for stat_key, stat_value in self.storage["positive"].items():  # type: ignore
                if stat_key != "n_gt_spans":
                    metrics[f"{stat_key}"] = stat_value / n_gt_spans

        if self.storage["positive_aux"] is not None:
            for storage_idx, storage_item in enumerate(self.storage["positive_aux"]):  # type: ignore
                n_gt_spans = storage_item["n_gt_spans"]  # type: ignore
                for stat_key, stat_value in storage_item.items():  # type: ignore
                    if stat_key != "n_gt_spans":
                        metrics[f"aux_{storage_idx}_{stat_key}"] = stat_value / n_gt_spans  # noqa: WPS220
        return metrics
