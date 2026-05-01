"""Helpers for building JSON-friendly experiment and evaluation reports."""

CLASS_NAME_MAP = {
    "healthy": "healthy",
    "melanoma": "melanoma",
    "nevus": "nevus",
    "chrpe": "chrpe",
}


def to_jsonable(value):
    """Recursively convert values into JSON-safe Python primitives."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _ci_triplet_to_pct(summary):
    """Convert a ``(mean, lower, upper)`` metric summary from 0..1 to percentage."""

    mean, lower, upper = summary
    return {
        "mean": mean * 100.0,
        "lower": lower * 100.0,
        "upper": upper * 100.0,
    }


def format_bootstrap_metrics(raw_metrics):
    """Normalize bootstrap output into explicit percentage-based JSON sections."""

    metric_key_map = {
        "accuracy": "accuracy_overall",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1_score": "f1_macro",
    }

    bootstrap_ci_pct = {}
    for raw_key, target_key in metric_key_map.items():
        bootstrap_ci_pct[target_key] = _ci_triplet_to_pct(raw_metrics[raw_key])

    class_metrics = {}
    for raw_name, class_name in CLASS_NAME_MAP.items():
        class_metrics[class_name] = {}
        for raw_key, target_key in metric_key_map.items():
            metric_key = target_key.replace("_overall", "").replace("_macro", "")
            class_metrics[class_name][metric_key] = _ci_triplet_to_pct(
                (
                    raw_metrics[f"{raw_key}_{raw_name}"][0],
                    raw_metrics[f"{raw_key}_{raw_name}"][1][0],
                    raw_metrics[f"{raw_key}_{raw_name}"][1][1],
                )
            )

    return {
        "bootstrap_ci_pct": to_jsonable(bootstrap_ci_pct),
        "class_one_vs_rest_bootstrap_ci_pct": to_jsonable(class_metrics),
    }


def format_test_metrics(point_metrics_pct, raw_bootstrap_metrics=None):
    """Build a clear evaluation report with explicit metric conventions.

    Returned sections use these conventions:
    - ``point_metrics_pct``:
      - ``accuracy_overall``
      - ``precision_weighted``
      - ``recall_weighted``
      - ``f1_weighted``
    - ``bootstrap_ci_pct``:
      - ``accuracy_overall``
      - ``precision_macro``
      - ``recall_macro``
      - ``f1_macro``
    - ``class_one_vs_rest_bootstrap_ci_pct``:
      per-class one-vs-rest summaries for accuracy / precision / recall / f1
    """

    report = {
        "point_metrics_pct": to_jsonable(point_metrics_pct),
    }
    if raw_bootstrap_metrics is not None:
        report.update(format_bootstrap_metrics(raw_bootstrap_metrics))
    return report
