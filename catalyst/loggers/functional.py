# def format_metric(name: str, value: float) -> str:
#     """Format metric.
#
#     Metric will be returned in the scientific format if 4
#     decimal chars are not enough (metric value lower than 1e-4).
#
#     Args:
#         name: metric name
#         value: value of metric
#
#     Returns:
#         str: formatted metric
#     """
#     if value < 1e-4:
#         return f"{name}={value:1.3e}"
#     return f"{name}={value:.4f}"
