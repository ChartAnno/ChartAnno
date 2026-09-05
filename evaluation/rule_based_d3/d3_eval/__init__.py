"""D3/SVG adapter for the ChartAnno rule-based annotation evaluator."""

def extract_diffed_d3_bundle(*args, **kwargs):
    from .pipeline import extract_diffed_d3_bundle as _func
    return _func(*args, **kwargs)

__all__ = ["extract_diffed_d3_bundle"]

