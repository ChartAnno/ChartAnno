"""Standalone SVG adapter for the ChartAnno rule-based evaluator."""

def extract_diffed_svg_bundle(*args, **kwargs):
    from .pipeline import extract_diffed_svg_bundle as _func
    return _func(*args, **kwargs)

__all__ = ["extract_diffed_svg_bundle"]
