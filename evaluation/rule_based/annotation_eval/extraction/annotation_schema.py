"""Shared annotation schema keys."""

ANNOTATION_TYPES = (
    ("1_enclosure", "enclosure"),
    ("2_connector", "connector"),
    ("3_text", "text"),
    ("4_glyph", "glyph"),
    ("5_color", "color"),
    ("6_indicator", "indicator"),
    ("7_geometric", "geometric"),
)

FEATURE_ORDER = tuple(key for key, _ in ANNOTATION_TYPES)


def new_annotation_dict() -> dict[str, list]:
    return {key: [] for key in FEATURE_ORDER}
