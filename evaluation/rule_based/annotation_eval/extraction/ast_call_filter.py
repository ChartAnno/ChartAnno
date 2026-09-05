"""AST-based filtering for plotting script source.

This module supports two optional source-level filters:
1. Strip grid-related calls (e.g., ax.grid / xaxis.grid / yaxis.grid).
2. Remove *base chart drawing* calls that are already present in a removed-reference script.

For removed-reference dedupe we use STRICT signature matching:
- function attribute chain (e.g. ``ax.plot`` vs ``plt.plot`` are different),
- positional args AST,
- keyword args AST (and their written order).
Only exact signature matches are removed.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Iterable


# Base chart primitives only; annotation-oriented calls are intentionally excluded.
BASE_DRAW_CALL_NAMES = {
    "plot",
    "scatter",
    "bar",
    "barh",
    "fill_between",
    "fill_betweenx",
    "text",
    "figtext",
    "annotate",
    "axhline",
    "hlines",
    "axvline",
    "vlines",
    "axhspan",
    "axvspan",
}

GRID_CALL_NAMES = {"grid"}


def _attr_chain_name(func: ast.AST) -> str:
    parts = []
    cur = func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)


def _call_leaf_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _node_key(node: ast.AST | None) -> str:
    if node is None:
        return "None"
    # include_attributes=False keeps key stable across files.
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _call_signature(call: ast.Call) -> str:
    func_name = _attr_chain_name(call.func) or _call_leaf_name(call)
    args_key = ",".join(_node_key(arg) for arg in call.args)
    kw_pairs = [(kw.arg or "", _node_key(kw.value)) for kw in call.keywords]
    kwargs_key = ",".join(f"{k}={v}" for k, v in kw_pairs)
    return f"{func_name}|args=[{args_key}]|kwargs=[{kwargs_key}]"


def _is_grid_call(call: ast.Call) -> bool:
    return _call_leaf_name(call) in GRID_CALL_NAMES


def _is_base_drawing_call(call: ast.Call) -> bool:
    return _call_leaf_name(call) in BASE_DRAW_CALL_NAMES


def _iter_top_level_call_nodes(tree: ast.AST) -> Iterable[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            yield node.value
            continue
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            yield node.value
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            yield node.value


def collect_drawing_call_counter(source: str) -> Counter[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Counter()

    counter: Counter[str] = Counter()
    for call in _iter_top_level_call_nodes(tree):
        if _is_grid_call(call):
            continue
        if not _is_base_drawing_call(call):
            continue
        counter[_call_signature(call)] += 1
    return counter


class _DrawingCallFilter(ast.NodeTransformer):
    def __init__(
        self,
        removed_counter: Counter[str] | None = None,
        strip_grid_calls: bool = False,
    ):
        super().__init__()
        self.removed_counter = removed_counter or Counter()
        self.strip_grid_calls = strip_grid_calls

    def _should_drop_call(self, call: ast.Call) -> bool:
        if self.strip_grid_calls and _is_grid_call(call):
            return True

        if not self.removed_counter:
            return False
        if not _is_base_drawing_call(call):
            return False

        sig = _call_signature(call)
        if self.removed_counter.get(sig, 0) <= 0:
            return False
        self.removed_counter[sig] -= 1
        return True

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if self._should_drop_call(node.value):
                return None
        return node

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Call) and self._should_drop_call(node.value):
            return None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Call) and self._should_drop_call(node.value):
            return None
        return node


def apply_ast_filters(
    source: str,
    removed_source: str | None = None,
    strip_grid_calls: bool = False,
    remove_removed_drawing_calls: bool = False,
) -> str:
    if not strip_grid_calls and not remove_removed_drawing_calls:
        return source

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    removed_counter: Counter[str] | None = None
    if remove_removed_drawing_calls and removed_source:
        removed_counter = collect_drawing_call_counter(removed_source)

    transformer = _DrawingCallFilter(
        removed_counter=removed_counter,
        strip_grid_calls=strip_grid_calls,
    )
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        return source


__all__ = [
    "apply_ast_filters",
    "collect_drawing_call_counter",
]
