"""Expression-string reward DSL with a safe-eval AST whitelist."""

from __future__ import annotations

import ast
from collections.abc import Callable

import numpy as np

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression, ast.Expr,
    ast.Constant,
    ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Subscript, ast.Slice, ast.Tuple,
    ast.IfExp,
)

_ALLOWED_NAMES = frozenset({"state", "agent_feats", "True", "False"})
_SAFE_GLOBALS: dict = {"__builtins__": {}}


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"reward expression uses disallowed syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise ValueError(f"reward expression references unknown name: {node.id!r}")


def compile_reward(expression: str) -> Callable[[np.ndarray, int], float]:
    """Compile a reward expression string into a callable ``(features, state) -> float``.

    Raises :class:`ValueError` if the expression uses disallowed syntax,
    references unknown names, or fails to parse.
    """
    tree = ast.parse(expression.strip(), mode="eval")
    _validate_ast(tree)
    code = compile(tree, "<reward>", "eval")

    def fn(agent_feats: np.ndarray, state: int) -> float:
        return float(eval(code, _SAFE_GLOBALS, {"agent_feats": agent_feats, "state": state}))

    fn.__doc__ = f"Compiled reward expression: {expression.strip()}"
    return fn


class ScriptedReward:
    """Reward function compiled from an expression string, with ``.source`` preserved for logging and prompting."""

    def __init__(self, expression: str) -> None:
        self.source = expression.strip()
        self._fn = compile_reward(self.source)

    def __call__(self, agent_feats: np.ndarray, state: int) -> float:
        return self._fn(agent_feats, state)

    def __repr__(self) -> str:
        return f"ScriptedReward({self.source!r})"


def is_valid_expression(expression: str, n_features: int = 1) -> bool:
    """True if ``expression`` parses, validates, and runs on dummy inputs."""
    try:
        fn = compile_reward(expression)
    except (SyntaxError, ValueError):
        return False
    try:
        feats = np.zeros(max(1, n_features), dtype=np.float32)
        for s in (0, 1):
            fn(feats, s)
        return True
    except Exception:
        return False
