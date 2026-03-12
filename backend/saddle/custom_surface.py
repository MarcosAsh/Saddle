"""
AST-based safe evaluator for user-defined surface expressions.

Only allows arithmetic, a small set of math functions, and the
variables x, y, pi, e. Everything else is rejected.
"""

from __future__ import annotations

import ast
import math
from typing import Callable

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

ALLOWED_FUNCTIONS = {"sin", "cos", "exp", "log", "sqrt", "abs", "tan", "tanh"}
ALLOWED_NAMES = {"x", "y", "pi", "e"}


def _validate_node(node: ast.AST) -> None:
    """Walk the AST and reject anything unsafe."""
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
    elif isinstance(node, ast.BinOp):
        _validate_node(node.left)
        _validate_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        _validate_node(node.operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Only simple function calls allowed, got {ast.dump(node.func)}")
        if node.func.id not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{node.func.id}' not allowed. Use: {', '.join(sorted(ALLOWED_FUNCTIONS))}")
        if len(node.args) != 1:
            raise ValueError(f"Function '{node.func.id}' takes exactly 1 argument")
        if node.keywords:
            raise ValueError("Keyword arguments not allowed")
        _validate_node(node.args[0])
    elif isinstance(node, ast.Name):
        if node.id not in ALLOWED_NAMES and node.id not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Name '{node.id}' not allowed. Use: x, y, pi, e")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants allowed, got {type(node.value).__name__}")
    else:
        raise ValueError(f"Expression type '{type(node).__name__}' not allowed")


def _parse_expr(expr: str) -> ast.Expression:
    """Parse and validate an expression string."""
    tree = ast.parse(expr, mode="eval")
    _validate_node(tree)
    return tree


def eval_custom_grid_np(
    expr: str,
    bounds: tuple[float, float, float, float],
    resolution: int,
) -> NDArray[np.float64]:
    """Evaluate expression on a grid using numpy. Returns (resolution, resolution) array."""
    _parse_expr(expr)  # validate first

    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xs, ys)

    ns = {
        "x": X, "y": Y,
        "pi": np.pi, "e": np.e,
        "sin": np.sin, "cos": np.cos, "exp": np.exp,
        "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
        "tan": np.tan, "tanh": np.tanh,
    }
    # eval is safe here: we validated the AST above and only allow known names
    result = eval(compile(ast.parse(expr, mode="eval"), "<custom>", "eval"), {"__builtins__": {}}, ns)
    return np.asarray(result, dtype=np.float64)


def make_custom_jax_fn(expr: str) -> Callable:
    """
    Build a differentiable JAX function from an expression string.

    Returns a function f(params) -> scalar where params = [x, y].
    """
    tree = _parse_expr(expr)

    def _compile_node(node: ast.AST) -> str:
        """Compile AST node back to string with jnp replacements."""
        if isinstance(node, ast.Expression):
            return _compile_node(node.body)
        if isinstance(node, ast.BinOp):
            left = _compile_node(node.left)
            right = _compile_node(node.right)
            ops = {
                ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
                ast.Div: "/", ast.Pow: "**", ast.Mod: "%",
            }
            op = ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Operator {type(node.op).__name__} not supported")
            return f"({left} {op} {right})"
        if isinstance(node, ast.UnaryOp):
            operand = _compile_node(node.operand)
            if isinstance(node.op, ast.USub):
                return f"(-{operand})"
            if isinstance(node.op, ast.UAdd):
                return f"(+{operand})"
            raise ValueError(f"Unary operator {type(node.op).__name__} not supported")
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            arg = _compile_node(node.args[0])
            return f"_jnp_{node.func.id}({arg})"
        if isinstance(node, ast.Name):
            if node.id == "x":
                return "_p0"
            if node.id == "y":
                return "_p1"
            if node.id == "pi":
                return str(math.pi)
            if node.id == "e":
                return str(math.e)
            raise ValueError(f"Unknown name: {node.id}")
        if isinstance(node, ast.Constant):
            return repr(node.value)
        raise ValueError(f"Cannot compile {type(node).__name__}")

    code_str = _compile_node(tree)

    # Build the function with jnp operations
    fn_globals = {
        "__builtins__": {},
        "_jnp_sin": jnp.sin, "_jnp_cos": jnp.cos, "_jnp_exp": jnp.exp,
        "_jnp_log": jnp.log, "_jnp_sqrt": jnp.sqrt, "_jnp_abs": jnp.abs,
        "_jnp_tan": jnp.tan, "_jnp_tanh": jnp.tanh,
    }

    func_code = f"def _custom_fn(params):\n    _p0, _p1 = params[0], params[1]\n    return {code_str}"
    exec(compile(func_code, "<custom_jax>", "exec"), fn_globals)
    return fn_globals["_custom_fn"]
