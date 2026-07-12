"""Reusable tools for the agent."""

from __future__ import annotations

import ast
import operator
from typing import Literal

from langchain.tools import tool

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression_node(
    node: ast.AST,
) -> int | float:
    """Evaluate a restricted arithmetic syntax tree."""
    if isinstance(node, ast.Expression):
        return evaluate_expression_node(
            node.body
        )

    if isinstance(node, ast.Constant):
        value = node.value

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return value

        raise ValueError(
            "Only numeric constants are supported."
        )

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in BINARY_OPERATORS:
            raise ValueError(
                "The expression contains an unsupported operator."
            )

        left_value = evaluate_expression_node(
            node.left
        )

        right_value = evaluate_expression_node(
            node.right
        )

        if (
            operator_type is ast.Pow
            and abs(right_value) > 10
        ):
            raise ValueError(
                "The exponent is too large."
            )

        operation = BINARY_OPERATORS[
            operator_type
        ]

        return operation(
            left_value,
            right_value,
        )

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in UNARY_OPERATORS:
            raise ValueError(
                "The expression contains an unsupported unary operator."
            )

        operation = UNARY_OPERATORS[
            operator_type
        ]

        return operation(
            evaluate_expression_node(
                node.operand
            )
        )

    raise ValueError(
        "The expression contains unsupported syntax."
    )


@tool
def calculator(
    expression: str,
) -> str:
    """
    Calculate an arithmetic expression.

    Supported operators:
    +, -, *, /, //, %, and **

    Example input:
    (25 + 63) * 4
    """
    if len(expression) > 200:
        raise ValueError(
            "The expression is too long."
        )

    syntax_tree = ast.parse(
        expression,
        mode="eval",
    )

    result = evaluate_expression_node(
        syntax_tree
    )

    return str(result)


@tool
def format_text(
    mode: Literal[
        "uppercase",
        "lowercase",
        "titlecase",
    ],
    text: str,
) -> str:
    """
    Format text as uppercase, lowercase, or title case.
    """
    if mode == "uppercase":
        return text.upper()

    if mode == "lowercase":
        return text.lower()

    return text.title()