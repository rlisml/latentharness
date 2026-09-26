"""Minimal AST helpers for `harness/markers.py::interface_alignment_present`
(clean distribution).

The upstream `harness/classify_errors_mbpp.py` is an MBPP error-analysis CLI
(E1-0 machinery, out of scope here). Only the two pure-AST functions consumed
by the TESTCONSULT marker are needed; they are extracted verbatim (bodies and
docstrings unchanged) so `harness.markers` behaves byte-identically. The
upstream module additionally depends on `harness/classify_errors.py`, which is
not carried over.
"""

import ast


def defined_names(code: str):
    """Function/class names defined anywhere in the extracted code (AST walk)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def called_names_in_asserts(asserts):
    """Names invoked by the test_list assert strings (Call nodes: plain names and
    attribute bases; `Pair(5, 24)` -> Pair, `math.sqrt(x)` -> math)."""
    called = set()
    for a in asserts:
        try:
            tree = ast.parse(a)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    called.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    base = fn.value
                    if isinstance(base, ast.Name):
                        called.add(base.id)
    return called
