"""
Code similarity computation module for sneaky verification.

Computes similarity between two Python code snippets using a weighted
combination of multiple features:

Feature weights:
- opcode_bigram: 0.71
- ast_edge: 0.17
- control_flow: 0.08
- literal: 0.03

All functions handle syntax errors gracefully by falling back to
token-based similarity.
"""

from __future__ import annotations

import ast
import dis
import io
import math
import tokenize
from collections import Counter
from typing import Any, Optional, Set, Tuple


# Feature weights for the composite similarity score
WEIGHTS = {
    "opcode_bigram": 0.71,
    "ast_edge": 0.17,
    "control_flow": 0.08,
    "literal": 0.03,
}


def canonicalize(code: str) -> Optional[str]:
    """
    Canonicalize Python code by parsing and unparsing.

    Returns canonical code string or None on syntax error.
    This normalizes whitespace and formatting while preserving semantics.

    Args:
        code: Python source code string

    Returns:
        Canonical code string, or None if code has syntax errors
    """
    if not code or not code.strip():
        return None

    try:
        tree = ast.parse(code)
        # Unparse back to normalized form
        return ast.unparse(tree)
    except SyntaxError:
        return None


def _get_tokens(code: str) -> list[str]:
    """
    Tokenize Python code, filtering out noise tokens.

    Returns list of token strings (type:value format).
    """
    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            # Skip comments, encoding, whitespace-related tokens
            if tok.type in (
                tokenize.COMMENT,
                tokenize.ENCODING,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                continue
            # Create token identifier
            tokens.append(f"{tok.type}:{tok.string}")
    except tokenize.TokenError:
        # Fallback: split on whitespace
        return [f"raw:{w}" for w in code.split()]
    return tokens


def _jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """
    Compute Jaccard similarity between two sets.

    Returns 1.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 1.0


def _cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    """
    Compute cosine similarity between two Counter objects (as vectors).

    Returns 1.0 if both counters are empty.
    """
    if not counter_a and not counter_b:
        return 1.0
    if not counter_a or not counter_b:
        return 0.0

    # Get all keys
    all_keys = set(counter_a.keys()) | set(counter_b.keys())

    # Compute dot product and magnitudes
    dot_product = sum(counter_a.get(k, 0) * counter_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v * v for v in counter_a.values()))
    mag_b = math.sqrt(sum(v * v for v in counter_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 1.0 if mag_a == mag_b else 0.0

    return dot_product / (mag_a * mag_b)


def _token_jaccard(code_a: str, code_b: str) -> float:
    """
    Fallback token-based Jaccard similarity for syntax errors.
    """
    tokens_a = set(_get_tokens(code_a))
    tokens_b = set(_get_tokens(code_b))
    return _jaccard_similarity(tokens_a, tokens_b)


def _get_opcodes(code: str) -> list[str]:
    """
    Get bytecode opcode names from Python code.

    Returns empty list on compilation error.
    """
    try:
        code_obj = compile(code, "<string>", "exec")
        opcodes = []

        def extract_opcodes(co):
            for instr in dis.get_instructions(co):
                opcodes.append(instr.opname)
            # Recursively handle nested code objects
            for const in co.co_consts:
                if hasattr(const, "co_code"):
                    extract_opcodes(const)

        extract_opcodes(code_obj)
        return opcodes
    except (SyntaxError, ValueError, TypeError):
        return []


def _get_bigrams(items: list[str]) -> Set[Tuple[str, str]]:
    """
    Generate bigrams from a list of items.
    """
    if len(items) < 2:
        return set()
    return {(items[i], items[i + 1]) for i in range(len(items) - 1)}


def opcode_bigram_jaccard(code_a: str, code_b: str) -> float:
    """
    Compute opcode bigram Jaccard similarity between two code snippets.

    Compiles both code snippets to bytecode, extracts opcode sequences,
    generates bigrams, and computes Jaccard similarity.

    Falls back to token-based similarity on syntax errors.

    Args:
        code_a: First Python source code string
        code_b: Second Python source code string

    Returns:
        Similarity score in [0, 1] where 1.0 means identical
    """
    opcodes_a = _get_opcodes(code_a)
    opcodes_b = _get_opcodes(code_b)

    # Fall back to token similarity if either fails to compile
    if not opcodes_a or not opcodes_b:
        return _token_jaccard(code_a, code_b)

    bigrams_a = _get_bigrams(opcodes_a)
    bigrams_b = _get_bigrams(opcodes_b)

    return _jaccard_similarity(bigrams_a, bigrams_b)


def _get_ast_edges(code: str) -> Set[Tuple[str, str, str]]:
    """
    Extract AST edges from Python code.

    Returns set of (parent_type, child_type, field_name) tuples.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    edges = set()

    def visit(node: ast.AST):
        parent_type = node.__class__.__name__
        for field_name, child in ast.iter_fields(node):
            if isinstance(child, ast.AST):
                child_type = child.__class__.__name__
                edges.add((parent_type, child_type, field_name))
                visit(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, ast.AST):
                        item_type = item.__class__.__name__
                        edges.add((parent_type, item_type, field_name))
                        visit(item)

    visit(tree)
    return edges


def ast_edge_jaccard(code_a: str, code_b: str) -> float:
    """
    Compute AST edge Jaccard similarity between two code snippets.

    Extracts AST edges (parent-child relationships with field names)
    and computes Jaccard similarity.

    Falls back to token-based similarity on syntax errors.

    Args:
        code_a: First Python source code string
        code_b: Second Python source code string

    Returns:
        Similarity score in [0, 1] where 1.0 means identical
    """
    edges_a = _get_ast_edges(code_a)
    edges_b = _get_ast_edges(code_b)

    # Fall back to token similarity if either fails to parse
    if not edges_a and not edges_b:
        # Both empty - could be empty code or syntax errors
        if not code_a.strip() and not code_b.strip():
            return 1.0
        return _token_jaccard(code_a, code_b)
    if not edges_a or not edges_b:
        return _token_jaccard(code_a, code_b)

    return _jaccard_similarity(edges_a, edges_b)


def _get_control_flow_counts(code: str) -> Counter:
    """
    Count control flow constructs in Python code.

    Returns Counter of control flow node types.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()

    control_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Match,  # Python 3.10+
        ast.Break,
        ast.Continue,
        ast.Return,
        ast.Raise,
        ast.ExceptHandler,
    )

    counts: Counter = Counter()

    for node in ast.walk(tree):
        if isinstance(node, control_nodes):
            counts[node.__class__.__name__] += 1

    return counts


def control_flow_cosine(code_a: str, code_b: str) -> float:
    """
    Compute control flow cosine similarity between two code snippets.

    Counts control flow constructs (if, for, while, try, etc.) and
    computes cosine similarity between the count vectors.

    Falls back to token-based similarity on syntax errors.

    Args:
        code_a: First Python source code string
        code_b: Second Python source code string

    Returns:
        Similarity score in [0, 1] where 1.0 means identical
    """
    counts_a = _get_control_flow_counts(code_a)
    counts_b = _get_control_flow_counts(code_b)

    # If both have no control flow, they're similar in that respect
    if not counts_a and not counts_b:
        return 1.0

    # If one has control flow and the other doesn't, check for syntax errors
    if (not counts_a or not counts_b) and (
        (counts_a and not counts_b) or (counts_b and not counts_a)
    ):
        # Check if it's due to syntax error
        try:
            ast.parse(code_a)
        except SyntaxError:
            return _token_jaccard(code_a, code_b)
        try:
            ast.parse(code_b)
        except SyntaxError:
            return _token_jaccard(code_a, code_b)

    return _cosine_similarity(counts_a, counts_b)


def _get_literals(code: str) -> Counter:
    """
    Extract literal values from Python code.

    Returns Counter of (literal_type, literal_value) tuples.
    For strings, we use a hash to avoid issues with long strings.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()

    literals: Counter = Counter()

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):
                # Use hash for strings to handle long strings
                literals[("str", hash(value) % (10**9))] += 1
            elif isinstance(value, (int, float, bool, type(None))):
                literals[(type(value).__name__, value)] += 1
            # Skip bytes and other exotic types

    return literals


def literal_signature_cosine(code_a: str, code_b: str) -> float:
    """
    Compute literal signature cosine similarity between two code snippets.

    Extracts literal values (strings, numbers, booleans) and computes
    cosine similarity between the literal frequency vectors.

    Returns 1.0 if neither code has literals (no difference to detect).

    Falls back to token-based similarity on syntax errors.

    Args:
        code_a: First Python source code string
        code_b: Second Python source code string

    Returns:
        Similarity score in [0, 1] where 1.0 means identical
    """
    literals_a = _get_literals(code_a)
    literals_b = _get_literals(code_b)

    # If both have no literals, they're identical in literal signature
    if not literals_a and not literals_b:
        return 1.0

    # Check for syntax errors
    if not literals_a:
        try:
            ast.parse(code_a)
        except SyntaxError:
            return _token_jaccard(code_a, code_b)

    if not literals_b:
        try:
            ast.parse(code_b)
        except SyntaxError:
            return _token_jaccard(code_a, code_b)

    return _cosine_similarity(literals_a, literals_b)


def compute_similarity(code_a: str, code_b: str) -> float:
    """
    Compute weighted similarity between two Python code snippets.

    Uses a static feature ladder with the following weights:
    - opcode_bigram: 0.71
    - ast_edge: 0.17
    - control_flow: 0.08
    - literal: 0.03

    All component functions handle syntax errors gracefully by falling
    back to token-based similarity.

    Args:
        code_a: First Python source code string
        code_b: Second Python source code string

    Returns:
        Similarity score in [0, 1] where 1.0 means identical
    """
    # Handle edge cases
    if not code_a and not code_b:
        return 1.0
    if not code_a or not code_b:
        # One is empty, one is not
        if not code_a.strip() and not code_b.strip():
            return 1.0
        return 0.0

    # Compute individual feature similarities
    opcode_sim = opcode_bigram_jaccard(code_a, code_b)
    ast_sim = ast_edge_jaccard(code_a, code_b)
    control_sim = control_flow_cosine(code_a, code_b)
    literal_sim = literal_signature_cosine(code_a, code_b)

    # Weighted combination
    total = (
        WEIGHTS["opcode_bigram"] * opcode_sim
        + WEIGHTS["ast_edge"] * ast_sim
        + WEIGHTS["control_flow"] * control_sim
        + WEIGHTS["literal"] * literal_sim
    )

    # Normalize by sum of weights (should be ~1.0 but handle floating point)
    weight_sum = sum(WEIGHTS.values())
    return total / weight_sum
