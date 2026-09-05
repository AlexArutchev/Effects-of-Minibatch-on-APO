"""Small benchmark-specific utilities with no dependency on the GEPA source tree."""

from __future__ import annotations


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the final answer from the last well-formed LaTex boxed expression."""
    marker = r"\boxed{"
    start_marker = solution.rfind(marker)
    if start_marker == -1:
        return None
    start = start_marker + len(marker)
    depth = 1
    index = start
    while index < len(solution) and depth:
        depth += (solution[index] == "{") - (solution[index] == "}")
        index += 1
    return solution[start : index - 1].strip() if depth == 0 else None
