# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the answer from a \\boxed{...} expression in a LaTeX solution.

    Handles nested braces correctly (e.g., \\boxed{\\frac{1}{2}}).
    Returns the innermost boxed content, or None if not found.
    """
    marker = r"\boxed{"
    idx = solution.rfind(marker)  # use last occurrence
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return solution[start : i - 1].strip()


def init_dataset(
    levels: list[str] | None = None,
    train_size: int = 100,
    val_size: int = 100,
    seed: int = 0,
):
    """Load and split the Hendrycks MATH benchmark.

    Args:
        levels: Problem difficulty levels to include (integers).
            Defaults to [4, 5] to avoid ceiling effects.
        train_size: Number of training examples (half used for train, half for val).
        val_size: Number of validation examples.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (trainset, valset, testset).
        Each example is a dict with keys:
          - "input": problem text
          - "answer": "### <boxed answer>"
          - "additional_context": {"solution": ..., "subject": ..., "level": ...}
            (training split only; test split omits additional_context)
    """
    import random

    from datasets import load_dataset

    if levels is None:
        levels = [4, 5]

    ds = load_dataset("nlile/hendrycks-MATH-benchmark")

    def keep(ex):
        return ex["level"] in levels and extract_boxed_answer(ex["solution"]) is not None

    train_filtered = [x for x in ds["train"] if keep(x)]
    test_filtered = [x for x in ds["test"] if keep(x)]

    rng = random.Random(seed)
    rng.shuffle(train_filtered)
    rng.shuffle(test_filtered)

    def make_train_example(x):
        return {
            "input": x["problem"],
            "additional_context": {
                "solution": x["solution"],
                "subject": x["subject"],
                "level": x["level"],
            },
            "answer": "### " + extract_boxed_answer(x["solution"]),
        }

    def make_test_example(x):
        return {
            "input": x["problem"],
            "answer": "### " + extract_boxed_answer(x["solution"]),
            # subject stored for per-type analysis but not exposed to the model
            "subject": x["subject"],
            "level": x["level"],
        }

    pool = [make_train_example(x) for x in train_filtered]
    trainset = pool[:train_size]
    valset = pool[train_size : train_size + val_size]

    testset = [make_test_example(x) for x in test_filtered[:200]]

    return trainset, valset, testset
