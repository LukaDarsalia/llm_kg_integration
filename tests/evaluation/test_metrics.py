from llm_kg.evaluation.metrics import exact_match, f1_score, normalize_text, recall_at_k


def test_normalize_lowercases_and_strips_articles_and_punct() -> None:
    assert normalize_text("The Cat's pajamas!") == "cats pajamas"
    assert normalize_text("  An apple. ") == "apple"
    assert normalize_text("a b") == "b"


def test_exact_match_simple() -> None:
    assert exact_match("Paris", ["paris"]) == 1.0
    assert exact_match("Paris", ["London"]) == 0.0


def test_exact_match_any_of_many() -> None:
    assert exact_match("NYC", ["New York", "NYC", "the big apple"]) == 1.0


def test_exact_match_with_articles() -> None:
    assert exact_match("the answer", ["answer"]) == 1.0


def test_f1_partial_overlap() -> None:
    # Use non-article tokens so SQuAD normalization doesn't drop any.
    # pred 2/3 tokens correct, recall 2/2 → P=2/3, R=1, F1 = 2 * (2/3) / (1 + 2/3) = 0.8
    score = f1_score("alpha beta gamma", ["alpha beta"])
    assert abs(score - 0.8) < 1e-9


def test_f1_no_overlap() -> None:
    assert f1_score("alpha", ["beta"]) == 0.0


def test_f1_empty_prediction() -> None:
    assert f1_score("", ["something"]) == 0.0


def test_f1_picks_max_across_golds() -> None:
    # against "x" → 0; against "x y" → 1 → max is 1
    assert f1_score("x y", ["x", "x y"]) == 1.0


def test_recall_at_k_full_recall() -> None:
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c1", "c3"]
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_truncation() -> None:
    retrieved = ["c1", "c2", "c3"]
    relevant = ["c3"]
    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_no_relevant() -> None:
    # By convention, recall is 1.0 when there are no relevant items (vacuously true).
    assert recall_at_k(["a", "b"], [], k=2) == 1.0
