from __future__ import annotations


def SAFE_SCORE(score: float) -> float:
    try:
        score = float(score)
    except:
        return 0.01

    if score <= 0:
        return 0.01

    if score >= 1:
        return 0.99

    return score


def clamp_score(score: float) -> float:
    """Alias for SAFE_SCORE for backward compatibility."""
    return SAFE_SCORE(score)


def safe_ratio_score(correct: int, total: int) -> float:
    if total == 0:
        score = 0.01
    else:
        score = correct / total
    return SAFE_SCORE(score)
