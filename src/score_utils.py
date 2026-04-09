from __future__ import annotations


def SAFE_SCORE(score: float) -> float:
    """
    STRICT global safe score function.
    Ensures ALL scores satisfy: 0 < score < 1 (never 0.0 or 1.0).
    """
    EPS = 1e-6

    try:
        score = float(score)
    except:
        return 0.5  # safe fallback

    if score <= 0:
        return 0.01 + EPS
    if score >= 1:
        return 0.99 - EPS

    return max(0.01 + EPS, min(0.99 - EPS, score))


def clamp_score(score: float) -> float:
    """Alias for SAFE_SCORE for backward compatibility."""
    return SAFE_SCORE(score)


def safe_ratio_score(correct: int, total: int) -> float:
    if total == 0:
        score = 0.01
    else:
        score = correct / total
    return SAFE_SCORE(score)
