"""The Champion-Challenger selection rule.  Pure Python, no heavy imports."""
from __future__ import annotations

from config import RULE

_EPS = 1e-9  # so a gain of exactly `margin` counts despite floating-point error


def select_champion(a: dict, b: dict, rule: dict = RULE) -> dict:
    """Decide between Model A and Model B.

    1. delta = metric(B) - metric(A) on the primary metric (higher is better).
    2. delta >= +margin  -> B wins on the primary metric.
       delta <= -margin  -> A wins on the primary metric.
    3. |delta| < margin  -> the lower tie_breaker value (average latency) wins.
    4. Exactly equal latency -> the incumbent (A) is kept.
    """
    metric, margin, tie = rule["primary_metric"], rule["margin"], rule["tie_breaker"]
    delta = b[metric] - a[metric]
    if delta >= margin - _EPS:
        winner, basis = "B", "primary_metric"
        reason = (f"{metric}(B) - {metric}(A) = {delta:+.4f} >= margin {margin}; "
                  "B beats A by more than the minimum margin.")
    elif delta <= -(margin - _EPS):
        winner, basis = "A", "primary_metric"
        reason = (f"{metric}(B) - {metric}(A) = {delta:+.4f} <= -{margin}; "
                  "A stays ahead by more than the minimum margin.")
    else:
        la, lb = a[tie], b[tie]
        if lb < la:
            winner, basis = "B", "latency_tiebreak"
        elif la < lb:
            winner, basis = "A", "latency_tiebreak"
        else:
            winner, basis = rule["incumbent"], "incumbent_tie"
        reason = (f"|{metric} difference| = {abs(delta):.4f} < margin {margin}; "
                  f"{tie}: A = {la:.2f}, B = {lb:.2f}; lower wins -> {winner} "
                  f"({basis}).")
    return {
        "winner": winner,
        "loser": "B" if winner == "A" else "A",
        "basis": basis,
        "delta_primary_metric": delta,
        "primary_metric": metric,
        "margin": margin,
        "tie_breaker": tie,
        "reason": reason,
    }
