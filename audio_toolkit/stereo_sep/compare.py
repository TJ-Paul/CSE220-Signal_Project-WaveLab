"""A/B comparison harness across all five methods.

The reason this exists: added mathematics is not self-justifying. A pipeline
with five evidence streams is only worth its complexity if it beats the
two-line baseline on the SAME file, and the only way to know is to run both.

This harness was not decorative during development -- it caught three real
defects that would otherwise have shipped, each of which made a "more
advanced" method score WORSE than a simpler one:

  1. A fused confidence mapped through a logistic saturated, because most
     energy in a mix is centre-panned. The method scored below plain
     Mid/Side.
  2. Feeding the HPSS harmonic stream as evidence FOR vocal argued to remove
     every pitched instrument, so adding HPSS cost ~5 dB.
  3. Estimating accompaniment as max() over noisy lower bounds biased the
     estimate upward and collapsed the mask toward "keep everything".

Keep the baselines. They are the control group.
"""
from __future__ import annotations

import numpy as np

from .config import SeparationConfig
from .pipeline import separate, METHODS
from .evaluate import evaluate_result, proxy_metrics, bss_eval


def compare_methods(
    y: np.ndarray,
    sr: int,
    cfg: SeparationConfig | None = None,
    methods: list[str] | None = None,
    true_instrumental: np.ndarray | None = None,
    true_vocals: np.ndarray | None = None,
) -> dict:
    """Run every method on one mixture and tabulate the results.

    With ground truth, reports SDR/SIR/SAR per stem. Without it, reports
    proxy measures plus a do-nothing reference.

    THE DO-NOTHING REFERENCE IS THE IMPORTANT COLUMN. A separation method
    that scores below the untouched mixture is actively harmful, and without
    that column it is easy to celebrate a method that merely looks plausible.
    Two methods here did fall below it during development and were rebuilt.
    """
    cfg = cfg or SeparationConfig()
    methods = methods or list(METHODS)
    has_truth = true_instrumental is not None and true_vocals is not None

    out: dict = {"methods": [], "hasGroundTruth": has_truth}

    if has_truth:
        out["doNothing"] = {
            "instrumental": bss_eval(y, true_instrumental, true_vocals),
            "vocals": bss_eval(y, true_vocals, true_instrumental),
            "note": "The unprocessed mixture, scored as if it were each stem. "
                    "Any method below this is worse than not separating.",
        }

    for m in methods:
        result = separate(y, sr, method=m, cfg=cfg)
        entry: dict = {
            "method": m,
            "methodName": result.method_name,
            "assumptions": result.assumptions,
            "proxy": proxy_metrics(y, result.instrumental, result.vocals, sr),
        }
        if has_truth:
            entry["bss"] = evaluate_result(result, true_instrumental, true_vocals)
        out["methods"].append(entry)

    return out


def format_table(comparison: dict) -> str:
    """Render a comparison as a fixed-width table for a terminal."""
    lines = []
    if comparison["hasGroundTruth"]:
        lines.append(f"{'method':<34} | {'iSDR':>6} {'iSIR':>6} {'iSAR':>6} "
                     f"| {'vSDR':>6} {'vSIR':>6} {'vSAR':>6}")
        lines.append("-" * 82)
        dn = comparison.get("doNothing")
        if dn:
            i, v = dn["instrumental"], dn["vocals"]
            lines.append(f"{'(do nothing - reference)':<34} | {i['sdrDb']:6.2f} "
                         f"{i['sirDb']:6.2f} {i['sarDb']:6.2f} | {v['sdrDb']:6.2f} "
                         f"{v['sirDb']:6.2f} {v['sarDb']:6.2f}")
        for e in comparison["methods"]:
            i = e["bss"]["instrumental"]; v = e["bss"]["vocals"]
            lines.append(f"{e['method'] + ' ' + e['methodName'][:30]:<34} | "
                         f"{i['sdrDb']:6.2f} {i['sirDb']:6.2f} {i['sarDb']:6.2f} | "
                         f"{v['sdrDb']:6.2f} {v['sirDb']:6.2f} {v['sarDb']:6.2f}")
        lines.append("")
        lines.append("SDR overall quality | SIR how much of the other source leaked "
                     "| SAR how many artefacts were introduced")
        lines.append("A method can raise SIR while lowering SAR: that is the "
                     "suppression-vs-artefact trade, not an improvement.")
    else:
        lines.append(f"{'method':<34} | {'suppr dB':>9} {'stemCorr':>9} "
                     f"{'energyCons':>11} {'holes':>7}")
        lines.append("-" * 76)
        for e in comparison["methods"]:
            p = e["proxy"]
            lines.append(f"{e['method'] + ' ' + e['methodName'][:30]:<34} | "
                         f"{p['vocalBandSuppressionDb']:9.2f} {p['stemCorrelation']:9.3f} "
                         f"{p['energyConservation']:11.3f} {p['spectralHoleFraction']:7.3f}")
        lines.append("")
        lines.append("PROXY MEASURES ONLY - none of these indicates quality. "
                     "Suppression is maximised by deleting the band entirely. "
                     "Use them to find WHERE to listen, then judge by ear.")
    return "\n".join(lines)
