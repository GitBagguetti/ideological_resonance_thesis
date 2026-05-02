"""
EN–German semantic axis comparison and PCA × semantic-association visualizations.

Uses hardcoded paths to w2v models, axis JSONs, entity mapping, and thesis motif
data (ideological_resonance_thesis). Optional pole alignment uses English as
the reference and can sign-flip German PCA and/or semantic components when their
actor projections correlate negatively with English.

**Environment:** This module imports ``helpers`` (and transitively word2vec analysis
stacks). Use a Conda env with Python 3.10+ (e.g. ``word2vec`` at
``~/.conda/envs/word2vec/``) and install **plotly** (required at import time by
``helpers.py``) plus numpy, pandas, scipy, scikit-learn, matplotlib, and gensim.
Example::

    ~/.conda/envs/word2vec/bin/python /path/to/w2v/analysis/modules/semanalysis_plots.py
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex

from matplotlib.figure import Figure
from scipy.stats import kendalltau, spearmanr

_MODULES_DIR = Path(__file__).resolve().parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

import helpers  # noqa: E402
import semaxis_util  # noqa: E402
import semanalysis_util as su  # noqa: E402

# EN−GE diff: light green at 0 (aligned); light red at ±abs_max (large |EN−GE|), inspired by
# RdBu_r(0.75) which maps data=+0.5 in [−1,+1] to a coral, lightened for the poles.
def _cmap_en_ge_diff() -> LinearSegmentedColormap:
    rdbu = plt.get_cmap("RdBu_r")
    t_hi = 0.75  # (0.5+1)/2 — same index as "data=0.5" on a [−1,+1] scale
    c = rdbu(t_hi)[:3]
    pole = tuple(0.55 * c[i] + 0.45 for i in range(3))  # light coral red
    pole_hex = to_hex(pole, keep_alpha=False)
    green = "#b8d9b0"  # light green: alignment
    return LinearSegmentedColormap.from_list(
        "en_ge_diff",
        [
            (0.0, pole_hex),
            (0.5, green),
            (1.0, pole_hex),
        ],
        N=256,
    )


_CMAP_ZERO_EMPHASIS: LinearSegmentedColormap = _cmap_en_ge_diff()


def _symmetric_vlim_absmax(values: np.ndarray) -> float:
    """|vmin| = |vmax| = max absolute value; fallback 1.0 for empty/flat data."""
    v = float(np.nanmax(np.abs(np.asarray(values, dtype=np.float64))))
    if not np.isfinite(v) or v <= 0:
        return 1.0
    return v


# ---------------------------------------------------------------------------
# Hardcoded resource locations
# ---------------------------------------------------------------------------
_W2V_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS_DIR = _W2V_ROOT / "analysis"
PLOTS_DIR = _ANALYSIS_DIR / "plots"
_REPO_ROOT = Path(__file__).resolve().parents[3]

ENTITY_MAPPING_PATH = helpers.ENTITY_MAPPING_2_PATH
AXES_EN_PATH = _ANALYSIS_DIR / "axes_en.json"
AXES_DE_PATH = _ANALYSIS_DIR / "axes_de.json"
MOTIFS_EN_CSV = _REPO_ROOT / "data" / "motifs_en_filtered.csv"
MOTIFS_DE_CSV = _REPO_ROOT / "data" / "motifs_de_filtered.csv"

MODEL_EN_NAME = "2_w2v_min10"
MODEL_DE_NAME = "2_w2v_min10_de"

ACTORS_PCT: Tuple[str, int, int] = ("pct", 0, 100)
ACTIONS_PCT: Tuple[str, float, float] = ("pct", 0.2, 10)

EN_FILTER_MASK_SUFFIXES: Tuple[str, ...] = ("_it", "_de", "good", "bad")
DE_FILTER_MASK_SUFFIXES: Tuple[str, ...] = (
    "_it",
    "alt_politics_us",
    "security_us",
    "biomed_us",
    "mainstreammedia_us",
    "altmedia_us",
    "antivax_us",
    "good",
    "bad",
)

DEFAULT_METRIC = "correlation_abs"
DEFAULT_FLIP_AXIS = "false"
FLIP_AXIS_CHOICES: Tuple[str, ...] = ("true", "pca", "pca_trump_anchor", "false")
TRUMP_ANCHOR_CANONICAL = "Donald Trump"
TRUMP_ANCHOR_TOKEN = "donald_trump"
TRUMP_ANCHOR_TARGETS: Dict[str, Tuple[float, float]] = {
    "regular": (1.0, -1.0),
    "distorted": (1.0, 1.0),
}

FNAME_BARPLOT = "semaxis_en_de_bars.png"
# Stacked: regular (rownorm) + distorted PCA rows in one figure each
FNAME_PCA_HEAT_COMBINED = "pca_semantic_heatmap_combined.png"
FNAME_PCA_DIFF_COMBINED = "pca_semantic_diff_combined.png"
# Semantic–semantic axis actor-projection correlation matrices
FNAME_SEM_CORR_EN_DE = "semantic_axes_correlation_en_de.png"
FNAME_SEM_CORR_DIFF = "semantic_axes_correlation_en_minus_de.png"


def _scipy_stat_pvalue(result: Any) -> Tuple[float, float]:
    if hasattr(result, "statistic") and hasattr(result, "pvalue"):
        return float(result.statistic), float(result.pvalue)
    a, b = result
    return float(a), float(b)


def _normalize_flip_axis(flip_axis: Any) -> str:
    """Normalize runtime/imported flip-axis controls to one of the CLI choices."""
    if isinstance(flip_axis, bool):
        return "true" if flip_axis else "false"
    mode = str(flip_axis).strip().lower()
    if mode not in FLIP_AXIS_CHOICES:
        raise ValueError(
            f"flip_axis must be one of {FLIP_AXIS_CHOICES}; got {flip_axis!r}"
        )
    return mode


def pairwise_correlation_metrics(
    s: np.ndarray, p: np.ndarray, top_k: int = 20
) -> Dict[str, Any]:
    """
    Same metric keys as ``compare_semantic_to_pca`` for two aligned 1D arrays
    (e.g. EN vs DE semantic projections on comparison units). Does not apply an
    internal axis flip; use after external DE sign-alignment.
    """
    s = np.asarray(s, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = int(min(len(s), len(p)))
    if m < 2:
        return {
            "correlation": float("nan"),
            "correlation_abs": float("nan"),
            "aligned_correlation": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_rho_abs": float("nan"),
            "spearman_p": float("nan"),
            "kendall_tau": float("nan"),
            "kendall_tau_abs": float("nan"),
            "kendall_p": float("nan"),
            "top_k_overlap": float("nan"),
            "top_k": 1,
            "inverse_squared_distance": 0.0,
            "mean_absolute_error": float("nan"),
            "sum_squared_distance": float("nan"),
        }
    s = s[:m]
    p = p[:m]
    correlation = float(np.corrcoef(s, p)[0, 1])
    rho, rho_p = _scipy_stat_pvalue(spearmanr(s, p))
    tau, tau_p = _scipy_stat_pvalue(kendalltau(s, p))
    n = len(s)
    k_eff = int(min(max(top_k, 1), n))
    top_sem = set(np.argsort(s, kind="mergesort")[-k_eff:])
    top_pca = set(np.argsort(p, kind="mergesort")[-k_eff:])
    top_k_overlap = len(top_sem & top_pca) / k_eff if k_eff > 0 else float("nan")
    if correlation < 0:
        pfa = -p
        aligned_correlation = -correlation
    else:
        pfa = p
        aligned_correlation = correlation
    sq = float(np.sum((s - pfa) ** 2))
    return {
        "correlation": correlation,
        "correlation_abs": abs(correlation) if not np.isnan(correlation) else float("nan"),
        "aligned_correlation": float(aligned_correlation),
        "spearman_rho": float(rho),
        "spearman_rho_abs": abs(rho) if not np.isnan(rho) else float("nan"),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_tau_abs": abs(tau) if not np.isnan(tau) else float("nan"),
        "kendall_p": float(tau_p),
        "top_k_overlap": float(top_k_overlap),
        "top_k": k_eff,
        "inverse_squared_distance": 1.0 / sq if sq > 0 else (np.inf if sq == 0 else float("nan")),
        "mean_absolute_error": float(np.mean(np.abs(s - pfa))),
        "sum_squared_distance": sq,
    }


def _metric_label(metric: str) -> str:
    labels = {
        "correlation": "Pearson r",
        "correlation_abs": "|Pearson r|",
        "aligned_correlation": "aligned Pearson r",
        "spearman_rho": "Spearman rho",
        "spearman_rho_abs": "|Spearman rho|",
        "spearman_p": "Spearman p",
        "kendall_tau": "Kendall tau",
        "kendall_tau_abs": "|Kendall tau|",
        "kendall_p": "Kendall p",
        "top_k_overlap": "top-k overlap",
        "inverse_squared_distance": "inverse squared distance",
        "mean_absolute_error": "mean absolute error",
        "sum_squared_distance": "sum squared distance",
    }
    return labels.get(metric, metric)


def _is_signed_correlation_metric(metric: str) -> bool:
    return metric in {"correlation", "spearman_rho", "kendall_tau"}


def _is_unit_interval_metric(metric: str) -> bool:
    return metric in {
        "correlation_abs",
        "aligned_correlation",
        "spearman_rho_abs",
        "spearman_p",
        "kendall_tau_abs",
        "kendall_p",
        "top_k_overlap",
    }


def _metric_plot_limits(values: np.ndarray, metric: str) -> Tuple[float, float]:
    if _is_signed_correlation_metric(metric):
        return -1.0, 1.0
    if _is_unit_interval_metric(metric):
        return 0.0, 1.0
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        pad = 0.5 if lo == 0 else abs(lo) * 0.1
    else:
        pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _pairwise_metric_value(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    metrics = pairwise_correlation_metrics(a[mask], b[mask], top_k=20)
    if metric not in metrics:
        raise KeyError(f"Unknown metric {metric!r}. Keys: {sorted(metrics.keys())}")
    value = metrics[metric]
    return float(value) if value is not None else float("nan")


def is_spiritual_category(category: str) -> bool:
    c = str(category)
    return c.endswith("good") or c.endswith("bad")


def is_locale_it(category: str) -> bool:
    return str(category).endswith("_it")


def is_locale_us(category: str) -> bool:
    return str(category).endswith("_us")


def is_locale_de(category: str) -> bool:
    return str(category).endswith("_de")


def is_general_category(category: str) -> bool:
    c = str(category)
    return not (c.endswith("_us") or c.endswith("_de") or c.endswith("_it"))


def build_canonical_to_category(
    en_df: pd.DataFrame, de_df: pd.DataFrame
) -> pd.Series:
    en_map = en_df.drop_duplicates("canonical").set_index("canonical")["category"]
    de_map = de_df.drop_duplicates("canonical").set_index("canonical")["category"]
    for c, cat in de_map.items():
        if c not in en_map.index:
            en_map[c] = cat
    return en_map


def build_entity_inclusion_tuples(
    canonicals: Iterable[str],
    cat_series: pd.Series,
    kv_en,
    kv_de,
    c2t: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """
    (canonical, 'included'|'excluded', reason) using spiritual / it / general OOV
    and locale OOV rules.
    """
    rows: List[Tuple[str, str, str]] = []
    for name in sorted(set(canonicals)):
        cat = str(cat_series.get(name, "unknown"))
        if is_spiritual_category(cat):
            rows.append((name, "excluded", "spiritual"))
            continue
        if is_locale_it(cat):
            rows.append((name, "excluded", "locale_it"))
            continue
        t = c2t.get(name)
        if t is None:
            rows.append((name, "excluded", "not_in_mapping"))
            continue
        in_en = t in kv_en
        in_de = t in kv_de
        if is_general_category(cat):
            if not in_en or not in_de:
                rows.append((name, "excluded", "general_oov"))
                continue
        elif is_locale_us(cat):
            if not in_en:
                rows.append((name, "excluded", "oov_en"))
                continue
        elif is_locale_de(cat):
            if not in_de:
                rows.append((name, "excluded", "oov_de"))
                continue
        if not in_en:
            rows.append((name, "excluded", "oov_en"))
            continue
        if not in_de:
            rows.append((name, "excluded", "oov_de"))
            continue
        rows.append((name, "included", "ok"))
    return rows


def entity_report_dataframe(
    records: List[Tuple[str, str, str]], analysis: str = ""
) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=["name", "status", "reason"])
    if analysis:
        df["analysis"] = analysis
    return df


def load_motif_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    en = pd.read_csv(MOTIFS_EN_CSV)
    de = pd.read_csv(MOTIFS_DE_CSV)
    en_f = helpers.filter_top_actions(en, actors=ACTORS_PCT, actions=ACTIONS_PCT)
    de_f = helpers.filter_top_actions(de, actors=ACTORS_PCT, actions=ACTIONS_PCT)
    return en_f, de_f


def load_axes_paired() -> List[Tuple[str, str, List[Tuple[str, str]], List[Tuple[str, str]]]]:
    with open(AXES_EN_PATH, "r", encoding="utf-8") as f:
        en: Dict[str, List] = json.load(f)
    with open(AXES_DE_PATH, "r", encoding="utf-8") as f:
        de: Dict[str, List] = json.load(f)
    paired: List[Tuple[str, str, List[Tuple[str, str]], List[Tuple[str, str]]]] = []
    for dk, d_pairs in de.items():
        if not dk.endswith("_de"):
            continue
        base = dk[:-3]
        if base not in en:
            continue
        e_list = [tuple(t) for t in en[base]]
        d_list = [tuple(t) for t in d_pairs]
        paired.append((base, dk, e_list, d_list))
    return paired


def _load_axis_json(path: Path) -> Dict[str, List[Tuple[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, List] = json.load(f)
    return {axis: [tuple(pair) for pair in pairs] for axis, pairs in raw.items()}


def _resolve_crossaxis_selection(
    crossaxis: str,
    axes_en: Dict[str, List[Tuple[str, str]]],
    axes_de: Dict[str, List[Tuple[str, str]]],
) -> Tuple[str, str, str, str]:
    """
    Resolve CLI shorthand such as ``reveal_holy`` to two shared base axis names.

    Selectors are English positive-pole words. Axis keys themselves are also
    accepted when they do not introduce ambiguity.
    """
    selector = str(crossaxis).strip().lower()
    parts = [part for part in selector.split("_") if part]
    if len(parts) != 2:
        raise ValueError(
            "--crossaxis must contain exactly two positive-pole words separated "
            f"by '_', e.g. 'reveal_holy'; got {crossaxis!r}"
        )

    pole_to_axes: Dict[str, List[str]] = defaultdict(list)
    for axis_name, pairs in axes_en.items():
        if "_" in axis_name:
            pole_to_axes[axis_name.split("_", 1)[0].lower()].append(axis_name)
        pole_to_axes[axis_name.lower()].append(axis_name)
        for pos_word, _neg_word in pairs:
            pole_to_axes[str(pos_word).lower()].append(axis_name)

    resolved: List[str] = []
    for pole in parts:
        candidates = sorted(set(pole_to_axes.get(pole, [])))
        paired_candidates = [axis for axis in candidates if f"{axis}_de" in axes_de]
        if not paired_candidates:
            raise ValueError(
                f"Could not resolve crossaxis pole {pole!r}. Available positive poles: "
                f"{sorted(k for k, v in pole_to_axes.items() if v)}"
            )
        if len(paired_candidates) > 1:
            raise ValueError(
                f"Crossaxis pole {pole!r} is ambiguous; matches {paired_candidates}."
            )
        resolved.append(paired_candidates[0])

    if resolved[0] == resolved[1]:
        raise ValueError(f"--crossaxis must select two different axes; got {resolved[0]!r}.")
    return resolved[0], resolved[1], parts[0], parts[1]


def build_sem_axis_dicts(
    paired: Sequence[Tuple[str, str, List[Tuple[str, str]], List[Tuple[str, str]]]],
    w2v_en,
    w2v_de,
) -> Tuple[Dict[str, semaxis_util.SemAxis], Dict[str, semaxis_util.SemAxis]]:
    en_axes: Dict[str, semaxis_util.SemAxis] = {}
    de_axes: Dict[str, semaxis_util.SemAxis] = {}
    for base, de_key, en_pairs, de_pairs in paired:
        try:
            en_axes[base] = semaxis_util.SemAxis(en_pairs, w2v_en, name=base)
            de_axes[base] = semaxis_util.SemAxis(de_pairs, w2v_de, name=de_key)
        except (ValueError, KeyError) as e:
            warnings.warn(f"Skipping axis {base!r}: {e}", UserWarning, stacklevel=2)
    return en_axes, de_axes


def _locale_basename_for_agg(cat: str) -> Optional[str]:
    c = str(cat)
    if c.endswith("_us"):
        return c[: -len("_us")]
    if c.endswith("_de"):
        return c[: -len("_de")]
    return None


def run_cross_language_axis_bars(
    metric: str = DEFAULT_METRIC,
    flip_axis: Any = DEFAULT_FLIP_AXIS,
) -> Tuple[pd.DataFrame, Figure, pd.Series, str]:
    """
    Barplot: one bar per shared semantic axis — association between EN and DE actor
    projections (optionally DE concept sign aligned to EN on shared actors, then
    one value per shared actor plus one aggregate per matched *_us / *_de basename).
    """
    flip_mode = _normalize_flip_axis(flip_axis)
    flip_semantic_axes = flip_mode == "true"
    w2v_en = helpers.load_trained_w2v_keyed_vectors(MODEL_EN_NAME)
    w2v_de = helpers.load_trained_w2v_keyed_vectors(MODEL_DE_NAME)
    c2t = helpers.load_canonical_to_w2v_token(str(ENTITY_MAPPING_PATH))
    en_f, de_f = load_motif_frames()
    cat_map = build_canonical_to_category(en_f, de_f)
    all_names = set(en_f["canonical"].unique()) | set(de_f["canonical"].unique())
    rec = build_entity_inclusion_tuples(all_names, cat_map, w2v_en, w2v_de, c2t)
    report_df = entity_report_dataframe(rec, "cross_lang_bars")
    included = {n for n, st, _ in rec if st == "included"}
    if not included:
        raise ValueError("No included actors for cross-language bars.")
    shared_indiv = sorted(
        n for n in included
        if c2t.get(n) in w2v_en and c2t.get(n) in w2v_de
    )
    us_by_base: Dict[str, List[str]] = defaultdict(list)
    de_by_base: Dict[str, List[str]] = defaultdict(list)
    for n in included:
        c = str(cat_map.get(n, ""))
        b = _locale_basename_for_agg(c)
        if b is None:
            continue
        t = c2t.get(n)
        if t is None:
            continue
        if c.endswith("_us") and t in w2v_en:
            us_by_base[b].append(n)
        if c.endswith("_de") and t in w2v_de:
            de_by_base[b].append(n)
    en_emb_shared = su.actor_embeddings_from_w2v_entities(
        w2v_en, shared_indiv, c2t, str(ENTITY_MAPPING_PATH)
    )
    de_emb_shared = su.actor_embeddings_from_w2v_entities(
        w2v_de, shared_indiv, c2t, str(ENTITY_MAPPING_PATH)
    )
    pair_axes = load_axes_paired()
    en_smax, de_smax = build_sem_axis_dicts(pair_axes, w2v_en, w2v_de)
    bases = sorted(en_smax.keys() & de_smax.keys())
    metric_vals: Dict[str, float] = {}
    sign_lines: List[str] = []
    for base in bases:
        c_en = np.asarray(en_smax[base].concept_vector, dtype=np.float64)
        c_de = np.asarray(de_smax[base].concept_vector, dtype=np.float64)
        c_en = c_en / (np.linalg.norm(c_en) or 1.0)
        c_de = c_de / (np.linalg.norm(c_de) or 1.0)
        s_en = su.actor_proj(en_emb_shared, c_en, rescale=True)
        s_de = su.actor_proj(de_emb_shared, c_de, rescale=True)
        com = [x for x in s_en.index if x in s_de.index]
        a_en = np.array([s_en.loc[x] for x in com], dtype=np.float64)
        a_de = np.array([s_de.loc[x] for x in com], dtype=np.float64)
        r_align = float(np.corrcoef(a_en, a_de)[0, 1]) if len(com) >= 2 else 1.0
        if flip_semantic_axes and (not np.isnan(r_align)) and r_align < 0:
            c_de = -c_de
            sign_lines.append(
                f"{base}: DE semantic axis sign flipped (Pearson on shared n={len(com)}: {r_align:.4f})"
            )
        else:
            if flip_semantic_axes:
                sign_lines.append(f"{base}: no DE flip (n_shared={len(com)}).")
            else:
                sign_lines.append(
                    f"{base}: semantic flip disabled by flip_axis={flip_mode} "
                    f"(r={r_align:.4f}, n={len(com)})"
                )
        s_en = su.actor_proj(en_emb_shared, c_en, rescale=True)
        s_de = su.actor_proj(de_emb_shared, c_de, rescale=True)
        en_list: List[float] = []
        de_list: List[float] = []
        for n in com:
            en_list.append(float(s_en.loc[n]))
            de_list.append(float(s_de.loc[n]))
        common_b = sorted(
            b for b in us_by_base if b in de_by_base
            and len(us_by_base[b]) and len(de_by_base[b])
        )
        for b in common_b:
            e_us = su.actor_embeddings_from_w2v_entities(
                w2v_en, us_by_base[b], c2t, str(ENTITY_MAPPING_PATH)
            )
            e_d = su.actor_embeddings_from_w2v_entities(
                w2v_de, de_by_base[b], c2t, str(ENTITY_MAPPING_PATH)
            )
            su.actor_proj(e_us, c_en, rescale=True)
            en_list.append(float(su.actor_proj(e_us, c_en, rescale=True).mean()))
            de_list.append(float(su.actor_proj(e_d, c_de, rescale=True).mean()))
        a_en = np.array(en_list, dtype=np.float64)
        a_de = np.array(de_list, dtype=np.float64)
        m = pairwise_correlation_metrics(a_en, a_de, top_k=20)
        if metric not in m:
            raise KeyError(f"Unknown metric {metric!r}. Keys: {sorted(m.keys())}")
        metric_vals[base] = float(m[metric]) if m[metric] is not None else float("nan")
    s_series = pd.Series(metric_vals, name=metric)
    s_sorted = s_series.sort_values(ascending=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(s_sorted)) ))
    neg_cmap = LinearSegmentedColormap.from_list(
        "cross_lang_axis_bars_negative", ["#f4b6b6", "#7f0000"]
    )
    pos_cmap = LinearSegmentedColormap.from_list(
        "cross_lang_axis_bars_positive", ["#bdd7e7", "#08519c"]
    )
    bar_colors = [
        neg_cmap(min(abs(float(v)), 1.0))
        if np.isfinite(v) and float(v) < 0
        else pos_cmap(min(float(v), 1.0)) if np.isfinite(v) else "#d9d9d9"
        for v in s_sorted
    ]
    s_sorted.plot(kind="barh", ax=ax, color=bar_colors)
    ax.set_xlabel(f"{_metric_label(metric)} (actor distributions in EN vs DE)")
    ax.set_ylabel("semantic axis (EN key)")
    ax.set_title("EN–DE association of actor positions on matched semantic axes")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / FNAME_BARPLOT, dpi=150)
    note = f"EN reference. flip_axis={flip_mode}. " + " ".join(sign_lines[:3])
    if len(sign_lines) > 3:
        note += f" … (+{len(sign_lines) - 3} more, see full docstring / logs)."
    return report_df, fig, s_sorted, note


def _align_de_to_en_pca_and_sem(
    pca_en: np.ndarray,
    pca_de: np.ndarray,
    idx_en: pd.Index,
    idx_de: pd.Index,
    sem_en: Dict[str, np.ndarray],
    sem_de: Dict[str, np.ndarray],
    w2v_en,
    w2v_de,
    c2t: Dict[str, str],
    flip_axis: Any = DEFAULT_FLIP_AXIS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """
    Optionally flip German PC1/PC2 and DE semantic concepts so that, on common
    actors, Pearson(EN, DE) >= 0. Returns modified pca_de and sem_de plus a log.
    """
    flip_mode = _normalize_flip_axis(flip_axis)
    flip_pca_axes = flip_mode in ("true", "pca")
    flip_semantic_axes = flip_mode == "true"
    pca_de = pca_de.copy()
    sem_de = {k: v.copy() for k, v in sem_de.items()}
    common = [n for n in idx_en if n in idx_de]
    if not common:
        return (
            pca_de,
            sem_de,
            f"flip_axis={flip_mode}; no common actors for cross-language sign alignment",
        )
    i_en = [idx_en.get_loc(n) for n in common]
    i_de = [idx_de.get_loc(n) for n in common]
    log: List[str] = []
    for k in (0, 1):
        if pca_en.shape[1] <= k:
            continue
        a = pca_en[i_en, k]
        b = pca_de[i_de, k]
        r = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 2 else 1.0
        if flip_pca_axes and (not np.isnan(r)) and r < 0:
            pca_de[:, k] *= -1.0
            log.append(f"PC{k+1}: GE flipped (r={r:.4f}, n={len(a)})")
        else:
            if flip_pca_axes:
                log.append(f"PC{k+1}: GE not flipped (r={r:.4f})")
            else:
                log.append(f"PC{k+1}: GE flip disabled (r={r:.4f})")
    if not flip_semantic_axes:
        log.append(f"semantic axis flips disabled by flip_axis={flip_mode}")
        return pca_de, sem_de, " | ".join(log)
    en_emb = su.actor_embeddings_from_w2v_entities(
        w2v_en, common, c2t, str(ENTITY_MAPPING_PATH)
    )
    de_emb = su.actor_embeddings_from_w2v_entities(
        w2v_de, common, c2t, str(ENTITY_MAPPING_PATH)
    )
    for name in list(sem_de.keys()):
        c_en = np.asarray(sem_en[name], dtype=np.float64)
        c_d = np.asarray(sem_de[name], dtype=np.float64)
        c_e = c_en / (np.linalg.norm(c_en) or 1.0)
        c_d_n = c_d / (np.linalg.norm(c_d) or 1.0)
        s1 = su.actor_proj(en_emb, c_e, rescale=True)
        s2 = su.actor_proj(de_emb, c_d_n, rescale=True)
        com2 = [n for n in common if n in s1.index and n in s2.index]
        a = np.array([s1.loc[n] for n in com2], dtype=np.float64)
        b = np.array([s2.loc[n] for n in com2], dtype=np.float64)
        if len(a) < 2:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if (not np.isnan(r)) and r < 0:
            sem_de[name] = -c_d
            log.append(f"sem {name}: GE flipped (r={r:.4f})")
    return pca_de, sem_de, " | ".join(log)


def _subset_pca_to_embeddings(
    pca_scores: np.ndarray,
    matrix_index: pd.Index,
    emb_index: pd.Index,
) -> np.ndarray:
    """
    ``visualize_pca_matrix`` returns scores for every matrix row, but
    ``actor_embeddings_from_w2v_entities`` drops OOV/missing-map actors. Select
    PCA rows in ``emb_index`` order so lengths match the embedding table.
    """
    if len(emb_index) == 0:
        return pca_scores[:0, :]
    rows: List[int] = []
    for n in emb_index:
        loc = matrix_index.get_loc(n)
        if isinstance(loc, (slice, np.ndarray)):
            loc = int(np.ravel(np.asarray(loc))[0])
        rows.append(int(loc))
    return pca_scores[np.array(rows, dtype=np.intp), :]


def _pc_sem_matrix_transposed(
    sem_axes: Dict[str, np.ndarray],
    emb: pd.DataFrame,
    pca_scores: np.ndarray,
    metric: str,
) -> pd.DataFrame:
    m = su.association_matrix(
        sem_axes,
        emb,
        pca_scores,
        n_components=2,
        actor_names=emb.index,
        semantic_index_style="canonical",
        entity_mapping_path=str(ENTITY_MAPPING_PATH),
        metric=metric,
    )
    return m.T


def _entity_report_pca_actors(
    matrix_index: pd.Index,
    kv,
    c2t: Dict[str, str],
    which: str,
) -> pd.DataFrame:
    rec: List[Tuple[str, str, str]] = []
    for n in matrix_index:
        t = c2t.get(n)
        if t is None:
            rec.append((n, "excluded", "not_in_mapping"))
        elif t not in kv:
            rec.append((n, "excluded", f"oov_{which}"))
        else:
            rec.append((n, "included", "ok_for_w2v"))
    return entity_report_dataframe(rec, f"pca_{which}")


def _stack_pca_rows_regular_and_distorted(
    m_reg: pd.DataFrame, m_dist: pd.DataFrame
) -> pd.DataFrame:
    """
    Stack row-normal (``rows``) and distorted (``both``) PC×semantic frames
    (4 rows: PC1_regular, PC2_regular, PC1_distorted, PC2_distorted).
    """
    a = m_reg.rename(index={n: f"{n}_regular" for n in m_reg.index})
    b = m_dist.rename(index={n: f"{n}_distorted" for n in m_dist.index})
    return pd.concat([a, b], axis=0)


def _plot_side_by_side_heatmaps_combined(
    m_en: pd.DataFrame, m_de: pd.DataFrame, title: str, out_path: Path, metric: str
) -> None:
    """EN | GE; one horizontal colorbar at bottom center."""
    is_abs = "abs" in metric or metric.endswith("_abs")
    val_min, val_max = (0.0, 1.0) if is_abs else (-1.0, 1.0)
    nrows = m_en.shape[0]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, max(6, 0.5 + 0.6 * nrows)))
    im0 = None
    for ax, mat, lab in ((ax0, m_en, "EN"), (ax1, m_de, "GE")):
        M = np.asarray(mat.values, dtype=np.float64)
        im0 = ax.imshow(
            M,
            aspect="auto",
            cmap="RdBu_r" if not is_abs else "Reds",
            vmin=val_min,
            vmax=val_max,
        )
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(list(mat.columns), rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(list(mat.index), fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="black"
                    )
        ax.set_title(lab, fontsize=11)
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2)
    cax = fig.add_axes([0.3, 0.06, 0.4, 0.025])
    cbar = fig.colorbar(im0, cax=cax, orientation="horizontal")
    cbar.set_label(metric, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pca_semantic_lollipop_combined(
    m_en: pd.DataFrame, m_de: pd.DataFrame, title: str, out_path: Path, metric: str
) -> None:
    """EN/GE endpoint comparisons for each stacked PCA × semantic-axis cell."""
    m_de = m_de.reindex(index=m_en.index, columns=m_en.columns)
    semantic_axes = sorted(m_en.columns)
    pc_group_labels = {
        "PC1_regular": "Regular, PC1:\nPolitical-Epistemic",
        "PC2_regular": "Regular, PC2:\nInstitution-Individual",
        "PC1_distorted": "Distorted, PC1:\nPolitical-Epistemic",
        "PC2_distorted": "Distorted, PC2:\nGood-Bad",
    }
    rows: List[Tuple[str, str, float, float, int]] = []
    y_pos = 0
    group_centers: Dict[str, float] = {}
    group_separators: List[float] = []
    for pc_label in list(m_en.index):
        group_start = y_pos
        for semantic_axis in semantic_axes:
            en_val = float(m_en.loc[pc_label, semantic_axis])
            de_val = float(m_de.loc[pc_label, semantic_axis])
            rows.append((pc_label, semantic_axis, en_val, de_val, y_pos))
            y_pos += 1
        group_centers[pc_label] = (group_start + y_pos - 1) / 2.0
        if pc_label != m_en.index[-1]:
            group_separators.append(y_pos - 0.5)

    nrows = len(rows)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.32 * nrows + 1.5)))
    for _pc_label, _semantic_axis, en_val, de_val, y in rows:
        if np.isfinite(en_val) and np.isfinite(de_val):
            ax.hlines(
                y=y,
                xmin=en_val,
                xmax=de_val,
                color="0.65",
                alpha=0.8,
                linewidth=1.2,
                zorder=1,
            )
        if np.isfinite(en_val):
            ax.scatter(en_val, y, color="#2ca02c", s=28, label="English", zorder=2)
        if np.isfinite(de_val):
            ax.scatter(de_val, y, color="#ff7f0e", s=28, label="German", zorder=2)

    for sep in group_separators:
        ax.axhline(sep, color="0.75", linestyle="--", linewidth=0.8, zorder=0)

    for pc_label, center in group_centers.items():
        ax.text(
            -1.34,
            center,
            pc_group_labels.get(pc_label, pc_label),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            rotation=90,
            rotation_mode="anchor",
            clip_on=False,
        )

    ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks(np.arange(-1.0, 1.0001, 0.25))
    ax.set_yticks([row[4] for row in rows])
    ax.set_yticklabels([row[1] for row in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(f"{_metric_label(metric)} (actor distributions on PCs vs semantic axes)")
    ax.set_ylabel("Semantic axis within PCA component", labelpad=80)
    ax.yaxis.set_label_coords(-0.34, 0.5)
    ax.set_title(title, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    unique_handles: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique_handles.setdefault(label, handle)
    ax.legend(
        unique_handles.values(),
        unique_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.92, bottom=0.13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _within_language_semantic_corr_df(
    emb: pd.DataFrame,
    sem_axes: Dict[str, np.ndarray],
    axis_order: List[str],
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """
    Selected association metric for actor-projection vectors between each pair of
    semantic axes, within one language.
    """
    n = len(axis_order)
    cols = []
    for name in axis_order:
        s = su.actor_proj(emb, sem_axes[name], rescale=True)
        aligned = s.reindex(emb.index)
        cols.append(aligned.values.astype(np.float64))
    X = np.column_stack(cols)
    R = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            a, b = X[:, i], X[:, j]
            c_ij = _pairwise_metric_value(a, b, metric)
            R[i, j] = c_ij
            R[j, i] = c_ij
    return pd.DataFrame(R, index=axis_order, columns=axis_order)


def _mask_to_strict_lower_triangle(R: np.ndarray) -> np.ndarray:
    """
    Show only the strict lower triangle (i > j). Upper triangle and main diagonal
    are set to NaN (diagonal is trivial: r=1 with self, or 0 in EN−GE diff of diagonals).
    """
    out = R.astype(float).copy()
    n = out.shape[0]
    for i in range(n):
        for j in range(n):
            if i <= j:
                out[i, j] = np.nan
    return out


def _plot_semantic_correlation_en_de(
    r_en: pd.DataFrame, r_de: pd.DataFrame, out_path: Path, subtitle: str, metric: str
) -> None:
    order = list(r_en.index)
    r_de = r_de.reindex(index=order, columns=order)
    Me = _mask_to_strict_lower_triangle(r_en.values)
    Md = _mask_to_strict_lower_triangle(r_de.values)
    vmin, vmax = _metric_plot_limits(np.concatenate([Me.ravel(), Md.ravel()]), metric)
    cmap = "RdBu_r" if _is_signed_correlation_metric(metric) else "Reds"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 8))
    im0 = None
    for ax, M, lab in ((ax0, Me, "EN"), (ax1, Md, "GE")):
        Mplot = np.ma.array(M, mask=~np.isfinite(M))
        im0 = ax.imshow(
            Mplot,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        n = M.shape[0]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(order, fontsize=6)
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center", fontsize=4, color="black"
                    )
        ax.set_title(f"{lab}: semantic–semantic (actors)", fontsize=10)
    fig.suptitle(
        f"Within-language: {_metric_label(metric)} between actor projection vectors "
        "(strict lower; no diagonal)\n"
        + subtitle,
        fontsize=9,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.2)
    cax = fig.add_axes([0.3, 0.07, 0.4, 0.025])
    cbar = fig.colorbar(im0, cax=cax, orientation="horizontal")
    cbar.set_label(_metric_label(metric), fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_semantic_correlation_diff(
    r_en: pd.DataFrame, r_de: pd.DataFrame, out_path: Path, metric: str
) -> None:
    order = list(r_en.index)
    r_de = r_de.reindex(index=order, columns=order)
    rows: List[Tuple[str, float, float]] = []
    for i, axis_1 in enumerate(order):
        for j, axis_2 in enumerate(order):
            if i <= j:
                continue
            en_val = float(r_en.loc[axis_1, axis_2])
            ge_val = float(r_de.loc[axis_1, axis_2])
            if np.isfinite(en_val) or np.isfinite(ge_val):
                rows.append((f"{axis_1} vs {axis_2}", en_val, ge_val))

    def abs_en_ge_difference(row: Tuple[str, float, float]) -> float:
        _label, en_val, ge_val = row
        if not (np.isfinite(en_val) and np.isfinite(ge_val)):
            return -np.inf
        return en_val - ge_val

    rows.sort(key=abs_en_ge_difference, reverse=True)
    top_n = 5
    top_rows = rows[:top_n]
    bottom_rows = rows[-top_n:] if len(rows) > top_n else []
    rows = top_rows + bottom_rows
    semantic_diff_group_labels = {
        "top": "Top 5 deltas",
        "bottom": "Bottom 5 deltas",
    }

    nrows = len(rows)
    all_values = np.array([v for _label, en_val, ge_val in rows for v in (en_val, ge_val)])
    xmin, xmax = _metric_plot_limits(all_values, metric)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * nrows + 1.5)))
    for y, (_label, en_val, ge_val) in enumerate(rows):
        if np.isfinite(en_val) and np.isfinite(ge_val):
            ax.hlines(
                y=y,
                xmin=en_val,
                xmax=ge_val,
                color="0.65",
                alpha=0.8,
                linewidth=1.2,
                zorder=1,
            )
        if np.isfinite(en_val):
            ax.scatter(en_val, y, color="#2ca02c", s=28, label="English", zorder=2)
        if np.isfinite(ge_val):
            ax.scatter(ge_val, y, color="#ff7f0e", s=28, label="German", zorder=2)

    if top_rows and bottom_rows:
        ax.axhline(
            len(top_rows) - 0.5,
            color="0.75",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )
        x_text = xmin + 0.02 * (xmax - xmin)
        ax.text(
            x_text,
            -0.28,
            semantic_diff_group_labels["top"],
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            clip_on=True,
        )
        ax.text(
            x_text,
            len(top_rows) - 0.22,
            semantic_diff_group_labels["bottom"],
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            clip_on=True,
        )

    if xmin <= 0 <= xmax:
        ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlim(xmin, xmax)
    if (xmin, xmax) in {(-1.0, 1.0), (0.0, 1.0)}:
        ax.set_xticks(np.arange(xmin, xmax + 0.0001, 0.25))
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels([row[0] for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{_metric_label(metric)} (actor distributions on semantic axes)")
    ax.set_ylabel(r"$\bf{Semantic\ axes\ compared}$", labelpad=68)
    ax.yaxis.set_label_coords(-0.37, 0.5)
    ax.set_title(
        f"EN vs GE: differences in within-language semantic-axis correlation ({_metric_label(metric)})",
        fontsize=10,
    )
    handles, labels = ax.get_legend_handles_labels()
    unique_handles: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique_handles.setdefault(label, handle)
    ax.legend(
        unique_handles.values(),
        unique_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.92, bottom=0.13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _format_ranked_series(s: pd.Series, n: int = 10, ascending: bool = False) -> str:
    """Compact display for the top/bottom ranked labels and signed scores."""
    s = s.dropna()
    ranked = s.nsmallest(n) if ascending else s.nlargest(n)
    return "; ".join(f"{idx} ({val:.3f})" for idx, val in ranked.items())


def _pole_terms(
    pairs: Sequence[Tuple[str, str]], pole: str, sign: float = 1.0
) -> str:
    """Return only the single antonym terms that define the displayed pole."""
    if (pole == "positive" and sign >= 0) or (pole == "negative" and sign < 0):
        idx = 0
    else:
        idx = 1
    terms = sorted({str(pair[idx]) for pair in pairs})
    return "; ".join(terms)


def _axis_sign_against_original(current: np.ndarray, original: np.ndarray) -> float:
    """Detect whether a vector/score column was sign-flipped relative to original."""
    if np.allclose(current, -original, equal_nan=True):
        return -1.0
    return 1.0


def _find_trump_anchor_actor(matrix_index: pd.Index, c2t: Dict[str, str]) -> Optional[str]:
    """Find the canonical actor row used for Donald Trump anchoring."""
    if TRUMP_ANCHOR_CANONICAL in matrix_index:
        return TRUMP_ANCHOR_CANONICAL
    normalized_target = TRUMP_ANCHOR_TOKEN.lower()
    for actor in matrix_index:
        actor_s = str(actor)
        actor_norm = actor_s.lower().replace(" ", "_")
        token = str(c2t.get(actor_s, "")).lower()
        if actor_norm == normalized_target or token == normalized_target:
            return actor_s
    return None


def _apply_trump_anchor_to_pca(
    pca_scores: np.ndarray,
    matrix_index: pd.Index,
    pca_type: str,
    language: str,
    c2t: Dict[str, str],
) -> Tuple[np.ndarray, List[float], str]:
    """
    Orient PCA scores so Donald Trump's sign matches the requested pole per PCA type.

    Returns the oriented scores, per-component multipliers, and a compact log.
    """
    out = pca_scores.copy()
    signs = [1.0 for _ in range(out.shape[1])]
    targets = TRUMP_ANCHOR_TARGETS[pca_type]
    anchor_actor = _find_trump_anchor_actor(matrix_index, c2t)
    if anchor_actor is None:
        return (
            out,
            signs,
            f"{language} {pca_type}: {TRUMP_ANCHOR_CANONICAL!r} not found for PCA anchor",
        )
    anchor_idx = matrix_index.get_loc(anchor_actor)
    if isinstance(anchor_idx, (slice, np.ndarray)):
        anchor_idx = int(np.ravel(np.asarray(anchor_idx))[0])
    log: List[str] = []
    for k, target_sign in enumerate(targets):
        if out.shape[1] <= k:
            continue
        score = float(out[int(anchor_idx), k])
        should_flip = (target_sign > 0 and score < 0) or (target_sign < 0 and score > 0)
        if should_flip:
            out[:, k] *= -1.0
            signs[k] = -1.0
            score = -score
            log.append(
                f"{language} {pca_type} PC{k+1}: flipped; "
                f"{anchor_actor}={score:.4f}, target_sign={target_sign:+.0f}"
            )
        else:
            log.append(
                f"{language} {pca_type} PC{k+1}: not flipped; "
                f"{anchor_actor}={score:.4f}, target_sign={target_sign:+.0f}"
            )
    return out, signs, " | ".join(log)


def _append_pole_report_rows(
    rows: List[Dict[str, str]],
    language: str,
    pca_type: str,
    pca,
    pca_scores: np.ndarray,
    pca_signs: Sequence[float],
    actor_names: Sequence[str],
    feature_names: Sequence[str],
    sem_axes: Dict[str, np.ndarray],
    sem_pairs: Dict[str, Sequence[Tuple[str, str]]],
    sem_signs: Dict[str, float],
    emb: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """Append PCA and semantic pole definitions plus top actor placements."""
    for k in range(min(2, pca_scores.shape[1], len(pca.components_))):
        axis_label = f"PC{k + 1}_{pca_type}"
        loadings = pd.Series(
            np.asarray(pca.components_[k], dtype=np.float64) * float(pca_signs[k]),
            index=list(feature_names),
        )
        actor_scores = pd.Series(
            np.asarray(pca_scores[:, k], dtype=np.float64), index=list(actor_names)
        )
        for pole, ascending in (("negative", True), ("positive", False)):
            rows.append(
                {
                    "language": language,
                    "pca_type": pca_type,
                    "axis_kind": "PCA",
                    "axis": axis_label,
                    "pole": pole,
                    "pole_definition": _format_ranked_series(
                        loadings, n=top_n, ascending=ascending
                    ),
                    "top_actors": _format_ranked_series(
                        actor_scores, n=top_n, ascending=ascending
                    ),
                }
            )

    for axis_name in sorted(sem_axes):
        if axis_name not in sem_pairs:
            continue
        projections = su.actor_proj(
            emb, np.asarray(sem_axes[axis_name], dtype=np.float64), rescale=True
        )
        sign = float(sem_signs.get(axis_name, 1.0))
        for pole, ascending in (("negative", True), ("positive", False)):
            rows.append(
                {
                    "language": language,
                    "pca_type": pca_type,
                    "axis_kind": "semantic",
                    "axis": axis_name,
                    "pole": pole,
                    "pole_definition": _pole_terms(sem_pairs[axis_name], pole, sign),
                    "top_actors": _format_ranked_series(
                        projections, n=top_n, ascending=ascending
                    ),
                }
            )


def run_semantic_correlation_heatmaps(
    emb_en: pd.DataFrame,
    emb_de: pd.DataFrame,
    sem_en: Dict[str, np.ndarray],
    sem_de: Dict[str, np.ndarray],
    metric: str = DEFAULT_METRIC,
) -> None:
    """
    EN | GE side-by-side within-language axis association matrices, then EN−GE
    endpoint comparison. Uses the selected ``metric`` for pairwise semantic-axis
    actor-projection vectors. Uses the same **base** axis names for both; actor sets are
    ``emb_en`` / ``emb_de`` (typically from row-normalized PCA filter, w2v-OOV
    already dropped in embeddings). Plots use the strict lower triangle
    (diagonal masked: no *i*=*j*).
    """
    order = sorted(set(sem_en.keys()) & set(sem_de.keys()))
    if not order:
        return
    r_en = _within_language_semantic_corr_df(emb_en, sem_en, order, metric=metric)
    r_de = _within_language_semantic_corr_df(emb_de, sem_de, order, metric=metric)
    _plot_semantic_correlation_en_de(
        r_en,
        r_de,
        PLOTS_DIR / FNAME_SEM_CORR_EN_DE,
        "EN/GE w2v actor rows: row-normalized PCA slice (thesis filters).",
        metric,
    )
    _plot_semantic_correlation_diff(r_en, r_de, PLOTS_DIR / FNAME_SEM_CORR_DIFF, metric)


def run_pca_semantic_heatmaps_and_diff(
    metric: str = DEFAULT_METRIC,
    flip_axis: Any = DEFAULT_FLIP_AXIS,
    print_pole_report: bool = True,
) -> Tuple[
    List[pd.DataFrame],
    str,
    str,
    str,
    str,
    str,
    str,
]:
    """
    Build EN/GE PCAs for row-normalized and distorted actor–action matrices,
    optionally align GE axes to EN, stack PC1/2_regular and PC1/2_distorted in
    one heatmap (EN|GE) and one diff plot, then build semantic–semantic
    correlation figures (EN|GE and EN−GE diff).
    """
    flip_mode = _normalize_flip_axis(flip_axis)
    w2v_en = helpers.load_trained_w2v_keyed_vectors(MODEL_EN_NAME)
    w2v_de = helpers.load_trained_w2v_keyed_vectors(MODEL_DE_NAME)
    c2t = helpers.load_canonical_to_w2v_token(str(ENTITY_MAPPING_PATH))
    en_f, de_f = load_motif_frames()
    mask_en = ~en_f["category"].str.endswith(EN_FILTER_MASK_SUFFIXES)
    mask_de = ~de_f["category"].str.endswith(DE_FILTER_MASK_SUFFIXES)
    reports: List[pd.DataFrame] = []
    pair_axes = load_axes_paired()
    axis_pairs_en = {
        base: en_pairs for base, _de_key, en_pairs, _de_pairs in pair_axes
    }
    axis_pairs_de = {
        base: de_pairs for base, _de_key, _en_pairs, de_pairs in pair_axes
    }
    en_smax, de_smax = build_sem_axis_dicts(pair_axes, w2v_en, w2v_de)
    sem_en = {k: np.asarray(v.concept_vector, dtype=np.float64) for k, v in en_smax.items()}
    sem_de = {k: np.asarray(v.concept_vector, dtype=np.float64) for k, v in de_smax.items()}
    blocks: List[Tuple[pd.DataFrame, pd.DataFrame, str]] = []
    pole_report_rows: List[Dict[str, str]] = []
    emb_en_reg: Optional[pd.DataFrame] = None
    emb_de_reg: Optional[pd.DataFrame] = None
    for norm_label, en_norm, de_norm in (
        ("row-normalized (rows)", "rows", "rows"),
        ("distorted (both)", "both", "both"),
    ):
        t_en, x_en, _, mat_en, _ = helpers.visualize_pca_matrix(
            helpers.create_actor_action_matrix(
                en_f, ("action", "canonical", "category"), filter_mask=mask_en, normalize=en_norm
            ),
            n_components=2,
            actors=True,
            actions=False,
        )
        t_de, x_de, _, mat_de, _ = helpers.visualize_pca_matrix(
            helpers.create_actor_action_matrix(
                de_f, ("action", "canonical", "category"), filter_mask=mask_de, normalize=de_norm
            ),
            n_components=2,
            actors=True,
            actions=False,
        )
        rep_en = _entity_report_pca_actors(mat_en.index, w2v_en, c2t, "en")
        rep_de = _entity_report_pca_actors(mat_de.index, w2v_de, c2t, "de")
        rep_en["pca_type"] = norm_label
        rep_de["pca_type"] = norm_label
        reports.append(pd.concat([rep_en, rep_de], ignore_index=True))
        emb_en = su.actor_embeddings_from_w2v_entities(
            w2v_en, mat_en.index, c2t, str(ENTITY_MAPPING_PATH)
        )
        emb_de = su.actor_embeddings_from_w2v_entities(
            w2v_de, mat_de.index, c2t, str(ENTITY_MAPPING_PATH)
        )
        pca_g = x_de.copy()
        s_g = {k: v.copy() for k, v in sem_de.items()}
        label_short = "regular" if en_norm == "rows" else "distorted"
        pca_g, s_g, loga = _align_de_to_en_pca_and_sem(
            x_en,
            pca_g,
            mat_en.index,
            mat_de.index,
            sem_en,
            s_g,
            w2v_en,
            w2v_de,
            c2t,
            flip_axis=flip_mode,
        )
        pca_signs_en = [1.0 for _ in range(x_en.shape[1])]
        if flip_mode == "pca_trump_anchor":
            x_en, pca_signs_en, loga_en = _apply_trump_anchor_to_pca(
                x_en,
                mat_en.index,
                label_short,
                "EN",
                c2t,
            )
            pca_g, pca_signs_de, loga_de = _apply_trump_anchor_to_pca(
                pca_g,
                mat_de.index,
                label_short,
                "GE",
                c2t,
            )
            loga = f"{loga} | {loga_en} | {loga_de}"
        else:
            pca_signs_de = [
                _axis_sign_against_original(pca_g[:, k], x_de[:, k])
                for k in range(min(pca_g.shape[1], x_de.shape[1]))
            ]
        sem_signs_de = {
            name: _axis_sign_against_original(s_g[name], sem_de[name])
            for name in s_g
            if name in sem_de
        }
        x_en_w2v = _subset_pca_to_embeddings(x_en, mat_en.index, emb_en.index)
        pca_g_w2v = _subset_pca_to_embeddings(pca_g, mat_de.index, emb_de.index)
        m_en = _pc_sem_matrix_transposed(sem_en, emb_en, x_en_w2v, metric)
        m_ge = _pc_sem_matrix_transposed(s_g, emb_de, pca_g_w2v, metric)
        m_ge = m_ge.reindex(index=m_en.index, columns=m_en.columns)
        blocks.append((m_en, m_ge, loga))
        _append_pole_report_rows(
            pole_report_rows,
            "EN",
            label_short,
            t_en,
            x_en,
            pca_signs_en,
            list(mat_en.index),
            list(mat_en.columns),
            sem_en,
            axis_pairs_en,
            {name: 1.0 for name in sem_en},
            emb_en,
        )
        _append_pole_report_rows(
            pole_report_rows,
            "GE",
            label_short,
            t_de,
            pca_g,
            pca_signs_de,
            list(mat_de.index),
            list(mat_de.columns),
            s_g,
            axis_pairs_de,
            sem_signs_de,
            emb_de,
        )
        if en_norm == "rows":
            emb_en_reg = emb_en
            emb_de_reg = emb_de
    if print_pole_report and pole_report_rows:
        pole_report = pd.DataFrame(pole_report_rows)
        print("\n=== PCA/Semantic pole definitions and top actor placements ===")
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            180,
            "display.width",
            240,
        ):
            print(pole_report.to_string(index=False))
    b0, b1 = blocks[0], blocks[1]
    m_en_c = _stack_pca_rows_regular_and_distorted(b0[0], b1[0])
    m_ge_c = _stack_pca_rows_regular_and_distorted(b0[1], b1[1])
    log_s = f"Regular: {b0[2][:180]} …\nDistorted: {b1[2][:180]}"
    _plot_side_by_side_heatmaps_combined(
        m_en_c,
        m_ge_c,
        f"PCA vs semantic — {metric} (stacked: PC1/2 regular + PC1/2 distorted)\n{log_s}",
        PLOTS_DIR / FNAME_PCA_HEAT_COMBINED,
        metric,
    )
    _plot_pca_semantic_lollipop_combined(
        m_en_c,
        m_ge_c,
        f"Comparing correlations between PCA and semantic axes (Spearman rho)",
        PLOTS_DIR / FNAME_PCA_DIFF_COMBINED,
        metric,
    )
    if emb_en_reg is not None and emb_de_reg is not None:
        run_semantic_correlation_heatmaps(
            emb_en_reg,
            emb_de_reg,
            sem_en,
            sem_de,
            metric=metric,
        )
    s_reps = "\n\n".join(r.to_string() for r in reports)
    pnote = (
        f"flip_axis={flip_mode}. Semantic–semantic: row-norm actor slice."
    )
    return (
        reports,
        s_reps,
        str((PLOTS_DIR / FNAME_PCA_HEAT_COMBINED).resolve()),
        str((PLOTS_DIR / FNAME_PCA_DIFF_COMBINED).resolve()),
        str((PLOTS_DIR / FNAME_SEM_CORR_EN_DE).resolve()),
        str((PLOTS_DIR / FNAME_SEM_CORR_DIFF).resolve()),
        pnote,
    )


def _normalized_w2v_vector(kv, token: str) -> np.ndarray:
    vec = np.asarray(kv[token], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError(f"Vector for {token!r} has zero norm.")
    return vec / norm


def _axis_projection_frame(
    kv,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    actor_token_to_canonical: Dict[str, str],
    axis_words: Iterable[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen_actor_labels: set = set()
    for token, canonical in sorted(actor_token_to_canonical.items(), key=lambda item: item[1]):
        if token not in kv or canonical in seen_actor_labels:
            continue
        vec = _normalized_w2v_vector(kv, token)
        rows.append(
            {
                "label": canonical,
                "kind": "actor",
                "x": float(vec @ axis_x),
                "y": float(vec @ axis_y),
            }
        )
        seen_actor_labels.add(canonical)

    for word in sorted(set(str(w) for w in axis_words)):
        if word not in kv:
            continue
        vec = _normalized_w2v_vector(kv, word)
        rows.append(
            {
                "label": word,
                "kind": "axis word",
                "x": float(vec @ axis_x),
                "y": float(vec @ axis_y),
            }
        )
    return pd.DataFrame(rows, columns=["label", "kind", "x", "y"])


def _all_axis_words(axis_defs: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    return [word for pairs in axis_defs.values() for pair in pairs for word in pair]


def _format_crossaxis_projection_report(
    df: pd.DataFrame,
    language: str,
    x_axis_name: str,
    y_axis_name: str,
    x_positive: str,
    y_positive: str,
) -> str:
    actor_count = int((df["kind"] == "actor").sum())
    word_count = int((df["kind"] == "axis word").sum())
    header = (
        f"{language}: {x_axis_name} (+{x_positive}) x "
        f"{y_axis_name} (+{y_positive}); "
        f"actors={actor_count}, axis_words={word_count}"
    )
    body = df.sort_values(["kind", "label"]).to_string(index=False)
    return f"{header}\n{body}"


def run_crossaxis_projection_report(crossaxis: str) -> str:
    """
    Report actors and all axis-defining words on the 2-D plane formed by two
    selected semantic axes. ``crossaxis`` uses English positive poles, e.g.
    ``reveal_holy`` -> x: reveal_hide, y: holy_unholy.
    """
    axes_en = _load_axis_json(AXES_EN_PATH)
    axes_de = _load_axis_json(AXES_DE_PATH)
    x_base, y_base, x_pole, y_pole = _resolve_crossaxis_selection(
        crossaxis,
        axes_en,
        axes_de,
    )
    x_de = f"{x_base}_de"
    y_de = f"{y_base}_de"

    w2v_en = helpers.load_trained_w2v_keyed_vectors(MODEL_EN_NAME)
    w2v_de = helpers.load_trained_w2v_keyed_vectors(MODEL_DE_NAME)
    token_to_canonical = helpers.load_w2v_token_to_canonical(str(ENTITY_MAPPING_PATH))

    x_axis_en = semaxis_util.SemAxis(axes_en[x_base], w2v_en, name=x_base).concept_vector
    y_axis_en = semaxis_util.SemAxis(axes_en[y_base], w2v_en, name=y_base).concept_vector
    x_axis_de = semaxis_util.SemAxis(axes_de[x_de], w2v_de, name=x_de).concept_vector
    y_axis_de = semaxis_util.SemAxis(axes_de[y_de], w2v_de, name=y_de).concept_vector

    x_axis_en = x_axis_en / (np.linalg.norm(x_axis_en) or 1.0)
    y_axis_en = y_axis_en / (np.linalg.norm(y_axis_en) or 1.0)
    x_axis_de = x_axis_de / (np.linalg.norm(x_axis_de) or 1.0)
    y_axis_de = y_axis_de / (np.linalg.norm(y_axis_de) or 1.0)

    en_df = _axis_projection_frame(
        w2v_en,
        x_axis_en,
        y_axis_en,
        token_to_canonical,
        _all_axis_words(axes_en),
    )
    de_df = _axis_projection_frame(
        w2v_de,
        x_axis_de,
        y_axis_de,
        token_to_canonical,
        _all_axis_words(axes_de),
    )

    en_report = _format_crossaxis_projection_report(
        en_df,
        "EN",
        x_base,
        y_base,
        x_pole,
        y_pole,
    )
    de_report = _format_crossaxis_projection_report(
        de_df,
        "GE",
        x_de,
        y_de,
        axes_de[x_de][0][0],
        axes_de[y_de][0][0],
    )
    return "\n\n".join(
        [
            "=== Cross-axis 2-D projection coordinates ===",
            f"Selector: {crossaxis}",
            en_report,
            de_report,
        ]
    )


def run_all(
    metric: str = DEFAULT_METRIC,
    flip_axis: Any = DEFAULT_FLIP_AXIS,
    crossaxis: Optional[str] = None,
) -> None:
    """CLI entry: barplot, then PCA heatmaps and diff matrices. Prints entity tables."""
    flip_mode = _normalize_flip_axis(flip_axis)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    r1, _fig, sser, snote = run_cross_language_axis_bars(
        metric=metric, flip_axis=flip_mode
    )
    print("=== (1) Cross-language bars entity report ===")
    print(r1.to_string())
    print("\nFlip-axis alignment (first lines):", snote)
    rlist, srep, p1, p2, p3, p4, pnote = run_pca_semantic_heatmaps_and_diff(
        metric=metric,
        flip_axis=flip_mode,
    )
    print("\n=== (2)–(4) PCA + semantic / entity reports (EN+GE) ===")
    print(srep)
    print(
        "\nFigure paths:\n  PCA heat (combined):",
        p1,
        "\n  PCA diff (combined):",
        p2,
        "\n  Semantic r (EN|GE):",
        p3,
        "\n  Semantic Δr (EN-GE):",
        p4,
        "\n",
        pnote,
    )
    if crossaxis:
        print("\n" + run_crossaxis_projection_report(crossaxis))


def main() -> None:
    ap = argparse.ArgumentParser(description="EN–GE semantic axis & PCA analysis")
    ap.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Key from semanalysis_util.compare_semantic_to_pca (default: correlation_abs). "
        "Use 'correlation' for signed heatmaps and interpretable EN−GE diffs.",
    )
    ap.add_argument(
        "--flip_axis",
        choices=FLIP_AXIS_CHOICES,
        default=DEFAULT_FLIP_AXIS,
        help=(
            "German axis sign alignment mode: 'true' flips PCA and semantic axes "
            "as in the legacy behavior; 'pca' flips only PCA axes; 'false' "
            "disables sign flips (default); 'pca_trump_anchor' orients EN and GE "
            "PCA axes using Donald Trump as an anchor."
        ),
    )
    ap.add_argument(
        "--crossaxis",
        default=None,
        help=(
            "Optional two-axis projection selector, written as two English "
            "positive-pole words separated by '_', e.g. 'reveal_holy'. When set, "
            "prints 2-D actor and axis-word projection coordinates."
        ),
    )
    args = ap.parse_args()
    run_all(metric=args.metric, flip_axis=args.flip_axis, crossaxis=args.crossaxis)


if __name__ == "__main__":
    main()
