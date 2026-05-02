"""
Semantic Analysis Utility Functions

Actor semantics use trained Word2Vec KeyedVectors (entity token lookups via
actor_embeddings_from_w2v_entities). PCA from helpers.visualize_pca_matrix uses
canonical actor row labels; comparison helpers reindex w2v-token series to
canonical by default using pca/modules/entities/entities_mapping_2.json.

All functions assume projection inputs are L2-normalized where noted.

COMPATIBILITY WITH PCA (helpers.py):
    The comparison functions in this module are designed to work with the PCA implementation
    in helpers.visualize_pca_matrix(), which:
    - Column-wise standardizes the actor-action matrix (z-scores: mean=0, std=1)
    - Fits PCA on the standardized matrix
    - Returns raw (unscaled) pca_scores and the pca object
    - Rescales coordinates to [-1, 1] only for visualization
    
    To ensure compatibility:
    - Use actor_embeddings_from_w2v_entities() for trained w2v + canonical PCA rows;
      use compare_semantic_to_pca(..., semantic_index_style="canonical") when embeddings
      are canonical-indexed, or default "w2v_token" with automatic reindexing.
    - Pass raw pca_scores (2nd return value) and pca object (1st return value) to comparison functions
    - The rescale parameters in comparison functions match the visualization rescaling approach

    Legacy: actor_embd() (verb-weighted profiles) is optional for old workflows.
"""

import importlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Union, Tuple, Optional, Iterable, Any
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, kendalltau


def _helpers():
    return importlib.import_module("helpers")


def _scipy_stat_pvalue(result: Any) -> Tuple[float, float]:
    """Normalize scipy.stats result (tuple or object with statistic/pvalue)."""
    if hasattr(result, "statistic") and hasattr(result, "pvalue"):
        return float(result.statistic), float(result.pvalue)
    a, b = result
    return float(a), float(b)


def actor_embeddings_from_w2v_entities(
    w2v_model: Any,
    actors: Iterable,
    canonical_to_w2v_token: Optional[Dict[str, str]] = None,
    entity_mapping_path: Optional[str] = None,
    normalize_vectors: bool = True,
    index_style: str = "canonical",
) -> pd.DataFrame:
    """
    One embedding row per actor via KeyedVectors lookup (trained w2v only).

    index_style ``canonical`` (default): actors are PCA names; resolve tokens via
    entities_mapping_2.json. Index of the result is canonical.

    index_style ``w2v_token``: actors are vocabulary keys; index is those keys.

    Skips unmapped or OOV actors (warns with counts).
    """
    h = _helpers()
    if index_style not in ("canonical", "w2v_token"):
        raise ValueError("index_style must be 'canonical' or 'w2v_token'")

    if index_style == "canonical" and canonical_to_w2v_token is None:
        canonical_to_w2v_token = h.load_canonical_to_w2v_token(entity_mapping_path)

    rows = []
    idx_out = []
    missing_map = 0
    missing_oov = 0
    if hasattr(w2v_model, "vector_size"):
        dim = int(w2v_model.vector_size)
    else:
        k0 = w2v_model.index_to_key[0]
        dim = int(np.asarray(w2v_model[k0]).shape[-1])

    for actor in actors:
        if index_style == "canonical":
            token = canonical_to_w2v_token.get(actor)
            if token is None:
                missing_map += 1
                continue
            label = actor
        else:
            token = actor
            label = actor

        if token not in w2v_model:
            missing_oov += 1
            continue

        v = np.asarray(w2v_model[token], dtype=float).copy()
        if normalize_vectors:
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v /= nrm
        rows.append(v)
        idx_out.append(label)

    if missing_map or missing_oov:
        import warnings

        warnings.warn(
            f"actor_embeddings_from_w2v_entities: skipped missing_map={missing_map}, "
            f"oov={missing_oov}; kept {len(rows)} actors.",
            UserWarning,
            stacklevel=2,
        )

    if not rows:
        raise ValueError("No actor vectors could be built (check map and w2v vocabulary).")

    return pd.DataFrame(
        rows,
        index=pd.Index(idx_out, name=None),
        columns=[f"dim_{i}" for i in range(dim)],
    )


def actor_embd(
    matrix: pd.DataFrame,
    verb_embeddings: Dict[str, np.ndarray],
    normalize: str = 'both'
) -> pd.DataFrame:
    """
    [Legacy] Weighted average of per-verb embedding vectors from an external dict.

    Prefer actor_embeddings_from_w2v_entities() with trained KeyedVectors for the
    current pipeline (no verb parquet).

    This function creates embeddings for actors by taking a weighted average of the
    embeddings of the actions they perform. The weighting can be normalized by row
    (actor), column (action), both (geometric mean), or standardized to match PCA preprocessing.
    
    COMPATIBILITY NOTE: The PCA implementation in helpers.py uses column-wise standardization
    (z-scores: mean=0, std=1) before fitting PCA. For maximum comparability with PCA results,
    use normalize='standardize'. For other applications, 'both' (geometric mean) is recommended.
    
    Args:
        matrix: Actor-action matrix where rows are actors and columns are actions.
                Values represent counts or frequencies of actions performed by actors.
        verb_embeddings: Dictionary mapping action/verb strings to their embedding
                        vectors (numpy arrays). All vectors should be pre-normalized
                        to unit length (2-norm = 1).
        normalize: Normalization method for matrix cell values. Options:
                  - 'row': Normalize by row (each actor's actions sum to 1)
                  - 'column': Normalize by column (each action sums to 1 across actors)
                  - 'both': Geometric mean of row and column normalization
                  - 'standardize': Column-wise z-score standardization (mean=0, std=1)
                                   - matches PCA preprocessing in helpers.py
                  - 'none': Use raw values without normalization
                  
    Returns:
        DataFrame with actor embeddings where rows are actors and columns are
        embedding dimensions (dim_0, dim_1, ..., dim_n).
        
    Example:
        >>> matrix = pd.DataFrame({'run': [5, 2], 'jump': [1, 3]}, 
        ...                       index=['Actor1', 'Actor2'])
        >>> embeddings = {'run': np.array([0.1, 0.2, 0.3]),
        ...               'jump': np.array([0.4, 0.5, 0.6])}
        >>> # For comparison with PCA:
        >>> actor_embeddings = actor_embd(matrix, embeddings, normalize='standardize')
        >>> # For general use:
        >>> actor_embeddings = actor_embd(matrix, embeddings, normalize='both')
    """
    from sklearn.preprocessing import StandardScaler
    
    # Apply normalization based on the specified method
    if normalize == 'none':
        normalized_matrix = matrix.copy()
    elif normalize == 'row':
        # Row normalization: each actor's actions sum to 1
        normalized_matrix = matrix.div(matrix.sum(axis=1), axis=0)
    elif normalize == 'column':
        # Column normalization: each action sums to 1 across actors
        normalized_matrix = matrix.div(matrix.sum(axis=0), axis=1)
    elif normalize == 'both':
        # Geometric mean of row and column normalization
        row_norms = matrix.div(matrix.sum(axis=1), axis=0)
        col_norms = matrix.div(matrix.sum(axis=0), axis=1)
        normalized_matrix = np.sqrt(row_norms * col_norms)
    elif normalize == 'standardize':
        # Column-wise standardization (z-scores) - matches PCA preprocessing in helpers.py
        scaler = StandardScaler(with_mean=True, with_std=True)
        standardized_values = scaler.fit_transform(matrix.values)
        normalized_matrix = pd.DataFrame(
            standardized_values,
            index=matrix.index,
            columns=matrix.columns
        )
    else:
        raise ValueError(
            f"normalize must be 'none', 'row', 'column', 'both', or 'standardize', got '{normalize}'"
        )
    
    # Filter to verbs that have embeddings
    available_verbs = [v for v in matrix.columns if v in verb_embeddings]
    missing_verbs = [v for v in matrix.columns if v not in verb_embeddings]
    
    if missing_verbs:
        print(f"Warning: {len(missing_verbs)} verbs not found in embeddings (will be skipped)")
        if len(missing_verbs) <= 10:
            print(f"  Missing verbs: {missing_verbs}")
    
    if not available_verbs:
        raise ValueError("No verbs in matrix have embeddings!")
    
    # Get embedding dimension from the first available embedding
    embedding_dim = next(iter(verb_embeddings.values())).shape[0]
    
    # Initialize actor embeddings matrix
    actor_embeddings = np.zeros((len(matrix), embedding_dim))
    
    # Calculate weighted average for each actor
    for i, actor in enumerate(matrix.index):
        # Get weights for this actor (normalized values for available verbs)
        weights = normalized_matrix.loc[actor, available_verbs].values
        
        # Skip if all weights are zero or NaN
        if not np.any(weights > 0) or np.all(np.isnan(weights)):
            continue
        
        # Stack verb embeddings into a matrix (one row per verb)
        verb_emb_matrix = np.vstack([verb_embeddings[v] for v in available_verbs])
        
        # Handle NaN values in weights
        weights = np.nan_to_num(weights)
        weight_sum = weights.sum()
        
        # Calculate weighted average
        if weight_sum > 0:
            actor_embeddings[i] = (weights @ verb_emb_matrix) / weight_sum
    
    # Convert to DataFrame with appropriate column names
    embedding_df = pd.DataFrame(
        actor_embeddings,
        index=matrix.index,
        columns=[f'dim_{i}' for i in range(embedding_dim)]
    )
    
    return embedding_df


def actor_proj(
    actor_embeddings: pd.DataFrame,
    semantic_axis: np.ndarray,
    rescale: bool = True
) -> pd.Series:
    """
    Project actors onto a semantic axis using dot product.
    
    This implements the projection formula from the B&J paper (p.5):
    sim(vec_actor, [[vec_concept_axis]]) = vec_actor · [[vec_concept_axis]]
    
    where [[v]] denotes normalization: [[v]] = v / ||v||_2
    
    Args:
        actor_embeddings: DataFrame with actor embeddings (rows=actors, cols=embedding dims)
        semantic_axis: Concept axis vector (should already be normalized if following B&J)
        rescale: If True, rescale positions to [-1, 1] range for visualization
        
    Returns:
        Series with actor positions along the axis (index=actors, values=positions)
        
    Example:
        >>> positions = actor_proj(actor_embeddings, concept_axis, rescale=True)
    """
    # Ensure semantic_axis is normalized (as per B&J notation [[v]])
    norm = np.linalg.norm(semantic_axis)
    if norm > 0:
        semantic_axis_normalized = semantic_axis / norm
    else:
        raise ValueError("Semantic axis has zero norm - cannot normalize")
    
    # Convert embeddings to numpy array
    embeddings_array = actor_embeddings.values
    
    # Project onto axis (dot product)
    # This gives sim(vec_actor, [[vec_concept_axis]])
    positions = embeddings_array @ semantic_axis_normalized
    
    # Rescale to [-1, 1] if requested
    if rescale:
        max_abs = np.max(np.abs(positions))
        if max_abs > 0:
            positions = positions / max_abs
    
    # Convert to Series
    positions_series = pd.Series(positions, index=actor_embeddings.index, name='projection')
    
    return positions_series


def compare_semantic_to_pca(
    semantic_positions: pd.Series,
    pca_scores: np.ndarray,
    pca_component: int,
    actor_names: pd.Index,
    rescale_pca: bool = True,
    semantic_index_style: str = "w2v_token",
    token_to_canonical: Optional[Dict[str, str]] = None,
    entity_mapping_path: Optional[str] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Compare actor positions on a semantic axis to their positions on a PCA component.
    
    This function is adapted from embedding_util.py to work with word2vec-based
    semantic axes. It calculates correlation and a normalized distance metric.
    
    COMPATIBILITY WITH helpers.py PCA:
    - Expects RAW pca_scores as returned by visualize_pca_matrix() (2nd return value)
    - These scores come from PCA on column-standardized data (z-scores)
    - rescale_pca=True rescales to [-1, 1] per component (matching visualization approach)
    - actor_names / PCA rows use canonical actor labels; semantic_positions often use
      w2v underscore tokens. Default semantic_index_style="w2v_token" reindexes
      semantic positions to canonical via helpers.load_w2v_token_to_canonical.
    
    Args:
        semantic_positions: Actor positions on the semantic axis (from actor_proj)
        pca_scores: PCA scores array (actors x components) - RAW scores from pca.fit_transform()
                    or the 2nd return value from helpers.visualize_pca_matrix()
        pca_component: Which PCA component to compare against (0-indexed)
        actor_names: Index of actor names (must match pca_scores rows)
        rescale_pca: If True, rescale PCA scores to [-1, 1] to match semantic axis
                     (matches the scale_to_unit approach in helpers.py)
        semantic_index_style: "w2v_token" (reindex to canonical) or "canonical" (no reindex)
        token_to_canonical: optional precomputed map (underscore token -> canonical name)
        entity_mapping_path: optional path to entities_mapping_2.json
        top_k: Size of the top set for ``top_k_overlap`` (intersection size / k_eff).
        
    Returns:
        Dictionary with metrics:
        - 'correlation': Pearson correlation (can be negative if axes point opposite ways)
        - 'correlation_abs': Absolute Pearson correlation (alignment strength, 0-1)
        - 'aligned_correlation': Pearson after auto-alignment (always non-negative)
        - 'spearman_rho', 'spearman_rho_abs': Spearman rank correlation and |rho|
        - 'spearman_p': two-sided p-value for Spearman
        - 'kendall_tau', 'kendall_tau_abs': Kendall's tau-b and |tau|
        - 'kendall_p': two-sided p-value for Kendall
        - 'top_k_overlap': |top_k_sem ∩ top_k_pca| / k_eff (positional top-k on aligned scores;
          k_eff = min(top_k, n)); computed before PCA sign flip, like Pearson/Spearman/Kendall
        - 'top_k': k_eff used for overlap
        - 'inverse_squared_distance': 1 / sum((sem_pos - pca_pos)^2) after alignment
        - 'mean_absolute_error': Mean absolute difference after alignment
        - 'sum_squared_distance': sum((sem_pos - pca_pos)^2) after alignment
        
        Note: Distance metrics are computed AFTER auto-aligning the axes based on 
        Pearson correlation sign. Spearman, Kendall, and top_k_overlap use the same
        paired scores as Pearson (before PCA flip).
        
    Example:
        >>> # Get PCA results from helpers.py
        >>> pca, actor_scores, feature_names, matrix, fig = visualize_pca_matrix(matrix_tuple)
        >>> # Compare semantic axis to PC1
        >>> metrics = compare_semantic_to_pca(sem_positions, actor_scores, 0, matrix.index)
        >>> print(f"Alignment strength: {metrics['correlation_abs']:.3f}")
        >>> print(f"Direction: {'same' if metrics['correlation'] > 0 else 'opposite'}")
        >>> print(f"Quality (inverse dist): {metrics['inverse_squared_distance']:.3f}")
    """
    h = _helpers()
    sem = semantic_positions.copy()
    if semantic_index_style == "w2v_token":
        t2c = (
            token_to_canonical
            if token_to_canonical is not None
            else h.load_w2v_token_to_canonical(entity_mapping_path)
        )
        sem = h.reindex_series_to_canonical(sem, t2c, drop_unmapped=True)
    elif semantic_index_style != "canonical":
        raise ValueError(
            "semantic_index_style must be 'w2v_token' or 'canonical'"
        )

    # Extract PCA component scores
    pca_positions = pca_scores[:, pca_component]
    
    # Rescale PCA to [-1, 1] to match semantic axis scale
    if rescale_pca:
        max_abs = np.max(np.abs(pca_positions))
        if max_abs > 0:
            pca_positions = pca_positions / max_abs
    
    # Create Series for PCA positions with matching index
    pca_series = pd.Series(pca_positions, index=actor_names, name=f'PC{pca_component+1}')
    
    # Align the two series (in case of any mismatches)
    common_actors = sem.index.intersection(pca_series.index)
    sem_aligned = sem.loc[common_actors]
    pca_aligned = pca_series.loc[common_actors]

    nan_pack = {
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
        "top_k": int(min(max(top_k, 1), max(len(common_actors), 0))),
        "inverse_squared_distance": 0.0,
        "mean_absolute_error": float("nan"),
        "sum_squared_distance": float("nan"),
    }

    if len(common_actors) < 2:
        return nan_pack

    s = np.asarray(sem_aligned.values, dtype=np.float64)
    p = np.asarray(pca_aligned.values, dtype=np.float64)

    # Pearson (same as before)
    correlation = np.corrcoef(s, p)[0, 1]

    # Spearman & Kendall on the same paired scores as Pearson (before PCA flip)
    rho, rho_p = _scipy_stat_pvalue(spearmanr(s, p))
    tau, tau_p = _scipy_stat_pvalue(kendalltau(s, p))

    n = len(s)
    k_eff = int(min(max(top_k, 1), n))
    top_sem = set(np.argsort(s, kind="mergesort")[-k_eff:])
    top_pca = set(np.argsort(p, kind="mergesort")[-k_eff:])
    top_k_overlap = len(top_sem & top_pca) / k_eff if k_eff > 0 else float("nan")

    # AUTO-ALIGN AXES: If correlation is negative, flip the PCA axis
    # This ensures distance metrics make sense regardless of arbitrary axis orientation
    if correlation < 0:
        pca_aligned = -pca_aligned
        aligned_correlation = -correlation
    else:
        aligned_correlation = correlation

    squared_distances = (sem_aligned - pca_aligned) ** 2
    sum_squared_dist = float(squared_distances.sum())

    if sum_squared_dist > 0:
        inverse_squared_distance = 1.0 / sum_squared_dist
    else:
        inverse_squared_distance = np.inf

    mean_absolute_error = float(np.abs(sem_aligned - pca_aligned).mean())

    return {
        "correlation": correlation,
        "correlation_abs": abs(correlation),
        "aligned_correlation": aligned_correlation,
        "spearman_rho": rho,
        "spearman_rho_abs": abs(rho),
        "spearman_p": rho_p,
        "kendall_tau": tau,
        "kendall_tau_abs": abs(tau),
        "kendall_p": tau_p,
        "top_k_overlap": top_k_overlap,
        "top_k": k_eff,
        "inverse_squared_distance": inverse_squared_distance,
        "mean_absolute_error": mean_absolute_error,
        "sum_squared_distance": sum_squared_dist,
    }


def association_matrix(
    semantic_axes: Dict[str, np.ndarray],
    actor_embeddings: pd.DataFrame,
    pca_scores: np.ndarray,
    n_components: int = 5,
    actor_names: Optional[pd.Index] = None,
    semantic_index_style: str = "w2v_token",
    token_to_canonical: Optional[Dict[str, str]] = None,
    entity_mapping_path: Optional[str] = None,
    metric: str = "correlation_abs",
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Build a matrix of association scores between semantic axes (rows) and PCA components (cols).

    Each cell is one scalar from :func:`compare_semantic_to_pca`, selected by ``metric``
    (a key in that function's return dict), e.g. ``correlation_abs``, ``spearman_rho_abs``,
    ``top_k_overlap``, etc.

    COMPATIBILITY WITH helpers.py PCA:
    - Expects RAW pca_scores as returned by visualize_pca_matrix() (2nd return value)

    Args:
        semantic_axes: axis name -> concept vector
        actor_embeddings: rows = actors (same order/labels as PCA)
        pca_scores: (n_actors × n_components) raw PCA scores
        n_components: how many leading PCs to include as columns
        actor_names: row index for PCA (default: actor_embeddings.index)
        semantic_index_style: passed to compare_semantic_to_pca
        token_to_canonical: optional map for reindexing semantic positions
        entity_mapping_path: optional path to entities_mapping_2.json
        metric: which key to read from compare_semantic_to_pca (default: correlation_abs)
        top_k: passed to compare_semantic_to_pca (for top_k_overlap)

    Returns:
        DataFrame (axes × PC1..PCk) of the chosen metric.
    """
    if actor_names is None:
        actor_names = actor_embeddings.index

    axis_names = list(semantic_axes.keys())
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    out = np.zeros((len(axis_names), n_components))

    for i, axis_name in enumerate(axis_names):
        semantic_positions = actor_proj(
            actor_embeddings,
            semantic_axes[axis_name],
            rescale=True,
        )
        for j in range(n_components):
            metrics = compare_semantic_to_pca(
                semantic_positions,
                pca_scores,
                j,
                actor_names,
                rescale_pca=True,
                semantic_index_style=semantic_index_style,
                token_to_canonical=token_to_canonical,
                entity_mapping_path=entity_mapping_path,
                top_k=top_k,
            )
            if metric not in metrics:
                raise KeyError(
                    f"Unknown metric {metric!r}. Valid keys include: {sorted(metrics.keys())}"
                )
            out[i, j] = metrics[metric]

    return pd.DataFrame(out, index=axis_names, columns=pc_names)


def compare_verb_loadings(
    semantic_axis: np.ndarray,
    verb_embeddings: Dict[str, np.ndarray],
    pca: PCA,
    pca_component: int,
    verb_names: list,
    n_top: int = 25,
    n_random: int = 10,
    rescale: bool = True
) -> Dict[str, float]:
    """
    Compare verb positions on a semantic axis to their PCA loadings.
    
    This is heavily adapted from embedding_util.py. Instead of comparing loadings directly,
    it projects the top N verbs from each PCA pole and N random middle verbs onto the
    semantic axis and calculates distances between their positions on the PCA component
    and the projected axis.
    
    COMPATIBILITY WITH helpers.py PCA:
    - Expects the PCA object as returned by visualize_pca_matrix() (1st return value)
    - Extracts RAW loadings from pca.components_[pca_component]
    - PCA was fit on column-standardized data (z-scores) in helpers.py
    - rescale=True rescales both to [-1, 1] for comparison (matching visualization approach)
    - verb_names should be the feature names from PCA (3rd return value from visualize_pca_matrix)
    
    Args:
        semantic_axis: The semantic axis vector (from SemAxis.concept_vector or anch2conceptvec)
        verb_embeddings: Dictionary mapping verbs to embedding vectors
        pca: Fitted PCA object (1st return value from helpers.visualize_pca_matrix)
        pca_component: Which PCA component to compare (0-indexed)
        verb_names: List of verb names corresponding to PCA features
                    (3rd return value from helpers.visualize_pca_matrix)
        n_top: Number of verbs to select from each pole (high/low loadings)
        n_random: Number of verbs to randomly sample from middle range
        rescale: If True, rescale both PCA and semantic positions to comparable scale [-1, 1]
        
    Returns:
        Dictionary with:
        - 'inverse_squared_distance': 1 / sum((pca_loading - sem_projection)^2) after alignment
        - 'correlation': Original correlation (can be negative if axes point opposite ways)
        - 'correlation_abs': Absolute correlation (alignment strength, 0-1)
        - 'aligned_correlation': Correlation after auto-alignment (always positive)
        - 'sum_squared_distance': sum((pca_loading - sem_projection)^2) after alignment
        - 'selected_verbs': List of selected verb names
        - 'pca_loadings': PCA loadings for selected verbs (rescaled if rescale=True)
        - 'semantic_projections': Semantic projections after alignment (rescaled if rescale=True)
        
        Note: Axes are auto-aligned before computing distance metrics to ensure meaningful
        comparison regardless of arbitrary orientation.
        
    Example:
        >>> # Get PCA results from helpers.py
        >>> pca, actor_scores, feature_names, matrix, fig = visualize_pca_matrix(matrix_tuple)
        >>> verb_embds = helpers.embeddings_dict_from_keyed_vectors(w2v_model, feature_names)
        >>> metrics = compare_verb_loadings(axis.concept_vector, verb_embds, pca, 0, feature_names)
        >>> print(f"Correlation: {metrics['correlation']:.3f}")
    """
    # Get PCA loadings for the specified component
    pca_loadings = pca.components_[pca_component]
    
    # Rescale PCA loadings to [-1, 1] if requested
    if rescale:
        max_abs = np.max(np.abs(pca_loadings))
        if max_abs > 0:
            pca_loadings_scaled = pca_loadings / max_abs
        else:
            pca_loadings_scaled = pca_loadings
    else:
        pca_loadings_scaled = pca_loadings
    
    # Sort verbs by their PCA loadings
    sorted_indices = np.argsort(pca_loadings)
    
    # Select verbs: top N from each pole + N random from middle
    top_negative = sorted_indices[:n_top]  # Lowest loadings
    top_positive = sorted_indices[-n_top:]  # Highest loadings
    
    # Select random verbs from middle range
    middle_start = n_top
    middle_end = len(sorted_indices) - n_top
    if middle_end > middle_start and n_random > 0:
        middle_range = sorted_indices[middle_start:middle_end]
        n_random_actual = min(n_random, len(middle_range))
        random_middle = np.random.choice(middle_range, size=n_random_actual, replace=False)
    else:
        random_middle = np.array([])
    
    # Combine selected indices
    selected_indices = np.concatenate([top_negative, random_middle, top_positive])
    selected_verbs = [verb_names[i] for i in selected_indices]
    
    # Filter to verbs that have embeddings
    available_verbs = [v for v in selected_verbs if v in verb_embeddings]
    available_indices = [i for i, v in zip(selected_indices, selected_verbs) 
                        if v in verb_embeddings]
    
    if len(available_verbs) == 0:
        raise ValueError("No selected verbs have embeddings!")
    
    # Get PCA loadings for available verbs
    pca_loads_selected = pca_loadings_scaled[available_indices]
    
    # Normalize semantic axis
    sem_axis_norm = semantic_axis / np.linalg.norm(semantic_axis)
    
    # Project verbs onto semantic axis
    semantic_projections = []
    for verb in available_verbs:
        # Verb embedding (already normalized)
        verb_emb = verb_embeddings[verb]
        # Project onto semantic axis
        projection = np.dot(verb_emb, sem_axis_norm)
        semantic_projections.append(projection)
    
    semantic_projections = np.array(semantic_projections)
    
    # Rescale semantic projections to [-1, 1] if requested
    if rescale:
        max_abs = np.max(np.abs(semantic_projections))
        if max_abs > 0:
            semantic_projections = semantic_projections / max_abs
    
    # Calculate correlation first
    correlation = np.corrcoef(pca_loads_selected, semantic_projections)[0, 1]
    
    # AUTO-ALIGN: If correlation is negative, flip semantic projections
    # This ensures distance metrics work regardless of arbitrary axis orientation
    if correlation < 0:
        semantic_projections = -semantic_projections
        aligned_correlation = -correlation
    else:
        aligned_correlation = correlation
    
    # Calculate distance metrics on aligned axes
    squared_distances = (pca_loads_selected - semantic_projections) ** 2
    sum_squared_dist = squared_distances.sum()
    
    # Calculate inverse (avoid division by zero)
    if sum_squared_dist > 0:
        inverse_squared_distance = 1.0 / sum_squared_dist
    else:
        inverse_squared_distance = np.inf
    
    return {
        'inverse_squared_distance': inverse_squared_distance,
        'correlation': correlation,  # Original correlation (can be negative)
        'correlation_abs': abs(correlation),  # Alignment strength
        'aligned_correlation': aligned_correlation,  # Always positive
        'sum_squared_distance': sum_squared_dist,
        'selected_verbs': available_verbs,
        'pca_loadings': pca_loads_selected,
        'semantic_projections': semantic_projections  # After alignment
    }


def visualize_projection(
    actor_positions: pd.Series,
    axis_name: str = "Semantic Axis",
    categories: pd.Series = None,
    figsize: Tuple[int, int] = (1000, 600),
    semantic_axis = None
) -> go.Figure:
    """
    Visualize the projection of actors onto a semantic axis using plotly.
    
    Creates an interactive scatter plot where actors are positioned along a semantic
    axis, with hover labels showing actor names and their position values. Actors are
    sorted by category on the y-axis.
    
    Args:
        actor_positions: Series with actor positions (index=actor names, values=positions)
        axis_name: Name of the semantic axis for plot title
        categories: Optional Series mapping actors to categories for color coding
        figsize: Tuple of (width, height) for the figure in pixels
        semantic_axis: Optional SemAxis object. If provided, shows the top 3 antonym pairs
                      with highest parallelism on the plot to indicate axis definition
        
    Returns:
        Plotly Figure object
        
    Example:
        >>> fig = visualize_projection(positions, "Transparency Axis", categories, 
        ...                           semantic_axis=transparency_axis)
        >>> fig.show()
    """
    # Sort actors by category first, then by position within category
    if categories is not None:
        # Create a DataFrame for easier sorting
        plot_df = pd.DataFrame({
            'position': actor_positions,
            'category': categories.reindex(actor_positions.index)
        })
        # Sort by category, then by position
        plot_df = plot_df.sort_values(['category', 'position'])
        sorted_positions = plot_df['position']
        sorted_categories = plot_df['category']
    else:
        # Sort by position only
        sorted_positions = actor_positions.sort_values()
        sorted_categories = None
    
    # Create figure
    fig = go.Figure()
    
    # If categories provided, color by category
    if categories is not None:
        # Track y-position
        current_y = 0
        
        # Plot each category separately
        for category in sorted_categories.unique():
            mask = sorted_categories == category
            positions_cat = sorted_positions[mask]
            
            # Create y-values (continuous across categories)
            y_values = np.arange(current_y, current_y + len(positions_cat))
            current_y += len(positions_cat)
            
            fig.add_trace(go.Scatter(
                x=positions_cat.values,
                y=y_values,
                mode='markers+text',
                name=str(category),
                text=positions_cat.index,
                textposition='middle right',
                textfont=dict(size=10),
                marker=dict(size=10),
                hovertemplate='<b>%{text}</b><br>' +
                             f'{axis_name}: %{{x:.3f}}<br>' +
                             f'Category: {category}<br>' +
                             '<extra></extra>'
            ))
    else:
        # Plot all actors in one trace
        y_values = np.arange(len(sorted_positions))
        
        fig.add_trace(go.Scatter(
            x=sorted_positions.values,
            y=y_values,
            mode='markers+text',
            name='Actors',
            text=sorted_positions.index,
            textposition='middle right',
            textfont=dict(size=10),
            marker=dict(size=10, color='blue'),
            hovertemplate='<b>%{text}</b><br>' +
                         f'{axis_name}: %{{x:.3f}}<br>' +
                         '<extra></extra>'
        ))
    
    # Add antonym pairs if semantic_axis is provided
    if semantic_axis is not None and hasattr(semantic_axis, 'pair_scores'):
        # Get top 3 pairs by parallelism
        if semantic_axis.pair_scores is not None:
            top_pairs = semantic_axis.get_best_pairs(n=3)
            
            # Create annotation text
            positive_words = []
            negative_words = []
            for (pos_word, neg_word), score in top_pairs:
                positive_words.append(pos_word)
                negative_words.append(neg_word)
            
            # Add annotations at the extremes of the x-axis
            # Positive pole (right side)
            fig.add_annotation(
                x=1.0,
                y=1.05,
                xref='x',
                yref='paper',
                text='<b>Positive pole:</b><br>' + '<br>'.join(positive_words),
                showarrow=False,
                font=dict(size=11, color='darkgreen'),
                align='right',
                xanchor='right',
                bgcolor='rgba(200, 255, 200, 0.8)',
                bordercolor='darkgreen',
                borderwidth=1,
                borderpad=4
            )
            
            # Negative pole (left side)
            fig.add_annotation(
                x=-1.0,
                y=1.05,
                xref='x',
                yref='paper',
                text='<b>Negative pole:</b><br>' + '<br>'.join(negative_words),
                showarrow=False,
                font=dict(size=11, color='darkred'),
                align='left',
                xanchor='left',
                bgcolor='rgba(255, 200, 200, 0.8)',
                bordercolor='darkred',
                borderwidth=1,
                borderpad=4
            )
    
    # Update layout
    y_axis_title = "Actors (sorted by category)" if categories is not None else "Actors (sorted by position)"
    
    fig.update_layout(
        title=dict(
            text=f"Actor Projections on {axis_name}",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=f"Position on {axis_name}",
        yaxis_title=y_axis_title,
        template='plotly_white',
        showlegend=True if categories is not None else False,
        height=figsize[1],
        width=figsize[0],
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        hovermode='closest'
    )
    
    return fig

