import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objects as go_fig
import plotly.colors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Base directory for external files (same folder as this module)
_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))

# Trained Word2Vec runs: w2v/analysis/modules -> parents[2] == w2v/
_W2V_ROOT = Path(__file__).resolve().parents[2]
W2V_MODELS_DIR = _W2V_ROOT / "models"

# Repo root: w2v/analysis/modules -> parents[3] == ideological_resonance_thesis
_REPO_ROOT = Path(__file__).resolve().parents[3]
ENTITY_MAPPING_2_PATH = _REPO_ROOT / "pca" / "modules" / "entities" / "entities_mapping_2.json"


def load_w2v_token_to_canonical(path=None):
    """
    Flatten pca/modules/entities/entities_mapping_2.json: underscore entity key -> canonical_name.

    Structure: category -> { entity_key -> { canonical_name, variations } }.
    """
    p = Path(path) if path is not None else ENTITY_MAPPING_2_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Entity mapping not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for cat_val in data.values():
        if not isinstance(cat_val, dict):
            continue
        for entity_key, entity_obj in cat_val.items():
            if not isinstance(entity_obj, dict):
                continue
            if "canonical_name" not in entity_obj:
                continue
            cn = entity_obj["canonical_name"]
            if entity_key in out and out[entity_key] != cn:
                warnings.warn(
                    f"Duplicate w2v entity key {entity_key!r} with differing "
                    f"canonical_name; keeping last ({cn!r}).",
                    UserWarning,
                    stacklevel=2,
                )
            out[entity_key] = cn
    return out


def load_canonical_to_w2v_token(path=None):
    """
    Inverse map: canonical_name -> underscore w2v token (first token wins per canonical).
    """
    forward = load_w2v_token_to_canonical(path)
    inv = {}
    for token, canonical in forward.items():
        if canonical not in inv:
            inv[canonical] = token
        elif inv[canonical] != token:
            warnings.warn(
                f"Multiple w2v tokens for canonical {canonical!r}: "
                f"{inv[canonical]!r} and {token!r}; keeping first.",
                UserWarning,
                stacklevel=2,
            )
    return inv


def reindex_series_to_canonical(series, token_to_canonical, drop_unmapped=True):
    """
    Map a Series indexed by w2v underscore tokens to canonical actor names.
    If duplicate canonicals appear, the last token's value wins.
    """
    mapped = {}
    for idx, val in series.items():
        c = token_to_canonical.get(idx)
        if c is None:
            if not drop_unmapped:
                mapped[idx] = val
            continue
        mapped[c] = val
    return pd.Series(mapped, name=series.name)


def reindex_series_to_w2v_token(series, canonical_to_token, drop_unmapped=True):
    """
    Map a Series indexed by canonical names to w2v underscore tokens.
    """
    mapped = {}
    for idx, val in series.items():
        t = canonical_to_token.get(idx)
        if t is None:
            if not drop_unmapped:
                mapped[idx] = val
            continue
        mapped[t] = val
    return pd.Series(mapped, name=series.name)


def load_trained_w2v_keyed_vectors(model_name, models_dir=None):
    """
    Load vectors for a trained run saved by train_w2v_cpu.save_model under
    ``(models_dir or W2V_MODELS_DIR) / model_name /``.

    Tries ``{model_name}.bin``, ``{model_name}.vec``, then full gensim ``{model_name}.model``.
    """
    from gensim.models import KeyedVectors, Word2Vec

    root = Path(models_dir) if models_dir is not None else W2V_MODELS_DIR
    run_dir = root.resolve() / model_name
    stem = run_dir.name
    bin_p = run_dir / f"{stem}.bin"
    vec_p = run_dir / f"{stem}.vec"
    model_p = run_dir / f"{stem}.model"
    tried = [str(bin_p), str(vec_p), str(model_p)]

    if bin_p.is_file():
        return KeyedVectors.load_word2vec_format(str(bin_p), binary=True)
    if vec_p.is_file():
        return KeyedVectors.load_word2vec_format(str(vec_p), binary=False)
    if model_p.is_file():
        return Word2Vec.load(str(model_p)).wv

    raise FileNotFoundError(
        "No Word2Vec artifacts found. Tried:\n  " + "\n  ".join(tried)
    )


def embeddings_dict_from_keyed_vectors(kv, keys, normalize=True):
    """
    Build {key: np.ndarray} for keys present in KeyedVectors ``kv``.
    Optionally L2-normalize each vector (recommended before projections).
    """
    out = {}
    for k in keys:
        if k not in kv:
            continue
        v = np.asarray(kv[k], dtype=float).copy()
        if normalize:
            n = np.linalg.norm(v)
            if n > 0:
                v /= n
        out[k] = v
    return out


def filter_top_actions(df, actors=None, actions=None):
    """
    Filter dataframe to keep only rows with actors and/or actions in specified frequency ranges.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'Entity' and 'action' columns
        actors (tuple): (mode, lb, ub) for filtering actors, where:
            mode (str): Either 'n' for top N actors or 'pct' for percentage range
            lb (int/float): Lower bound (inclusive) - either N or percentage
            ub (int/float): Upper bound (exclusive) - either N or percentage
        actions (tuple): (mode, lb, ub) for filtering actions, same format as actors
        
    Returns:
        pd.DataFrame: Filtered dataframe containing only rows matching specified ranges
    """
    filtered_df = df.copy()
    
    # Helper function to get filtered items based on mode and bounds
    def get_filtered_items(counts, mode, lb, ub):
        total = len(counts)
        
        if mode not in ['n', 'pct']:
            raise ValueError("Mode must be either 'n' or 'pct'")
            
        if lb >= ub:
            raise ValueError("Lower bound must be less than upper bound")
            
        if mode == 'n':
            if lb < 0 or ub > total:
                raise ValueError(f"Bounds must be between 0 and {total}")
            return counts.iloc[lb:ub].index
        else:
            if lb < 0 or ub > 100:
                raise ValueError("Percentage bounds must be between 0 and 100")
            n_lb = int(total * lb / 100)
            n_ub = int(total * ub / 100)
            return counts.iloc[n_lb:n_ub].index
    
    # Filter by actors if specified
    if actors is not None:
        mode, lb, ub = actors
        actor_counts = df['canonical'].value_counts()
        top_actors = get_filtered_items(actor_counts, mode, lb, ub)
        filtered_df = filtered_df[filtered_df['canonical'].isin(top_actors)]
        print(f"Filtered to {len(top_actors)} actors ({mode} mode, {lb}-{ub})")
    
    # Filter by actions if specified
    if actions is not None:
        mode, lb, ub = actions
        action_counts = df['action'].value_counts()
        top_actions = get_filtered_items(action_counts, mode, lb, ub)
        filtered_df = filtered_df[filtered_df['action'].isin(top_actions)]
        print(f"Filtered to {len(top_actions)} actions ({mode} mode, {lb}-{ub})")
        
    return filtered_df


def create_actor_action_matrix(
    df,
    columns_tuple,
    filter_mask=None,
    binarize=False,
    normalize='none',
    category_actors=False,
    category_verbs=False,
):
    """
    Create a matrix where rows are actors and columns are actions.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_tuple (tuple): (action_column, actor_column, category_column) names.
                             category_column is optional - if provided, will be used
                             for coloring in visualization
        filter_mask (pd.Series, optional): Boolean mask for filtering df
        binarize (bool, optional): If True, use binary values (0/1) to indicate if an actor
                                 performed an action at least once.
        normalize (str, optional): Normalization method to use. One of:
                                'none': No normalization (raw counts)
                                'rows': Normalize by actor (each row sums to 1)
                                'columns': Normalize by action (each column sums to 1)
                                'both': Normalize both rows and columns (geometric mean)
        category_actors (bool, optional): If True, build rows by actor category instead of 
                                individual actors. Defaults to False.
        category_verbs (bool, optional): If True, build columns by verb category instead of
                                individual verbs. Defaults to False.
        
    Returns:
        tuple: (matrix, categories) where:
               - matrix is pd.DataFrame of actor-action relationships
               - categories is pd.Series mapping actors to their categories (only returned when
                 category_actors is False and a category_column was provided)
    """
    if binarize and normalize != 'none':
        raise ValueError("Cannot use binarize=True with normalization as this would eliminate all meaningful differences")
    
    if normalize not in ['none', 'rows', 'columns', 'both']:
        raise ValueError("normalize must be one of: 'none', 'rows', 'columns', 'both'")
    
    # Apply filter if provided
    if filter_mask is not None:
        df = df[filter_mask].copy()
    
    # Extract column names
    if len(columns_tuple) == 3:
        action_col, actor_col, category_col = columns_tuple
    else:
        action_col, actor_col = columns_tuple
        category_col = None

    # Load verb mappings if needed
    if category_verbs:
        import json
        with open(os.path.join(_MODULES_DIR, 'translations', 'actions_mapping.json'), 'r', encoding='utf-8') as f:
            verb_map = json.load(f)
            
        # Create mappings
        verb_to_category = {}
        verb_to_canonical = {}
        for category, verbs in verb_map.items():
            for verb_key, verb_data in verbs.items():
                canonical = verb_data['canonical_name']
                # Map canonical name
                verb_to_category[canonical.lower()] = category
                verb_to_canonical[canonical.lower()] = canonical
                # Map variations
                for variation in verb_data['variations']:
                    verb_to_category[variation.lower()] = category
                    verb_to_canonical[variation.lower()] = canonical
        
        # Map verbs to their canonical forms
        df = df.copy()
        df[action_col] = df[action_col].str.lower().map(verb_to_canonical).fillna(df[action_col])

    # Decide what our row and column identifiers should be
    if category_actors:
        if category_col is None:
            raise ValueError("category_actors=True requires a category_column in columns_tuple")
        row_id_col = category_col  # build rows by category
        categories = None          # already aggregated; no separate colouring information
    else:
        row_id_col = actor_col
        # If a category column exists, keep it for colouring later
        categories = (
            df.groupby(actor_col)[category_col].first() if category_col is not None else None
        )
    
    # Create pivot table
    if binarize:
        matrix = pd.crosstab(
            df[row_id_col], 
            df[action_col],
            normalize=False
        )
        matrix = (matrix > 0).astype(int)  # Convert to binary 0/1
    else:
        matrix = pd.crosstab(
            df[row_id_col],
            df[action_col],
        )
        
        # Apply normalization if requested
        if normalize == 'rows':
            matrix = matrix.div(matrix.sum(axis=1), axis=0)
        elif normalize == 'columns':
            matrix = matrix.div(matrix.sum(axis=0), axis=1)
        elif normalize == 'both':
            # Calculate both normalizations and take geometric mean
            row_norms = matrix.div(matrix.sum(axis=1), axis=0)
            col_norms = matrix.div(matrix.sum(axis=0), axis=1)
            matrix = np.sqrt(row_norms * col_norms)
    
    # If using verb categories, aggregate columns by category
    if category_verbs:
        # Create mapping series for verbs to categories
        verb_categories = pd.Series(verb_to_category)
        # Group columns by category and sum
        matrix = matrix.groupby(verb_categories, axis=1).sum()
    
    # If rows represent categories, ensure we return a mapping so downstream plotting can
    # still colour-code them. Use an identity mapping (category → category).
    if category_actors:
        categories = pd.Series(matrix.index, index=matrix.index, name="category")

    return matrix, categories


def visualize_pca_matrix(matrix_tuple, n_components=2, actors=True, actions=False, scale_to_unit=True, title='', translate=None, logscale=False):
    """
    Create a *single* PCA biplot for an actor-action matrix following the methodological
    steps described in the reference paper (see ``pca_descr.txt``):

    1. Centre & standardise **each action column** (relative frequencies → z-scores).  This
       is equivalent to performing PCA on the correlation matrix of actions, mirroring the
       *"centered and standardised relative frequencies"* in the paper.
    2. Fit **one** PCA on the standardised matrix.  Actor coordinates are the usual
       component scores; action coordinates are the component *loadings* (eigenvectors).
    3. Rescale both sets of coordinates to the range ``[-1, 1]`` (default) so that actors
       and actions share the same frame – the paper rescales by the maximum absolute score
       on each axis which leads to axes spanning roughly one unit in either direction.
    4. Optionally apply log scaling to the coordinates while maintaining the [-1, 1] range,
       with 0 as the center point.

    Parameters
    ----------
    matrix_tuple : tuple | pd.DataFrame
        Either the ``(matrix, categories)`` tuple returned by
        ``create_actor_action_matrix`` **or** the ``pd.DataFrame`` directly.
        The dataframe must contain actors on rows and actions on columns.
    n_components : int, default 2
        Number of principal components to compute.
    actors : bool, default True
        Whether to plot actor points.
    actions : bool, default False
        Whether to plot action points.
    scale_to_unit : bool, default True
        Whether to rescale coordinates to [-1, 1] range.
    title : str, default ''
        Optional title for the plot.
    translate : str | None, default None
        Language to translate verbs from. Options: None (no translation), 'de' (German), 'it' (Italian).
        Translations are loaded from translations_{lang}.json files.
    logscale : bool, default False
        Whether to apply log scaling to the coordinates while maintaining the [-1, 1] range.
        The log scale is applied from the center (0) outwards in both positive and negative directions.

    Returns
    -------
    tuple: (pca, actor_scores, feature_names, matrix, fig) where:
        - pca: Fitted PCA object
        - actor_scores: PCA scores for actors
        - feature_names: List of feature names
        - matrix: Original matrix
        - fig: Plotly figure object (only when n_components=2, else None)
    """
    # ------------------------------------------------------------------
    config = {
    'toImageButtonOptions': {
    'format': 'svg', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

    # Guards & unpacking
    if not actors and not actions:
        raise ValueError("At least one of actors or actions must be True")

    if isinstance(matrix_tuple, tuple):
        matrix, categories = matrix_tuple
    else:
        matrix = matrix_tuple
        categories = None

    import numpy as np
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import json
    import pandas as pd

    # Load translations if requested
    translations = {}
    translation_table = []  # Store translations for export
    if translate in ['de', 'it']:
        # First load the JSON translations
        try:
            with open(os.path.join(_MODULES_DIR, 'translations', f'translations_{translate}.json'), 'r', encoding='utf-8') as f:
                translations = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load translations for {translate}: {e}")
            translations = {}
            
        # Then try to load existing Excel translations file
        excel_path = os.path.join(_MODULES_DIR, 'translations', f'translations_{translate}_2.xlsx')
        try:
            existing_df = pd.read_excel(excel_path)
            # Convert existing translations to list of dicts for consistency
            translation_table = existing_df.to_dict('records')
        except Exception as e:
            print(f"Note: No existing translations Excel file found or could not read it: {e}")
            translation_table = []

    # ------------------------------------------------------------------
    # 1) Column-wise centring & scaling (z-scores)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(matrix.values)

    # ------------------------------------------------------------------
    # 2) PCA on the standardised matrix
    n_components = min(n_components, min(X_std.shape))
    pca = PCA(n_components=n_components)
    actor_scores = pca.fit_transform(X_std)        # shape: n_actors × k
    action_loadings = pca.components_.T            # shape: n_actions × k

    def _create_log_scale_params():
        """Create parameters for log-scaled axes."""
        # Create tick positions for positive and negative sides
        pos_ticks = np.array([0.001, 0.01, 0.1, 0.5, 1.0])
        neg_ticks = -pos_ticks[::-1]
        
        # Add 0 and combine
        tick_positions = np.concatenate([neg_ticks, [0], pos_ticks])
        
        # Create tick labels
        tick_labels = []
        for x in tick_positions:
            if abs(x) < 0.01 and x != 0:
                tick_labels.append(f"{x:.0e}")
            elif x == 0:
                tick_labels.append("0")
            else:
                tick_labels.append(f"{x:.2f}")
        
        return tick_positions, tick_labels

    def _symlog_transform(x):
        """
        Apply symmetric log transform to values while preserving -1 to 1 range.
        Uses a modified log transform that gives more space to larger values.
        """
        # Convert input to numpy array if it isn't already
        is_scalar = np.isscalar(x)
        x_arr = np.array([x]) if is_scalar else np.array(x)
        
        # Parameters
        linthresh = 0.05    # Smaller linear threshold to start log scaling even earlier
        linscale = 0.3     # More space for linear region to spread out central points
        
        # Initialize output array
        out = np.zeros_like(x_arr, dtype=float)
        
        # Split into positive, negative, and zero regions
        pos_mask = x_arr > linthresh
        neg_mask = x_arr < -linthresh
        lin_mask = ~(pos_mask | neg_mask)
        
        # Linear region (small values)
        out[lin_mask] = x_arr[lin_mask] * linscale / linthresh
        
        # Log region (positive values)
        if np.any(pos_mask):
            # Map [linthresh, 1] to [linscale, 1] logarithmically
            log_base = np.log(1/linthresh)
            out[pos_mask] = linscale + (1 - linscale) * np.log(x_arr[pos_mask]/linthresh) / log_base
            
        # Log region (negative values)
        if np.any(neg_mask):
            # Mirror the positive transformation
            log_base = np.log(1/linthresh)
            out[neg_mask] = -(linscale + (1 - linscale) * np.log(-x_arr[neg_mask]/linthresh) / log_base)
        
        return out[0] if is_scalar else out

    # ------------------------------------------------------------------
    # 3) Optional rescaling to [-1, 1] (facilitates plotting)
    #    Separate rescaling for actors and actions so each block spans ±1 independently.
    if scale_to_unit and n_components == 2:
        # Rescale actors
        max_actor = np.max(np.abs(actor_scores), axis=0)
        max_actor[max_actor == 0] = 1  # avoid division by zero
        actor_coords = actor_scores / max_actor

        # Rescale actions independently
        max_action = np.max(np.abs(action_loadings), axis=0)
        max_action[max_action == 0] = 1
        action_coords = action_loadings / max_action
    else:
        actor_coords = actor_scores
        action_coords = action_loadings

    # 4) Optional log scaling while maintaining -1 to 1 range
    if logscale and n_components == 2:
        # Transform data points using log scale
        def transform_coords(coords):
            pos_mask = coords > 0
            neg_mask = coords < 0
            transformed = np.zeros_like(coords)
            transformed[pos_mask] = np.exp(np.log(2) * coords[pos_mask]) - 1
            transformed[neg_mask] = -(np.exp(np.log(2) * -coords[neg_mask]) - 1)
            return transformed

        # Transform all coordinates
        actor_coords = transform_coords(actor_coords)
        action_coords = transform_coords(action_coords)

    # Helper: which labels to show (top & bottom 25 % by frequency)
    def _select_actor_labels(coords):
        """
        Select actors to label based on their coordinates:
        - 10 highest magnitude loadings at each pole of each axis
        - Random 20% sample from the middle range
        """
        n_poles = 10  # Number of extreme actors to label at each pole
        middle_pct = 0.20  # Percentage of middle-range actors to randomly label
        
        selected = set()
        for axis in range(2):  # PC1 and PC2
            # Sort indices by absolute coordinates
            axis_coords = coords[:, axis]
            sorted_idx = np.argsort(axis_coords)
            
            # Add extreme ends (positive and negative)
            selected.update(sorted_idx[-n_poles:])  # Most positive
            selected.update(sorted_idx[:n_poles])   # Most negative
            
            # Add random selection from middle range
            middle_start = n_poles
            middle_end = len(sorted_idx) - n_poles
            if middle_end > middle_start:  # Only if we have middle range
                middle_idx = sorted_idx[middle_start:middle_end]
                n_middle = int(len(middle_idx) * middle_pct)
                if n_middle > 0:  # Only if we want at least one middle actor
                    middle_selected = np.random.choice(middle_idx, size=n_middle, replace=False)
                    selected.update(middle_selected)
        
        return selected

    def _select_action_labels(loadings):
        """
        Select actions to show based on their loadings:
        - 15 highest and 15 lowest on each axis
        - 10 random from middle range on each axis
        """
        n_extreme = 20  # Number of extreme words per pole
        n_middle = 20   # Number of random middle-range words
        
        selected = set()
        for axis in range(2):  # PC1 and PC2
            # Sort indices by absolute loadings
            axis_loadings = loadings[:, axis]
            sorted_idx = np.argsort(axis_loadings)
            
            # Add extreme ends (positive and negative)
            selected.update(sorted_idx[-n_extreme:])  # Most positive
            selected.update(sorted_idx[:n_extreme])   # Most negative
            
            # Add random selection from middle range
            middle_start = n_extreme
            middle_end = len(sorted_idx) - n_extreme
            if middle_end > middle_start:  # Only if we have middle range
                middle_idx = sorted_idx[middle_start:middle_end]
                if len(middle_idx) > n_middle:
                    middle_selected = np.random.choice(middle_idx, size=n_middle, replace=False)
                    selected.update(middle_selected)
        
        return selected

    actor_lbl_idx = _select_actor_labels(actor_coords)
    action_lbl_idx = _select_action_labels(action_loadings) if actions else set()

    # ------------------------------------------------------------------
    # 4) Plot (only when 2 components were requested)
    fig = None
    if n_components == 2:
        fig = go.Figure()

        # ---- Actors ----
        if actors:
            if categories is not None:
                # Calculate number of unique categories for legend grouping
                n_cats = len(categories.unique())
                
                for cat in categories.unique():
                    mask = categories == cat
                    # When using category_actors=True, always show all labels
                    if matrix.index.equals(categories.index):  # category_actors=True
                        text = matrix.index[mask]  # Show all category labels
                    else:
                        # Map global indices to category-local indices for label selection
                        cat_indices = np.where(mask)[0]
                        text = []
                        for i, lbl in enumerate(matrix.index[mask]):
                            global_idx = cat_indices[i]  # Convert local index to global
                            text.append(lbl if global_idx in actor_lbl_idx else "")
                    fig.add_trace(
                        go.Scatter(
                            x=actor_coords[mask, 0],
                            y=actor_coords[mask, 1],
                            mode="markers+text",
                            text=text,
                            name=str(cat),
                            hovertext=matrix.index[mask],
                            textposition="bottom center",
                        )
                    )
            else:
                # Select actors to label using the new function
                actor_lbl_idx = _select_actor_labels(actor_coords)
                text = [lbl if i in actor_lbl_idx else "" for i, lbl in enumerate(matrix.index)]
                fig.add_trace(
                    go.Scatter(
                        x=actor_coords[:, 0],
                        y=actor_coords[:, 1],
                        mode="markers+text",
                        text=text,
                        name="Actors",
                        hovertext=matrix.index,
                        textposition="bottom center",
                    )
                )

        # ---- Actions ----
        if actions:
            # First get the indices of actions to display
            action_lbl_idx = _select_action_labels(action_loadings)
            
            # Translate action labels if requested
            if translations:
                # Create display and hover labels with translations
                display_labels = []
                hover_labels = []
                for i, verb in enumerate(matrix.columns):
                    # Try to find translation, use original if not found
                    translation = translations.get(verb.lower(), verb)
                    display_labels.append(translation if translation != verb else verb)
                    hover_labels.append(f"{verb} → {translation}" if translation != verb else verb)
                    
                    # Only store translations for verbs that will be displayed
                    if i in action_lbl_idx and translation != verb:
                        translation_table.append({
                            translate.upper(): verb,  # 'DE' or 'IT'
                            'Translation': translation
                        })
            else:
                display_labels = matrix.columns
                hover_labels = matrix.columns

            # Create text array with selected labels only
            text = [lbl if i in action_lbl_idx else "" for i, lbl in enumerate(display_labels)]

            # Only plot text labels at the action positions, no markers/dots
            fig.add_trace(
                go.Scatter(
                    x=action_coords[:, 0],
                    y=action_coords[:, 1],
                    mode="text",
                    text=text,
                    name="Actions",
                    # marker is omitted so no dots are shown
                    hovertext=hover_labels,
                    textposition="bottom center",
                    showlegend=True,
                )
            )

        # ---- Layout ----
        ev = pca.explained_variance_ratio_ * 100
        x_title = f"PC1 — {ev[0]:.1f}% variance"
        y_title = f"PC2 — {ev[1]:.1f}% variance"

        if logscale and scale_to_unit:
            # Store original coordinates for hover text
            orig_actor_coords = actor_coords.copy() if actors else None
            orig_action_coords = action_coords.copy() if actions else None
            
            # Apply symlog transform to coordinates
            if actors:
                actor_coords = np.column_stack([
                    _symlog_transform(actor_coords[:, 0]),
                    _symlog_transform(actor_coords[:, 1])
                ])
            if actions:
                action_coords = np.column_stack([
                    _symlog_transform(action_coords[:, 0]),
                    _symlog_transform(action_coords[:, 1])
                ])

            # Create tick positions with more detail in the critical regions
            tick_positions = np.array([-1.0, -0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5, 1.0])
            tick_positions_transformed = _symlog_transform(tick_positions)
            
            # Create tick labels (showing original values)
            tick_labels = []
            for x in tick_positions:
                if x == 0:
                    tick_labels.append("0")
                elif abs(x) < 0.01:
                    tick_labels.append(f"{x:.3f}")
                else:
                    tick_labels.append(f"{x:.2f}")
            
            # Create figure with transformed coordinates
            fig = go.Figure()

            # Plot actors with transformed coordinates
            if actors:
                if categories is not None:
                    for cat in categories.unique():
                        mask = categories == cat
                        # When using category_actors=True, always show all labels
                        if matrix.index.equals(categories.index):  # category_actors=True
                            text = matrix.index[mask]  # Show all category labels
                        else:
                            # Map global indices to category-local indices for label selection
                            cat_indices = np.where(mask)[0]
                            text = []
                            for i, lbl in enumerate(matrix.index[mask]):
                                global_idx = cat_indices[i]  # Convert local index to global
                                text.append(lbl if global_idx in actor_lbl_idx else "")
                        
                        # Add trace with transformed coordinates but original hover text
                        hover_text = [f"Original: ({x:.3f}, {y:.3f})" for x, y in orig_actor_coords[mask]]
                        fig.add_trace(
                            go.Scatter(
                                x=actor_coords[mask, 0],
                                y=actor_coords[mask, 1],
                                mode="markers+text",
                                text=text,
                                name=str(cat),
                                hovertext=hover_text,
                                textposition="bottom center",
                            )
                        )
                else:
                    # Select actors to label using the helper function
                    text = [lbl if i in actor_lbl_idx else "" for i, lbl in enumerate(matrix.index)]
                    hover_text = [f"Original: ({x:.3f}, {y:.3f})" for x, y in orig_actor_coords]
                    fig.add_trace(
                        go.Scatter(
                            x=actor_coords[:, 0],
                            y=actor_coords[:, 1],
                            mode="markers+text",
                            text=text,
                            name="Actors",
                            hovertext=hover_text,
                            textposition="bottom center",
                        )
                    )

            # Plot actions with transformed coordinates
            if actions:
                # First get the indices of actions to display
                action_lbl_idx = _select_action_labels(action_loadings)
                
                # Handle translations if requested
                if translations:
                    display_labels = []
                    hover_labels = []
                    for i, verb in enumerate(matrix.columns):
                        translation = translations.get(verb.lower(), verb)
                        display_labels.append(translation if translation != verb else verb)
                        hover_labels.append(f"{verb} → {translation}" if translation != verb else verb)
                        
                        # Only store translations for verbs that will be displayed
                        if i in action_lbl_idx and translation != verb:
                            translation_table.append({
                                translate.upper(): verb,  # 'DE' or 'IT'
                                'Translation': translation
                            })
                else:
                    display_labels = matrix.columns
                    hover_labels = matrix.columns

                # Create text array with selected labels only
                text = [lbl if i in action_lbl_idx else "" for i, lbl in enumerate(display_labels)]
                hover_text = [f"Original: ({x:.3f}, {y:.3f})" for x, y in orig_action_coords]

                fig.add_trace(
                    go.Scatter(
                        x=action_coords[:, 0],
                        y=action_coords[:, 1],
                        mode="text",
                        text=text,
                        name="Actions",
                        hovertext=hover_text,
                        textposition="bottom center",
                        showlegend=True,
                    )
                )
            
            # Update layout with symlog-scaled axes
            axis_common = dict(
                tickmode='array',
                tickvals=tick_positions_transformed,
                ticktext=tick_labels,
                range=[-1.07, 1.07],
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor='black',
                linewidth=2,
                mirror=True,        # draws lines on all 4 sides → black box frame
            )
            
            fig.update_layout(
                title=title,
                template="plotly_white",
                xaxis=dict(**axis_common, title=dict(text=x_title)),
                yaxis=dict(**axis_common, title=dict(text=y_title)),
                hovermode="closest",
                legend=dict(
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            # Dashed zero lines (x- and y-axes through the origin) for actions-only plots
            if not actors and actions:
                r = 1.07
                for shape_kwargs in [
                    dict(x0=-r, x1=r, y0=0, y1=0),   # horizontal zero line
                    dict(x0=0,  x1=0, y0=-r, y1=r),   # vertical zero line
                ]:
                    fig.add_shape(
                        type='line',
                        line=dict(color='black', dash='dash', width=1),
                        **shape_kwargs,
                    )
        else:
            # Regular linear scale layout
            if not actors and actions:
                # For actions-only plots, use black border frame style
                tick_positions_raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                tick_labels = ["-1.00", "-0.50", "0", "0.50", "1.00"]
                axis_common = dict(
                    tickmode='array',
                    tickvals=tick_positions_raw,
                    ticktext=tick_labels,
                    range=[-1.07, 1.07],
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    linecolor='black',
                    linewidth=2,
                    mirror=True,        # draws lines on all 4 sides → black box frame
                )
                fig.update_layout(
                    title=title,
                    template="plotly_white",
                    xaxis=dict(**axis_common, title=dict(text=x_title)),
                    yaxis=dict(**axis_common, title=dict(text=y_title)),
                    hovermode="closest",
                    legend=dict(
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                # Dashed zero lines (x- and y-axes through the origin)
                r = 1.07
                for shape_kwargs in [
                    dict(x0=-r, x1=r, y0=0, y1=0),   # horizontal zero line
                    dict(x0=0,  x1=0, y0=-r, y1=r),   # vertical zero line
                ]:
                    fig.add_shape(
                        type='line',
                        line=dict(color='black', dash='dash', width=1),
                        **shape_kwargs,
                    )
            else:
                # For other cases, use original layout
                fig.update_layout(
                    title=title,
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    template="plotly_white",
                    xaxis=dict(range=[-1.05, 1.05] if scale_to_unit else None),
                    yaxis=dict(range=[-1.05, 1.05] if scale_to_unit else None),
                    hovermode="closest",
                    legend=dict(
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
        # fig.show() # Commented out for use in notebook.

    # ------------------------------------------------------------------
    # Save translation table if translations were used
    if translations and len(translation_table) > 0:
        # Create DataFrame from collected translations
        df_translations = pd.DataFrame(translation_table)
        
        # Remove duplicates based on the original text column (DE or IT)
        # Keep the first occurrence which will be from the existing file if it was there
        df_translations = df_translations.drop_duplicates(
            subset=[translate.upper()], 
            keep='first'
        ).sort_values(translate.upper())
        
        # Save to Excel, overwriting the previous file
        output_path = os.path.join(_MODULES_DIR, 'translations', f'translations_{translate}_2.xlsx')
        df_translations.to_excel(output_path, index=False)

    return pca, actor_scores, matrix.columns.tolist(), matrix, fig
