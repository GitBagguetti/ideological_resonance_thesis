import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.graph_objects as go_fig
import plotly.colors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Base directory for external files (same folder as this module)
_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Constants for neat_pca_viz
# ---------------------------------------------------------------------------
ENTITIES_SHORT_PATH = os.path.join(_MODULES_DIR, 'entities', 'entities_short.json')
ENTITIES_CATEGORIES_PATH = os.path.join(_MODULES_DIR, 'entities', 'entities_categories.json')

# Maximum characters per line for verb lists inside area annotations
MAX_LINE_LEN = 20

# Marker (dot) sizes
MARKER_SIZE_DEFAULT = 12         # grey / non-highlighted entities
MARKER_SIZE_HIGHLIGHTED = 16     # highlighted entities (viz_entities mode)

# Font sizes for 1500×1500 pixel output
ENTITY_LABEL_SIZE = 26           # individual entity labels (small increase)
AREA_ANNOTATION_FONT_SIZE = 26   # area annotations (low-to-medium increase)
AXIS_ANNOTATION_FONT_SIZE = 30   # pole descriptions (medium increase)
LEGEND_FONT_SIZE = 22            # legend (substantial increase)
TICK_FONT_SIZE = 22              # axis tick labels (substantial increase)
AXIS_TITLE_FONT_SIZE = 26        # PC1/PC2 axis titles (substantial increase)


# ---------------------------------------------------------------------------
# Helper functions for neat_pca_viz
# ---------------------------------------------------------------------------

def _symlog_transform(x):
    """
    Symmetric log transform that maps [-1, 1] → [-1, 1], compressing the
    centre and expanding the tails.  Accepts scalars or numpy arrays.
    """
    is_scalar = np.isscalar(x)
    x_arr = np.array([x], dtype=float) if is_scalar else np.asarray(x, dtype=float)

    linthresh = 0.05
    linscale  = 0.3
    out = np.zeros_like(x_arr)

    pos_mask = x_arr >  linthresh
    neg_mask = x_arr < -linthresh
    lin_mask = ~(pos_mask | neg_mask)

    out[lin_mask] = x_arr[lin_mask] * linscale / linthresh

    if np.any(pos_mask):
        log_base = np.log(1 / linthresh)
        out[pos_mask] = (
            linscale
            + (1 - linscale) * np.log(x_arr[pos_mask] / linthresh) / log_base
        )

    if np.any(neg_mask):
        log_base = np.log(1 / linthresh)
        out[neg_mask] = -(
            linscale
            + (1 - linscale) * np.log(-x_arr[neg_mask] / linthresh) / log_base
        )

    return float(out[0]) if is_scalar else out


def _wrap_verb_list(verbs, max_len=MAX_LINE_LEN):
    """
    Return *verbs* as a comma-separated HTML string, wrapping to a new
    ``<br>`` line whenever the current line would exceed *max_len* characters.

    Example
    -------
    >>> _wrap_verb_list(["reveal", "expose", "attack", "control"], max_len=20)
    'reveal, expose,<br>attack, control'
    """
    lines, current = [], ""
    for v in verbs:
        sep   = ", " if current else ""
        token = sep + v
        if current and len(current) + len(token) > max_len:
            lines.append(current)
            current = v
        else:
            current += token
    if current:
        lines.append(current)
    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Main visualization function: neat_pca_viz
# ---------------------------------------------------------------------------

def neat_pca_viz(
    matrix_tuple,
    n_components=2,
    scale_to_unit=True,
    title='',
    logscale=False,
    viz_entities=None,
    area_annotations=None,
    axis_annotations=None,
):
    """
    Create a PCA scatter plot for an actor-action matrix (actors only).

    Methodological steps
    --------------------
    1. Column-wise centring & standardisation (z-scores on each action).
    2. PCA fit; actor coordinates = component scores, rescaled to [-1, 1].
    3. Optional symmetric-log transform of coordinates (``logscale=True``).
    4. Plot actors as coloured dots with short entity names (or highlighted
       subset via ``viz_entities``).

    Parameters
    ----------
    matrix_tuple : tuple | pd.DataFrame
        ``(matrix, categories)`` from ``create_actor_action_matrix``, or the
        ``pd.DataFrame`` directly (actors × actions).
    n_components : int, default 2
        Number of PCA components.  Plotting only happens when this equals 2.
    scale_to_unit : bool, default True
        Rescale actor scores to the [-1, 1] range before plotting.
    title : str, default ''
        Plot title.
    logscale : bool, default False
        Apply a symmetric-log transform to the coordinates.  Annotation
        ``pos`` and ``axis_annotations`` positions are specified in the
        original PCA data space; the function transforms them automatically.
    viz_entities : list[str] | None, default None
        Canonical actor names to highlight.  When provided:
        - Listed entities → category-coloured dot + full canonical name label.
        - All other entities → light-grey dot, no label.
        When ``None`` every actor is plotted with its colour and short name
        from ``entities_short.json``.
    area_annotations : list[dict] | None, default None
        Annotations that label a region of the plot.  Each dict must contain:

        ``"pos"`` : tuple (x, y)
            Position in PCA data space ([-1, 1] × [-1, 1]).
        ``"title"`` : list [display_title, category_name]
            ``display_title`` is rendered bold and coloured by the category;
            ``category_name`` is used to look up the colour.
        ``"verbs"`` : list[str]
            Verbs shown below the title, dark-grey, wrapped to ``MAX_LINE_LEN``
            characters per line.

    axis_annotations : dict | None, default None
        Labels for the four poles of the PCA axes.  Keys (all optional):

        ``"top_lab"`` / ``"bot_lab"`` : str or (str, (x, y))
            Label for the positive / negative pole of the y-axis.
            Rendered horizontally, centred on the x = 0 line by default.
        ``"left_lab"`` / ``"right_lab"`` : str or (str, (x, y))
            Label for the negative / positive pole of the x-axis.
            Rendered vertically (bottom-to-top / top-to-bottom) by default.

        When the optional ``(x, y)`` position is omitted, sensible defaults
        at the respective poles are used.  All labels are large, bold, grey,
        and upper-cased automatically.

    Returns
    -------
    tuple : (pca, actor_scores, feature_names, matrix, fig)
    """
    config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'custom_image',
            'scale': 10,
        }
    }

    # ------------------------------------------------------------------
    # Unpack input
    # ------------------------------------------------------------------
    if isinstance(matrix_tuple, tuple):
        matrix, categories = matrix_tuple
    else:
        matrix     = matrix_tuple
        categories = None

    # ------------------------------------------------------------------
    # Load short entity names and category labels
    # ------------------------------------------------------------------
    try:
        with open(ENTITIES_SHORT_PATH, 'r', encoding='utf-8') as fh:
            short_names = json.load(fh)
    except Exception as exc:
        print(f"Warning: could not load entities_short.json: {exc}")
        short_names = {}

    try:
        with open(ENTITIES_CATEGORIES_PATH, 'r', encoding='utf-8') as fh:
            category_labels = json.load(fh)
    except Exception as exc:
        print(f"Warning: could not load entities_categories.json: {exc}")
        category_labels = {}

    def _full_label(cat):
        return category_labels.get(cat, str(cat))

    # ------------------------------------------------------------------
    # 1) Standardise columns
    # ------------------------------------------------------------------
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std  = scaler.fit_transform(matrix.values)

    # ------------------------------------------------------------------
    # 2) PCA
    # ------------------------------------------------------------------
    n_components = min(n_components, min(X_std.shape))
    pca          = PCA(n_components=n_components)
    actor_scores = pca.fit_transform(X_std)

    # ------------------------------------------------------------------
    # 3) Scale to [-1, 1]
    # ------------------------------------------------------------------
    if scale_to_unit and n_components == 2:
        max_actor = np.max(np.abs(actor_scores), axis=0)
        max_actor[max_actor == 0] = 1
        actor_coords = actor_scores / max_actor
    else:
        actor_coords = actor_scores.copy()

    # Keep a copy in original (pre-logscale) space for hover text
    orig_actor_coords = actor_coords.copy()

    # ------------------------------------------------------------------
    # 4) Tick arrays + optional symlog transform
    # ------------------------------------------------------------------
    if logscale and n_components == 2:
        actor_coords = np.column_stack([
            _symlog_transform(actor_coords[:, 0]),
            _symlog_transform(actor_coords[:, 1]),
        ])
        tick_positions_raw = np.array(
            [-1.0, -0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        tick_vals   = _symlog_transform(tick_positions_raw)
        tick_labels = []
        for x in tick_positions_raw:
            if x == 0:
                tick_labels.append("0")
            elif abs(x) < 0.01:
                tick_labels.append(f"{x:.3f}")
            else:
                tick_labels.append(f"{x:.2f}")
    else:
        tick_positions_raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        tick_vals          = tick_positions_raw.copy()
        tick_labels        = ["-1.00", "-0.50", "0", "0.50", "1.00"]

    axis_range = [-1.07, 1.07]

    # ------------------------------------------------------------------
    # Early exit for non-2D case
    # ------------------------------------------------------------------
    if n_components != 2:
        return pca, actor_scores, matrix.columns.tolist(), matrix, None

    # ------------------------------------------------------------------
    # 5) Category → colour map
    # ------------------------------------------------------------------
    color_palette = plotly.colors.qualitative.D3
    unique_cats   = list(categories.unique()) if categories is not None else []
    cat_color_map = {
        cat: color_palette[i % len(color_palette)]
        for i, cat in enumerate(unique_cats)
    }

    # Fast lookup: actor name → row index
    name_to_idx = {name: i for i, name in enumerate(matrix.index)}

    def _hover(name):
        i = name_to_idx[name]
        return (
            f"{name} "
            f"({orig_actor_coords[i, 0]:.3f}, {orig_actor_coords[i, 1]:.3f})"
        )

    # ------------------------------------------------------------------
    # 6) Build figure & add actor traces
    # ------------------------------------------------------------------
    fig     = go_fig.Figure()
    viz_set = set(viz_entities) if viz_entities is not None else None

    if viz_set is not None:
        # ── grey trace for all non-highlighted entities ──────────────────
        non_hl_idx = [i for i, name in enumerate(matrix.index) if name not in viz_set]
        if non_hl_idx:
            fig.add_trace(go_fig.Scatter(
                x=actor_coords[non_hl_idx, 0],
                y=actor_coords[non_hl_idx, 1],
                mode="markers",
                marker=dict(color='lightgrey', size=MARKER_SIZE_DEFAULT),
                showlegend=False,
                hovertext=[_hover(matrix.index[i]) for i in non_hl_idx],
                hoverinfo='text',
                name='',
            ))

        # ── per-category traces for highlighted entities ─────────────────
        if categories is not None:
            # Only categories that have at least one highlighted entity
            cats_with_hl = [
                cat for cat in unique_cats
                if any(
                    name in viz_set and categories.iloc[i] == cat
                    for i, name in enumerate(matrix.index)
                )
            ]
            for cat in sorted(cats_with_hl, key=_full_label):
                hl_idx = [
                    i for i, name in enumerate(matrix.index)
                    if name in viz_set and categories.iloc[i] == cat
                ]
                if not hl_idx:
                    continue
                labels = [short_names.get(matrix.index[i], matrix.index[i]) for i in hl_idx]
                fig.add_trace(go_fig.Scatter(
                    x=actor_coords[hl_idx, 0],
                    y=actor_coords[hl_idx, 1],
                    mode="markers+text",
                    text=labels,
                    textfont=dict(size=ENTITY_LABEL_SIZE),
                    name=_full_label(cat),
                    marker=dict(color=cat_color_map.get(cat, '#333333'), size=MARKER_SIZE_HIGHLIGHTED),
                    hovertext=[_hover(matrix.index[i]) for i in hl_idx],
                    hoverinfo='text',
                    textposition="bottom center",
                ))
        else:
            # No category information — single highlighted trace
            hl_idx = [i for i, name in enumerate(matrix.index) if name in viz_set]
            if hl_idx:
                labels = [short_names.get(matrix.index[i], matrix.index[i]) for i in hl_idx]
                fig.add_trace(go_fig.Scatter(
                    x=actor_coords[hl_idx, 0],
                    y=actor_coords[hl_idx, 1],
                    mode="markers+text",
                    text=labels,
                    textfont=dict(size=ENTITY_LABEL_SIZE),
                    name="Highlighted",
                    marker=dict(color=color_palette[0], size=MARKER_SIZE_HIGHLIGHTED),
                    hovertext=[_hover(matrix.index[i]) for i in hl_idx],
                    hoverinfo='text',
                    textposition="bottom center",
                ))

    else:
        # ── default: all entities with short names and category colours ──
        if categories is not None:
            for cat in sorted(unique_cats, key=_full_label):
                mask        = (categories == cat).values
                cat_indices = np.where(mask)[0]
                actor_names = matrix.index[mask]
                labels      = [short_names.get(name, name) for name in actor_names]
                fig.add_trace(go_fig.Scatter(
                    x=actor_coords[mask, 0],
                    y=actor_coords[mask, 1],
                    mode="markers+text",
                    text=labels,
                    textfont=dict(size=ENTITY_LABEL_SIZE),
                    name=_full_label(cat),
                    marker=dict(color=cat_color_map.get(cat, '#333333'), size=MARKER_SIZE_DEFAULT),
                    hovertext=[_hover(matrix.index[i]) for i in cat_indices],
                    hoverinfo='text',
                    textposition="bottom center",
                ))
        else:
            # No category information — single trace
            labels = [short_names.get(name, name) for name in matrix.index]
            fig.add_trace(go_fig.Scatter(
                x=actor_coords[:, 0],
                y=actor_coords[:, 1],
                mode="markers+text",
                text=labels,
                textfont=dict(size=ENTITY_LABEL_SIZE),
                name="Actors",
                marker=dict(color=color_palette[0], size=MARKER_SIZE_DEFAULT),
                hovertext=[_hover(name) for name in matrix.index],
                hoverinfo='text',
                textposition="bottom center",
            ))

    # ------------------------------------------------------------------
    # 7) Axis style: black box frame + dashed zero lines
    # ------------------------------------------------------------------
    ev      = pca.explained_variance_ratio_ * 100
    x_title = f"PC1 — {ev[0]:.1f}% variance"
    y_title = f"PC2 — {ev[1]:.1f}% variance"

    axis_common = dict(
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_labels,
        range=axis_range,
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor='black',
        linewidth=2,
        mirror=True,        # draws lines on all 4 sides → black box frame
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis=dict(**axis_common, title=dict(text=x_title, font=dict(size=AXIS_TITLE_FONT_SIZE))),
        yaxis=dict(**axis_common, title=dict(text=y_title, font=dict(size=AXIS_TITLE_FONT_SIZE))),
        hovermode="closest",
        margin=dict(t=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )

    # Dashed zero lines (x- and y-axes through the origin)
    r = axis_range[1]
    for shape_kwargs in [
        dict(x0=-r, x1=r, y0=0, y1=0),   # horizontal zero line
        dict(x0=0,  x1=0, y0=-r, y1=r),   # vertical zero line
    ]:
        fig.add_shape(
            type='line',
            line=dict(color='black', dash='dash', width=1),
            **shape_kwargs,
        )

    # ------------------------------------------------------------------
    # 8) area_annotations
    # ------------------------------------------------------------------
    if area_annotations:
        for ann in area_annotations:
            pos        = ann.get('pos', (0, 0))
            title_list = ann.get('title', ['', ''])
            verbs      = ann.get('verbs', [])

            title_text = title_list[0] if len(title_list) > 0 else ''
            cat_name   = title_list[1] if len(title_list) > 1 else ''

            # Resolve category colour (case-insensitive match)
            cat_color = '#333333'
            for k, v in cat_color_map.items():
                if str(k).lower() == str(cat_name).lower():
                    cat_color = v
                    break

            # Build HTML annotation text
            title_html = (
                f"<b><span style='color:{cat_color}'>{title_text}</span></b>"
            )
            verbs_html = (
                f"<span style='color:#555555'>{_wrap_verb_list(verbs)}</span>"
                if verbs else ""
            )
            ann_text = title_html + ("<br>" + verbs_html if verbs_html else "")

            # Convert pos to display coordinates
            x_pos, y_pos = float(pos[0]), float(pos[1])
            if logscale:
                x_pos = _symlog_transform(x_pos)
                y_pos = _symlog_transform(y_pos)

            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                xref='x',
                yref='y',
                text=ann_text,
                showarrow=False,
                align='left',
                valign='top',
                bgcolor='rgba(255,255,255,0.7)',
                borderpad=4,
                font=dict(size=AREA_ANNOTATION_FONT_SIZE),
            )

    # ------------------------------------------------------------------
    # 9) axis_annotations
    # ------------------------------------------------------------------
    if axis_annotations:
        # (default_x, default_y, textangle)
        pole_defaults = {
            'top_lab':   ( 0,  1,   0),   # horizontal, top of y-axis
            'bot_lab':   ( 0, -1,   0),   # horizontal, bottom of y-axis
            'left_lab':  (-1,  0, -90),   # vertical bottom-to-top, left of x-axis
            'right_lab': ( 1,  0,  90),   # vertical top-to-bottom, right of x-axis
        }

        for key, (def_x, def_y, angle) in pole_defaults.items():
            if key not in axis_annotations:
                continue

            val = axis_annotations[key]

            # val may be a plain string or (string,) or (string, (x, y))
            if isinstance(val, str):
                label        = val
                x_pos, y_pos = float(def_x), float(def_y)
            elif isinstance(val, (tuple, list)):
                label = val[0]
                if len(val) > 1 and val[1] is not None:
                    x_pos, y_pos = float(val[1][0]), float(val[1][1])
                else:
                    x_pos, y_pos = float(def_x), float(def_y)
            else:
                continue

            # Apply symlog transform to position if needed
            if logscale:
                x_pos = _symlog_transform(x_pos)
                y_pos = _symlog_transform(y_pos)

            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                xref='x',
                yref='y',
                text=f"<b>{label.upper()}</b>",
                showarrow=False,
                textangle=angle,
                font=dict(size=AXIS_ANNOTATION_FONT_SIZE, color='grey'),
            )

    # ------------------------------------------------------------------
    # fig.show(config=config)
    return pca, actor_scores, matrix.columns.tolist(), matrix, fig


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

def add_traces_and_axes(fig_from, fig_to, row, col):
    """
    Helper function to add all traces from a source Plotly figure (`fig_from`) to a subplot
    in a target figure (`fig_to`) at the specified row and column. Also copies axis ranges
    and axis titles (including explained variance, if present) from the source to the target subplot.

    Args:
        fig_from (plotly.graph_objs.Figure): The source figure from which to copy traces and axis settings.
        fig_to (plotly.graph_objs.Figure): The target figure (usually a subplot figure) to which traces and axes are added.
        row (int): The row index of the subplot in `fig_to`.
        col (int): The column index of the subplot in `fig_to`.

    Notes:
        - If a trace is a Scatter and has text labels, the text font size is increased for better readability.
        - Axis ranges and titles are copied if present in the source figure.
    """
    for trace in fig_from.data:
        # If trace is a Scatter and has textfont, increase font size
        if isinstance(trace, go.Scatter):
            # Try to increase textfont size for labels (if present)
            trace = trace.update(textfont=dict(size=14))  # Slightly larger than default (~12-14)
        fig_to.add_trace(trace, row=row, col=col)
    # Copy all axis properties
    if 'xaxis' in fig_from.layout:
        xaxis_props = fig_from.layout['xaxis']
        update_dict = {}
        
        # Copy all relevant axis properties
        for prop in ['range', 'type', 'tickmode', 'tickvals', 'ticktext', 'title', 'gridwidth', 'gridcolor', 'zeroline', 'zerolinewidth', 'zerolinecolor']:
            if prop in xaxis_props:
                if prop == 'title':
                    update_dict['title_text'] = xaxis_props[prop].text if hasattr(xaxis_props[prop], 'text') else xaxis_props[prop]
                else:
                    update_dict[prop] = xaxis_props[prop]
        
        fig_to.update_xaxes(row=row, col=col, **update_dict)

    if 'yaxis' in fig_from.layout:
        yaxis_props = fig_from.layout['yaxis']
        update_dict = {}
        
        # Copy all relevant axis properties
        for prop in ['range', 'type', 'tickmode', 'tickvals', 'ticktext', 'title', 'gridwidth', 'gridcolor', 'zeroline', 'zerolinewidth', 'zerolinecolor']:
            if prop in yaxis_props:
                if prop == 'title':
                    update_dict['title_text'] = yaxis_props[prop].text if hasattr(yaxis_props[prop], 'text') else yaxis_props[prop]
                else:
                    update_dict[prop] = yaxis_props[prop]
        
        fig_to.update_yaxes(row=row, col=col, **update_dict)




def analyze_pca_components(pca, feature_names, scree=False, axes_words=(False, 2)):
    """
    Analyze PCA components in detail with publication-grade visualizations.
    
    Args:
        pca: Fitted sklearn.decomposition.PCA object
        feature_names: List of feature names corresponding to PCA columns
        scree (bool): If True, display scree plot of top 8 components
        axes_words (tuple): (bool, N) - If bool True, show top 25 words for each pole of top N axes
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 1. Scree plot
    if scree:
        n_comp = min(8, len(pca.explained_variance_ratio_))
        var_explained = pca.explained_variance_ratio_[:n_comp] * 100
        cumulative_var = np.cumsum(var_explained)
        
        fig = go.Figure()
        
        # Bar chart for individual variance
        fig.add_trace(
            go.Bar(
                x=list(range(1, n_comp + 1)),
                y=var_explained,
                name='Individual',
                text=[f'{v:.1f}%' for v in var_explained],
                textposition='auto',
            )
        )
        
        # Line for cumulative variance
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_comp + 1)),
                y=cumulative_var,
                mode='lines+markers',
                name='Cumulative',
                text=[f'{v:.1f}%' for v in cumulative_var],
                textposition='top center',
                line=dict(color='red')
            )
        )
        
        fig.update_layout(
            title='Explained Variance by Principal Component',
            xaxis_title='Principal Component',
            yaxis_title='Variance Explained (%)',
            template='plotly_white',
            showlegend=True,
            yaxis_range=[0, 100],
            width=800,
            height=500,
            font=dict(size=14),
            xaxis=dict(dtick=1)  # Force whole number ticks
        )
        
        fig.show()
    
    # 2. Top words table
    if axes_words[0]:
        n_axes = min(axes_words[1], len(pca.components_))
        n_words = 50
        
        # Create figure with subplots for each component
        for i in range(n_axes):
            loadings = pca.components_[i]
            sorted_idx = np.argsort(loadings)
            
            # Create two-column table
            neg_words = [(feature_names[idx], loadings[idx]) for idx in sorted_idx[:n_words]]
            pos_words = [(feature_names[idx], loadings[idx]) for idx in sorted_idx[-n_words:][::-1]]
            
            # Format as markdown table
            print(f"\nPC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% variance)")
            print("| Negative Pole | Loading | Loading | Positive Pole |")
            print("|---------------|----------|----------|---------------|")
            for (neg_word, neg_load), (pos_word, pos_load) in zip(neg_words, pos_words):
                print(f"| {neg_word:13} | {neg_load:8.3f} | {pos_load:8.3f} | {pos_word:13} |")

# ------------------------------------------------------------------
