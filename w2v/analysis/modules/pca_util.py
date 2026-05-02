import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = Path(__file__).resolve().parents[3]


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
        with open(_REPO_ROOT / 'pca' / 'modules' / 'translations' / 'actions_mapping.json', 'r', encoding='utf-8') as f:
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
