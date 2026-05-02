import pandas as pd
import json

"""
PART 1: Loading and preprocessing QAnon NER data
"""

# Load NER data
ner_df = pd.read_parquet('/project/ssd-stu-research/ploertscher/thesis_code/text-processing/data/full_filt/5_posts_ner.parquet')

# Convert columns to categorical to save memory
categorical_cols = ['entity_type']
for col in categorical_cols:
    ner_df[col] = ner_df[col].astype('category')

print(f"Loaded NER data shape: {ner_df.shape}")

# Join with lang_df to keep only rows present in lang_df
ner_df = ner_df.merge(lang_df, 
                      on='posts_id',
                      how='inner')
print(f"NER data shape after joining with lang_df: {ner_df.shape}")

# Load entity mapping functions from util_pd.py
import sys
sys.path.append('/project/ssd-stu-research/ploertscher/thesis_code/text-processing')
from util_pd import load_entity_variations
import json

# Load entity mapping data
entities_mapping_path = '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/pca/modules/entities/entities_mapping_2.json'
with open(entities_mapping_path, 'r', encoding='utf-8') as f:
    entities_mapping = json.load(f)

# Create helper functions for entity processing
def build_entity_mappings(entities_mapping):
    """
    Build mappings from entity variations to canonical names and groups.
    
    Returns:
        tuple: (variation_to_canonical, canonical_to_group)
    """
    variation_to_canonical = {}
    canonical_to_group = {}
    
    for group_name, entities in entities_mapping.items():
        for entity_key, entity_data in entities.items():
            canonical_name = entity_data['canonical_name']
            variations = entity_data['variations']
            
            # Map all variations to canonical name (case-insensitive)
            for variation in variations:
                variation_to_canonical[variation.lower()] = canonical_name
            
            # Map canonical name to group
            canonical_to_group[canonical_name] = group_name
    
    return variation_to_canonical, canonical_to_group

# Build the mappings
variation_to_canonical, canonical_to_group = build_entity_mappings(entities_mapping)

print(f"Built mappings for {len(variation_to_canonical)} entity variations")
print(f"Mapped to {len(canonical_to_group)} canonical entities")
print(f"Across {len(set(canonical_to_group.values()))} entity groups")

# Process entities in batches
print("\nProcessing entities in batches...")
batch_size = 100000
n_batches = len(ner_df) // batch_size + 1

filtered_dfs = []
original_non_loc_count = 0

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(ner_df))
    
    batch_df = ner_df.iloc[start_idx:end_idx].copy()
    
    # Count non-location entities
    original_non_loc_count += len(batch_df[batch_df['entity_type'] != 'LOC'])
    
    # Filter entities that exist in the mapping
    batch_df['entity_value_lower'] = batch_df['entity_value'].str.lower()
    batch_filtered = batch_df[batch_df['entity_value_lower'].isin(list(variation_to_canonical.keys()))].copy()
    
    if len(batch_filtered) > 0:
        # Map to canonical names
        batch_filtered.loc[:, 'entity_canonical'] = batch_filtered['entity_value_lower'].map(variation_to_canonical)
        
        # Add NER group columns
        batch_filtered.loc[:, 'ner_group'] = batch_filtered['entity_canonical'].map(canonical_to_group)
        batch_filtered.loc[:, 'ner_group_generic'] = batch_filtered['ner_group'].str.replace(r'_(us|de|it|intl|good|bad)$', '', regex=True)
        
        # Clean up temporary columns
        batch_filtered = batch_filtered.drop(['entity_value_lower'], axis=1)
        
        filtered_dfs.append(batch_filtered)
    
    print(f"Processed batch {i+1}/{n_batches}", end='\r')

# Combine all filtered batches
ner_df_filtered = pd.concat(filtered_dfs, ignore_index=True)

print(f"\nOriginal non-location entities: {original_non_loc_count:,}")
print(f"Filtered entities: {len(ner_df_filtered):,}")
print(f"Kept {len(ner_df_filtered)/original_non_loc_count*100:.1f}% of non-location entities")

# Display results
print(f"\nFinal dataset shape: {ner_df_filtered.shape}")
print(f"\nEntities by group:")
print(ner_df_filtered['ner_group'].value_counts())

print(f"\nEntities by generic group:")
print(ner_df_filtered['ner_group_generic'].value_counts())

print(f"\nTop 10 canonical entities:")
print(ner_df_filtered['entity_canonical'].value_counts().head(10))

print(f"\nSample of processed data:")
print(ner_df_filtered[['posts_id', 'entity_type', 'entity_value', 'entity_canonical', 'ner_group', 'ner_group_generic']].head(10))


"""
PART 2: Loading and preprocessing media NER data
"""
def process_entities(df, fpath):
    """
    Process named entities by mapping variations to canonical names and groups.
    
    Args:
        df: DataFrame containing entity data with columns 'entity_value' and 'entity_type'
        fpath: Path to JSON file containing entity mappings
        
    Returns:
        DataFrame with added columns 'entity_canonical' and 'ner_group'
    """
    # Load entity mapping data
    with open(fpath, 'r', encoding='utf-8') as f:
        entities_mapping = json.load(f)

    # Build mappings
    variation_to_canonical = {}
    canonical_to_group = {}
    nvar = 0
    ncan = 0

    for group_name, entities in entities_mapping.items():
        for entity_key, entity_data in entities.items():
            canonical_name = entity_data['canonical_name']
            variations = entity_data['variations']
            
            # Map variations to canonical name (case-insensitive)
            for variation in variations:
                variation_to_canonical[variation.lower()] = canonical_name
                nvar += 1
            
            # Map canonical name to group
            canonical_to_group[canonical_name] = group_name
            ncan += 1
    
    print(f"Total number of variations: {nvar}")
    print(f"Total number of canonical names: {ncan}")

    # Filter out location entities
    df = df[df['entity_type'] != 'LOC'].copy()
    
    # Create lowercase entity values
    df['entity_value_lower'] = df['entity_value'].str.lower()
    
    # Split into matched and unmatched entities
    matched_mask = df['entity_value_lower'].isin(variation_to_canonical.keys())
    matched_df = df[matched_mask].copy()
    unmatched_df = df[~matched_mask].copy()
    
    # Process matched entities
    if len(matched_df) > 0:
        matched_df['entity_canonical'] = matched_df['entity_value_lower'].map(variation_to_canonical)
        matched_df['ner_group'] = matched_df['entity_canonical'].map(canonical_to_group)
    
    # Set NA for unmatched entities
    if len(unmatched_df) > 0:
        unmatched_df['entity_canonical'] = pd.NA
        unmatched_df['ner_group'] = pd.NA
    
    # Combine matched and unmatched
    result_df = pd.concat([matched_df, unmatched_df], ignore_index=True)
    
    # Clean up temporary column
    result_df = result_df.drop('entity_value_lower', axis=1)
    
    print(f"Total entities processed: {len(result_df):,}")
    print(f"Entities mapped to canonical names: {len(matched_df):,}")
    print(f"Unmapped entities: {len(unmatched_df):,}")
    
    return result_df

print("\nLoading media NER data...")
media_df = pd.read_parquet('/project/ssd-stu-research/ploertscher/thesis_code/text-processing/data/poltext_cased/ner_40.parquet').rename(columns={'entity': 'entity_value', 'ner_label': 'entity_type'})
print(f"Loaded media NER data shape: {media_df.shape}")
print(f"Columns: {media_df.columns}")

# Replace entity type codes with full names
entity_type_map = {
    'C': 'LOC',
    'G': 'ORG', 
    'R': 'PER'
}
media_df['entity_type'] = media_df['entity_type'].map(entity_type_map)

# Convert columns to appropriate types
print("\nConverting media data columns to appropriate types...")

# Convert source and polbias to categorical
media_df['source'] = pd.Categorical(media_df['source'])
media_df['polbias'] = pd.Categorical(media_df['polbias'])
media_df['entity_type'] = pd.Categorical(media_df['entity_type'])

# Convert entity_value to string
media_df['entity_value'] = media_df['entity_value'].astype(str)

# Convert year to datetime 
media_df['year'] = pd.to_datetime(media_df['year'], format='%Y')

print("Data types after conversion:")
print(media_df[['source', 'polbias', 'entity_type', 'year']].dtypes)

media_df = process_entities(media_df, '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/pca/modules/entities/entities_mapping_2.json')


"""
PART 3: Analyzing top 25 entities distribution across corpora
"""
print("\nAnalyzing top 25 entities distribution across corpora...")

def get_ranked_entities(df, entity_col='entity_value', filter_col=None, filter_val=None, n=25):
    """Get top n entities with their percentages for a corpus"""
    if filter_col and filter_val:
        df = df[df[filter_col] == filter_val]
    # Remove rows where entity contains only punctuation/whitespace (no letters/numbers) and handle NAs
    mask = df[entity_col].notna() & ~df[entity_col].str.match(r'^[^\w]+$', na=False)
    df = df[mask]
    dist = df[entity_col].value_counts(normalize=True)
    return [(entity, f"{pct*100:.2f}%") for entity, pct in dist.nlargest(n).items()]

# 1. Unfiltered data analysis
print("\nTop 25 entities by rank in each corpus (unfiltered data):")

# Calculate ranked distributions for each corpus
unfiltered_rankings = {
    'Media': get_ranked_entities(media_df),
    'All Conspiracy': get_ranked_entities(ner_df),
    'EN Conspiracy': get_ranked_entities(ner_df, filter_col='lang_det', filter_val='en'),
    'DE Conspiracy': get_ranked_entities(ner_df, filter_col='lang_det', filter_val='de'),
    'IT Conspiracy': get_ranked_entities(ner_df, filter_col='lang_det', filter_val='it')
}

# Create DataFrame with ranked entities and percentages
unfiltered_table = pd.DataFrame(index=range(1, 26))
for corpus, rankings in unfiltered_rankings.items():
    unfiltered_table[corpus] = [f"{entity} ({pct})" for entity, pct in rankings]

print("\nRanked entities with percentages by corpus (unfiltered data):")
pd.set_option('display.max_colwidth', None)
print(unfiltered_table.to_string())

# 2. Filtered data analysis
print("\nTop 25 entities by rank in each corpus (filtered data):")

# Calculate ranked distributions for filtered data
filtered_rankings = {
    'Media': get_ranked_entities(media_df, entity_col='entity_canonical'),
    'All Conspiracy': get_ranked_entities(ner_df, entity_col='entity_canonical'),
    'EN Conspiracy': get_ranked_entities(ner_df[ner_df['lang_det'] == 'en'], entity_col='entity_canonical'),
    'DE Conspiracy': get_ranked_entities(ner_df[ner_df['lang_det'] == 'de'], entity_col='entity_canonical'),
    'IT Conspiracy': get_ranked_entities(ner_df[ner_df['lang_det'] == 'it'], entity_col='entity_canonical')
}

# Create DataFrame with ranked entities and percentages
filtered_table = pd.DataFrame(index=range(1, 26))
for corpus, rankings in filtered_rankings.items():
    filtered_table[corpus] = [f"{entity} ({pct})" for entity, pct in rankings]

print("\nRanked entities with percentages by corpus (filtered data):")
print(filtered_table.to_string())

# 3. Relative frequency comparison
print("\nComparing relative frequencies between media and conspiracy corpora:")

def get_entity_stats(df, entity_col='entity_canonical'):
    """Get entity counts and frequencies"""
    counts = df[entity_col].value_counts()
    freqs = df[entity_col].value_counts(normalize=True)
    return pd.DataFrame({'count': counts, 'freq': freqs})

# Get overall media corpus stats first
media_stats_all = get_entity_stats(media_df)

# Get language-specific media stats with filtering
media_stats_en = get_entity_stats(media_df[~media_df['ner_group'].str.endswith(('_it', '_de'), na=False)])
media_stats_de = get_entity_stats(media_df[~media_df['ner_group'].str.endswith('_it', na=False)])
media_stats_it = get_entity_stats(media_df[~media_df['ner_group'].str.endswith('_de', na=False)])

# Function to calculate relative frequency ratios
def get_freq_ratios(consp_df, media_stats, lang=None):
    if lang == 'en':
        consp_df = consp_df[
            (consp_df['lang_det'] == 'en') & 
            ~consp_df['ner_group'].str.endswith(('_it', '_de'), na=False)
        ]
    elif lang == 'de':
        consp_df = consp_df[
            (consp_df['lang_det'] == 'de') &
            ~consp_df['ner_group'].str.endswith('_it', na=False)
        ]
    elif lang == 'it':
        consp_df = consp_df[
            (consp_df['lang_det'] == 'it') &
            ~consp_df['ner_group'].str.endswith('_de', na=False)
        ]
    
    consp_stats = get_entity_stats(consp_df)
    
    # Combine stats
    combined = pd.DataFrame({
        'media_count': media_stats['count'],
        'media_freq': media_stats['freq'],
        'consp_count': consp_stats['count'],
        'consp_freq': consp_stats['freq']
    }).fillna(0)
    
    # Calculate ratio (adding small constant to avoid division by zero)
    combined['ratio'] = combined['consp_freq'] / (combined['media_freq'] + 1e-10)
    
    return combined.sort_values('ratio', ascending=False)

# Calculate ratios for overall comparison first
comparisons = {
    'Media vs All Conspiracy': get_freq_ratios(ner_df, media_stats_all),
    'Media vs EN': get_freq_ratios(ner_df, media_stats_en, 'en'),
    'Media vs DE': get_freq_ratios(ner_df, media_stats_de, 'de'),
    'Media vs IT': get_freq_ratios(ner_df, media_stats_it, 'it')
}

# Display top 75 entities with highest ratios for each comparison
for name, comparison in comparisons.items():
    print(f"\n{name} Conspiracy Comparison (Top 75):")
    top75 = comparison.head(75)
    formatted = pd.DataFrame({
        'Entity': top75.index,
        'Media Count': top75['media_count'].astype(int),
        'Media Freq': top75['media_freq'].map('{:.2%}'.format),
        'Conspiracy Count': top75['consp_count'].astype(int),
        'Conspiracy Freq': top75['consp_freq'].map('{:.2%}'.format),
        'Relative Ratio': top75['ratio'].map('{:.2f}'.format)
    })
    print(formatted.to_string())

# Reset display options to default
pd.reset_option('display.max_colwidth')


"""
PART 4: Analyzing English conspiracy vs mainstream media entity frequency ratios
"""
import matplotlib.pyplot as plt
import numpy as np

print("\nAnalyzing English conspiracy vs mainstream media entity frequency ratios...")

def get_entity_stats(df, entity_col='entity_canonical'):
    """Get entity counts and frequencies"""
    counts = df[entity_col].value_counts()
    freqs = df[entity_col].value_counts(normalize=True)
    return pd.DataFrame({'count': counts, 'freq': freqs})

# Get English-only, non-DE/IT media stats
media_stats_en = get_entity_stats(
    media_df[~media_df['ner_group'].str.endswith(('_it', '_de'), na=False)]
)

# Get English-only, non-DE/IT conspiracy stats
consp_en_df = ner_df[
    (ner_df['lang_det'] == 'en') &
    ~ner_df['ner_group'].str.endswith(('_it', '_de'), na=False)
]
consp_stats_en = get_entity_stats(consp_en_df)

# Combine stats
combined = pd.DataFrame({
    'media_count': media_stats_en['count'],
    'media_freq': media_stats_en['freq'],
    'consp_count': consp_stats_en['count'],
    'consp_freq': consp_stats_en['freq']
}).fillna(0)

# Calculate ratio (adding small constant to avoid division by zero)
combined['ratio'] = combined['consp_freq'] / (combined['media_freq'] + 1e-10)

# Filter: only entities with at least 0.01% in media, then take top 20 by ratio
filtered = combined[
    (combined['media_freq'] >= 0.0001)  # 0.01% = 0.0001
].sort_values('ratio', ascending=False).head(12)

print(f"\nTop 20 entities by ratio (with >0.01% freq in media):")
pd.set_option('display.max_colwidth', None)
print(filtered[['media_count', 'media_freq', 'consp_count', 'consp_freq', 'ratio']].to_string())

# Prepare excluded entities: those with <0.01% in MSM, ranked by conspiracy freq, top 10
excluded = combined[
    (combined['media_freq'] < 0.0001) & (combined['consp_count'] > 0)
].sort_values('consp_count', ascending=False).head(10)
excluded_names = excluded.index.tolist()
excluded_counts = excluded['consp_count'].tolist()

# Barplot
entities = filtered.index.tolist()
ratios = filtered['ratio'].values

plt.figure(figsize=(max(10, len(entities)*0.7), 6))
bars = plt.bar(range(len(entities)), ratios, color='skyblue', log=True)

plt.xticks(range(len(entities)), entities, rotation=45, ha='right', fontsize=10)
plt.ylabel('Relative Frequency Ratio (log scale)')
plt.xlabel('Entity')
plt.title('Entities with Highest Relative Frequency Ratio\n(English Conspiracy vs Mainstream Media)')

plt.tight_layout()
plt.subplots_adjust(top=1.05)  # Add more distance between the top frame and the plot

# Annotate ratio on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        height * 1.05, 
        f"{ratios[i]:.0f}", 
        ha='center', va='bottom', fontsize=9, rotation=0
    )

# Add excluded box in top right
if len(excluded_names) > 0:
    # Format: Entity (count)
    box_lines = [f"{i+1}. {name}" for i, name in enumerate(excluded_names)]
    # Make the title bold using TeX markup
    box_text = r"$\bf{Excluded\ (<0.01\%\ in\ MSM)}$" + "\n" + "\n".join(box_lines)
    plt.gca().text(
        0.775, 0.97, box_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        va='top', ha='left',
        bbox=dict(boxstyle='square,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
    )

plt.show()

# Reset display options to default
pd.reset_option('display.max_colwidth')