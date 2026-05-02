import pandas as pd
import json
import os

"""
PART 1: Loading and preprocessing QAnon NER data
"""

# Load language-classified posts for merge and date filtering
lang_df = pd.read_parquet(
    '/project/ssd-stu-research/ploertscher/thesis_code/text-processing/data/full_filt/4_posts_langclass.parquet'
)
lang_df = lang_df[(lang_df['date'].dt.year >= 2019) & (lang_df['date'].dt.year <= 2022)]

# Load NER data
ner_df = pd.read_parquet(
    '/project/ssd-stu-research/ploertscher/thesis_code/text-processing/data/full_filt/5_posts_ner.parquet'
)

# Convert columns to categorical to save memory
ner_df['entity_type'] = ner_df['entity_type'].astype('category')

print(f"Loaded NER data shape: {ner_df.shape}")

# Join with lang_df to keep only rows present in lang_df
ner_df = ner_df.merge(lang_df, on='posts_id', how='inner')
print(f"NER data shape after joining with lang_df: {ner_df.shape}")

# Remove Italian data (posts in Italian)
ner_df = ner_df[ner_df['lang_det'] != 'it'].copy()
print(f"NER data shape after removing Italian: {ner_df.shape}")

# Load entity mapping data
entities_mapping_path = '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/pca/modules/entities/entities_mapping_2.json'
with open(entities_mapping_path, 'r', encoding='utf-8') as f:
    entities_mapping = json.load(f)


def build_entity_mappings(entities_mapping):
    """Build mappings from entity variations to canonical names and groups."""
    variation_to_canonical = {}
    canonical_to_group = {}

    for group_name, entities in entities_mapping.items():
        for entity_key, entity_data in entities.items():
            canonical_name = entity_data['canonical_name']
            variations = entity_data['variations']

            for variation in variations:
                variation_to_canonical[variation.lower()] = canonical_name
            canonical_to_group[canonical_name] = group_name

    return variation_to_canonical, canonical_to_group


variation_to_canonical, canonical_to_group = build_entity_mappings(entities_mapping)

# Process entities in batches
print("\nProcessing entities in batches...")
batch_size = 100000
n_batches = len(ner_df) // batch_size + 1
filtered_dfs = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(ner_df))
    batch_df = ner_df.iloc[start_idx:end_idx].copy()

    batch_df['entity_value_lower'] = batch_df['entity_value'].str.lower()
    batch_filtered = batch_df[
        batch_df['entity_value_lower'].isin(list(variation_to_canonical.keys()))
    ].copy()

    if len(batch_filtered) > 0:
        batch_filtered.loc[:, 'entity_canonical'] = batch_filtered['entity_value_lower'].map(
            variation_to_canonical
        )
        batch_filtered.loc[:, 'ner_group'] = batch_filtered['entity_canonical'].map(
            canonical_to_group
        )
        batch_filtered = batch_filtered.drop(['entity_value_lower'], axis=1)
        filtered_dfs.append(batch_filtered)

    print(f"Processed batch {i+1}/{n_batches}", end='\r')

# Combine and remove Italian entity groups
ner_df = pd.concat(filtered_dfs, ignore_index=True)
ner_df = ner_df[~ner_df['ner_group'].str.endswith('_it', na=False)].copy()
print(f"\nFinal QAnon NER shape: {ner_df.shape}")

# Save to data folder
data_dir = '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/data'
os.makedirs(data_dir, exist_ok=True)
ner_df.to_csv(os.path.join(data_dir, 'qanon_ner.csv'), index=False)
print(f"Saved to {data_dir}/qanon_ner.csv")


"""
PART 2: Loading and preprocessing media NER data
"""


def process_entities(df, fpath):
    """Map entity variations to canonical names and groups."""
    with open(fpath, 'r', encoding='utf-8') as f:
        entities_mapping = json.load(f)

    variation_to_canonical = {}
    canonical_to_group = {}

    for group_name, entities in entities_mapping.items():
        for entity_key, entity_data in entities.items():
            canonical_name = entity_data['canonical_name']
            variations = entity_data['variations']
            for variation in variations:
                variation_to_canonical[variation.lower()] = canonical_name
            canonical_to_group[canonical_name] = group_name

    df = df[df['entity_type'] != 'LOC'].copy()
    df['entity_value_lower'] = df['entity_value'].str.lower()
    matched_mask = df['entity_value_lower'].isin(variation_to_canonical.keys())
    matched_df = df[matched_mask].copy()
    unmatched_df = df[~matched_mask].copy()

    if len(matched_df) > 0:
        matched_df['entity_canonical'] = matched_df['entity_value_lower'].map(
            variation_to_canonical
        )
        matched_df['ner_group'] = matched_df['entity_canonical'].map(canonical_to_group)
    if len(unmatched_df) > 0:
        unmatched_df['entity_canonical'] = pd.NA
        unmatched_df['ner_group'] = pd.NA

    result_df = pd.concat([matched_df, unmatched_df], ignore_index=True)
    result_df = result_df.drop('entity_value_lower', axis=1)
    return result_df


print("\nLoading media NER data...")
media_df = pd.read_parquet(
    '/project/ssd-stu-research/ploertscher/thesis_code/text-processing/data/poltext_cased/ner_40.parquet'
).rename(columns={'entity': 'entity_value', 'ner_label': 'entity_type'})

entity_type_map = {'C': 'LOC', 'G': 'ORG', 'R': 'PER'}
media_df['entity_type'] = media_df['entity_type'].map(entity_type_map)
media_df['entity_value'] = media_df['entity_value'].astype(str)

media_df = process_entities(
    media_df,
    '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/pca/modules/entities/entities_mapping_2.json'
)

# Remove Italian entity groups
media_df = media_df[~media_df['ner_group'].str.endswith('_it', na=False)].copy()
print(f"Final media NER shape: {media_df.shape}")

# Save to data folder
media_df.to_csv(os.path.join(data_dir, 'media_ner.csv'), index=False)
print(f"Saved to {data_dir}/media_ner.csv")


"""
PART 3: Load preprocessed data for analysis
(Run this part instead of Parts 1 & 2 when re-running analysis without re-processing)
"""

# data_dir = '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/data'
# ner_df = pd.read_csv(os.path.join(data_dir, 'qanon_ner.csv'))
# ner_df['lang_det'] = ner_df['lang_det'].astype('category')
# media_df = pd.read_csv(os.path.join(data_dir, 'media_ner.csv'))


"""
PART 3b: Ranked entities with percentages by corpus (filtered data, top 10, no Italian)
"""


def get_ranked_entities(df, entity_col='entity_canonical', filter_col=None, filter_val=None, n=10):
    """Get top n entities with their percentages for a corpus"""
    if filter_col and filter_val:
        df = df[df[filter_col] == filter_val]
    mask = df[entity_col].notna() & ~df[entity_col].str.match(r'^[^\w]+$', na=False)
    df = df[mask]
    dist = df[entity_col].value_counts(normalize=True)
    return [(entity, f"{pct*100:.2f}%") for entity, pct in dist.nlargest(n).items()]


filtered_rankings = {
    'Media': get_ranked_entities(media_df, entity_col='entity_canonical', n=10),
    'All Conspiracy': get_ranked_entities(ner_df, entity_col='entity_canonical', n=10),
    'EN Conspiracy': get_ranked_entities(ner_df[ner_df['lang_det'] == 'en'], entity_col='entity_canonical', n=10),
    'DE Conspiracy': get_ranked_entities(ner_df[ner_df['lang_det'] == 'de'], entity_col='entity_canonical', n=10),
}

filtered_table = pd.DataFrame(index=range(1, 11))
for corpus, rankings in filtered_rankings.items():
    filtered_table[corpus] = [f"{entity} ({pct})" for entity, pct in rankings]

print("\nRanked entities with percentages by corpus (filtered data, top 10, no Italian):")
pd.set_option('display.max_colwidth', None)
print(filtered_table.to_string())
pd.reset_option('display.max_colwidth')


"""
PART 4: Analyzing English conspiracy vs mainstream media entity frequency ratios
"""

import matplotlib.pyplot as plt


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
    (ner_df['lang_det'] == 'en')
    & ~ner_df['ner_group'].str.endswith(('_it', '_de'), na=False)
]
consp_stats_en = get_entity_stats(consp_en_df)

# Combine stats
combined = pd.DataFrame({
    'media_count': media_stats_en['count'],
    'media_freq': media_stats_en['freq'],
    'consp_count': consp_stats_en['count'],
    'consp_freq': consp_stats_en['freq']
}).fillna(0)

combined['ratio'] = combined['consp_freq'] / (combined['media_freq'] + 1e-10)

# Filter: only entities with at least 0.01% in media, then take top 12 by ratio
filtered = combined[
    (combined['media_freq'] >= 0.0001)
].sort_values('ratio', ascending=False).head(12)

# Prepare excluded entities: those with <0.01% in MSM, ranked by conspiracy freq, top 10
excluded = combined[
    (combined['media_freq'] < 0.0001) & (combined['consp_count'] > 0)
].sort_values('consp_count', ascending=False).head(10)
excluded_names = excluded.index.tolist()

# Barplot
entities = filtered.index.tolist()
ratios = filtered['ratio'].values

plt.figure(figsize=(max(10, len(entities) * 0.7), 6))
bars = plt.bar(range(len(entities)), ratios, color='skyblue', log=True)

plt.xticks(range(len(entities)), entities, rotation=45, ha='right', fontsize=10)
plt.ylabel('Relative Frequency Ratio (log scale)')
plt.xlabel('Entity')
plt.title('Entities with Highest Relative Frequency Ratio\n(English Conspiracy vs Mainstream Media)')

plt.tight_layout()
plt.subplots_adjust(top=1.05)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height * 1.05,
        f"{ratios[i]:.0f}",
        ha='center', va='bottom', fontsize=9, rotation=0
    )

if len(excluded_names) > 0:
    box_lines = [f"{i+1}. {name}" for i, name in enumerate(excluded_names)]
    box_text = r"$\bf{Excluded\ (<0.01\%\ in\ MSM)}$" + "\n" + "\n".join(box_lines)
    plt.gca().text(
        0.775, 0.97, box_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        va='top', ha='left',
        bbox=dict(boxstyle='square,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
    )

plt.show()
