# Imports
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import os
from text_cleaning import create_text_cleaner

# Data paths
INPUT_PATH = '/project/ssd-stu-research/ploertscher/thesis_code/text-processing/data/full_filt/4_posts_og_content_en.csv'
OUTPUT_PATH = '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/data/clean_final_en.csv'
ENTITIES_PATH = '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/pca/modules/entities/entities_mapping_2.json'
STOPWORDS_PATH = '/project/ssd-stu-research/ploertscher/thesis_code/ideological_resonance_thesis/w2v/preprocessing/stopwords_en.py'

# Text column name in the CSV
TEXT_COLUMN = 'content'
OUTPUT_COLUMN = 'cleaned_text'

# Number of parallel processes (set to None to use all available cores)
N_PROCESSES = None  # Will auto-detect from SLURM or use all cores

def process_chunk(args):
    """
    Worker function to process a chunk of the dataframe.
    Must be defined at module level for multiprocessing.
    
    Args:
        args: Tuple of (chunk_df, chunk_id, entities_path, stopwords_path)
    
    Returns:
        DataFrame with cleaned text column added
    """
    chunk_df, chunk_id, entities_path, stopwords_path = args
    
    # Create cleaner with pre-loaded resources
    cleaner = create_text_cleaner(
        entities_path=entities_path,
        stopwords_path=stopwords_path,
        remove_emoji=True,
        min_token_length=2
    )
    
    # Apply cleaning to the text column
    chunk_df = chunk_df.copy()
    chunk_df[OUTPUT_COLUMN] = chunk_df[TEXT_COLUMN].apply(cleaner)
    
    # Print progress for this chunk
    print(f"[INFO] Chunk {chunk_id} completed: {len(chunk_df):,} rows processed")
    
    return chunk_df


def clean_dataframe_parallel(df, n_processes=None, min_sentence_tokens=3):
    """
    Clean text data in parallel using multiprocessing.
    
    Args:
        df: Input DataFrame
        n_processes: Number of parallel processes (None = auto-detect)
        min_sentence_tokens: Minimum number of tokens required to keep a row (default: 3)
    
    Returns:
        DataFrame with cleaned text column added (rows with < min_sentence_tokens removed)
    """
    # Determine number of processes
    if n_processes is None:
        # Try to get from SLURM environment
        n_processes = int(os.environ.get('SLURM_NTASKS', 0))
        if n_processes == 0:
            # Fall back to CPU count
            n_processes = cpu_count()
    
    print(f"[INFO] Using {n_processes} parallel processes")
    
    # Split dataframe into chunks (one per process)
    chunk_size = len(df) // n_processes
    if chunk_size == 0:
        chunk_size = len(df)
        n_processes = 1
    
    chunks = []
    for i in range(n_processes):
        start_idx = i * chunk_size
        if i == n_processes - 1:
            # Last chunk gets any remaining rows
            end_idx = len(df)
        else:
            end_idx = start_idx + chunk_size
        
        chunk = df.iloc[start_idx:end_idx]
        chunks.append((chunk, i + 1, ENTITIES_PATH, STOPWORDS_PATH))
    
    print(f"[INFO] Split data into {len(chunks)} chunks of ~{chunk_size:,} rows each")
    
    # Process chunks in parallel
    if n_processes == 1:
        # Single process - no need for Pool overhead
        print("[INFO] Single process mode (no parallel overhead)")
        results = [process_chunk(chunks[0])]
    else:
        # Multi-process mode
        print(f"[INFO] Starting parallel processing...")
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_chunk, chunks)
    
    # Combine results
    print(f"[INFO] Combining {len(results)} processed chunks...")
    df_cleaned = pd.concat(results, ignore_index=True)
    
    # Filter rows with fewer than min_sentence_tokens tokens
    if min_sentence_tokens > 0:
        initial_count = len(df_cleaned)
        
        # Count tokens in each cleaned text (split by whitespace)
        token_counts = df_cleaned[OUTPUT_COLUMN].str.split().str.len().fillna(0).astype(int)
        
        # Filter to keep only rows with >= min_sentence_tokens
        df_cleaned = df_cleaned[token_counts >= min_sentence_tokens]
        
        dropped = initial_count - len(df_cleaned)
        if dropped > 0:
            print(f"[INFO] Filtered out {dropped:,} rows with < {min_sentence_tokens} tokens ({dropped/initial_count*100:.1f}% of data)")
            print(f"[INFO] Retained {len(df_cleaned):,} rows ({len(df_cleaned)/initial_count*100:.1f}% of data)")
    
    return df_cleaned


def main():
    print(f"[INFO] Starting preprocessing pipeline for English data")
    print(f"[INFO] Input: {INPUT_PATH}")
    print(f"[INFO] Output: {OUTPUT_PATH}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory created/verified: {output_dir}")
    
    # Load the data
    print(f"[INFO] Loading data from CSV...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {len(df):,} rows")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Check if text column exists
    if TEXT_COLUMN not in df.columns:
        print(f"[ERROR] Column '{TEXT_COLUMN}' not found. Available columns: {list(df.columns)}")
        return 1
    
    # Clean text using parallel processing
    # This preserves all original columns (including post IDs) and adds a cleaned_text column
    # Remove emojis and stopwords as specified
    print(f"[INFO] Cleaning text data with parallel processing...")
    df_cleaned = clean_dataframe_parallel(df, n_processes=N_PROCESSES)
    
    # Save to CSV (preserves all columns including IDs)
    print(f"[INFO] Saving cleaned data to CSV...")
    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    
    # Print statistics
    non_empty = (df_cleaned[OUTPUT_COLUMN].str.len() > 0).sum()
    empty = len(df_cleaned) - non_empty
    
    print(f"[SUCCESS] Preprocessing complete!")
    print(f"[INFO] Total rows: {len(df_cleaned):,}")
    print(f"[INFO] Rows with cleaned text: {non_empty:,}")
    print(f"[INFO] Rows with empty cleaned text: {empty:,}")
    print(f"[INFO] Output file: {OUTPUT_PATH}")
    print(f"[INFO] All original columns (including IDs) preserved")
    
    return 0

if __name__ == '__main__':
    exit(main())
