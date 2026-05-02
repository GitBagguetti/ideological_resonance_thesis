"""
Text cleaning module for word2vec preprocessing.

This module provides functions to clean and preprocess text data for word2vec training,
with support for entity normalization, stopword removal, and emoji handling.

MESSAGE BOUNDARIES:
    Word2vec learns word embeddings based on context windows. To prevent the context
    window from spanning across different messages (which would create spurious 
    co-occurrences), this module treats each message as a separate "sentence".
    
    For word2vec training, use:
    - get_sentences_for_word2vec(): Returns List[List[str]] format for gensim
    - get_tokenized_corpus(): Returns iterator of token lists (memory efficient)
    - CLI with --output-format tokens: Outputs one tokenized message per line

Usage:
    CLI:
        python text_cleaning.py input.csv output.csv --entities-path entities.json --stopwords-path stopwords.txt --remove-emoji
        python text_cleaning.py input.csv output.csv --keep-emoji
    
    Python (for word2vec training):
        from text_cleaning import get_sentences_for_word2vec
        from gensim.models import Word2Vec
        
        # Get sentences with proper message boundaries
        sentences = get_sentences_for_word2vec(
            df, 
            text_column='content',
            entities_path='entities.json'
        )
        
        # Train word2vec - context windows won't cross message boundaries
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=5)
    
    Python (basic cleaning):
        from text_cleaning import create_text_cleaner, clean_dataframe
        
        # Stopwords can be loaded from .py module or .txt file
        cleaner = create_text_cleaner(entities_path='entities.json', stopwords_path='stopwords_en.py')
        cleaned = cleaner("Some text with Donald Trump mentioned")
        
        df = clean_dataframe(df, text_column='content', entities_path='entities.json')
"""

import re
import json
import argparse
import importlib.util
from pathlib import Path
from typing import Optional, Callable, Set, Dict, List, Union
import pandas as pd


# =============================================================================
# Emoji Handling
# =============================================================================

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    print("[WARNING] emoji library not available. Using fallback regex for emoji removal.")


# Fallback emoji pattern for when emoji library is not available
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002300-\U000023FF"  # misc technical
    "\U00002B50-\U00002B55"  # stars
    "\U0001F004-\U0001F0CF"  # game symbols
    "\U0001F7E0-\U0001F7EB"  # colored circles
    "\U00002194-\U00002199"  # arrows
    "\U000021A9-\U000021AA"  # arrows
    "\U0000203C-\U0000203C"  # exclamation marks
    "\U00002049-\U00002049"  # question marks
    "\U00002122-\U00002122"  # trademark
    "\U00002139-\U00002139"  # info
    "\U0001F004"  # mahjong
    "\U0001F0CF"  # joker
    "]+",
    flags=re.UNICODE
)


def remove_emojis(text: str) -> str:
    """Remove emojis from text using emoji library or fallback regex."""
    if EMOJI_AVAILABLE:
        return emoji.replace_emoji(text, replace=' ')
    else:
        return EMOJI_PATTERN.sub(' ', text)


# =============================================================================
# Precompiled Regex Patterns
# =============================================================================

# URL patterns (including common shorteners and Telegram links)
URL_PATTERN = re.compile(
    r'https?://[^\s<>"\']+|'
    r'www\.[^\s<>"\']+|'
    r't\.me/[^\s<>"\']+|'
    r'bit\.ly/[^\s<>"\']+|'
    r'cutt\.ly/[^\s<>"\']+|'
    r'rumble\.com/[^\s<>"\']+',
    re.IGNORECASE
)

# Telegram-specific patterns
MENTION_PATTERN = re.compile(r'@\w+', re.UNICODE)
HASHTAG_PATTERN = re.compile(r'#\w+', re.UNICODE)

# Pattern for cleaning special characters (preserves underscores for entity tokens)
# We'll use this after entity replacement
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s]', re.UNICODE)

# Pattern for multiple whitespaces
WHITESPACE_PATTERN = re.compile(r'\s+')

# Pattern for number-only tokens
NUMBERS_ONLY_PATTERN = re.compile(r'^\d+$')

# Pattern for newlines (often encoded as \n in the data)
NEWLINE_PATTERN = re.compile(r'\\n|\n|\r')


# =============================================================================
# Resource Loading Functions
# =============================================================================

def load_stopwords(stopwords_source: Union[str, Set[str], None]) -> Set[str]:
    """
    Load stopwords from various sources.
    
    Supports:
    - Python module (.py file) with STOP_WORDS set (e.g., stopwords_en.py)
    - Text file (.txt) with one word per line
    - Direct set of stopwords
    - None (returns empty set)
    
    Args:
        stopwords_source: Path to stopwords file (.py or .txt), 
                         a set of stopwords, or None
        
    Returns:
        Set of lowercase stopwords
    """
    # Handle None or empty input
    if stopwords_source is None:
        return set()
    
    # If already a set, return it directly
    if isinstance(stopwords_source, set):
        print(f"[INFO] Using provided stopwords set ({len(stopwords_source)} words)")
        return stopwords_source
    
    # Handle file path
    path = Path(stopwords_source)
    
    if not path.exists():
        print(f"[WARNING] Stopwords file not found: {stopwords_source}")
        return set()
    
    # Load from Python module (.py file)
    if path.suffix == '.py':
        try:
            spec = importlib.util.spec_from_file_location("stopwords_module", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for STOP_WORDS or STOPWORDS attribute
            if hasattr(module, 'STOP_WORDS'):
                stopwords = set(word.lower() for word in module.STOP_WORDS)
            elif hasattr(module, 'STOPWORDS'):
                stopwords = set(word.lower() for word in module.STOPWORDS)
            else:
                print(f"[WARNING] No STOP_WORDS or STOPWORDS found in {stopwords_source}")
                return set()
            
            print(f"[INFO] Loaded {len(stopwords)} stopwords from {stopwords_source}")
            return stopwords
        except Exception as e:
            print(f"[ERROR] Failed to load stopwords from {stopwords_source}: {e}")
            return set()
    
    # Load from text file (.txt or other)
    stopwords = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith('#'):  # Allow comments
                stopwords.add(word)
    
    print(f"[INFO] Loaded {len(stopwords)} stopwords from {stopwords_source}")
    return stopwords


def build_entity_mapping(entities_path: str) -> Dict[str, str]:
    """
    Build a mapping from entity variations to canonical names.
    
    The canonical names are converted to lowercase with spaces/hyphens replaced
    by underscores, so they appear as single tokens for word2vec.
    
    Args:
        entities_path: Path to the entities JSON file
        
    Returns:
        Dictionary mapping lowercase variations to canonical tokens
    """
    mapping = {}
    
    path = Path(entities_path)
    if not path.exists():
        print(f"[WARNING] Entities file not found: {entities_path}")
        return mapping
    
    with open(path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    for category, entities_dict in entities.items():
        for entity_key, entity_data in entities_dict.items():
            canonical = entity_data.get('canonical_name', '')
            if not canonical:
                continue
                
            # Convert canonical name to a single token:
            # - lowercase
            # - replace spaces and hyphens with underscores
            # - remove other special characters
            canonical_token = canonical.lower()
            canonical_token = re.sub(r'[\s\-]+', '_', canonical_token)
            canonical_token = re.sub(r'[^\w]', '', canonical_token)
            
            # Map all variations to this canonical token
            variations = entity_data.get('variations', [])
            for variation in variations:
                # Store lowercase version as key for case-insensitive matching
                variation_lower = variation.lower()
                
                # Only add if not already mapped (prefer first occurrence)
                if variation_lower not in mapping:
                    mapping[variation_lower] = canonical_token
    
    print(f"[INFO] Loaded {len(mapping)} entity variations from {entities_path}")
    return mapping


def build_entity_pattern(entity_mapping: Dict[str, str]) -> Optional[re.Pattern]:
    """
    Build a compiled regex pattern for matching entity variations.
    
    Patterns are sorted by length (longest first) to ensure that longer 
    matches like "Hillary Clinton" take precedence over shorter ones like "Clinton".
    
    Args:
        entity_mapping: Dictionary mapping variations to canonical names
        
    Returns:
        Compiled regex pattern or None if mapping is empty
    """
    if not entity_mapping:
        return None
    
    # Sort by length (longest first) to match longer phrases before shorter ones
    sorted_variations = sorted(entity_mapping.keys(), key=len, reverse=True)
    
    # Build pattern parts with proper escaping
    # Use word boundaries, but handle multi-word phrases correctly
    pattern_parts = []
    for variation in sorted_variations:
        escaped = re.escape(variation)
        pattern_parts.append(escaped)
    
    # Create pattern with word boundaries
    # Use (?<!\w) and (?!\w) instead of \b for better Unicode support
    pattern = r'(?<![a-zA-Z0-9_])(' + '|'.join(pattern_parts) + r')(?![a-zA-Z0-9_])'
    
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


# =============================================================================
# Text Cleaning Functions
# =============================================================================

def clean_text(
    text: str,
    entity_pattern: Optional[re.Pattern] = None,
    entity_mapping: Optional[Dict[str, str]] = None,
    stopwords: Optional[Set[str]] = None,
    remove_emoji: bool = True,
    min_token_length: int = 2
) -> str:
    """
    Clean a single text string for word2vec training.
    
    Processing order (optimized for word2vec):
    1. Handle newlines and basic normalization
    2. Replace entity variations with canonical names (case-insensitive)
    3. Remove URLs
    4. Remove @mentions and #hashtags
    5. Handle emojis based on parameter
    6. Remove special characters (preserve underscores for entities)
    7. Lowercase everything
    8. Tokenize
    9. Remove stopwords
    10. Remove number-only tokens and short tokens
    
    Args:
        text: Input text string
        entity_pattern: Compiled regex pattern for entity matching
        entity_mapping: Dictionary mapping lowercase variations to canonical names
        stopwords: Set of stopwords to remove
        remove_emoji: Whether to remove emojis (default: True)
        min_token_length: Minimum token length to keep (default: 2)
    
    Returns:
        Cleaned text string with tokens separated by spaces
    """
    # Handle None/NaN values
    if not isinstance(text, str):
        if pd.isna(text):
            return ''
        text = str(text)
    
    if not text.strip():
        return ''
    
    # 1. Normalize newlines and whitespace
    text = NEWLINE_PATTERN.sub(' ', text)
    
    # 2. Replace entity variations with canonical names (before lowercasing!)
    # This allows case-insensitive matching but produces consistent output
    if entity_pattern and entity_mapping:
        def replace_entity(match):
            matched_text = match.group(1).lower()
            return entity_mapping.get(matched_text, matched_text)
        
        text = entity_pattern.sub(replace_entity, text)
    
    # 3. Remove URLs (do this early to avoid partial matching issues)
    text = URL_PATTERN.sub(' ', text)
    
    # 4. Remove @mentions and #hashtags
    text = MENTION_PATTERN.sub(' ', text)
    text = HASHTAG_PATTERN.sub(' ', text)
    
    # 5. Handle emojis
    if remove_emoji:
        text = remove_emojis(text)
    
    # 6. Remove special characters (keep alphanumeric, underscores, spaces)
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)
    
    # 7. Lowercase everything
    text = text.lower()
    
    # 8. Normalize whitespace and tokenize
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    tokens = text.split()
    
    # 9. Filter tokens
    # Convert stopwords to set for O(1) lookup (handles frozenset, set, list, or None)
    stopwords_lookup = set(stopwords) if stopwords else set()
    
    filtered_tokens = []
    for token in tokens:
        # Skip stopwords - explicit check with set membership
        if token in stopwords_lookup:
            continue
        
        # Skip number-only tokens
        if NUMBERS_ONLY_PATTERN.match(token):
            continue
        
        # Skip tokens that are too short
        if len(token) < min_token_length:
            continue
        
        filtered_tokens.append(token)
    
    return ' '.join(filtered_tokens)


def create_text_cleaner(
    entities_path: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    remove_emoji: bool = True,
    min_token_length: int = 2
) -> Callable[[str], str]:
    """
    Create a text cleaning function with pre-loaded resources.
    
    This factory function loads entity mappings and stopwords once,
    then returns a cleaning function that can be efficiently applied
    to many text strings.
    
    Args:
        entities_path: Path to entities JSON file (optional)
        stopwords_path: Path to stopwords file (optional)
        remove_emoji: Whether to remove emojis (default: True)
        min_token_length: Minimum token length to keep (default: 2)
    
    Returns:
        A function that takes a string and returns cleaned text
    
    Example:
        >>> cleaner = create_text_cleaner(
        ...     entities_path='entities.json',
        ...     stopwords_path='stopwords.txt',
        ...     remove_emoji=True
        ... )
        >>> cleaned = cleaner("President Trump announced...")
    """
    # Load resources once
    entity_mapping = None
    entity_pattern = None
    stopwords_set = None
    
    if entities_path:
        entity_mapping = build_entity_mapping(entities_path)
        entity_pattern = build_entity_pattern(entity_mapping)
    
    if stopwords_path:
        loaded_stopwords = load_stopwords(stopwords_path)
        # Convert to frozenset to ensure immutability and proper closure capture
        stopwords_set = frozenset(loaded_stopwords) if loaded_stopwords else None
    
    # Use default argument binding to ensure proper closure capture
    def cleaner(
        text: str,
        _entity_pattern=entity_pattern,
        _entity_mapping=entity_mapping,
        _stopwords=stopwords_set,
        _remove_emoji=remove_emoji,
        _min_token_length=min_token_length
    ) -> str:
        return clean_text(
            text,
            entity_pattern=_entity_pattern,
            entity_mapping=_entity_mapping,
            stopwords=_stopwords,
            remove_emoji=_remove_emoji,
            min_token_length=_min_token_length
        )
    
    return cleaner


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str,
    output_column: str = 'cleaned_text',
    entities_path: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    remove_emoji: bool = True,
    min_token_length: int = 2,
    chunk_size: int = 100000,
    drop_empty: bool = False,
    min_sentence_tokens: int = 3,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Clean text data in a pandas DataFrame.
    
    Processes in chunks for memory efficiency with large DataFrames.
    Logs progress for monitoring long-running operations.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text to clean
        output_column: Name of column for cleaned text output (default: 'cleaned_text')
        entities_path: Path to entities JSON file (optional)
        stopwords_path: Path to stopwords file (optional)
        remove_emoji: Whether to remove emojis (default: True)
        min_token_length: Minimum token length to keep (default: 2)
        chunk_size: Number of rows to process at once for progress logging
        drop_empty: Whether to drop rows with empty cleaned text (default: False)
        min_sentence_tokens: Minimum number of tokens required to keep a row (default: 3)
        inplace: Whether to modify the DataFrame in place (default: False)
    
    Returns:
        DataFrame with cleaned text column added (rows with < min_sentence_tokens removed)
    
    Example:
        >>> df = pd.read_csv('messages.csv')
        >>> df = clean_dataframe(
        ...     df, 
        ...     text_column='content',
        ...     entities_path='entities.json',
        ...     stopwords_path='stopwords.txt',
        ...     min_sentence_tokens=3  # Remove messages with < 3 tokens
        ... )
    """
    if not inplace:
        df = df.copy()
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    total_rows = len(df)
    print(f"[INFO] Starting text cleaning for {total_rows:,} rows...")
    
    # Create cleaner function with pre-loaded resources
    cleaner = create_text_cleaner(
        entities_path=entities_path,
        stopwords_path=stopwords_path,
        remove_emoji=remove_emoji,
        min_token_length=min_token_length
    )
    
    # Process with progress logging for large DataFrames
    if total_rows > chunk_size:
        results = []
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk = df.iloc[i:end_idx]
            chunk_result = chunk[text_column].apply(cleaner)
            results.append(chunk_result)
            
            progress = (end_idx / total_rows) * 100
            print(f"[INFO] Progress: {end_idx:,}/{total_rows:,} rows ({progress:.1f}%)")
        
        df[output_column] = pd.concat(results)
    else:
        df[output_column] = df[text_column].apply(cleaner)
    
    # Optionally drop empty results
    if drop_empty:
        initial_count = len(df)
        df = df[df[output_column].str.len() > 0]
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"[INFO] Dropped {dropped:,} rows with empty cleaned text")
    
    # Filter rows with fewer than min_sentence_tokens tokens
    if min_sentence_tokens > 0:
        initial_count = len(df)
        
        # Count tokens in each cleaned text (split by whitespace)
        token_counts = df[output_column].str.split().str.len().fillna(0).astype(int)
        
        # Filter to keep only rows with >= min_sentence_tokens
        df = df[token_counts >= min_sentence_tokens]
        
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"[INFO] Filtered out {dropped:,} rows with < {min_sentence_tokens} tokens ({dropped/initial_count*100:.1f}% of data)")
            print(f"[INFO] Retained {len(df):,} rows ({len(df)/initial_count*100:.1f}% of data)")
    
    print(f"[INFO] Text cleaning complete. Output column: '{output_column}'")
    return df


def clean_text_series(
    series: pd.Series,
    entities_path: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    remove_emoji: bool = True,
    min_token_length: int = 2
) -> pd.Series:
    """
    Clean a pandas Series of text data.
    
    Convenience function for cleaning a single Series without a full DataFrame.
    
    Args:
        series: pandas Series containing text data
        entities_path: Path to entities JSON file (optional)
        stopwords_path: Path to stopwords file (optional)
        remove_emoji: Whether to remove emojis (default: True)
        min_token_length: Minimum token length to keep (default: 2)
    
    Returns:
        Series with cleaned text
    """
    cleaner = create_text_cleaner(
        entities_path=entities_path,
        stopwords_path=stopwords_path,
        remove_emoji=remove_emoji,
        min_token_length=min_token_length
    )
    
    return series.apply(cleaner)


def get_tokens(
    text: str,
    entities_path: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    remove_emoji: bool = True,
    min_token_length: int = 2
) -> List[str]:
    """
    Clean text and return as a list of tokens.
    
    Useful for direct input to word2vec training which often expects
    lists of tokens rather than space-separated strings.
    
    Args:
        text: Input text string
        entities_path: Path to entities JSON file (optional)
        stopwords_path: Path to stopwords file (optional)
        remove_emoji: Whether to remove emojis (default: True)
        min_token_length: Minimum token length to keep (default: 2)
    
    Returns:
        List of cleaned tokens
    """
    cleaner = create_text_cleaner(
        entities_path=entities_path,
        stopwords_path=stopwords_path,
        remove_emoji=remove_emoji,
        min_token_length=min_token_length
    )
    
    cleaned = cleaner(text)
    return cleaned.split() if cleaned else []


# =============================================================================
# Word2Vec Training Functions (Message Boundary Aware)
# =============================================================================

def get_sentences_for_word2vec(
    df: pd.DataFrame,
    text_column: str,
    entities_path: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    remove_emoji: bool = True,
    min_token_length: int = 2,
    min_sentence_length: int = 2
) -> List[List[str]]:
    """
    Get cleaned sentences for word2vec training with proper message boundaries.
    
    Each message in the DataFrame becomes a separate "sentence" (list of tokens).
    This ensures that word2vec's context window does not span across different
    messages, which would create spurious word co-occurrences.
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text
        entities_path: Path to entities JSON file (optional)
        stopwords_path: Path to stopwords file (optional)
        remove_emoji: Whether to remove emojis (default: True)
        min_token_length: Minimum token length to keep (default: 2)
        min_sentence_length: Minimum number of tokens for a sentence to be included (default: 2)
    
    Returns:
        List of sentences, where each sentence is a list of tokens.
        Ready for direct use with gensim's Word2Vec.
    
    Example:
        >>> from gensim.models import Word2Vec
        >>> sentences = get_sentences_for_word2vec(df, 'content', entities_path='entities.json')
        >>> model = Word2Vec(sentences, vector_size=100, window=5, min_count=5)
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    print(f"[INFO] Preparing {len(df):,} messages for word2vec training...")
    
    # Create cleaner function with pre-loaded resources
    cleaner = create_text_cleaner(
        entities_path=entities_path,
        stopwords_path=stopwords_path,
        remove_emoji=remove_emoji,
        min_token_length=min_token_length
    )
    
    sentences = []
    for text in df[text_column]:
        cleaned = cleaner(text)
        if cleaned:
            tokens = cleaned.split()
            # Only include sentences with enough tokens
            if len(tokens) >= min_sentence_length:
                sentences.append(tokens)
    
    print(f"[INFO] Prepared {len(sentences):,} sentences (messages with {min_sentence_length}+ tokens)")
    return sentences


class TokenizedCorpus:
    """
    Memory-efficient iterator over tokenized messages for word2vec training.
    
    This class allows processing large datasets without loading all sentences
    into memory at once. It properly respects message boundaries by yielding
    one tokenized message at a time.
    
    Supports both DataFrame input and file-based input.
    
    Example:
        >>> corpus = TokenizedCorpus(df, 'content', entities_path='entities.json')
        >>> model = Word2Vec(corpus, vector_size=100, window=5, min_count=5)
        
        >>> # Or from a file (one message per line)
        >>> corpus = TokenizedCorpus.from_file('messages.txt', entities_path='entities.json')
        >>> model = Word2Vec(corpus, vector_size=100, window=5, min_count=5)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str,
        entities_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        remove_emoji: bool = True,
        min_token_length: int = 2,
        min_sentence_length: int = 2
    ):
        """
        Initialize the corpus iterator.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            entities_path: Path to entities JSON file (optional)
            stopwords_path: Path to stopwords file (optional)
            remove_emoji: Whether to remove emojis (default: True)
            min_token_length: Minimum token length to keep (default: 2)
            min_sentence_length: Minimum tokens for sentence inclusion (default: 2)
        """
        self.df = df
        self.text_column = text_column
        self.min_sentence_length = min_sentence_length
        
        # Pre-load resources once
        self.cleaner = create_text_cleaner(
            entities_path=entities_path,
            stopwords_path=stopwords_path,
            remove_emoji=remove_emoji,
            min_token_length=min_token_length
        )
    
    def __iter__(self):
        """Iterate over messages, yielding one tokenized sentence per message."""
        for text in self.df[self.text_column]:
            cleaned = self.cleaner(text)
            if cleaned:
                tokens = cleaned.split()
                if len(tokens) >= self.min_sentence_length:
                    yield tokens
    
    def __len__(self):
        """Return approximate count (actual may be less due to filtering)."""
        return len(self.df)
    
    @classmethod
    def from_file(
        cls,
        file_path: str,
        entities_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        remove_emoji: bool = True,
        min_token_length: int = 2,
        min_sentence_length: int = 2
    ) -> 'TokenizedCorpusFromFile':
        """
        Create a corpus iterator from a text file (one message per line).
        
        Args:
            file_path: Path to text file with one message per line
            entities_path: Path to entities JSON file (optional)
            stopwords_path: Path to stopwords file (optional)
            remove_emoji: Whether to remove emojis (default: True)
            min_token_length: Minimum token length to keep (default: 2)
            min_sentence_length: Minimum tokens for sentence inclusion (default: 2)
        
        Returns:
            TokenizedCorpusFromFile iterator
        """
        return TokenizedCorpusFromFile(
            file_path=file_path,
            entities_path=entities_path,
            stopwords_path=stopwords_path,
            remove_emoji=remove_emoji,
            min_token_length=min_token_length,
            min_sentence_length=min_sentence_length
        )


class TokenizedCorpusFromFile:
    """
    Memory-efficient iterator over tokenized messages from a file.
    
    Reads the file line by line, treating each line as a separate message.
    This ensures proper message boundary handling for word2vec training.
    """
    
    def __init__(
        self,
        file_path: str,
        entities_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        remove_emoji: bool = True,
        min_token_length: int = 2,
        min_sentence_length: int = 2
    ):
        self.file_path = file_path
        self.min_sentence_length = min_sentence_length
        
        # Pre-load resources once
        self.cleaner = create_text_cleaner(
            entities_path=entities_path,
            stopwords_path=stopwords_path,
            remove_emoji=remove_emoji,
            min_token_length=min_token_length
        )
    
    def __iter__(self):
        """Iterate over file lines, yielding one tokenized sentence per line."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned = self.cleaner(line.strip())
                if cleaned:
                    tokens = cleaned.split()
                    if len(tokens) >= self.min_sentence_length:
                        yield tokens


def save_corpus_for_word2vec(
    df: pd.DataFrame,
    text_column: str,
    output_path: str,
    entities_path: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    remove_emoji: bool = True,
    min_token_length: int = 2,
    min_sentence_length: int = 2
) -> int:
    """
    Save cleaned corpus to a file in word2vec-ready format.
    
    Output format: one sentence per line, tokens separated by spaces.
    Each line represents one message, preserving message boundaries.
    
    This file can be used directly with gensim's LineSentence:
        >>> from gensim.models import Word2Vec
        >>> from gensim.models.word2vec import LineSentence
        >>> sentences = LineSentence('corpus.txt')
        >>> model = Word2Vec(sentences, vector_size=100, window=5, min_count=5)
    
    Args:
        df: Input DataFrame
        text_column: Name of column containing text
        output_path: Path to output file
        entities_path: Path to entities JSON file (optional)
        stopwords_path: Path to stopwords file (optional)
        remove_emoji: Whether to remove emojis (default: True)
        min_token_length: Minimum token length to keep (default: 2)
        min_sentence_length: Minimum tokens for sentence inclusion (default: 2)
    
    Returns:
        Number of sentences written
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    print(f"[INFO] Saving corpus for word2vec training...")
    
    # Create cleaner function with pre-loaded resources
    cleaner = create_text_cleaner(
        entities_path=entities_path,
        stopwords_path=stopwords_path,
        remove_emoji=remove_emoji,
        min_token_length=min_token_length
    )
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    sentence_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(df[text_column]):
            cleaned = cleaner(text)
            if cleaned:
                tokens = cleaned.split()
                if len(tokens) >= min_sentence_length:
                    f.write(cleaned + '\n')
                    sentence_count += 1
            
            if (i + 1) % 100000 == 0:
                print(f"[INFO] Progress: {i + 1:,}/{len(df):,} messages processed")
    
    print(f"[INFO] Saved {sentence_count:,} sentences to {output_path}")
    print(f"[INFO] Use with gensim: LineSentence('{output_path}')")
    return sentence_count


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point for text cleaning."""
    parser = argparse.ArgumentParser(
        description='Clean text data for word2vec training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean a CSV file with entity normalization
  python text_cleaning.py input.csv output.csv --text-column content --entities-path entities.json

  # Clean a TXT file (one document per line) keeping emojis
  python text_cleaning.py input.txt output.txt --keep-emoji

  # OUTPUT FOR WORD2VEC TRAINING (recommended):
  # Creates a corpus file with one sentence per line (message boundaries preserved)
  python text_cleaning.py data.csv corpus.txt \\
      --text-column message \\
      --entities-path entities.json \\
      --stopwords-path stopwords.txt \\
      --word2vec-format \\
      --min-sentence-length 3

  # Then use in Python:
  #   from gensim.models import Word2Vec
  #   from gensim.models.word2vec import LineSentence
  #   model = Word2Vec(LineSentence('corpus.txt'), vector_size=100, window=5)

  # Full pipeline with all options
  python text_cleaning.py data.csv cleaned.csv \\
      --text-column message \\
      --entities-path entities_mapping.json \\
      --stopwords-path stopwords.txt \\
      --remove-emoji \\
      --min-token-length 2 \\
      --drop-empty
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input file (CSV or TXT)'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output file'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of text column for CSV files (default: "text")'
    )
    parser.add_argument(
        '--output-column',
        type=str,
        default='cleaned_text',
        help='Name of output column for CSV files (default: "cleaned_text")'
    )
    parser.add_argument(
        '--entities-path',
        type=str,
        default=None,
        help='Path to entities mapping JSON file'
    )
    parser.add_argument(
        '--stopwords-path',
        type=str,
        default=None,
        help='Path to stopwords file (one word per line)'
    )
    
    # Emoji handling - mutually exclusive options
    emoji_group = parser.add_mutually_exclusive_group()
    emoji_group.add_argument(
        '--remove-emoji',
        action='store_true',
        dest='remove_emoji',
        help='Remove emojis from text (default behavior)'
    )
    emoji_group.add_argument(
        '--keep-emoji',
        action='store_false',
        dest='remove_emoji',
        help='Keep emojis in text'
    )
    parser.set_defaults(remove_emoji=True)
    
    parser.add_argument(
        '--min-token-length',
        type=int,
        default=2,
        help='Minimum token length to keep (default: 2)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Chunk size for processing large files (default: 100000)'
    )
    parser.add_argument(
        '--drop-empty',
        action='store_true',
        help='Drop rows with empty cleaned text (CSV only)'
    )
    parser.add_argument(
        '--output-tokens-only',
        action='store_true',
        help='For CSV: output only the cleaned text column'
    )
    parser.add_argument(
        '--word2vec-format',
        action='store_true',
        help='Output in word2vec training format (one sentence per line, for use with gensim LineSentence)'
    )
    parser.add_argument(
        '--min-sentence-length',
        type=int,
        default=2,
        help='Minimum number of tokens for a sentence to be included (default: 2)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process based on file type
    if input_path.suffix.lower() == '.csv':
        print(f"[INFO] Processing CSV file: {input_path}")
        
        # Read CSV
        df = pd.read_csv(input_path)
        print(f"[INFO] Loaded {len(df):,} rows")
        
        if args.text_column not in df.columns:
            print(f"[ERROR] Column '{args.text_column}' not found. Available columns: {list(df.columns)}")
            return 1
        
        # Word2vec format: one sentence per line (preserves message boundaries)
        if args.word2vec_format:
            sentence_count = save_corpus_for_word2vec(
                df,
                text_column=args.text_column,
                output_path=str(output_path),
                entities_path=args.entities_path,
                stopwords_path=args.stopwords_path,
                remove_emoji=args.remove_emoji,
                min_token_length=args.min_token_length,
                min_sentence_length=args.min_sentence_length
            )
            print(f"[INFO] Word2vec corpus ready: {sentence_count:,} sentences")
        else:
            # Standard DataFrame cleaning
            df = clean_dataframe(
                df,
                text_column=args.text_column,
                output_column=args.output_column,
                entities_path=args.entities_path,
                stopwords_path=args.stopwords_path,
                remove_emoji=args.remove_emoji,
                min_token_length=args.min_token_length,
                chunk_size=args.chunk_size,
                drop_empty=args.drop_empty
            )
            
            # Save output
            if args.output_tokens_only:
                df[[args.output_column]].to_csv(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
            
            print(f"[INFO] Saved {len(df):,} rows to {output_path}")
        
    elif input_path.suffix.lower() == '.txt':
        print(f"[INFO] Processing TXT file: {input_path}")
        
        # Read text file (one document per line)
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"[INFO] Loaded {len(lines):,} lines")
        
        # Create cleaner
        cleaner = create_text_cleaner(
            entities_path=args.entities_path,
            stopwords_path=args.stopwords_path,
            remove_emoji=args.remove_emoji,
            min_token_length=args.min_token_length
        )
        
        # Clean each line with progress (each line = one message = one sentence)
        cleaned_lines = []
        for i, line in enumerate(lines):
            cleaned = cleaner(line.strip())
            if cleaned:
                tokens = cleaned.split()
                # Apply min_sentence_length filter for word2vec format
                if args.word2vec_format and len(tokens) < args.min_sentence_length:
                    continue
                cleaned_lines.append(cleaned)
            elif not args.drop_empty and not args.word2vec_format:
                cleaned_lines.append('')
            
            if (i + 1) % 100000 == 0:
                print(f"[INFO] Progress: {i + 1:,}/{len(lines):,} lines")
        
        # Save output (one sentence per line, preserving message boundaries)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')
        
        print(f"[INFO] Saved {len(cleaned_lines):,} lines to {output_path}")
        if args.word2vec_format:
            print(f"[INFO] Word2vec corpus ready. Use with gensim: LineSentence('{output_path}')")
        
    else:
        print(f"[ERROR] Unsupported file type: {input_path.suffix}. Use .csv or .txt")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
