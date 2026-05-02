"""
Word2Vec training script using gensim on CPU.

This script trains a Word2Vec model using skip-gram architecture on text data
that has been preprocessed using the text_cleaning module. The training data
should be in a CSV/Parquet file with a 'cleaned_text' column containing
space-separated tokens, where each row represents a separate message.

Message boundaries are preserved: each row in the dataframe is treated as a
separate sentence, so the context window will not span across different messages.

Usage:
    python train_w2v_cpu.py --input data/clean_final_en.csv --output models/my_run_name
    python train_w2v_cpu.py --input data.parquet --output models/my_run_name --vector-size 300 --epochs 10
    
    # With logging
    python train_w2v_cpu.py --input data.csv --output models/w2v --verbose
"""

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Iterator, List, Dict, Any

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for training progress."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )
    return logging.getLogger(__name__)


# =============================================================================
# Training Callbacks
# =============================================================================

class TrainingCallback(CallbackAny2Vec):
    """Callback to log training progress and track metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.epoch = 0
        self.logger = logger
        self.losses = []  # Track loss per epoch
        self.epoch_times = []  # Track time per epoch
        self._epoch_start = None
    
    def on_epoch_begin(self, model):
        import time
        self._epoch_start = time.time()
        self.logger.info(f"Epoch {self.epoch + 1} starting...")
    
    def on_epoch_end(self, model):
        import time
        self.epoch += 1
        
        # Track epoch time
        if self._epoch_start:
            epoch_time = time.time() - self._epoch_start
            self.epoch_times.append(epoch_time)
            self.logger.info(f"Epoch {self.epoch} completed in {epoch_time:.1f}s")
        else:
            self.logger.info(f"Epoch {self.epoch} completed.")
        
        # Track cumulative loss (gensim accumulates loss)
        if hasattr(model, 'get_latest_training_loss'):
            cumulative_loss = model.get_latest_training_loss()
            self.losses.append(cumulative_loss)
    
    def on_train_begin(self, model):
        self.logger.info("Training started.")
        # Enable loss computation
        model.running_training_loss = 0.0
    
    def on_train_end(self, model):
        self.logger.info("Training completed.")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return collected training metrics."""
        # Convert cumulative losses to per-epoch losses
        per_epoch_losses = []
        for i, loss in enumerate(self.losses):
            if i == 0:
                per_epoch_losses.append(loss)
            else:
                per_epoch_losses.append(loss - self.losses[i-1])
        
        return {
            'cumulative_losses': self.losses,
            'per_epoch_losses': per_epoch_losses,
            'epoch_times': self.epoch_times,
            'total_epochs': self.epoch
        }


# =============================================================================
# Data Loading
# =============================================================================

class SentenceIterator:
    """
    Memory-efficient iterator over sentences from a DataFrame.
    
    Each row in the 'cleaned_text' column is treated as a separate sentence,
    preserving message boundaries for word2vec training.
    
    Supports multiple iterations (required by gensim for vocabulary building
    and training phases).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = 'cleaned_text',
        min_sentence_length: int = 2
    ):
        """
        Initialize the sentence iterator.
        
        Args:
            df: DataFrame containing the text data
            text_column: Name of the column with cleaned text
            min_sentence_length: Minimum tokens required for inclusion
        """
        self.df = df
        self.text_column = text_column
        self.min_sentence_length = min_sentence_length
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame. "
                           f"Available columns: {list(df.columns)}")
    
    def __iter__(self) -> Iterator[List[str]]:
        """Iterate over sentences, yielding token lists."""
        for text in self.df[self.text_column]:
            if pd.isna(text) or not isinstance(text, str):
                continue
            
            tokens = text.strip().split()
            if len(tokens) >= self.min_sentence_length:
                yield tokens
    
    def __len__(self) -> int:
        """Return approximate sentence count (may be less due to filtering)."""
        return len(self.df)


def load_data(input_path: str, text_column: str = 'cleaned_text') -> pd.DataFrame:
    """
    Load data from CSV or Parquet file.
    
    Args:
        input_path: Path to input file (.csv or .parquet)
        text_column: Name of the text column to verify existence
        
    Returns:
        Loaded DataFrame
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load based on file extension
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path)
    elif path.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .parquet")
    
    # Verify text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. "
                        f"Available columns: {list(df.columns)}")
    
    return df


# =============================================================================
# Model Training
# =============================================================================

def train_word2vec(
    sentences: SentenceIterator,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = None,
    epochs: int = 5,
    negative: int = 5,
    seed: int = 42,
    callbacks: List[CallbackAny2Vec] = None,
    logger: logging.Logger = None
) -> Word2Vec:
    """
    Train a Word2Vec model using skip-gram architecture.
    
    Args:
        sentences: Iterable of sentences (lists of tokens)
        vector_size: Dimensionality of word vectors (default: 100)
        window: Context window size (default: 5)
        min_count: Minimum word frequency to be included (default: 5)
        workers: Number of worker threads (default: all CPUs)
        epochs: Number of training iterations (default: 5)
        negative: Number of negative samples (default: 5)
        seed: Random seed for reproducibility (default: 42)
        callbacks: List of training callbacks
        logger: Logger for progress messages
        
    Returns:
        Trained Word2Vec model
    """
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    if logger:
        logger.info(f"Training parameters:")
        logger.info(f"  - Architecture: Skip-gram (sg=1)")
        logger.info(f"  - Vector size: {vector_size}")
        logger.info(f"  - Window size: {window}")
        logger.info(f"  - Min count: {min_count}")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Negative samples: {negative}")
        logger.info(f"  - Workers: {workers}")
        logger.info(f"  - Random seed: {seed}")
    
    # Train the model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,  # Skip-gram architecture (as specified)
        negative=negative,
        seed=seed,
        callbacks=callbacks if callbacks else []
    )
    
    return model


def save_model(
    model: Word2Vec,
    output_path: str,
    save_word2vec_format: bool = True,
    logger: logging.Logger = None
) -> None:
    """
    Save the trained model in multiple formats under a run directory.
    
    ``output_path`` is the directory for this run (e.g. ``models/1_pilot_w2v_model``).
    The directory is created if missing. Artifacts are written as
    ``{dir_name}/{dir_name}.model``, ``.bin``, and optionally ``.vec``.
    
    Saves:
    1. Full gensim model (.model) - can be loaded and retrained
    2. Word vectors in word2vec text format (.vec) - compatible with other tools
    3. Word vectors in word2vec binary format (.bin)
    
    Args:
        model: Trained Word2Vec model
        output_path: Path to the model run directory (final path component = run name)
        save_word2vec_format: Whether to also save in word2vec text format
        logger: Logger for progress messages
    """
    run_dir = Path(output_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = run_dir.name
    base = run_dir / stem

    # Save full gensim model (can be loaded and retrained)
    model_path = base.with_suffix('.model')
    model.save(str(model_path))
    if logger:
        logger.info(f"Saved gensim model to: {model_path}")

    # Save word vectors in word2vec text format (for compatibility)
    if save_word2vec_format:
        vec_path = base.with_suffix('.vec')
        model.wv.save_word2vec_format(str(vec_path), binary=False)
        if logger:
            logger.info(f"Saved word vectors (text format) to: {vec_path}")

    # Save binary format as well (smaller file, faster loading)
    bin_path = base.with_suffix('.bin')
    model.wv.save_word2vec_format(str(bin_path), binary=True)
    if logger:
        logger.info(f"Saved word vectors (binary format) to: {bin_path}")


# =============================================================================
# Training Overview Plots
# =============================================================================

def create_training_plots(
    model: Word2Vec,
    training_metrics: Dict[str, Any],
    output_path: str,
    n_words_frequency: int = 50,
    n_words_pca: int = 100,
    logger: logging.Logger = None
) -> None:
    """
    Create and save training overview plots.
    
    Generates a figure with 4 subplots:
    1. Training loss over epochs
    2. Word frequency distribution (top N words)
    3. Embedding vector norms distribution
    4. 2D PCA projection of top word embeddings
    
    Args:
        model: Trained Word2Vec model
        training_metrics: Dictionary with training metrics from callback
        output_path: Base path for saving plots (will add _training_overview.png)
        n_words_frequency: Number of top words to show in frequency plot
        n_words_pca: Number of top words to include in PCA visualization
        logger: Logger for progress messages
    """
    if not PLOTTING_AVAILABLE:
        if logger:
            logger.warning("Plotting unavailable. Install matplotlib and scikit-learn: "
                          "pip install matplotlib scikit-learn")
        return
    
    if logger:
        logger.info("Generating training overview plots...")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Word2Vec Training Overview', fontsize=14, fontweight='bold')
    
    # 1. Training Loss Plot (top-left)
    ax1 = axes[0, 0]
    per_epoch_losses = training_metrics.get('per_epoch_losses', [])
    if per_epoch_losses:
        epochs = range(1, len(per_epoch_losses) + 1)
        ax1.plot(epochs, per_epoch_losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss per Epoch')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(list(epochs))
    else:
        ax1.text(0.5, 0.5, 'Loss data not available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Training Loss per Epoch')
    
    # 2. Word Frequency Distribution (top-right)
    ax2 = axes[0, 1]
    vocab = model.wv.index_to_key[:n_words_frequency]
    frequencies = [model.wv.get_vecattr(word, 'count') for word in vocab]
    
    ax2.barh(range(len(vocab)), frequencies, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(vocab)))
    ax2.set_yticklabels(vocab, fontsize=8)
    ax2.invert_yaxis()  # Most frequent at top
    ax2.set_xlabel('Frequency')
    ax2.set_title(f'Top {n_words_frequency} Words by Frequency')
    ax2.set_xscale('log')
    
    # 3. Embedding Vector Norms Distribution (bottom-left)
    ax3 = axes[1, 0]
    all_vectors = model.wv.vectors
    norms = np.linalg.norm(all_vectors, axis=1)
    
    ax3.hist(norms, bins=50, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axvline(np.mean(norms), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(norms):.2f}')
    ax3.axvline(np.median(norms), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {np.median(norms):.2f}')
    ax3.set_xlabel('Vector Norm (L2)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Embedding Vector Norms')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 2D PCA Projection of Top Words (bottom-right)
    ax4 = axes[1, 1]
    top_words = model.wv.index_to_key[:n_words_pca]
    word_vectors = np.array([model.wv[word] for word in top_words])
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(word_vectors)
    
    # Plot points
    ax4.scatter(coords[:, 0], coords[:, 1], c='purple', alpha=0.6, s=30)
    
    # Annotate top 20 words (to avoid clutter)
    n_annotate = min(20, len(top_words))
    for i in range(n_annotate):
        ax4.annotate(top_words[i], (coords[i, 0], coords[i, 1]), 
                    fontsize=8, alpha=0.8,
                    xytext=(3, 3), textcoords='offset points')
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax4.set_title(f'PCA of Top {n_words_pca} Word Embeddings')
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to plots directory
    output = Path(output_path)
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{output.stem}_training_overview.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Saved training overview plot to: {plot_path}")


def print_training_summary(
    model: Word2Vec,
    training_metrics: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Print a text-based training summary to the console.
    
    Args:
        model: Trained Word2Vec model
        training_metrics: Dictionary with training metrics from callback
        logger: Logger for output
    """
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    # Model statistics
    vocab_size = len(model.wv)
    vector_size = model.wv.vector_size
    
    logger.info(f"Vocabulary size: {vocab_size:,} words")
    logger.info(f"Vector dimensions: {vector_size}")
    logger.info(f"Total parameters: {vocab_size * vector_size:,}")
    
    # Training time
    epoch_times = training_metrics.get('epoch_times', [])
    if epoch_times:
        total_time = sum(epoch_times)
        avg_time = np.mean(epoch_times)
        logger.info(f"Total training time: {total_time:.1f}s")
        logger.info(f"Average time per epoch: {avg_time:.1f}s")
    
    # Loss statistics
    per_epoch_losses = training_metrics.get('per_epoch_losses', [])
    if per_epoch_losses:
        logger.info(f"Initial loss (epoch 1): {per_epoch_losses[0]:,.0f}")
        logger.info(f"Final loss (epoch {len(per_epoch_losses)}): {per_epoch_losses[-1]:,.0f}")
        if len(per_epoch_losses) > 1:
            reduction = (per_epoch_losses[0] - per_epoch_losses[-1]) / per_epoch_losses[0] * 100
            logger.info(f"Loss reduction: {reduction:.1f}%")
    
    # Vector statistics
    all_vectors = model.wv.vectors
    norms = np.linalg.norm(all_vectors, axis=1)
    logger.info(f"Vector norm - Mean: {np.mean(norms):.3f}, Std: {np.std(norms):.3f}")
    
    # Top words
    logger.info("-" * 60)
    logger.info("Top 10 most frequent words:")
    for i, word in enumerate(model.wv.index_to_key[:10], 1):
        freq = model.wv.get_vecattr(word, 'count')
        logger.info(f"  {i:2d}. {word:<20} (freq: {freq:,})")
    
    logger.info("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for Word2Vec training."""
    parser = argparse.ArgumentParser(
        description='Train Word2Vec model using skip-gram architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (creates ../models/my_w2v_run/my_w2v_run.{model,bin,vec})
  python train_w2v_cpu.py --input ../data/clean_final_en.csv --output ../models/my_w2v_run

  # Training with custom parameters and plots
  python train_w2v_cpu.py \\
      --input ../data/clean_final_en.csv \\
      --output ../models/w2v_300d \\
      --vector-size 300 \\
      --window 10 \\
      --epochs 10 \\
      --min-count 10 \\
      --plot \\
      --verbose

  # From parquet file
  python train_w2v_cpu.py --input data.parquet --output models/my_run --workers 8
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input file (CSV or Parquet) with cleaned text'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for this run (e.g. models/my_run); saves my_run.{model,bin,vec} inside it'
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--vector-size',
        type=int,
        default=100,
        help='Dimensionality of word vectors (default: 100)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='Context window size (default: 5)'
    )
    parser.add_argument(
        '--min-count',
        type=int,
        default=5,
        help='Minimum word frequency threshold (default: 5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--negative',
        type=int,
        default=5,
        help='Number of negative samples (default: 5)'
    )
    
    # Technical parameters
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker threads (default: all available CPUs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Data parameters
    parser.add_argument(
        '--text-column',
        type=str,
        default='cleaned_text',
        help='Name of the text column (default: cleaned_text)'
    )
    parser.add_argument(
        '--min-sentence-length',
        type=int,
        default=2,
        help='Minimum tokens per sentence (default: 2)'
    )
    
    # Output options
    parser.add_argument(
        '--no-word2vec-format',
        action='store_true',
        help='Skip saving in word2vec text format'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate training overview plots'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing training summary'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load data
    logger.info(f"Loading data from: {args.input}")
    df = load_data(args.input, args.text_column)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Create sentence iterator
    sentences = SentenceIterator(
        df,
        text_column=args.text_column,
        min_sentence_length=args.min_sentence_length
    )
    
    # Setup callbacks
    training_callback = TrainingCallback(logger)
    callbacks = [training_callback]
    
    # Train model
    logger.info("Starting Word2Vec training...")
    model = train_word2vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        negative=args.negative,
        seed=args.seed,
        callbacks=callbacks,
        logger=logger
    )
    
    # Get training metrics
    training_metrics = training_callback.get_metrics()
    
    # Log vocabulary statistics
    vocab_size = len(model.wv)
    logger.info(f"Vocabulary size: {vocab_size:,} words")
    
    # Save model
    save_model(
        model,
        args.output,
        save_word2vec_format=not args.no_word2vec_format,
        logger=logger
    )
    
    # Print training summary
    if not args.no_summary:
        print_training_summary(model, training_metrics, logger)
    
    # Generate plots
    if args.plot:
        create_training_plots(
            model=model,
            training_metrics=training_metrics,
            output_path=args.output,
            logger=logger
        )
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
