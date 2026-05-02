"""
Semantic Axis Utility Functions

This module provides functions and classes for constructing and analyzing semantic axes
in word embedding hyperspace, following the approach described in Baerg & Jamieson (2024).

All functions assume that word vectors in the word2vec embedding space are normalized
to have a length (2-norm) of 1.

References:
    Baerg, N. R., & Jamieson, W. (2024). "Semantic Axes in Word Embeddings"
    (referred to as "B&J" in code comments)

For self-trained corpora, antonym strings should match KeyedVectors keys (typically
underscore tokens from preprocessing). Use helpers.load_w2v_token_to_canonical() only
for display labels, not for axis arithmetic.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from gensim.models import KeyedVectors


def anch2vec(
    antonym_pair: Tuple[str, str],
    word2vec_model: KeyedVectors,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate an anchor vector from an antonym tuple.
    
    Implements equation 3 from B&J paper (p.9): vec(a_i, z_i) = vec(a_i) - vec(z_i)
    where the first word (a_i) is the positive pole and the second word (z_i) is
    the negative pole.
    
    Args:
        antonym_pair: Tuple of (positive_word, negative_word), e.g., ('good', 'bad')
        word2vec_model: Loaded word2vec KeyedVectors model with normalized vectors
        normalize: If True, normalize the resulting anchor vector (not in original B&J
                   but useful for some applications)
        
    Returns:
        Anchor vector as numpy array (shape: [embedding_dim])
        
    Raises:
        KeyError: If either word is not in the model's vocabulary
        
    Example:
        >>> model = KeyedVectors.load_word2vec_format('path/to/model')
        >>> anchor = anch2vec(('good', 'bad'), model)
    """
    a_word, z_word = antonym_pair
    
    # Get word vectors (already normalized in word2vec)
    try:
        vec_a = word2vec_model[a_word]
        vec_z = word2vec_model[z_word]
    except KeyError as e:
        raise KeyError(f"Word not found in vocabulary: {e}")
    
    # Ensure vectors are normalized (they should be, but we verify)
    vec_a = vec_a / np.linalg.norm(vec_a)
    vec_z = vec_z / np.linalg.norm(vec_z)
    
    # Calculate anchor vector: a - z
    anchor_vector = vec_a - vec_z
    
    # Optionally normalize the anchor vector
    if normalize:
        norm = np.linalg.norm(anchor_vector)
        if norm > 0:
            anchor_vector = anchor_vector / norm
    
    return anchor_vector


def anch2conceptvec(
    antonym_pairs: List[Tuple[str, str]],
    word2vec_model: KeyedVectors
) -> np.ndarray:
    """
    Calculate a concept vector from multiple antonym tuples.
    
    Implements equation 3 from B&J paper (p.9):
    [[vec(a_1, z_1) + vec(a_2, z_2) + ... + vec(a_i, z_i)]]
    
    where [[v]] denotes normalization: [[v]] = v / ||v||_2
    
    Args:
        antonym_pairs: List of (positive_word, negative_word) tuples
        word2vec_model: Loaded word2vec KeyedVectors model
        
    Returns:
        Normalized concept vector as numpy array (shape: [embedding_dim])
        
    Raises:
        ValueError: If no valid antonym pairs are found or if concept vector is zero
        
    Example:
        >>> pairs = [('good', 'bad'), ('right', 'wrong'), ('truth', 'lie')]
        >>> concept_vec = anch2conceptvec(pairs, model)
    """
    if not antonym_pairs:
        raise ValueError("At least one antonym pair is required")
    
    # Calculate anchor vectors for each pair
    anchor_vectors = []
    failed_pairs = []
    
    for pair in antonym_pairs:
        try:
            # Get anchor vector (not normalized individually)
            anchor = anch2vec(pair, word2vec_model, normalize=False)
            anchor_vectors.append(anchor)
        except KeyError:
            failed_pairs.append(pair)
            continue
    
    if not anchor_vectors:
        raise ValueError(
            f"No valid antonym pairs found. Failed pairs: {failed_pairs}"
        )
    
    if failed_pairs:
        print(f"Warning: Could not process {len(failed_pairs)} antonym pairs: {failed_pairs}")
    
    # Sum all anchor vectors
    concept_vector = np.sum(anchor_vectors, axis=0)
    
    # Normalize the sum (as per [[v]] notation in B&J)
    norm = np.linalg.norm(concept_vector)
    if norm > 0:
        concept_vector = concept_vector / norm
    else:
        raise ValueError(
            "Concept vector has zero norm. This may indicate that the antonym "
            "pairs cancel each other out."
        )
    
    return concept_vector


def axis_parallelism(
    antonym_pairs: List[Tuple[str, str]],
    word2vec_model: KeyedVectors
) -> float:
    """
    Calculate the parallelism metric for a set of antonym pairs.
    
    Implements equation 8 from B&J paper (p.13):
    parallelism(S) = 1/(n(n-1)) * sum_{i=1}^{n} sum_{j=1, j≠i}^{n} sim(z_i - a_i, z_j - a_j)
    
    where sim() is the cosine similarity.
    
    This metric measures how parallel the antonym pair vectors are to each other,
    with values close to 1 indicating high parallelism (all pairs point in similar
    directions) and values close to -1 indicating anti-parallelism.
    
    Args:
        antonym_pairs: List of (positive_word, negative_word) tuples
        word2vec_model: Loaded word2vec KeyedVectors model
        
    Returns:
        Parallelism score (float in range [-1, 1])
        
    Raises:
        ValueError: If fewer than 2 valid pairs are found
        
    Example:
        >>> pairs = [('good', 'bad'), ('right', 'wrong'), ('truth', 'lie')]
        >>> score = axis_parallelism(pairs, model)
        >>> print(f"Parallelism: {score:.3f}")
    """
    if len(antonym_pairs) < 2:
        raise ValueError("At least 2 antonym pairs are required for parallelism metric")
    
    # Calculate anchor vectors for each pair
    anchor_vectors = []
    failed_pairs = []
    
    for pair in antonym_pairs:
        try:
            # Note: We want z - a (negative pole - positive pole) as per B&J equation
            # So we reverse the pair
            reversed_pair = (pair[1], pair[0])
            anchor = anch2vec(reversed_pair, word2vec_model, normalize=True)
            anchor_vectors.append(anchor)
        except KeyError:
            failed_pairs.append(pair)
            continue
    
    if len(anchor_vectors) < 2:
        raise ValueError(
            f"Need at least 2 valid pairs for parallelism. Failed pairs: {failed_pairs}"
        )
    
    if failed_pairs:
        print(f"Warning: Could not process {len(failed_pairs)} pairs: {failed_pairs}")
    
    n = len(anchor_vectors)
    
    # Calculate all pairwise similarities
    total_similarity = 0.0
    count = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Cosine similarity (both vectors are already normalized)
                similarity = np.dot(anchor_vectors[i], anchor_vectors[j])
                total_similarity += similarity
                count += 1
    
    # Average similarity
    parallelism_score = total_similarity / count if count > 0 else 0.0
    
    return parallelism_score


def pair_parallelism(
    antonym_pairs: List[Tuple[str, str]],
    word2vec_model: KeyedVectors
) -> Dict[Tuple[str, str], float]:
    """
    Calculate pair-specific parallelism for each antonym pair.
    
    Implements equation 16 from B&J paper (p.34):
    parallelism((z_i, a_i); S) = 1/(n-1) * sum_{j=1, j≠i}^{n} sim(z_i - a_i, z_j - a_j)
    
    This measures how well each individual pair aligns with the other pairs,
    useful for identifying outlier pairs that don't fit the overall axis.
    
    Args:
        antonym_pairs: List of (positive_word, negative_word) tuples
        word2vec_model: Loaded word2vec KeyedVectors model
        
    Returns:
        Dictionary mapping each antonym pair to its parallelism score
        
    Raises:
        ValueError: If fewer than 2 valid pairs are found
        
    Example:
        >>> pairs = [('good', 'bad'), ('right', 'wrong'), ('truth', 'lie')]
        >>> scores = pair_parallelism(pairs, model)
        >>> for pair, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        ...     print(f"{pair}: {score:.3f}")
    """
    if len(antonym_pairs) < 2:
        raise ValueError("At least 2 antonym pairs are required for pair parallelism")
    
    # Calculate anchor vectors for each pair
    anchor_vectors = []
    valid_pairs = []
    failed_pairs = []
    
    for pair in antonym_pairs:
        try:
            # Note: We want z - a (negative pole - positive pole) as per B&J
            reversed_pair = (pair[1], pair[0])
            anchor = anch2vec(reversed_pair, word2vec_model, normalize=True)
            anchor_vectors.append(anchor)
            valid_pairs.append(pair)
        except KeyError:
            failed_pairs.append(pair)
            continue
    
    if len(valid_pairs) < 2:
        raise ValueError(
            f"Need at least 2 valid pairs. Failed pairs: {failed_pairs}"
        )
    
    if failed_pairs:
        print(f"Warning: Could not process {len(failed_pairs)} pairs: {failed_pairs}")
    
    n = len(valid_pairs)
    
    # Calculate pair-specific parallelism for each pair
    pair_scores = {}
    
    for i, pair_i in enumerate(valid_pairs):
        # Calculate average similarity with all other pairs
        total_similarity = 0.0
        count = 0
        
        for j, pair_j in enumerate(valid_pairs):
            if i != j:
                # Cosine similarity
                similarity = np.dot(anchor_vectors[i], anchor_vectors[j])
                total_similarity += similarity
                count += 1
        
        # Average similarity for this pair
        pair_score = total_similarity / count if count > 0 else 0.0
        pair_scores[pair_i] = pair_score
    
    return pair_scores


def find_antonym(
    existing_pairs: List[Tuple[str, str]],
    word: str,
    candidate_antonyms: List[str],
    word2vec_model: KeyedVectors,
    return_top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find the best antonym(s) for a word based on parallelism with existing pairs.
    
    This function tests each candidate antonym by creating a temporary pair with the
    given word, then calculating how well that pair's parallelism score fits with
    the existing anchor pairs. The candidates are ranked by the resulting parallelism
    value.
    
    Args:
        existing_pairs: List of existing (positive, negative) antonym tuples that
                       define the semantic axis
        word: The word for which to find an antonym
        candidate_antonyms: List of potential antonym words to test
        word2vec_model: Loaded word2vec KeyedVectors model
        return_top_k: Number of top candidates to return (default: 5)
        
    Returns:
        List of (antonym, parallelism_score) tuples, sorted by score (highest first)
        
    Raises:
        ValueError: If word is not in vocabulary or no valid candidates found
        
    Example:
        >>> existing = [('good', 'bad'), ('right', 'wrong')]
        >>> candidates = ['falsehood', 'deception', 'fiction', 'error']
        >>> best = find_antonym(existing, 'truth', candidates, model, return_top_k=3)
        >>> print(f"Best antonym for 'truth': {best[0][0]} (score: {best[0][1]:.3f})")
    """
    # Check if word is in vocabulary
    if word not in word2vec_model:
        raise ValueError(f"Word '{word}' not found in vocabulary")
    
    # Filter candidates to those in vocabulary
    valid_candidates = [c for c in candidate_antonyms if c in word2vec_model]
    invalid_candidates = [c for c in candidate_antonyms if c not in word2vec_model]
    
    if not valid_candidates:
        raise ValueError(
            f"No valid candidate antonyms found in vocabulary. "
            f"Invalid: {invalid_candidates}"
        )
    
    if invalid_candidates:
        print(f"Warning: Skipping {len(invalid_candidates)} candidates not in vocabulary")
    
    # Score each candidate
    candidate_scores = []
    
    for candidate in valid_candidates:
        # Create temporary pair
        # We need to determine which position (positive or negative) the word should be in
        # Try both orientations and use the one with higher parallelism
        
        pair1 = (word, candidate)  # word as positive pole
        pair2 = (candidate, word)  # word as negative pole
        
        # Calculate parallelism with existing pairs for both orientations
        try:
            # Test pair1
            test_pairs_1 = existing_pairs + [pair1]
            score1 = axis_parallelism(test_pairs_1, word2vec_model)
        except (ValueError, KeyError):
            score1 = -np.inf
        
        try:
            # Test pair2
            test_pairs_2 = existing_pairs + [pair2]
            score2 = axis_parallelism(test_pairs_2, word2vec_model)
        except (ValueError, KeyError):
            score2 = -np.inf
        
        # Use the orientation with higher parallelism
        best_score = max(score1, score2)
        best_pair = pair1 if score1 >= score2 else pair2
        
        candidate_scores.append((candidate, best_score, best_pair))
    
    # Sort by score (descending)
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results (candidate, score)
    top_k = min(return_top_k, len(candidate_scores))
    results = [(cand, score) for cand, score, _ in candidate_scores[:top_k]]
    
    return results


class SemAxis:
    """
    Class for constructing and managing semantic axes in word embedding space.
    
    This class encapsulates the functionality for creating semantic axes from
    antonym pairs, calculating parallelism metrics, and managing the axis state.
    
    Attributes:
        antonym_pairs: List of (positive, negative) word tuples defining the axis
        word2vec_model: The word2vec model containing the embeddings
        concept_vector: The normalized concept vector representing the axis
        parallelism_score: Overall parallelism of the antonym pairs
        pair_scores: Dictionary of pair-specific parallelism scores
        
    Example:
        >>> model = KeyedVectors.load_word2vec_format('path/to/model')
        >>> pairs = [('good', 'bad'), ('right', 'wrong'), ('truth', 'lie')]
        >>> axis = SemAxis(pairs, model)
        >>> print(f"Axis parallelism: {axis.parallelism_score:.3f}")
        >>> print(f"Concept vector shape: {axis.concept_vector.shape}")
    """
    
    def __init__(
        self,
        antonym_pairs: List[Tuple[str, str]],
        word2vec_model: KeyedVectors,
        name: Optional[str] = None
    ):
        """
        Initialize a semantic axis from antonym pairs.
        
        Args:
            antonym_pairs: List of (positive_word, negative_word) tuples
            word2vec_model: Loaded word2vec KeyedVectors model
            name: Optional name for the axis (e.g., "transparency", "trust")
        """
        self.antonym_pairs = antonym_pairs
        self.word2vec_model = word2vec_model
        self.name = name or "semantic_axis"
        
        # Calculate concept vector
        self.concept_vector = anch2conceptvec(antonym_pairs, word2vec_model)
        
        # Calculate parallelism metrics
        try:
            self.parallelism_score = axis_parallelism(antonym_pairs, word2vec_model)
            self.pair_scores = pair_parallelism(antonym_pairs, word2vec_model)
        except ValueError:
            # If only one pair, parallelism is not defined
            self.parallelism_score = None
            self.pair_scores = None
    
    def get_anchor_vectors(self, normalize: bool = True) -> List[np.ndarray]:
        """
        Get anchor vectors for all antonym pairs.
        
        Args:
            normalize: Whether to normalize each anchor vector
            
        Returns:
            List of anchor vectors
        """
        vectors = []
        for pair in self.antonym_pairs:
            try:
                vec = anch2vec(pair, self.word2vec_model, normalize=normalize)
                vectors.append(vec)
            except KeyError:
                continue
        return vectors
    
    def find_best_antonym(
        self,
        word: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the best antonym(s) for a word to add to this axis.
        
        Args:
            word: Word to find antonym for
            candidates: List of candidate antonyms
            top_k: Number of top results to return
            
        Returns:
            List of (antonym, score) tuples
        """
        return find_antonym(
            self.antonym_pairs,
            word,
            candidates,
            self.word2vec_model,
            return_top_k=top_k
        )

    def find_antonyms_fullsearch(
        self,
        anchor: str,
        top_n: int = 10,
        refine_pool: int = 200,
    ) -> Tuple[
        List[Tuple[str, float]],
        List[Tuple[str, float, float]],
    ]:
        """
        Search the full vocabulary for antonym words on the **opposite pole** from
        ``anchor`` on this axis.

        First, ``anchor`` is placed along ``self.concept_vector`` (unit) via
        ``cos_side = dot(norm(anchor), concept)``. That indicates which side of the
        axis the word occupies (the same "positive" / "negative" sense as
        (positive, negative) pairs in ``self.antonym_pairs``).

        **Fast pass (vectorized):** For each candidate ``c``, form the new-pair
        difference that matches the existing (pos, neg) → ``pos - neg`` direction:

        * If ``anchor`` lies on the **positive** side (``cos_side >= 0``), we seek
          candidates on the **negative** pole: pair ``(anchor, c)`` and score
          ``cos(norm(anchor - c), concept)``.
        * If ``anchor`` lies on the **negative** side (``cos_side < 0``), we seek
          candidates on the **positive** pole: pair ``(c, anchor)`` and score
          ``cos(norm(c - anchor), concept)``.

        High positive scores mean the difference vector points along the axis, toward
        the **opposite** pole; wrong-side or misaligned candidates rank lower (often
        negative).

        **Refine:** Among the top ``refine_pool`` fast candidates, rank by
        ``axis_parallelism`` for **only** the matching orientation: ``pairs + (anchor, c)``
        if ``cos_side >= 0``, else ``pairs + (c, anchor)``.

        Args:
            anchor: The vocabulary item to pair (word whose opposite pole is sought).
            top_n: Number of (word, score) entries to return in each result list.
            refine_pool: How many fast-pass candidates receive exact ``axis_parallelism``.

        Returns:
            (fast_top, refined_top) where:

            - fast_top: ``top_n`` items ``(word, align)`` from the vectorized pass
              (signed cosine, higher is better).
            - refined_top: ``top_n`` items ``(word, parallelism, raw_score)``
              after refinement; ``raw_score`` is the ``axis_parallelism`` value
              for the correct extended pair (same as ``parallelism``).

        Raises:
            ValueError: If ``anchor`` is OOV, concept vector is zero, or no candidates.
        """
        kv = self.word2vec_model
        existing_pairs = list(self.antonym_pairs)

        concept = np.asarray(self.concept_vector, dtype=np.float64)
        cn = np.linalg.norm(concept)
        if cn == 0:
            raise ValueError("concept_vector has zero norm")
        concept = concept / cn

        if anchor not in kv:
            raise ValueError(f"Anchor {anchor!r} not in model vocabulary")

        used = {anchor}
        for pos, neg in existing_pairs:
            used.add(pos)
            used.add(neg)

        vocab_ids = np.array(
            [i for i, w in enumerate(kv.index_to_key) if w not in used],
            dtype=np.intp,
        )
        if vocab_ids.size == 0:
            raise ValueError("No candidate words after excluding axis vocabulary + anchor")

        V = np.asarray(kv.vectors[vocab_ids], dtype=np.float64)
        u = np.asarray(kv[anchor], dtype=np.float64)
        un = np.linalg.norm(u)
        if un == 0:
            raise ValueError("anchor vector has zero norm")
        u = u / un

        # Which pole does anchor occupy? (Same sense as (pos, neg) vs concept in anch2conceptvec.)
        cos_side = float(u @ concept)
        on_positive_pole = cos_side >= 0.0
        if on_positive_pole:
            # (anchor, c) = (pos, neg) → pos - neg = u - c
            D = u[None, :] - V
        else:
            # (c, anchor) = (pos, neg) → c - u
            D = V - u[None, :]

        norms = np.linalg.norm(D, axis=1, keepdims=True)
        nonzero = norms.ravel() > 0
        safe = np.where(norms > 0, norms, 1.0)
        D_hat = D / safe
        align_scores = (D_hat @ concept).ravel()
        align_scores = np.where(nonzero, align_scores, 0.0)

        k = min(top_n, align_scores.size)
        part = np.argpartition(-align_scores, kth=k - 1)[:k]
        order_fast = part[np.argsort(-align_scores[part])]
        fast_top = [
            (kv.index_to_key[int(vocab_ids[i])], float(align_scores[i]))
            for i in order_fast
        ]

        pool = min(refine_pool, align_scores.size)
        part_r = np.argpartition(-align_scores, kth=pool - 1)[:pool]
        refine_candidates = [kv.index_to_key[int(vocab_ids[j])] for j in part_r]

        refined: List[Tuple[str, float, float]] = []
        for cand in refine_candidates:
            try:
                if on_positive_pole:
                    pnew: Tuple[str, str] = (anchor, cand)
                else:
                    pnew = (cand, anchor)
                raw = axis_parallelism(existing_pairs + [pnew], kv)
                refined.append((cand, raw, raw))
            except (ValueError, KeyError):
                continue

        refined.sort(key=lambda x: -x[1])
        refined_top = refined[:top_n]

        return fast_top, refined_top

    def add_pair(self, new_pair: Tuple[str, str]) -> 'SemAxis':
        """
        Create a new SemAxis with an additional antonym pair.
        
        Args:
            new_pair: (positive_word, negative_word) tuple to add
            
        Returns:
            New SemAxis instance with the added pair
        """
        new_pairs = self.antonym_pairs + [new_pair]
        return SemAxis(new_pairs, self.word2vec_model, self.name)
    
    def remove_pair(self, pair: Tuple[str, str]) -> 'SemAxis':
        """
        Create a new SemAxis with a pair removed.
        
        Args:
            pair: Antonym pair to remove
            
        Returns:
            New SemAxis instance without the specified pair
            
        Raises:
            ValueError: If pair is not in the current pairs list
        """
        if pair not in self.antonym_pairs:
            raise ValueError(f"Pair {pair} not found in current pairs")
        
        new_pairs = [p for p in self.antonym_pairs if p != pair]
        if not new_pairs:
            raise ValueError("Cannot remove the last pair from an axis")
        
        return SemAxis(new_pairs, self.word2vec_model, self.name)
    
    def get_worst_pairs(self, n: int = 3) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get the n pairs with lowest parallelism scores.
        
        Useful for identifying pairs that don't fit well with the axis and might
        be candidates for removal.
        
        Args:
            n: Number of worst pairs to return
            
        Returns:
            List of (pair, score) tuples sorted by score (lowest first)
        """
        if self.pair_scores is None:
            return []
        
        sorted_pairs = sorted(self.pair_scores.items(), key=lambda x: x[1])
        return sorted_pairs[:n]
    
    def get_best_pairs(self, n: int = 3) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get the n pairs with highest parallelism scores.
        
        Args:
            n: Number of best pairs to return
            
        Returns:
            List of (pair, score) tuples sorted by score (highest first)
        """
        if self.pair_scores is None:
            return []
        
        sorted_pairs = sorted(self.pair_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:n]
    
    def summary(self) -> str:
        """
        Get a summary of the semantic axis.
        
        Returns:
            String with axis statistics
        """
        lines = [
            f"Semantic Axis: {self.name}",
            f"Number of antonym pairs: {len(self.antonym_pairs)}",
            f"Concept vector dimension: {len(self.concept_vector)}",
        ]
        
        if self.parallelism_score is not None:
            lines.append(f"Overall parallelism: {self.parallelism_score:.3f}")
            
            if self.pair_scores:
                best_pairs = self.get_best_pairs(3)
                worst_pairs = self.get_worst_pairs(3)
                
                lines.append("\nBest pairs (highest parallelism):")
                for pair, score in best_pairs:
                    lines.append(f"  {pair}: {score:.3f}")
                
                lines.append("\nWorst pairs (lowest parallelism):")
                for pair, score in worst_pairs:
                    lines.append(f"  {pair}: {score:.3f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"SemAxis(name='{self.name}', n_pairs={len(self.antonym_pairs)})"


def find_antonyms_fullsearch(
    axis: SemAxis,
    anchor: str,
    top_n: int = 10,
    refine_pool: int = 200,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float, float]]]:
    """
    Full-vocabulary antonym search for ``anchor`` relative to ``axis``.
    See :meth:`SemAxis.find_antonyms_fullsearch`.
    """
    return axis.find_antonyms_fullsearch(anchor, top_n=top_n, refine_pool=refine_pool)
