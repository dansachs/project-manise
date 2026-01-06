#!/usr/bin/env python3
"""
Test Embeddings with Google Generative AI
Visualizes semantic separation between Ambonese and Indonesian using Google's text-embedding-004 model.
"""

import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_parallel_sentences(file_path: Path) -> List[Tuple[str, str]]:
    """
    Load parallel sentences from JSONL file.
    Returns list of (ambonese, indonesian) tuples.
    """
    logger = logging.getLogger(__name__)
    pairs = []
    
    if not file_path.exists():
        logger.warning(f"Input file not found: {file_path}")
        logger.info("Creating dummy data file...")
        create_dummy_data(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    ambonese = item.get('ambonese', '').strip()
                    indonesian = item.get('indonesian', '').strip()
                    if ambonese and indonesian:
                        pairs.append((ambonese, indonesian))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        
        logger.info(f"Loaded {len(pairs)} parallel sentence pairs")
        return pairs
    
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise


def create_dummy_data(file_path: Path) -> None:
    """Create dummy data file if input file is missing."""
    dummy_data = [
        {"ambonese": "beta pi ka pasar", "indonesian": "saya pergi ke pasar"},
        {"ambonese": "dia pung mama ada di ruma", "indonesian": "ibunya ada di rumah"},
        {"ambonese": "katong samua su makan", "indonesian": "kami semua sudah makan"},
        {"ambonese": "jang baku mara", "indonesian": "jangan saling marah"},
        {"ambonese": "beta seng tau", "indonesian": "saya tidak tahu"},
    ]
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in dummy_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created dummy data file: {file_path}")


def flatten_sentences(pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[int]]:
    """
    Flatten sentence pairs into single list with labels.
    Returns (texts, labels) where:
    - texts: [ambonese_1, indonesian_1, ambonese_2, indonesian_2, ...]
    - labels: [0, 1, 0, 1, ...] where 0=Ambonese, 1=Indonesian
    """
    texts = []
    labels = []
    
    for ambonese, indonesian in pairs:
        texts.append(ambonese)
        labels.append(0)  # 0 = Ambonese
        texts.append(indonesian)
        labels.append(1)  # 1 = Indonesian
    
    return texts, labels


def get_embeddings_batched(
    client: genai.Client,
    texts: List[str],
    model: str = "models/text-embedding-004",
    batch_size: int = 50,
    sleep_seconds: int = 10
) -> np.ndarray:
    """
    Get embeddings for texts using batch processing with rate limiting.
    
    Args:
        client: Google GenAI client
        texts: List of text strings to embed
        model: Model name for embeddings
        batch_size: Number of texts per batch
        sleep_seconds: Sleep time between batches (rate limiting)
    
    Returns:
        NumPy array of embeddings
    """
    logger = logging.getLogger(__name__)
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(texts)} texts in {total_batches} batches (batch size: {batch_size})")
    
    with tqdm(total=len(texts), desc="Getting embeddings") as pbar:
        for batch_idx in range(0, len(texts), batch_size):
            batch_num = (batch_idx // batch_size) + 1
            batch = texts[batch_idx:batch_idx + batch_size]
            
            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Retry logic for API calls
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Call embedding API
                    # Note: google-genai Client API for embeddings
                    response = client.models.embed_content(
                        model=model,
                        contents=batch
                    )
                    
                    # Extract embeddings from response
                    # The response structure varies, handle multiple cases
                    batch_embeddings = None
                    
                    if hasattr(response, 'embeddings'):
                        batch_embeddings = response.embeddings
                    elif hasattr(response, 'values'):
                        # Single embedding returned
                        batch_embeddings = [response.values]
                    elif isinstance(response, dict):
                        if 'embeddings' in response:
                            batch_embeddings = response['embeddings']
                        elif 'values' in response:
                            batch_embeddings = [response['values']]
                        elif 'embedding' in response:
                            batch_embeddings = [response['embedding']]
                    elif isinstance(response, list):
                        batch_embeddings = response
                    
                    # Handle different embedding formats
                    if batch_embeddings:
                        processed_embeddings = []
                        for emb in batch_embeddings:
                            if isinstance(emb, dict):
                                # Extract values from dict
                                processed_embeddings.append(
                                    emb.get('values', emb.get('embedding', emb.get('embedding_values', [])))
                                )
                            elif isinstance(emb, (list, np.ndarray)):
                                processed_embeddings.append(list(emb))
                            elif hasattr(emb, 'values'):
                                processed_embeddings.append(list(emb.values))
                            else:
                                # Try to convert to list
                                processed_embeddings.append(list(emb))
                        
                        all_embeddings.extend(processed_embeddings)
                        success = True
                        logger.info(f"Batch {batch_num}/{total_batches} completed successfully ({len(processed_embeddings)} embeddings)")
                        break
                    else:
                        raise ValueError("Could not extract embeddings from API response")
                
                except Exception as e:
                    logger.warning(f"Batch {batch_num} attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                    else:
                        logger.error(f"Batch {batch_num} failed after {max_retries} attempts, skipping")
                        # Add zero embeddings as placeholder to maintain indexing
                        for _ in batch:
                            all_embeddings.append([0.0] * 768)  # Default embedding dimension
            
            pbar.update(len(batch))
            
            # Rate limiting: sleep between batches (except after last batch)
            if batch_idx + batch_size < len(texts):
                logger.debug(f"Sleeping {sleep_seconds} seconds for rate limiting...")
                time.sleep(sleep_seconds)
    
    logger.info(f"Retrieved {len(all_embeddings)} embeddings")
    return np.array(all_embeddings)


def apply_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Apply PCA to reduce embedding dimensions.
    
    Args:
        embeddings: NumPy array of embeddings
        n_components: Number of principal components
    
    Returns:
        Reduced embeddings array
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Applying PCA to reduce from {embeddings.shape[1]} to {n_components} dimensions")
    
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance: {explained_variance.sum():.2%} "
                f"(PC1: {explained_variance[0]:.2%}, PC2: {explained_variance[1]:.2%})")
    
    return reduced_embeddings


def visualize_embeddings(
    reduced_embeddings: np.ndarray,
    labels: List[int],
    output_path: Path,
    title: str = "Semantic Separation between Ambonese and Indonesian"
) -> None:
    """
    Create visualization of embeddings with connecting lines.
    
    Args:
        reduced_embeddings: 2D embeddings from PCA
        labels: List of labels (0=Ambonese, 1=Indonesian)
        output_path: Path to save the plot
        title: Plot title
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating visualization...")
    
    labels_array = np.array(labels)
    ambonese_indices = labels_array == 0
    indonesian_indices = labels_array == 1
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot Ambonese points (red)
    ambonese_points = reduced_embeddings[ambonese_indices]
    ax.scatter(
        ambonese_points[:, 0],
        ambonese_points[:, 1],
        color='red',
        label='Ambonese',
        alpha=0.6,
        s=50
    )
    
    # Plot Indonesian points (blue)
    indonesian_points = reduced_embeddings[indonesian_indices]
    ax.scatter(
        indonesian_points[:, 0],
        indonesian_points[:, 1],
        color='blue',
        label='Indonesian',
        alpha=0.6,
        s=50
    )
    
    # Draw connecting lines between parallel sentences
    num_pairs = len(labels) // 2
    for i in range(num_pairs):
        ambonese_idx = i * 2
        indonesian_idx = i * 2 + 1
        
        if ambonese_idx < len(reduced_embeddings) and indonesian_idx < len(reduced_embeddings):
            ax.plot(
                [reduced_embeddings[ambonese_idx, 0], reduced_embeddings[indonesian_idx, 0]],
                [reduced_embeddings[ambonese_idx, 1], reduced_embeddings[indonesian_idx, 1]],
                color='gray',
                linestyle='--',
                alpha=0.3,
                linewidth=0.5
            )
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {output_path}")
    
    plt.close()


def main():
    """Main function."""
    # Get project root (parent of evaluation directory or current directory)
    script_dir = Path(__file__).parent
    if script_dir.name == 'evaluation':
        project_root = script_dir.parent
    else:
        project_root = Path.cwd()
    
    parser = argparse.ArgumentParser(
        description="Visualize semantic separation between Ambonese and Indonesian using Google embeddings"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=project_root / 'outputs' / 'parallel_sentences_100.jsonl',
        help='Path to input JSONL file (default: outputs/parallel_sentences_100.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=project_root / 'google_embedding_clusters.png',
        help='Path to output visualization (default: google_embedding_clusters.png)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/text-embedding-004',
        help='Embedding model name (default: models/text-embedding-004)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for embeddings (default: 50)'
    )
    parser.add_argument(
        '--sleep',
        type=int,
        default=10,
        help='Sleep seconds between batches for rate limiting (default: 10)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google API key (overrides environment variable)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        if args.api_key:
            api_key = args.api_key
        else:
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            logger.error("API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
                        "or use --api-key argument")
            sys.exit(1)
        
        # Initialize client
        logger.info("Initializing Google GenAI client...")
        client = genai.Client(api_key=api_key)
        
        # Load data
        logger.info(f"Loading parallel sentences from: {args.input}")
        pairs = load_parallel_sentences(args.input)
        
        if not pairs:
            logger.error("No sentence pairs found. Exiting.")
            sys.exit(1)
        
        # Flatten sentences
        texts, labels = flatten_sentences(pairs)
        logger.info(f"Flattened to {len(texts)} texts ({sum(1 for l in labels if l == 0)} Ambonese, "
                   f"{sum(1 for l in labels if l == 1)} Indonesian)")
        
        # Get embeddings
        embeddings = get_embeddings_batched(
            client=client,
            texts=texts,
            model=args.model,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep
        )
        
        if embeddings.size == 0:
            logger.error("Failed to retrieve embeddings. Exiting.")
            sys.exit(1)
        
        # Apply PCA
        reduced_embeddings = apply_pca(embeddings)
        
        # Visualize
        visualize_embeddings(
            reduced_embeddings=reduced_embeddings,
            labels=labels,
            output_path=args.output
        )
        
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"  Input pairs: {len(pairs)}")
        logger.info(f"  Total texts: {len(texts)}")
        logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  Output visualization: {args.output}")
        logger.info("=" * 60)
    
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

