#!/usr/bin/env python3
"""
Script 5: Extract Parallel Ambonese-Indonesian Sentences
Extracts sentence pairs from the cleaned dictionary JSON file and randomly selects
100 pairs to create a parallel corpus in JSONL format.
RUN THIS SCRIPT FIFTH (after 4_correct_ocr_typos.py)
"""

import json
import random
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm


def setup_logging(output_dir: Path, debug: bool = False, quiet: bool = False) -> Path:
    """Set up logging to file and console following project patterns."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if debug else (logging.ERROR if quiet else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file


def load_json_file(file_path: Path) -> Dict:
    """Load and parse the JSON file."""
    logging.info(f"Loading JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded JSON file with {len(data.get('entries', []))} entries")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        raise


def extract_sentence_pairs(data: Dict) -> List[Tuple[str, str]]:
    """Extract all valid sentence pairs from dictionary entries."""
    pairs = []
    entries = data.get('entries', [])
    
    logging.info(f"Extracting sentence pairs from {len(entries)} entries...")
    
    for entry in tqdm(entries, desc="Extracting pairs"):
        meanings = entry.get('meanings', [])
        for meaning in meanings:
            ambonese = meaning.get('ambonese_example') or ''
            indonesian = meaning.get('indonesian_translation') or ''
            
            # Strip whitespace and only include pairs where both sentences are non-empty
            ambonese = ambonese.strip() if ambonese else ''
            indonesian = indonesian.strip() if indonesian else ''
            
            # Only include pairs where both sentences are non-empty
            if ambonese and indonesian:
                pairs.append((ambonese, indonesian))
    
    logging.info(f"Extracted {len(pairs)} valid sentence pairs")
    return pairs


def select_random_pairs(pairs: List[Tuple[str, str]], count: int = 100) -> List[Tuple[str, str]]:
    """Randomly select the specified number of sentence pairs."""
    if len(pairs) <= count:
        logging.warning(f"Only {len(pairs)} pairs available, selecting all")
        return pairs
    
    logging.info(f"Randomly selecting {count} pairs from {len(pairs)} available pairs")
    selected = random.sample(pairs, count)
    return selected


def write_jsonl(output_path: Path, pairs: List[Tuple[str, str]]):
    """Write sentence pairs to JSONL file."""
    logging.info(f"Writing {len(pairs)} sentence pairs to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ambonese, indonesian in tqdm(pairs, desc="Writing JSONL"):
            pair_obj = {
                "ambonese": ambonese,
                "indonesian": indonesian
            }
            f.write(json.dumps(pair_obj, ensure_ascii=False) + '\n')
    
    logging.info(f"Successfully wrote {len(pairs)} pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract parallel Ambonese-Indonesian sentences from dictionary JSON"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/cleaned/progress_20260105_145525_original_cleaned_v2.json',
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/parallel_sentences_100.jsonl',
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of sentence pairs to extract (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output (errors only)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    workspace_root = Path(__file__).parent
    input_path = workspace_root / args.input
    output_path = workspace_root / args.output
    
    # Set up logging
    log_file = setup_logging(workspace_root / "outputs", debug=args.debug, quiet=args.quiet)
    logging.info(f"Starting parallel sentence extraction")
    logging.info(f"Log file: {log_file}")
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logging.info(f"Random seed set to: {args.seed}")
    
    # Load JSON file
    data = load_json_file(input_path)
    
    # Extract all sentence pairs
    pairs = extract_sentence_pairs(data)
    
    if not pairs:
        logging.error("No valid sentence pairs found in the input file")
        sys.exit(1)
    
    # Select random pairs
    selected_pairs = select_random_pairs(pairs, count=args.count)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to JSONL
    write_jsonl(output_path, selected_pairs)
    
    logging.info("Parallel sentence extraction completed successfully")


if __name__ == '__main__':
    main()

