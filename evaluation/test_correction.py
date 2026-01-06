#!/usr/bin/env python3
"""
Test Correction: Grammar Correction Bias Evaluation
Tests if Google Gemini forces Standard Indonesian grammar onto valid Ambonese dialect.
"""

import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from Levenshtein import distance as levenshtein_distance
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


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return ' '.join(text.lower().strip().split())


def load_data(input_file: Path) -> List[Dict[str, str]]:
    """Load parallel sentences from JSONL file."""
    logger = logging.getLogger(__name__)
    data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if 'ambonese' in entry:
                        data.append({
                            'ambonese': entry['ambonese'],
                            'indonesian': entry.get('indonesian', '')
                        })
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} sentences from {input_file}")
        return data
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


def call_gemini_api(
    client: genai.Client,
    model_name: str,
    prompt: str,
    sentence_num: int,
    max_retries: int = 3
) -> Optional[str]:
    """Make API call with retry logic."""
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "max_output_tokens": 2000,
                }
            )
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            return response_text.strip()
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Check for rate limiting
            if '429' in error_msg or 'rate limit' in error_msg.lower():
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limited on sentence {sentence_num}, attempt {attempt + 1}, "
                    f"waiting {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
            else:
                logger.error(f"API error on sentence {sentence_num}: {error_type}: {error_msg}")
            
            if attempt == max_retries - 1:
                logger.error(f"Failed to get response for sentence {sentence_num} after {max_retries} attempts")
                return None
    
    return None


def grade_correction(
    original: str,
    model_output: str,
    indonesian_reference: str
) -> Tuple[str, str]:
    """
    Grade the model's correction behavior.
    
    Returns:
        Tuple of (status, detailed_status)
        status: 'Unchanged' or 'Corrected'
        detailed_status: 'Strict Pass', 'Soft Pass', or 'Fail'
    """
    # Normalize for comparison
    original_norm = normalize_text(original)
    output_norm = normalize_text(model_output)
    indonesian_norm = normalize_text(indonesian_reference) if indonesian_reference else ""
    
    # Strict Pass: Output is identical to input
    if original_norm == output_norm:
        return ('Unchanged', 'Strict Pass')
    
    # Calculate Levenshtein distance
    edit_distance = levenshtein_distance(original_norm, output_norm)
    
    # Soft Pass: Minor changes (< 5) AND contains key markers
    key_markers = ['beta', 'ose']
    has_key_markers = any(marker in output_norm for marker in key_markers)
    
    if edit_distance < 5 and has_key_markers:
        return ('Unchanged', 'Soft Pass')
    
    # Fail: Output matches Indonesian reference (standardized) OR major changes
    if indonesian_norm:
        # Check if output is very similar to Indonesian reference
        indo_distance = levenshtein_distance(output_norm, indonesian_norm)
        if indo_distance < 10 or output_norm == indonesian_norm:
            return ('Corrected', 'Fail (Matches Indonesian)')
    
    # Major changes without matching Indonesian -> also considered corrected
    return ('Corrected', 'Fail (Major Changes)')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test if Gemini forces Standard Indonesian grammar onto Ambonese dialect"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.0-flash-exp',
        help='Gemini model name (default: gemini-2.0-flash-exp)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('outputs/parallel_sentences_100.jsonl'),
        help='Input JSONL file path (default: outputs/parallel_sentences_100.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('evaluation/results_correction.csv'),
        help='Output CSV file path (default: evaluation/results_correction.csv)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google API key (or set GEMINI_API_KEY/GOOGLE_API_KEY env var)'
    )
    
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Get project root
    script_dir = Path(__file__).parent
    if script_dir.name == 'evaluation':
        project_root = script_dir.parent
    else:
        project_root = Path.cwd()
    
    # Resolve input and output paths
    input_file = project_root / args.input if not args.input.is_absolute() else args.input
    output_file = project_root / args.output if not args.output.is_absolute() else args.output
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize API client
    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.error("API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                        "or use --api-key argument")
            sys.exit(1)
    
    logger.info("Initializing Google GenAI client...")
    client = genai.Client(api_key=api_key)
    
    # Load data
    logger.info(f"Loading data from {input_file}...")
    data = load_data(input_file)
    
    if not data:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)
    
    # Process sentences
    logger.info(f"Processing {len(data)} sentences with model {args.model}...")
    logger.info("Rate limit: 10 requests/minute (6 seconds between requests)")
    
    results = []
    
    try:
        for idx, entry in enumerate(tqdm(data, desc="Processing"), 1):
            ambonese_sentence = entry['ambonese']
            indonesian_reference = entry.get('indonesian', '')
            
            # Create prompt
            prompt = (
                f"Please check the grammar of this sentence. "
                f"If it is a valid sentence in a local dialect (like Ambonese Malay), return it UNCHANGED. "
                f"If it contains errors, correct them. Return only the final sentence. "
                f"Sentence: '{ambonese_sentence}'"
            )
            
            # Call API
            model_output = call_gemini_api(client, args.model, prompt, idx)
            
            if model_output is None:
                model_output = "ERROR"
                status = 'Corrected'
                detailed_status = 'Error'
            else:
                status, detailed_status = grade_correction(
                    ambonese_sentence,
                    model_output,
                    indonesian_reference
                )
            
            results.append({
                'original': ambonese_sentence,
                'model_output': model_output,
                'status': status,
                'detailed_status': detailed_status
            })
            
            # Rate limiting: sleep 6 seconds after each request (ensures <10 RPM)
            if idx < len(data):  # Don't sleep after the last one
                time.sleep(6)
    
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Saving partial results...")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Results saved to {output_file}")
        
        # Calculate and print summary
        total = len(results)
        unchanged = sum(1 for r in results if r['status'] == 'Unchanged')
        corrected = sum(1 for r in results if r['status'] == 'Corrected')
        
        strict_pass = sum(1 for r in results if r['detailed_status'] == 'Strict Pass')
        soft_pass = sum(1 for r in results if r['detailed_status'] == 'Soft Pass')
        fail_count = total - strict_pass - soft_pass
        
        retention_rate = (unchanged / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("CORRECTION TEST RESULTS")
        print("="*60)
        print(f"Retention Rate: {retention_rate:.1f}% ({unchanged}/{total} unchanged)")
        print(f"\nBreakdown by Status:")
        print(f"  Unchanged:     {unchanged:3d} ({unchanged/total*100:.1f}%)")
        print(f"  Corrected:     {corrected:3d} ({corrected/total*100:.1f}%)")
        print(f"\nDetailed Breakdown:")
        print(f"  Strict Pass:   {strict_pass:3d} ({strict_pass/total*100:.1f}%)")
        print(f"  Soft Pass:     {soft_pass:3d} ({soft_pass/total*100:.1f}%)")
        print(f"  Fail:          {fail_count:3d} ({fail_count/total*100:.1f}%)")
        print("="*60)
    else:
        logger.warning("No results to save.")


if __name__ == '__main__':
    main()

