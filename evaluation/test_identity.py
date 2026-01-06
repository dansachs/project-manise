#!/usr/bin/env python3
"""
Test Identity: Language Identification Evaluation
Determines if Google Gemini correctly identifies Ambonese Malay text or conflates it with Indonesian.
"""

import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
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


def grade_identity(predicted_label: str) -> str:
    """
    Grade the predicted language label.
    
    Returns:
        'Pass' if Ambonese-related terms found
        'Fail' if Indonesian without Ambonese context
        'Hallucination' otherwise
    """
    label_lower = predicted_label.lower()
    
    # Check for Ambonese indicators
    ambonese_terms = ['ambonese', 'ambon', 'melayu ambon']
    if any(term in label_lower for term in ambonese_terms):
        return 'Pass'
    
    # Check for Indonesian (without Ambonese context)
    indonesian_terms = ['indonesian', 'bahasa indonesia']
    if any(term in label_lower for term in indonesian_terms):
        return 'Fail'
    
    # Otherwise, it's a hallucination (wrong language)
    return 'Hallucination'


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test if Gemini correctly identifies Ambonese Malay text"
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
        default=Path('evaluation/results_identity.csv'),
        help='Output CSV file path (default: evaluation/results_identity.csv)'
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
            
            # Create prompt
            prompt = (
                f"Identify the language of the following sentence. "
                f"Return ONLY the language name, nothing else. "
                f"Sentence: '{ambonese_sentence}'"
            )
            
            # Call API
            predicted_label = call_gemini_api(client, args.model, prompt, idx)
            
            if predicted_label is None:
                predicted_label = "ERROR"
                score = "Fail"
            else:
                score = grade_identity(predicted_label)
            
            results.append({
                'input_sentence': ambonese_sentence,
                'predicted_label': predicted_label,
                'score': score
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
        passed = sum(1 for r in results if r['score'] == 'Pass')
        failed = sum(1 for r in results if r['score'] == 'Fail')
        hallucinated = sum(1 for r in results if r['score'] == 'Hallucination')
        
        recognition_rate = (passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("IDENTITY TEST RESULTS")
        print("="*60)
        print(f"Ambonese Recognition Rate: {recognition_rate:.1f}% ({passed}/{total} passed)")
        print(f"\nBreakdown:")
        print(f"  Pass:           {passed:3d} ({passed/total*100:.1f}%)")
        print(f"  Fail:           {failed:3d} ({failed/total*100:.1f}%)")
        print(f"  Hallucination:  {hallucinated:3d} ({hallucinated/total*100:.1f}%)")
        print("="*60)
    else:
        logger.warning("No results to save.")


if __name__ == '__main__':
    main()

