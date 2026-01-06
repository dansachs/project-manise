#!/usr/bin/env python3
"""
Script 4: OCR/Typo Correction for Dictionary Examples
Identifies and corrects strictly necessary OCR errors and typos in Ambonese Malay examples
and Indonesian translations, preserving dialectal variations in Ambonese while standardizing Indonesian.
Uses batch processing with Gemini API, incremental JSONL saves, and robust error handling.
RUN THIS SCRIPT FOURTH (after 3_clean_placeholders.py)
"""

import json
import re
import argparse
import logging
import shutil
import tempfile
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
from google import genai


# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert Computational Linguist specializing in Low-Resource Language preservation (specifically Ambonese Malay) and Indonesian standard copyediting.

Your task is to review a batch of dictionary examples (Ambonese Malay sentences and their Indonesian translations) and identify ONLY strictly necessary errors caused by OCR (Optical Character Recognition) or typos.

### 1. GUIDELINES FOR AMBONESE (Target: Conservative Preservation)
* **Goal:** Preserve the authentic dialect.
* **DO NOT** standardize spelling variations. If the text says "kalo" or "kalau", "pi" or "pigi", "sa" or "saja", LEAVE IT ALONE. These are valid dialectal markers.
* **DO** fix obvious OCR failures.
    * *Example:* "makanq" -> "makang" (q/g confusion)
    * *Example:* "b3ta" -> "beta" (number/letter confusion)
    * *Example:* "oran g" -> "orang" (split words)
* **DO** fix spacing and punctuation.
    * *Example:* "beta , dia" -> "beta, dia" (remove space before comma)

### 2. GUIDELINES FOR INDONESIAN (Target: Standard Cleanliness)
* **Goal:** Standard Formal Indonesian (Baku).
* **DO** fix typos and non-standard abbreviations.
    * *Example:* "Sya tdk tau" -> "Saya tidak tahu"
    * *Example:* "perqi" -> "pergi"
* **DO** ensure capitalization is correct (Start of sentence, proper nouns).

### 3. OUTPUT FORMAT
You will receive a JSON list of objects. You must return a JSON object containing a list called `changes`.
* Include **ONLY** rows where a change was actually made.
* If a row requires no changes, do not include it in the output.

Your output schema must be:
{
  "changes": [
    {
      "unique_id": "string (from input)",
      "field": "ambonese_example" OR "indonesian_translation",
      "original_value": "string",
      "suggested_value": "string",
      "change_type": "OCR_FIX" | "TYPO" | "FORMATTING",
      "reason": "Short explanation"
    }
  ]
}

### 4. EXAMPLES OF DESIRED BEHAVIOR

**Input:**
[
  {"unique_id": "1", "headword": "pigi", "definition": "pergi", "ambonese_example": "Beta  pigi ka pasar", "indonesian_translation": "Saya perqi ke pasar"},
  {"unique_id": "2", "headword": "pung", "definition": "punya", "ambonese_example": "Dia pung mama", "indonesian_translation": "Ibunya dia"},
  {"unique_id": "3", "headword": "jang", "definition": "jangan", "ambonese_example": "Jang maen gila", "indonesian_translation": "Jangan main gila"}
]

**Correct Output:**
{
  "changes": [
    {
      "unique_id": "1",
      "field": "ambonese_example",
      "original_value": "Beta  pigi ka pasar",
      "suggested_value": "Beta pigi ka pasar",
      "change_type": "FORMATTING",
      "reason": "Removed double space"
    },
    {
      "unique_id": "1",
      "field": "indonesian_translation",
      "original_value": "Saya perqi ke pasar",
      "suggested_value": "Saya pergi ke pasar",
      "change_type": "OCR_FIX",
      "reason": "Fixed OCR error 'q' to 'g' in 'pergi'"
    }
  ]
}
(Note: ID 2 and 3 were omitted because they are correct. "pung" and "jang" are valid Ambonese and were NOT changed to "punya" or "jangan".)

### IMPORTANT: Return ONLY entries that require changes. Do not include entries that are already correct."""


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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file


def validate_input_file(input_path: Path) -> bool:
    """Validate input file exists and is readable."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Path is not a file: {input_path}")
    
    if not os.access(input_path, os.R_OK):
        raise PermissionError(f"Cannot read file: {input_path}")
    
    return True


def validate_json_structure(data: Dict) -> Tuple[bool, Optional[str]]:
    """Validate JSON has required structure before processing."""
    if not isinstance(data, dict):
        return False, "Root element must be a dictionary"
    
    if 'entries' not in data:
        return False, "Missing required key: 'entries'"
    
    if not isinstance(data['entries'], list):
        return False, "'entries' must be a list"
    
    # Validate entry structure
    for i, entry in enumerate(data['entries']):
        if not isinstance(entry, dict):
            return False, f"Entry {i} is not a dictionary"
        
        if 'headword' not in entry:
            return False, f"Entry {i} missing required key: 'headword'"
        
        if 'meanings' not in entry:
            return False, f"Entry {i} missing required key: 'meanings'"
        
        if not isinstance(entry['meanings'], list):
            return False, f"Entry {i} 'meanings' must be a list"
        
        for j, meaning in enumerate(entry['meanings']):
            if not isinstance(meaning, dict):
                return False, f"Entry {i}, meaning {j} is not a dictionary"
    
    return True, None


def slugify_headword(headword: str) -> str:
    """
    Robust slugification for stable IDs.
    Converts headword to lowercase, replaces spaces/special chars with underscores.
    """
    # Convert to lowercase
    slug = headword.lower()
    
    # Replace spaces and special characters with underscores
    slug = re.sub(r'[^\w\s-]', '_', slug)
    slug = re.sub(r'[\s_-]+', '_', slug)
    
    # Remove leading/trailing underscores
    slug = slug.strip('_')
    
    # If empty after slugification, use a default
    if not slug:
        slug = "unknown"
    
    return slug


def flatten_entries(data: Dict) -> List[Dict]:
    """
    Extract all ambonese_example + indonesian_translation pairs from entries.
    Returns list of dicts with unique_id, headword, definition, ambonese_example, indonesian_translation.
    """
    logger = logging.getLogger(__name__)
    entries = data.get('entries', [])
    flattened = []
    
    # Track headword occurrences for duplicate handling
    headword_counts = {}
    
    for entry in entries:
        headword = entry.get('headword') or ''
        slugified = slugify_headword(headword)
        
        meanings = entry.get('meanings', [])
        for sense_idx, meaning in enumerate(meanings):
            sense_number = meaning.get('sense_number') or (sense_idx + 1)
            definition = meaning.get('definition') or ''
            
            ambonese_example = (meaning.get('ambonese_example') or '').strip()
            indonesian_translation = (meaning.get('indonesian_translation') or '').strip()
            
            # Skip if both are empty
            if not ambonese_example and not indonesian_translation:
                continue
            
            # Generate unique ID: {slugified_headword}_{sense_number}_{example_index}
            # Handle duplicates by tracking occurrences
            key = f"{slugified}_{sense_number}"
            if key not in headword_counts:
                headword_counts[key] = 0
            else:
                headword_counts[key] += 1
            
            example_index = headword_counts[key]
            unique_id = f"{slugified}_{sense_number}_{example_index}"
            
            flattened.append({
                'unique_id': unique_id,
                'headword': headword,
                'definition': definition,
                'ambonese_example': ambonese_example,
                'indonesian_translation': indonesian_translation
            })
    
    logger.info(f"Flattened {len(entries)} entries into {len(flattened)} examples")
    return flattened


def create_batch_prompt(examples: List[Dict]) -> str:
    """
    Create prompt for batch of examples with context (headword + definition).
    """
    # Build input JSON array
    input_data = json.dumps(examples, ensure_ascii=False, indent=2)
    
    prompt = f"""{SYSTEM_PROMPT}

### INPUT BATCH

Review the following {len(examples)} dictionary examples. Return ONLY the entries that require changes:

{input_data}

Return your response as a valid JSON object with a "changes" array containing only entries that need corrections."""

    return prompt


def call_gemini_api(
    client: genai.Client,
    model_name: str,
    prompt: str,
    batch_num: int,
    max_retries: int = 3
) -> Optional[str]:
    """
    Make API call with JSON mode enabled and retry logic.
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "max_output_tokens": 4000,
                    "response_mime_type": "application/json",  # JSON mode
                }
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"API call for batch {batch_num} took {elapsed:.2f}s")
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            return response_text
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Check for rate limiting
            if '429' in error_msg or 'rate limit' in error_msg.lower():
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Rate limited on batch {batch_num}, attempt {attempt + 1}, "
                    f"waiting {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
            
            logger.error(
                f"API error on batch {batch_num}, attempt {attempt + 1}/{max_retries}: "
                f"{error_type}: {error_msg}"
            )
            
            if attempt == max_retries - 1:
                return None
            
            # Wait before retry
            time.sleep(2 ** attempt)
    
    return None


def parse_response(
    response_text: str,
    batch_num: int,
    original_examples: List[Dict]
) -> List[Dict]:
    """
    Parse JSON response, validate, and apply hallucination guard.
    Returns list of validated changes.
    """
    logger = logging.getLogger(__name__)
    changes = []
    
    try:
        # Parse JSON response
        response_data = json.loads(response_text)
        
        if not isinstance(response_data, dict):
            logger.error(f"Batch {batch_num}: Response is not a dictionary")
            return []
        
        if 'changes' not in response_data:
            logger.warning(f"Batch {batch_num}: No 'changes' key in response (might mean no changes needed)")
            return []
        
        response_changes = response_data.get('changes', [])
        if not isinstance(response_changes, list):
            logger.error(f"Batch {batch_num}: 'changes' is not a list")
            return []
        
        # Create mapping of unique_id to original examples for validation
        example_map = {ex['unique_id']: ex for ex in original_examples}
        
        # Validate each change
        for change in response_changes:
            # Validate required fields
            required_fields = ['unique_id', 'field', 'original_value', 'suggested_value', 'change_type', 'reason']
            missing_fields = [f for f in required_fields if f not in change]
            if missing_fields:
                logger.warning(f"Batch {batch_num}: Change missing fields {missing_fields}, skipping")
                continue
            
            unique_id = change['unique_id']
            field = change['field']
            original_value = change['original_value']
            suggested_value = change['suggested_value']
            
            # Validate field
            if field not in ['ambonese_example', 'indonesian_translation']:
                logger.warning(f"Batch {batch_num}: Invalid field '{field}' for {unique_id}, skipping")
                continue
            
            # Validate unique_id exists
            if unique_id not in example_map:
                logger.warning(f"Batch {batch_num}: Unknown unique_id '{unique_id}', skipping")
                continue
            
            # Validate original_value matches
            expected_original = example_map[unique_id].get(field, '')
            if original_value != expected_original:
                logger.warning(
                    f"Batch {batch_num}: Original value mismatch for {unique_id}.{field}. "
                    f"Expected: '{expected_original}', Got: '{original_value}'"
                )
                # Still process if close enough (minor whitespace differences)
                if original_value.strip() != expected_original.strip():
                    continue
            
            # Hallucination Guard: Check length difference
            original_len = len(original_value)
            suggested_len = len(suggested_value)
            
            if original_len == 0:
                logger.warning(f"Batch {batch_num}: Empty original_value for {unique_id}.{field}, skipping")
                continue
            
            length_ratio = abs(suggested_len - original_len) / original_len
            if length_ratio > 0.5:
                logger.warning(
                    f"Batch {batch_num}: Hallucination guard triggered for {unique_id}.{field}. "
                    f"Length difference: {length_ratio:.1%} (>50%). Original: {original_len} chars, "
                    f"Suggested: {suggested_len} chars. Discarding change."
                )
                continue
            
            # Validate change_type
            if change['change_type'] not in ['OCR_FIX', 'TYPO', 'FORMATTING']:
                logger.warning(
                    f"Batch {batch_num}: Invalid change_type '{change['change_type']}' for {unique_id}, "
                    f"defaulting to 'TYPO'"
                )
                change['change_type'] = 'TYPO'
            
            changes.append(change)
        
        logger.info(f"Batch {batch_num}: Validated {len(changes)}/{len(response_changes)} changes")
        
    except json.JSONDecodeError as e:
        logger.error(f"Batch {batch_num}: Failed to parse JSON response: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        return []
    except Exception as e:
        logger.error(f"Batch {batch_num}: Error parsing response: {e}")
        return []
    
    return changes


def append_to_jsonl(changes: List[Dict], jsonl_path: Path) -> bool:
    """
    Append change objects to JSONL file (incremental save).
    Returns True on success, False on failure.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure directory exists
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to file (create if doesn't exist)
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            for change in changes:
                json.dump(change, f, ensure_ascii=False)
                f.write('\n')
        
        logger.debug(f"Appended {len(changes)} changes to {jsonl_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to append to JSONL file {jsonl_path}: {e}")
        return False


def load_processed_ids(jsonl_path: Path) -> Set[str]:
    """
    Load already-processed unique_ids from existing JSONL file for resume capability.
    Returns set of unique_ids.
    """
    logger = logging.getLogger(__name__)
    processed_ids = set()
    
    if not jsonl_path.exists():
        return processed_ids
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    change = json.loads(line)
                    unique_id = change.get('unique_id')
                    if unique_id:
                        # Track by unique_id (not unique_id + field, since we want to skip the whole example)
                        processed_ids.add(unique_id)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} of {jsonl_path}: {e}")
        
        logger.info(f"Loaded {len(processed_ids)} already-processed IDs from {jsonl_path}")
        
    except Exception as e:
        logger.warning(f"Failed to load processed IDs from {jsonl_path}: {e}")
    
    return processed_ids


def consolidate_jsonl_to_json(jsonl_path: Path, json_path: Path) -> bool:
    """
    Convert JSONL file to consolidated JSON file with {"changes": [...]} format.
    Returns True on success, False on failure.
    """
    logger = logging.getLogger(__name__)
    
    try:
        changes = []
        
        if jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        change = json.loads(line)
                        changes.append(change)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line in {jsonl_path}: {e}")
        
        # Ensure directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file
        if json_path.exists():
            backup_path = json_path.with_suffix('.json.backup')
            shutil.copy(json_path, backup_path)
            logger.info(f"Backed up existing JSON to {backup_path}")
        
        # Write consolidated JSON
        output_data = {'changes': changes}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Consolidated {len(changes)} changes to {json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to consolidate JSONL to JSON: {e}")
        return False


def process_batches(
    client: genai.Client,
    model_name: str,
    examples: List[Dict],
    batch_size: int,
    jsonl_path: Path,
    processed_ids: Set[str]
) -> Tuple[int, int]:
    """
    Main processing loop with progress tracking, resume capability, and batch management.
    Returns (total_processed, total_changes).
    """
    logger = logging.getLogger(__name__)
    
    # Filter out already-processed examples
    remaining_examples = [
        ex for ex in examples
        if ex['unique_id'] not in processed_ids
    ]
    
    if not remaining_examples:
        logger.info("All examples already processed")
        return len(examples), 0
    
    if len(processed_ids) > 0:
        logger.info(f"Skipping {len(examples) - len(remaining_examples)} already-processed examples")
    
    total_changes = 0
    total_batches = (len(remaining_examples) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(remaining_examples)} examples in {total_batches} batches")
    
    with tqdm(total=len(remaining_examples), desc="Processing examples") as pbar:
        for batch_idx in range(0, len(remaining_examples), batch_size):
            batch_num = (batch_idx // batch_size) + 1
            batch = remaining_examples[batch_idx:batch_idx + batch_size]
            
            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} examples)")
            
            # Create prompt
            prompt = create_batch_prompt(batch)
            
            # Call API
            response_text = call_gemini_api(client, model_name, prompt, batch_num)
            
            if response_text is None:
                logger.error(f"Batch {batch_num} failed, skipping")
                pbar.update(len(batch))
                continue
            
            # Parse response
            changes = parse_response(response_text, batch_num, batch)
            
            if changes:
                # Append to JSONL
                if append_to_jsonl(changes, jsonl_path):
                    total_changes += len(changes)
                    logger.info(f"Batch {batch_num}: Saved {len(changes)} changes")
                else:
                    logger.error(f"Batch {batch_num}: Failed to save changes")
            
            pbar.update(len(batch))
    
    return len(examples), total_changes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Correct OCR errors and typos in dictionary examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input JSON file (parsed entries or cleaned entries)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('outputs/corrected'),
        help='Output directory for corrections (default: outputs/corrected)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=15,
        help='Number of examples per API call (default: 15)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.0-flash-exp',
        help='Gemini model name (default: gemini-2.0-flash-exp)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Gemini API key (or set GEMINI_API_KEY/GOOGLE_API_KEY env var)'
    )
    
    parser.add_argument(
        '--no-consolidate',
        action='store_true',
        help='Do not create consolidated JSON file (only JSONL)'
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
    
    # Setup logging
    output_dir = Path('outputs')
    log_file = setup_logging(output_dir, debug=args.debug, quiet=args.quiet)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("OCR/Typo Correction Script")
    logger.info("=" * 60)
    
    try:
        # Validate input
        logger.info(f"Input file: {args.input_file}")
        validate_input_file(args.input_file)
        
        # Load JSON
        logger.info("Loading JSON file...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        logger.info("Validating JSON structure...")
        is_valid, error_msg = validate_json_structure(data)
        if not is_valid:
            logger.error(f"JSON structure validation failed: {error_msg}")
            sys.exit(1)
        
        # Flatten entries
        logger.info("Flattening entries...")
        examples = flatten_entries(data)
        
        if not examples:
            logger.warning("No examples found in input file")
            sys.exit(0)
        
        # Setup API client
        if args.api_key:
            client = genai.Client(api_key=args.api_key)
        else:
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error("API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY env var")
                sys.exit(1)
            client = genai.Client(api_key=api_key)
        
        # Setup output paths
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_stem = args.input_file.stem.replace('_cleaned', '').replace('_original', '')
        jsonl_path = args.output_dir / f"corrections_{input_stem}_{timestamp}.jsonl"
        json_path = args.output_dir / f"corrections_{input_stem}_{timestamp}.json"
        
        logger.info(f"Output JSONL: {jsonl_path}")
        if not args.no_consolidate:
            logger.info(f"Output JSON: {json_path}")
        
        # Load already-processed IDs for resume
        processed_ids = load_processed_ids(jsonl_path)
        
        # Process batches
        logger.info(f"Starting batch processing (batch size: {args.batch_size})...")
        total_processed, total_changes = process_batches(
            client,
            args.model,
            examples,
            args.batch_size,
            jsonl_path,
            processed_ids
        )
        
        logger.info("=" * 60)
        logger.info(f"Processing complete!")
        logger.info(f"  Total examples: {total_processed}")
        logger.info(f"  Total changes: {total_changes}")
        logger.info(f"  JSONL file: {jsonl_path}")
        
        # Consolidate to JSON if requested
        if not args.no_consolidate:
            logger.info("Consolidating to JSON...")
            if consolidate_jsonl_to_json(jsonl_path, json_path):
                logger.info(f"  JSON file: {json_path}")
        
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

