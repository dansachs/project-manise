#!/usr/bin/env python3
"""
Script 3: Clean Placeholder Symbols
Cleans OCR errors in dictionary JSON file by standardizing placeholder symbols
(~ to --), fixing spacing issues, and replacing -- with headwords.
Includes comprehensive error handling, validation, logging, and safeguards.
RUN THIS SCRIPT THIRD (after 2_parse_dictionary_entries.py)
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
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# Compiled regex patterns for efficiency
TILDE_PATTERN = re.compile(r'~')
GLUED_BEFORE_DASH = re.compile(r'(\w)--')
GLUED_AFTER_DASH = re.compile(r'--(\w)')
MULTIPLE_SPACES = re.compile(r'\s+')
LEADING_DASH = re.compile(r'^--(\w)')


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
        
        # Validate meanings structure
        for j, meaning in enumerate(entry['meanings']):
            if not isinstance(meaning, dict):
                return False, f"Entry {i}, meaning {j} is not a dictionary"
            
            # ambonese_example and definition are optional but should be strings if present
            if 'ambonese_example' in meaning and not isinstance(meaning['ambonese_example'], (str, type(None))):
                return False, f"Entry {i}, meaning {j} 'ambonese_example' must be string or None"
            
            if 'definition' in meaning and not isinstance(meaning['definition'], (str, type(None))):
                return False, f"Entry {i}, meaning {j} 'definition' must be string or None"
    
    return True, None


def clean_placeholder_text(text: str) -> str:
    """
    Clean placeholder symbols and spacing in text.
    Handles None/empty strings gracefully.
    
    Steps:
    1. Replace ~ with --
    2. Fix spacing around --
    3. Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    # Step 1: Replace ~ with --
    text = TILDE_PATTERN.sub('--', text)
    
    # Step 2: Fix spacing around --
    # Handle glued before dash: word-- -> word --
    text = GLUED_BEFORE_DASH.sub(r'\1 --', text)
    
    # Handle glued after dash: --word -> -- word (but preserve leading --)
    # First handle cases where -- is at start of string
    text = LEADING_DASH.sub(r'-- \1', text)
    # Then handle other cases
    text = GLUED_AFTER_DASH.sub(r'-- \1', text)
    
    # Step 3: Normalize multiple spaces to single space
    text = MULTIPLE_SPACES.sub(' ', text)
    
    # Strip leading/trailing whitespace but preserve structure
    text = text.strip()
    
    return text


def replace_with_headword(text: str, headword: str) -> str:
    """
    Replace -- with headword in example text.
    Uses simple string replacement (not regex) for safety.
    Handles None/empty strings gracefully.
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    if not headword or not isinstance(headword, str):
        return text
    
    # Simple string replacement - no regex needed, no escaping needed
    # This is safe because we're replacing literal '--' string, not using regex
    return text.replace('--', headword)


def create_backup(file_path: Path) -> Path:
    """Create timestamped backup of file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path.parent / f"{file_path.stem}.backup_{timestamp}{file_path.suffix}"
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating backup: {backup_path}")
    
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup created successfully: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise


def check_disk_space(file_path: Path, required_mb: float = 100.0) -> bool:
    """Check if there's enough disk space for operations."""
    try:
        stat = os.statvfs(file_path.parent)
        available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        
        if available_mb < required_mb:
            logger = logging.getLogger(__name__)
            logger.warning(f"Low disk space: {available_mb:.1f} MB available, {required_mb} MB recommended")
            return False
        return True
    except (OSError, AttributeError):
        # statvfs not available on all systems, skip check
        return True


def process_json_file(input_path: Path, output_path: Optional[Path] = None, dry_run: bool = False) -> Dict:
    """
    Process JSON file with full error handling.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (if None, will be auto-generated in cleaned directory)
        dry_run: If True, don't write any files
    
    Returns dictionary with statistics about the processing.
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Initialize backup_path before try block to avoid UnboundLocalError
    backup_path = None
    
    # Determine output path if not provided
    if output_path is None:
        # Create cleaned directory in outputs
        cleaned_dir = input_path.parent.parent / 'cleaned'
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename based on input filename
        input_stem = input_path.stem
        # Remove .json if it's duplicated (e.g., file.json.json -> file.json)
        if input_stem.endswith('.json'):
            input_stem = input_stem[:-5]
        output_path = cleaned_dir / f"{input_stem}_cleaned.json"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output file: {output_path}")
    
    # Statistics tracking
    stats = {
        'entries_processed': 0,
        'meanings_processed': 0,
        'ambonese_examples_cleaned': 0,
        'definitions_cleaned': 0,
        'tildes_replaced': 0,
        'headwords_inserted': 0,
        'warnings': [],
        'errors': []
    }
    
    try:
        # Validate input file
        logger.info(f"Validating input file: {input_path}")
        validate_input_file(input_path)
        
        # Check disk space
        if not check_disk_space(input_path):
            logger.warning("Low disk space detected, but continuing...")
        
        # Check if output file already exists
        if output_path.exists() and not dry_run:
            logger.warning(f"Output file already exists: {output_path}")
            logger.warning("It will be overwritten")
        
        # Only create backup if we're modifying the input file in-place
        # (i.e., when output_path is the same as input_path)
        if not dry_run and output_path.resolve() == input_path.resolve():
            backup_path = create_backup(input_path)
        elif not dry_run:
            logger.info("Writing to new file, no backup needed")
        else:
            logger.info("DRY RUN MODE: No backup created, no file will be modified")
        
        # Load JSON
        logger.info("Loading JSON file...")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {e}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            raise
        
        # Validate JSON structure
        logger.info("Validating JSON structure...")
        is_valid, error_msg = validate_json_structure(data)
        if not is_valid:
            logger.error(f"JSON structure validation failed: {error_msg}")
            stats['errors'].append(f"Structure validation: {error_msg}")
            raise ValueError(f"Invalid JSON structure: {error_msg}")
        
        entries_count = len(data.get('entries', []))
        logger.info(f"JSON structure validated. Found {entries_count} entries")
        
        # Process entries
        entries = data.get('entries', [])
        logger.info(f"Processing {len(entries)} entries...")
        
        for entry_idx, entry in enumerate(tqdm(entries, desc="Processing entries")):
            try:
                headword = entry.get('headword', '')
                meanings = entry.get('meanings', [])
                
                # Validate headword
                if not headword or not isinstance(headword, str):
                    warning = f"Entry {entry_idx}: Missing or invalid headword, skipping headword replacement"
                    logger.warning(warning)
                    stats['warnings'].append(warning)
                    headword = None
                
                stats['entries_processed'] += 1
                
                # Process each meaning
                for meaning_idx, meaning in enumerate(meanings):
                    stats['meanings_processed'] += 1
                    
                    # Phase 1: Clean placeholders in ambonese_example
                    if 'ambonese_example' in meaning:
                        original_example = meaning.get('ambonese_example')
                        if original_example and isinstance(original_example, str):
                            # Count tildes before cleaning
                            tildes_count = original_example.count('~')
                            if tildes_count > 0:
                                stats['tildes_replaced'] += tildes_count
                            
                            # Clean the text
                            cleaned_example = clean_placeholder_text(original_example)
                            
                            # Phase 2: Replace -- with headword (only if headword exists)
                            if headword and '--' in cleaned_example:
                                dashes_count = cleaned_example.count('--')
                                stats['headwords_inserted'] += dashes_count
                                cleaned_example = replace_with_headword(cleaned_example, headword)
                            
                            # Update the meaning
                            if cleaned_example != original_example:
                                meaning['ambonese_example'] = cleaned_example
                                stats['ambonese_examples_cleaned'] += 1
                    
                    # Phase 1: Clean placeholders in definition
                    if 'definition' in meaning:
                        original_definition = meaning.get('definition')
                        if original_definition and isinstance(original_definition, str):
                            # Count tildes before cleaning
                            tildes_count = original_definition.count('~')
                            if tildes_count > 0:
                                stats['tildes_replaced'] += tildes_count
                            
                            # Clean the text (but don't replace with headword in definitions)
                            cleaned_definition = clean_placeholder_text(original_definition)
                            
                            # Update the meaning
                            if cleaned_definition != original_definition:
                                meaning['definition'] = cleaned_definition
                                stats['definitions_cleaned'] += 1
                
            except Exception as e:
                error_msg = f"Error processing entry {entry_idx}: {e}"
                logger.error(error_msg, exc_info=True)
                stats['errors'].append(error_msg)
                # Continue processing other entries
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        stats['processing_time_seconds'] = elapsed_time
        
        # Save results (unless dry run)
        if not dry_run:
            logger.info(f"Writing cleaned JSON to file: {output_path}")
            
            # Atomic write: write to temp file, then rename
            temp_file = None
            try:
                # Create temp file in same directory as output
                temp_fd, temp_file = tempfile.mkstemp(
                    suffix='.json',
                    dir=output_path.parent,
                    text=True
                )
                
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # Atomic rename
                shutil.move(temp_file, output_path)
                logger.info(f"Successfully wrote cleaned JSON to: {output_path}")
                
            except Exception as e:
                error_msg = f"Failed to write output file: {e}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
                
                # Clean up temp file if it exists
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                # If write failed and we were modifying in-place, restore from backup
                if backup_path and backup_path.exists() and output_path.resolve() == input_path.resolve():
                    logger.info("Restoring from backup due to write failure...")
                    shutil.copy2(backup_path, input_path)
                
                raise
        else:
            logger.info("DRY RUN MODE: File not modified")
        
        logger.info("Processing completed successfully")
        return stats
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        stats['errors'].append("Processing interrupted by user")
        
        # Restore from backup if it exists and we were modifying in-place
        if backup_path and backup_path.exists() and not dry_run and output_path.resolve() == input_path.resolve():
            logger.info("Restoring from backup due to interruption...")
            try:
                shutil.copy2(backup_path, input_path)
            except Exception as e:
                logger.error(f"Failed to restore from backup: {e}")
        
        raise
    
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}", exc_info=True)
        stats['errors'].append(f"Fatal error: {e}")
        
        # Restore from backup if it exists and we were modifying in-place
        if backup_path and backup_path.exists() and not dry_run and output_path.resolve() == input_path.resolve():
            logger.info("Restoring from backup due to error...")
            try:
                shutil.copy2(backup_path, input_path)
            except Exception as restore_error:
                logger.error(f"Failed to restore from backup: {restore_error}")
        
        raise


def print_statistics(stats: Dict, output_path: Optional[Path] = None):
    """Print processing statistics in a readable format."""
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    if output_path:
        print(f"Output file:              {output_path}")
    print(f"Entries processed:        {stats.get('entries_processed', 0):,}")
    print(f"Meanings processed:       {stats.get('meanings_processed', 0):,}")
    print(f"Ambonese examples cleaned: {stats.get('ambonese_examples_cleaned', 0):,}")
    print(f"Definitions cleaned:      {stats.get('definitions_cleaned', 0):,}")
    print(f"Tildes replaced (~ → --):  {stats.get('tildes_replaced', 0):,}")
    print(f"Headwords inserted:        {stats.get('headwords_inserted', 0):,}")
    
    if 'processing_time_seconds' in stats:
        elapsed = stats['processing_time_seconds']
        print(f"Processing time:          {elapsed:.2f} seconds")
    
    if stats.get('warnings'):
        print(f"\nWarnings: {len(stats['warnings'])}")
        for warning in stats['warnings'][:10]:  # Show first 10
            print(f"  - {warning}")
        if len(stats['warnings']) > 10:
            print(f"  ... and {len(stats['warnings']) - 10} more warnings")
    
    if stats.get('errors'):
        print(f"\nErrors: {len(stats['errors'])}")
        for error in stats['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    print("="*60 + "\n")


def main():
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description='Clean placeholder symbols in dictionary JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process the default file (outputs to outputs/cleaned/)
  python 3_clean_placeholders.py
  
  # Process a specific file (outputs to outputs/cleaned/)
  python 3_clean_placeholders.py --input outputs/progress/progress_20260105_145525_original.json
  
  # Specify custom output file
  python 3_clean_placeholders.py --input file.json --output cleaned_file.json
  
  # Dry run to preview changes
  python 3_clean_placeholders.py --dry-run
  
  # Debug mode
  python 3_clean_placeholders.py --debug
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('outputs/progress/progress_20260105_145525_original.json.json'),
        help='Input JSON file to process (default: outputs/progress/progress_20260105_145525_original.json.json)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output JSON file path (default: auto-generated in outputs/cleaned/ directory)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output directory for logs (default: outputs)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying the file'
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
    log_file = setup_logging(args.output_dir, debug=args.debug, quiet=args.quiet)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Starting placeholder cleaning script")
    logger.info("="*60)
    logger.info(f"Input file: {args.input}")
    if args.output:
        logger.info(f"Output file: {args.output}")
    else:
        logger.info("Output file: (auto-generated in outputs/cleaned/)")
    logger.info(f"Output directory (logs): {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Process the file
        stats = process_json_file(args.input, output_path=args.output, dry_run=args.dry_run)
        
        # Get the actual output path used (for display)
        actual_output_path = args.output
        if actual_output_path is None:
            # Reconstruct the auto-generated path
            cleaned_dir = args.input.parent.parent / 'cleaned'
            input_stem = args.input.stem
            if input_stem.endswith('.json'):
                input_stem = input_stem[:-5]
            actual_output_path = cleaned_dir / f"{input_stem}_cleaned.json"
        
        # Print statistics
        if not args.quiet:
            if not args.dry_run:
                print_statistics(stats, output_path=actual_output_path)
            else:
                print_statistics(stats)
        
        logger.info("Script completed successfully")
        
        # Exit with error code if there were errors
        if stats.get('errors'):
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        if not args.quiet:
            print("\n\nScript interrupted by user. Exiting...")
        sys.exit(130)  # Standard exit code for SIGINT
    
    except Exception as e:
        logger.error(f"Script failed with error: {e}", exc_info=True)
        if not args.quiet:
            print(f"\n\nError: {e}")
            print(f"See log file for details: {log_file}")
        sys.exit(1)


if __name__ == '__main__':
    main()

