#!/usr/bin/env python3
"""
Script 3.1: Replace " - " with Headword and "/" with "l"
Replaces " - " (space-dash-space) patterns in ambonese_example fields with the 
corresponding entry's headword, and replaces "/" with "l".
Includes comprehensive error handling, validation, logging, and safeguards.
RUN THIS SCRIPT AFTER 3_clean_placeholders.py
"""

import json
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


def replace_dash_space_dash(text: str, headword: str) -> str:
    """
    Replace " - " (space-dash-space) with headword (surrounded by spaces) in example text.
    Uses simple string replacement (not regex) for safety.
    Handles None/empty strings gracefully.
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    if not headword or not isinstance(headword, str):
        return text
    
    # Replace " - " with " headword " (space-headword-space)
    # This preserves the spacing structure
    return text.replace(" - ", f" {headword} ")


def replace_slash_with_l(text: str) -> str:
    """
    Replace "/" with "l" in example text.
    Uses simple string replacement (not regex) for safety.
    Handles None/empty strings gracefully.
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    # Simple string replacement - no regex needed
    # This is safe because we're replacing literal "/" string
    return text.replace("/", "l")


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
        # Remove _cleaned suffix if present, then add _cleaned_v2
        if input_stem.endswith('_cleaned'):
            input_stem = input_stem[:-8]
        output_path = cleaned_dir / f"{input_stem}_cleaned_v2.json"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output file: {output_path}")
    
    # Statistics tracking
    stats = {
        'entries_processed': 0,
        'meanings_processed': 0,
        'examples_modified': 0,
        'dash_space_dash_replacements': 0,
        'slash_replacements': 0,
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
                    
                    # Process ambonese_example
                    if 'ambonese_example' in meaning:
                        original_example = meaning.get('ambonese_example')
                        if original_example and isinstance(original_example, str):
                            modified_example = original_example
                            example_modified = False
                            
                            # Replace " - " with headword (if headword exists)
                            if headword and " - " in modified_example:
                                dash_count = modified_example.count(" - ")
                                stats['dash_space_dash_replacements'] += dash_count
                                modified_example = replace_dash_space_dash(modified_example, headword)
                                example_modified = True
                            
                            # Replace "/" with "l"
                            if "/" in modified_example:
                                slash_count = modified_example.count("/")
                                stats['slash_replacements'] += slash_count
                                modified_example = replace_slash_with_l(modified_example)
                                example_modified = True
                            
                            # Update the meaning if changes were made
                            if example_modified:
                                meaning['ambonese_example'] = modified_example
                                stats['examples_modified'] += 1
                
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
            logger.info(f"Writing processed JSON to file: {output_path}")
            
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
                logger.info(f"Successfully wrote processed JSON to: {output_path}")
                
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
    print(f"Examples modified:        {stats.get('examples_modified', 0):,}")
    print(f'" - " replacements:        {stats.get("dash_space_dash_replacements", 0):,}')
    print(f'"/" replacements:          {stats.get("slash_replacements", 0):,}')
    
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
        description='Replace " - " with headword and "/" with "l" in ambonese_example fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process the default file (outputs to outputs/cleaned/)
  python 3.1_replace_dash_space_dash.py
  
  # Process a specific file (outputs to outputs/cleaned/)
  python 3.1_replace_dash_space_dash.py --input outputs/cleaned/progress_20260105_145525_original_cleaned.json
  
  # Specify custom output file
  python 3.1_replace_dash_space_dash.py --input file.json --output processed_file.json
  
  # Dry run to preview changes
  python 3.1_replace_dash_space_dash.py --dry-run
  
  # Debug mode
  python 3.1_replace_dash_space_dash.py --debug
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('outputs/cleaned/progress_20260105_145525_original_cleaned.json'),
        help='Input JSON file to process (default: outputs/cleaned/progress_20260105_145525_original_cleaned.json)'
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
    logger.info("Starting dash-space-dash and slash replacement script")
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
            # Remove _cleaned suffix if present, then add _cleaned_v2
            if input_stem.endswith('_cleaned'):
                input_stem = input_stem[:-8]
            actual_output_path = cleaned_dir / f"{input_stem}_cleaned_v2.json"
        
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

