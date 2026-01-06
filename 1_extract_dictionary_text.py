#!/usr/bin/env python3
"""
Script 1: PDF Dictionary Text Extraction
Extracts text from dictionary_20260105.pdf starting at page 16.
Handles double-column layout and saves with timestamped output.
RUN THIS SCRIPT FIRST
"""

import pdfplumber
import argparse
import logging
import sys
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from collections import deque
from tqdm import tqdm


class TimeEstimator:
    """Calculate time estimates using moving average."""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.start_time = time.time()
    
    def update(self, elapsed_time):
        """Add a new processing time to the window."""
        self.times.append(elapsed_time)
    
    def get_eta(self, remaining_items):
        """Calculate estimated time remaining."""
        if not self.times:
            return None
        avg_time = sum(self.times) / len(self.times)
        return remaining_items * avg_time
    
    def get_elapsed(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS or MM:SS."""
        if seconds is None:
            return "N/A"
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def setup_logging(output_dir, debug=False, quiet=False):
    """Set up logging to file and console."""
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


def validate_pdf(pdf_path):
    """Validate PDF file exists and is readable."""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    
    # Try to open and check page count
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages < 16:
                raise ValueError(f"PDF has only {total_pages} pages, need at least 16")
            return total_pages
    except Exception as e:
        raise ValueError(f"Cannot read PDF file: {e}")


def extract_text_from_pdf(
    pdf_path: str,
    output_dir: Path,
    start_page: int = 16,
    debug: bool = False,
    dry_run: bool = False,
    checkpoint_interval: int = 50,
    no_progress: bool = False
):
    """
    Extract text from PDF starting at specified page.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files
        start_page: First page to extract (default: 16)
        debug: Enable debug mode
        dry_run: Don't write file, just show what would be extracted
        checkpoint_interval: Save checkpoint every N pages
        no_progress: Disable progress bar
    """
    logger = logging.getLogger(__name__)
    
    # Validate PDF
    logger.info(f"Validating PDF: {pdf_path}")
    total_pages = validate_pdf(pdf_path)
    logger.info(f"PDF validated: {total_pages} total pages")
    
    # Create output directory
    extractions_dir = output_dir / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"extraction_{timestamp}.txt"
    output_path = extractions_dir / output_filename
    
    if dry_run:
        logger.info("DRY RUN MODE: No file will be written")
        output_path = None
    else:
        logger.info(f"Output file: {output_path}")
    
    # Statistics
    stats = {
        'total_pages': total_pages,
        'pages_to_process': total_pages - start_page + 1,
        'pages_processed': 0,
        'pages_skipped': 0,
        'pages_with_errors': 0,
        'total_chars': 0,
        'errors': []
    }
    
    # Time estimator
    time_estimator = TimeEstimator(window_size=10)
    
    # Process PDF
    logger.info(f"Starting extraction from page {start_page} to {total_pages}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Use temp file for atomic writes
            if not dry_run:
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', delete=False, suffix='.txt'
                )
                temp_path = Path(temp_file.name)
            
            # Progress bar
            pages_to_process = total_pages - start_page + 1
            pbar = None if no_progress else tqdm(
                total=pages_to_process,
                desc="Extracting pages",
                unit="page",
                disable=no_progress
            )
            
            try:
                for page_idx in range(start_page - 1, total_pages):
                    page_num = page_idx + 1
                    start_time = time.time()
                    
                    try:
                        # Write page marker
                        page_marker = f"--- PAGE {page_num} ---\n\n"
                        if not dry_run:
                            temp_file.write(page_marker)
                        
                        page = pdf.pages[page_idx]
                        
                        # Double-column layout for pages 16+
                        page_width = page.width
                        page_height = page.height
                        center_x = page_width / 2
                        
                        # Extract LEFT half first
                        left_crop = page.crop((0, 0, center_x, page_height))
                        left_text = left_crop.extract_text() or ""
                        
                        # Extract RIGHT half second
                        right_crop = page.crop((center_x, 0, page_width, page_height))
                        right_text = right_crop.extract_text() or ""
                        
                        # Combine columns
                        combined_text = left_text
                        if right_text:
                            if combined_text:
                                combined_text += "\n\n"
                            combined_text += right_text
                        
                        # Validate extracted text
                        if combined_text:
                            # Sanitize: remove null bytes and control characters
                            combined_text = ''.join(
                                char for char in combined_text 
                                if ord(char) >= 32 or char in '\n\t'
                            )
                            
                            if not dry_run:
                                temp_file.write(combined_text)
                                temp_file.write("\n\n")
                            
                            # Write separator
                            separator = "\n" + "=" * 80 + "\n\n"
                            if not dry_run:
                                temp_file.write(separator)
                            
                            stats['total_chars'] += len(combined_text)
                            stats['pages_processed'] += 1
                            
                            # Debug: save sample pages
                            if debug and page_num in [start_page, start_page + 1, total_pages - 1, total_pages]:
                                debug_dir = output_dir / "debug" / "page_samples"
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                sample_file = debug_dir / f"page_{page_num}_sample.txt"
                                with open(sample_file, 'w', encoding='utf-8') as f:
                                    f.write(f"Page {page_num} Sample\n")
                                    f.write("=" * 80 + "\n\n")
                                    f.write(combined_text[:1000])  # First 1000 chars
                                logger.debug(f"Saved sample for page {page_num}")
                        else:
                            stats['pages_skipped'] += 1
                            logger.warning(f"Page {page_num}: No text extracted")
                        
                        # Update progress bar
                        if pbar:
                            elapsed = time.time() - start_time
                            time_estimator.update(elapsed)
                            eta = time_estimator.get_eta(pages_to_process - (page_idx - start_page + 2))
                            
                            pbar.update(1)
                            pbar.set_postfix({
                                'page': page_num,
                                'chars': stats['total_chars'],
                                'speed': f'{1/elapsed:.2f} pages/s' if elapsed > 0 else 'N/A'
                            })
                        
                        # Checkpoint
                        if not dry_run and (page_num - start_page + 1) % checkpoint_interval == 0:
                            temp_file.flush()
                            logger.info(f"Checkpoint: Processed {page_num - start_page + 1} pages")
                    
                    except Exception as e:
                        stats['pages_with_errors'] += 1
                        error_info = {
                            'page': page_num,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                        stats['errors'].append(error_info)
                        logger.error(f"Error processing page {page_num}: {e}", exc_info=debug)
                        
                        # Continue processing
                        if pbar:
                            pbar.update(1)
                
                # Close temp file
                if not dry_run:
                    temp_file.close()
                    
                    # Atomic move: temp file to final location
                    shutil.move(str(temp_path), str(output_path))
                    logger.info(f"Extraction complete! File saved: {output_path}")
                else:
                    logger.info("DRY RUN complete - no file written")
            
            finally:
                if pbar:
                    pbar.close()
    
    except Exception as e:
        logger.error(f"Fatal error during extraction: {e}", exc_info=True)
        raise
    
    # Print summary
    avg_chars_per_page = stats['total_chars'] / stats['pages_processed'] if stats['pages_processed'] > 0 else 0
    elapsed_time = time_estimator.get_elapsed()
    
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total pages in PDF: {stats['total_pages']}")
    print(f"Pages processed: {stats['pages_processed']}")
    print(f"Pages skipped (empty): {stats['pages_skipped']}")
    print(f"Pages with errors: {stats['pages_with_errors']}")
    print(f"Total characters extracted: {stats['total_chars']:,}")
    print(f"Average characters per page: {avg_chars_per_page:.0f}")
    print(f"Total elapsed time: {time_estimator.format_time(elapsed_time)}")
    
    if not dry_run:
        file_size = output_path.stat().st_size
        print(f"Output file size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        print(f"Output file: {output_path}")
    
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  Page {error['page']}: {error['error']}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more errors")
    
    print("=" * 80)
    
    return output_path if not dry_run else None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract text from dictionary PDF starting at page 16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 1_extract_dictionary_text.py
  python 1_extract_dictionary_text.py --pdf custom.pdf --start-page 20
  python 1_extract_dictionary_text.py --debug --verbose
  python 1_extract_dictionary_text.py --dry-run
        """
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        default='dictionary_20260105.pdf',
        help='PDF file path (default: dictionary_20260105.pdf)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--start-page',
        type=int,
        default=16,
        help='Start page number (default: 16)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (errors only, no progress bar)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't write file, just show what would be extracted"
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=50,
        help='Save checkpoint every N pages (default: 50)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar (for scripts/automation)'
    )
    
    args = parser.parse_args()
    
    # Validate start page
    if args.start_page < 1:
        print("Error: start-page must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    # Setup
    output_dir = Path(args.output_dir)
    log_file = setup_logging(output_dir, debug=args.debug, quiet=args.quiet)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PDF Dictionary Text Extraction")
    logger.info("=" * 80)
    logger.info(f"PDF: {args.pdf}")
    logger.info(f"Start page: {args.start_page}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    if args.debug:
        logger.info("Debug mode enabled")
    if args.dry_run:
        logger.info("Dry run mode enabled")
    
    try:
        output_path = extract_text_from_pdf(
            pdf_path=args.pdf,
            output_dir=output_dir,
            start_page=args.start_page,
            debug=args.debug,
            dry_run=args.dry_run,
            checkpoint_interval=args.checkpoint_interval,
            no_progress=args.quiet or args.no_progress
        )
        
        if output_path:
            print(f"\n✓ Success! Extraction saved to: {output_path}")
            sys.exit(0)
        else:
            print("\n✓ Dry run completed successfully")
            sys.exit(0)
    
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\n\nInterrupted by user. Partial extraction may be incomplete.")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

