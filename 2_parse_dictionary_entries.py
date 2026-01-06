#!/usr/bin/env python3
"""
Script 2: Dictionary Entry Parser
Uses Google Gemini API to parse dictionary entries into structured format.
Handles cross-page entries, progress tracking, and comprehensive error handling.
RUN THIS SCRIPT SECOND (after 1_extract_dictionary_text.py)
"""

import re
import json
import time
import argparse
import logging
import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from collections import deque
from tqdm import tqdm
from google import genai


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


def find_most_recent_extraction(extractions_dir: Path) -> Optional[Path]:
    """Find the most recent extraction file."""
    if not extractions_dir.exists():
        return None
    
    extraction_files = list(extractions_dir.glob("extraction_*.txt"))
    if not extraction_files:
        return None
    
    # Sort by modification time (most recent first)
    extraction_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return extraction_files[0]


def validate_input_file(input_path: Path) -> bool:
    """Validate input file exists and has correct format."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Path is not a file: {input_path}")
    
    # Check for page markers
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Read first 1000 chars
            if '--- PAGE' not in content:
                raise ValueError("Input file does not contain page markers (--- PAGE X ---)")
    except UnicodeDecodeError:
        raise ValueError("Input file is not UTF-8 encoded")
    
    return True


class EntryParser:
    """Parse dictionary entries using Gemini API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        output_dir: Path = None,
        debug: bool = False
    ):
        """
        Initialize the entry parser with Gemini API.
        
        Args:
            api_key: Google API key (if None, will try to get from environment)
            model_name: Name of the Gemini model to use
            output_dir: Directory for output files
            debug: Enable debug mode
        """
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            # Try GEMINI_API_KEY first (new standard), then GOOGLE_API_KEY (legacy)
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError(
                    "API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                    "environment variable or pass api_key parameter."
                )
            self.client = genai.Client(api_key=api_key)
        
        self.model_name = model_name
        self.entries = []
        self.processed_pages = set()
        self.failed_pages = []
        self.error_log = []
        self.output_dir = output_dir or Path("outputs")
        self.progress_dir = self.output_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'api_calls': 0,
            'api_success': 0,
            'api_errors': 0,
            'average_time_per_page': 0.0,
            'total_elapsed_time': 0.0,
            'average_entries_per_page': 0.0
        }
        
        # Progress file will be set when processing starts
        self.progress_file = None
        self.extraction_file = None
    
    def load_progress(self, progress_file: Path):
        """Load previously processed pages and entries."""
        self.progress_file = progress_file
        
        if not progress_file.exists():
            logger = logging.getLogger(__name__)
            logger.info("No existing progress file, starting fresh")
            return
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.entries = data.get('entries', [])
            self.processed_pages = set(data.get('processed_pages', []))
            self.failed_pages = data.get('failed_pages', [])
            self.error_log = data.get('error_log', [])
            self.extraction_file = data.get('extraction_file')
            self.stats = data.get('statistics', self.stats)
            
            logger = logging.getLogger(__name__)
            logger.info(
                f"Loaded progress: {len(self.entries)} entries from "
                f"{len(self.processed_pages)} pages"
            )
        except json.JSONDecodeError as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Progress file corrupted, starting fresh: {e}")
            # Try to backup corrupted file
            backup_file = progress_file.with_suffix('.json.backup')
            shutil.copy(progress_file, backup_file)
            logger.info(f"Corrupted file backed up to: {backup_file}")
    
    def save_progress(self, extraction_file: str, started_at: str):
        """Save current progress atomically."""
        if not self.progress_file:
            return
        
        data = {
            'extraction_file': extraction_file,
            'started_at': started_at,
            'last_updated': datetime.now().isoformat(),
            'total_pages': self.stats.get('total_pages', 0),
            'processed_pages': sorted(list(self.processed_pages)),
            'failed_pages': self.failed_pages,
            'entries': self.entries,
            'statistics': self.stats,
            'error_log': self.error_log
        }
        
        # Atomic write: write to temp file, then rename
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', delete=False, suffix='.json',
            dir=self.progress_dir
        )
        temp_path = Path(temp_file.name)
        
        try:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.close()
            
            # Backup previous progress file
            if self.progress_file.exists():
                backup_file = self.progress_file.with_suffix('.json.backup')
                shutil.copy(self.progress_file, backup_file)
            
            # Atomic move
            shutil.move(str(temp_path), str(self.progress_file))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving progress: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def clear_progress(self, progress_file: Path):
        """Clear all progress and reset state."""
        if progress_file.exists():
            # Backup before deleting
            backup_file = progress_file.with_suffix('.json.backup')
            shutil.copy(progress_file, backup_file)
            progress_file.unlink()
            logger = logging.getLogger(__name__)
            logger.info(f"Progress cleared. Previous file backed up to: {backup_file}")
        
        self.entries = []
        self.processed_pages = set()
        self.failed_pages = []
        self.error_log = []
        self.stats = {
            'total_entries': 0,
            'api_calls': 0,
            'api_success': 0,
            'api_errors': 0,
            'average_time_per_page': 0.0,
            'total_elapsed_time': 0.0,
            'average_entries_per_page': 0.0
        }
    
    def safe_api_call(
        self,
        prompt: str,
        page_num: int,
        max_retries: int = 3,
        timeout: int = 60
    ) -> Optional[str]:
        """
        Make API call with retry logic and error handling.
        
        Args:
            prompt: The prompt to send to Gemini
            page_num: Page number for logging
            max_retries: Maximum number of retry attempts
            timeout: API call timeout in seconds
            
        Returns:
            Response text or None if all retries failed
        """
        logger = logging.getLogger(__name__)
        
        for attempt in range(max_retries):
            try:
                self.stats['api_calls'] += 1
                start_time = time.time()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "max_output_tokens": 4000,
                    }
                )
                
                elapsed = time.time() - start_time
                logger.debug(f"API call for page {page_num} took {elapsed:.2f}s")
                
                response_text = response.text if hasattr(response, 'text') else str(response)
                self.stats['api_success'] += 1
                return response_text
                
            except Exception as e:
                self.stats['api_errors'] += 1
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check for rate limiting
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Rate limited on page {page_num}, attempt {attempt + 1}, "
                        f"waiting {wait_time}s..."
                    )
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                
                logger.error(
                    f"API error on page {page_num}, attempt {attempt + 1}/{max_retries}: "
                    f"{error_type}: {error_msg}"
                )
                
                if attempt == max_retries - 1:
                    # Last attempt failed
                    self.error_log.append({
                        'page': page_num,
                        'error': f'{error_type}: {error_msg}',
                        'error_type': error_type,
                        'attempts': attempt + 1
                    })
                    return None
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        return None
    
    def parse_entry_block(
        self,
        entry_text: str,
        page_num: int,
        is_continuation: bool = False,
        max_retries: int = 3,
        timeout: int = 60
    ) -> List[Dict]:
        """
        Use Gemini to parse an entry block into structured format.
        
        Args:
            entry_text: The text content of the entry
            page_num: Page number for context
            is_continuation: Whether this is a continuation from previous page
            max_retries: Maximum retry attempts
            timeout: API call timeout
            
        Returns:
            List of parsed entry dictionaries
        """
        logger = logging.getLogger(__name__)
        
        # Sanitize input text
        entry_text = ''.join(
            char for char in entry_text 
            if ord(char) >= 32 or char in '\n\t'
        )
        
        continuation_note = (
            "\nNOTE: This entry continues from the previous page."
            if is_continuation else ""
        )
        
        prompt = f"""You are parsing entries from an Ambonese-Indonesian dictionary (page {page_num}).

TASK: Parse ALL dictionary entries from the text into structured format. Each entry should have these parts:
1. headword: The main Ambonese word being defined
2. definition: The Indonesian translation/definition
3. ambonese_example: The example sentence in Ambonese (with -- placeholder where headword appears)
4. indonesian_translation: The Indonesian translation of the example

ENTRY STRUCTURE:
- Main entries: "headword number definition: Ambonese_example Indonesian_translation;"
- Sub-entries: "-- subword definition: Ambonese_example Indonesian_translation;"
- Some entries have multiple numbered definitions (1, 2, 3, etc.)
- Some entries only have headword and definition (no examples)
- The "--" in Ambonese examples is a placeholder for the headword

EXAMPLES:
Input: "ada 1 ada: kalu dudu digi balakang seng -- kadengarang kalau duduk di belakang tidak kedengaran; 2 hadir: sagala waktu beta -- setiap menit saya hadir;"
Output:
{{
  "headword": "ada",
  "meanings": [
    {{
      "sense_number": 1,
      "definition": "ada",
      "ambonese_example": "kalu dudu digi balakang seng -- kadengarang",
      "indonesian_translation": "kalau duduk di belakang tidak kedengaran"
    }},
    {{
      "sense_number": 2,
      "definition": "hadir",
      "ambonese_example": "sagala waktu beta --",
      "indonesian_translation": "setiap menit saya hadir"
    }}
  ]
}}

Input: "abu debu: makanang tu su pono -- makanan itu sudah penuh debu"
Output:
{{
  "headword": "abu",
  "meanings": [
    {{
      "sense_number": 1,
      "definition": "debu",
      "ambonese_example": "makanang tu su pono --",
      "indonesian_translation": "makanan itu sudah penuh debu"
    }}
  ]
}}

RULES:
1. Extract ALL entries from the text (main entries and sub-entries)
2. For entries with multiple meanings, create a "meanings" array
3. For simple entries, use a single meaning object
4. For sub-entries (starting with --), set "is_subentry": true
5. If an entry has no examples, set ambonese_example and indonesian_translation to empty strings
6. Clean up whitespace and line breaks in examples
7. Return a JSON array of all entries found

{continuation_note}

Entry text to parse:
{entry_text}

Return ONLY valid JSON array, no explanations:"""

        # Make API call with retries
        response_text = self.safe_api_call(prompt, page_num, max_retries, timeout)
        
        if not response_text:
            logger.warning(f"No response from API for page {page_num}")
            return []
        
        # Save API response for debugging
        if self.debug:
            debug_dir = self.output_dir / "debug" / "api_responses"
            debug_dir.mkdir(parents=True, exist_ok=True)
            response_file = debug_dir / f"page_{page_num}_response.json"
            try:
                with open(response_file, 'w', encoding='utf-8') as f:
                    json.dump({'prompt': prompt[:500], 'response': response_text}, f, indent=2)
            except Exception as e:
                logger.debug(f"Could not save API response: {e}")
        
        # Extract JSON from response (might have markdown code blocks)
        response_text = response_text.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            if len(lines) > 2:
                response_text = '\n'.join(lines[1:-1])
        elif response_text.startswith('```json'):
            lines = response_text.split('\n')
            if len(lines) > 2:
                response_text = '\n'.join(lines[1:-1])
        
        # Parse JSON
        try:
            parsed_entries = json.loads(response_text)
            if not isinstance(parsed_entries, list):
                parsed_entries = [parsed_entries]
            
            # Validate and add metadata
            validated_entries = []
            for entry in parsed_entries:
                # Validate required fields
                if 'headword' not in entry:
                    logger.warning(f"Entry missing headword on page {page_num}: {entry}")
                    continue
                
                # Ensure meanings array exists
                if 'meanings' not in entry:
                    # Try to convert single meaning to array
                    if 'definition' in entry:
                        entry['meanings'] = [{
                            'sense_number': 1,
                            'definition': entry.get('definition', ''),
                            'ambonese_example': entry.get('ambonese_example', ''),
                            'indonesian_translation': entry.get('indonesian_translation', '')
                        }]
                    else:
                        logger.warning(f"Entry missing meanings on page {page_num}: {entry}")
                        continue
                
                # Add metadata
                entry['page_number'] = page_num
                entry['source_text'] = entry_text[:500]  # Keep first 500 chars for reference
                
                validated_entries.append(entry)
            
            return validated_entries
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for page {page_num}: {e}")
            logger.debug(f"Response text (first 500 chars): {response_text[:500]}")
            
            self.error_log.append({
                'page': page_num,
                'error': f'JSON decode error: {e}',
                'error_type': 'JSONDecodeError',
                'response_preview': response_text[:500]
            })
            
            # Save problematic page for debugging
            if self.debug:
                debug_dir = self.output_dir / "debug" / "problematic_pages"
                debug_dir.mkdir(parents=True, exist_ok=True)
                problem_file = debug_dir / f"page_{page_num}_problem.txt"
                with open(problem_file, 'w', encoding='utf-8') as f:
                    f.write(f"Page {page_num} - JSON Parse Error\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Error: {e}\n\n")
                    f.write("Response:\n")
                    f.write(response_text)
                    f.write("\n\nOriginal Text:\n")
                    f.write(entry_text[:2000])
            
            return []
    
    def prepare_page_content(
        self,
        page_content: str,
        prev_page_tail: Optional[str] = None
    ) -> str:
        """
        Prepare page content for parsing, including context from previous page.
        
        Args:
            page_content: The text content of the current page
            prev_page_tail: Last 500 chars of previous page (for continuation detection)
            
        Returns:
            Prepared page text with context
        """
        # Clean up the page content
        lines = page_content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        page_text = '\n'.join(cleaned_lines)
        
        # If we have previous page context, prepend it (limit to 500 chars)
        if prev_page_tail:
            context_text = prev_page_tail[-500:] + '\n' + page_text
            return context_text
        
        return page_text
    
    def process_file(
        self,
        input_file: str,
        output_file: Path,
        progress_file: Path,
        delay_seconds: float = 1.0,
        start_page: int = 16,
        max_retries: int = 3,
        timeout: int = 60,
        max_pages: Optional[int] = None,
        no_progress: bool = False,
        dry_run: bool = False
    ):
        """
        Process the dictionary file page by page.
        
        Args:
            input_file: Path to dictionary file
            output_file: Path to save parsed entries (JSON format)
            progress_file: Path to progress file
            delay_seconds: Delay between API calls
            start_page: Page number to start from
            max_retries: Maximum retries for failed pages
            timeout: API call timeout
            max_pages: Maximum number of pages to process (None = process all)
            no_progress: Disable progress bar
            dry_run: Don't save, just test
        """
        logger = logging.getLogger(__name__)
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_file}")
        
        # Load existing progress
        self.load_progress(progress_file)
        
        logger.info(f"Reading file: {input_file}")
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by page markers
        page_sections = re.split(r'---\s*PAGE\s*(\d+)\s*---', content)
        
        # Parse pages
        pages = []
        for i in range(1, len(page_sections), 2):
            if i + 1 < len(page_sections):
                page_num = int(page_sections[i])
                page_content = page_sections[i + 1]
                # Clean up separators
                page_content = re.sub(r'={10,}', '', page_content).strip()
                if page_content:
                    pages.append({
                        'number': page_num,
                        'content': page_content
                    })
        
        total_pages = len([p for p in pages if p['number'] >= start_page])
        self.stats['total_pages'] = total_pages
        
        logger.info(f"Found {len(pages)} pages total, {total_pages} pages from page {start_page} onwards")
        
        if max_pages:
            logger.info(f"Test mode: Processing maximum {max_pages} pages")
        
        # Process pages
        pages_processed = 0
        prev_page_tail = None
        started_at = datetime.now().isoformat()
        time_estimator = TimeEstimator(window_size=10)
        
        # Filter pages to process
        pages_to_process = [p for p in pages if p['number'] >= start_page]
        if max_pages:
            pages_to_process = pages_to_process[:max_pages]
        
        # Progress bar
        pbar = None if no_progress else tqdm(
            total=len(pages_to_process),
            desc="Parsing entries",
            unit="page",
            disable=no_progress
        )
        
        try:
            for page in pages_to_process:
                page_num = page['number']
                page_content = page['content']
                
                # Skip already processed pages
                if page_num in self.processed_pages:
                    logger.debug(f"Page {page_num}: Already processed, skipping...")
                    prev_page_tail = page_content[-500:] if len(page_content) > 500 else page_content
                    if pbar:
                        pbar.update(1)
                    continue
                
                start_time = time.time()
                logger.info(f"Processing page {page_num}...")
                
                # Prepare page content with context from previous page
                prepared_content = self.prepare_page_content(page_content, prev_page_tail)
                has_continuation = prev_page_tail is not None
                
                # Process the entire page at once
                parsed_entries = []
                for attempt in range(max_retries):
                    parsed_entries = self.parse_entry_block(
                        prepared_content,
                        page_num,
                        is_continuation=has_continuation,
                        max_retries=1,  # We handle retries at this level
                        timeout=timeout
                    )
                    if parsed_entries:
                        break
                    elif attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries - 1} for page {page_num}...")
                        time.sleep(delay_seconds * 2)
                
                elapsed = time.time() - start_time
                time_estimator.update(elapsed)
                
                if parsed_entries:
                    self.entries.extend(parsed_entries)
                    self.stats['total_entries'] = len(self.entries)
                    logger.info(f"  Found {len(parsed_entries)} entry/entries")
                else:
                    logger.warning(f"  No entries parsed from page {page_num}")
                    self.failed_pages.append(page_num)
                
                # Update prev_page_tail for next iteration
                prev_page_tail = page_content[-500:] if len(page_content) > 500 else page_content
                
                self.processed_pages.add(page_num)
                pages_processed += 1
                
                # Update statistics
                if pages_processed > 0:
                    self.stats['average_time_per_page'] = (
                        sum(time_estimator.times) / len(time_estimator.times)
                        if time_estimator.times else 0
                    )
                    self.stats['average_entries_per_page'] = (
                        self.stats['total_entries'] / pages_processed
                    )
                    self.stats['total_elapsed_time'] = time_estimator.get_elapsed()
                
                # Save progress after each page (if not dry run)
                if not dry_run:
                    self.save_progress(input_path.name, started_at)
                
                # Update progress bar
                if pbar:
                    remaining = len(pages_to_process) - pages_processed
                    eta = time_estimator.get_eta(remaining)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'page': page_num,
                        'entries': self.stats['total_entries'],
                        'api_calls': self.stats['api_calls'],
                        'speed': f'{1/elapsed:.2f} pages/s' if elapsed > 0 else 'N/A'
                    })
                
                # Rate limiting
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        
        finally:
            if pbar:
                pbar.close()
        
        # Save final results
        if not dry_run:
            # Atomic write for output file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', delete=False, suffix='.json',
                dir=output_file.parent
            )
            temp_path = Path(temp_file.name)
            
            try:
                output_data = {
                    'extraction_file': input_path.name,
                    'started_at': started_at,
                    'completed_at': datetime.now().isoformat(),
                    'total_pages': self.stats['total_pages'],
                    'processed_pages': sorted(list(self.processed_pages)),
                    'failed_pages': self.failed_pages,
                    'entries': self.entries,
                    'statistics': self.stats,
                    'error_log': self.error_log
                }
                
                json.dump(output_data, temp_file, indent=2, ensure_ascii=False)
                temp_file.close()
                
                shutil.move(str(temp_path), str(output_file))
                logger.info(f"Final output saved to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving final output: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                raise
        
        # Print summary
        elapsed_time = time_estimator.get_elapsed()
        print("\n" + "=" * 80)
        print("PARSING SUMMARY")
        print("=" * 80)
        print(f"Total pages processed: {len(self.processed_pages)}")
        print(f"Pages with errors: {len(self.failed_pages)}")
        print(f"Total entries parsed: {self.stats['total_entries']}")
        print(f"Average entries per page: {self.stats['average_entries_per_page']:.2f}")
        print(f"API calls: {self.stats['api_calls']}")
        print(f"API success: {self.stats['api_success']}")
        print(f"API errors: {self.stats['api_errors']}")
        print(f"Average time per page: {self.stats['average_time_per_page']:.2f}s")
        print(f"Total elapsed time: {time_estimator.format_time(elapsed_time)}")
        
        if not dry_run:
            print(f"\nResults saved to: {output_file}")
            print(f"Progress saved to: {progress_file}")
        
        if self.error_log:
            print(f"\n⚠ Errors encountered: {len(self.error_log)}")
            for error in self.error_log[:5]:
                print(f"  Page {error['page']}: {error['error']}")
            if len(self.error_log) > 5:
                print(f"  ... and {len(self.error_log) - 5} more errors")
        
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse dictionary entries using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 2_parse_dictionary_entries.py
  python 2_parse_dictionary_entries.py --input custom.txt
  python 2_parse_dictionary_entries.py --test --debug
  python 2_parse_dictionary_entries.py --clear --start-page 20
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input extraction file (default: most recent in outputs/extractions)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--progress-dir',
        type=str,
        default=None,
        help='Progress directory (default: outputs/progress)'
    )
    parser.add_argument(
        '--start-page',
        type=int,
        default=16,
        help='Start from page N (default: 16)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Process only N pages (for testing)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between API calls in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Max retries per page (default: 3)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='API call timeout in seconds (default: 60)'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear progress and start fresh'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode (process 10 pages)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
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
        '--dry-run',
        action='store_true',
        help="Don't save, just test"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate input file only'
    )
    parser.add_argument(
        '--stats',
        type=str,
        default=None,
        help='Show statistics from progress file'
    )
    parser.add_argument(
        '--check',
        type=str,
        default=None,
        help='Check progress file integrity'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Pause on errors'
    )
    parser.add_argument(
        '--page',
        type=int,
        default=None,
        help='Process only page N'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar (for scripts/automation)'
    )
    
    args = parser.parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir)
    progress_dir = Path(args.progress_dir) if args.progress_dir else output_dir / "progress"
    parsed_dir = output_dir / "parsed"
    extractions_dir = output_dir / "extractions"
    
    # Setup logging
    log_file = setup_logging(output_dir, debug=args.debug, quiet=args.quiet)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Dictionary Entry Parser")
    logger.info("=" * 80)
    
    # Handle special commands
    if args.stats:
        progress_file = Path(args.stats)
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            stats = data.get('statistics', {})
            print("\nStatistics from progress file:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"Progress file not found: {progress_file}", file=sys.stderr)
            sys.exit(1)
        return
    
    if args.check:
        progress_file = Path(args.check)
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Progress file is valid: {progress_file}")
            print(f"  Entries: {len(data.get('entries', []))}")
            print(f"  Processed pages: {len(data.get('processed_pages', []))}")
        except Exception as e:
            print(f"✗ Progress file is corrupted: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")
        api_key = input("Enter your Google API key (or press Enter to exit): ").strip()
        if not api_key:
            print("Exiting...")
            sys.exit(1)
    
    # Find input file
    if args.input:
        input_file = Path(args.input)
        validate_input_file(input_file)
    else:
        input_file = find_most_recent_extraction(extractions_dir)
        if not input_file:
            print(
                f"Error: No extraction file found in {extractions_dir}",
                file=sys.stderr
            )
            print("Please run 1_extract_dictionary_text.py first.", file=sys.stderr)
            sys.exit(1)
        logger.info(f"Using most recent extraction: {input_file}")
    
    if args.validate:
        validate_input_file(input_file)
        print(f"✓ Input file is valid: {input_file}")
        return
    
    # Generate output filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = parsed_dir / f"entries_{timestamp}.json"
    progress_file = progress_dir / f"progress_{timestamp}.json"
    
    parsed_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    
    # Create parser
    parser_obj = EntryParser(
        api_key=api_key,
        model_name="gemini-2.0-flash-exp",
        output_dir=output_dir,
        debug=args.debug
    )
    
    # Clear progress if requested
    if args.clear:
        parser_obj.clear_progress(progress_file)
        print()
    
    # Test mode
    if args.test:
        args.max_pages = 10
        print("=" * 60)
        print("TEST MODE: Processing only 10 pages")
        print("=" * 60 + "\n")
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Progress file: {progress_file}")
    logger.info(f"Start page: {args.start_page}")
    logger.info(f"Log file: {log_file}")
    
    try:
        parser_obj.process_file(
            input_file=str(input_file),
            output_file=output_file,
            progress_file=progress_file,
            delay_seconds=args.delay,
            start_page=args.start_page,
            max_retries=args.max_retries,
            timeout=args.timeout,
            max_pages=args.max_pages,
            no_progress=args.quiet or args.no_progress,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            print(f"\n✓ Success! Results saved to: {output_file}")
        else:
            print("\n✓ Dry run completed successfully")
        
        sys.exit(0)
    
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\n\nInterrupted by user. Progress has been saved.")
        parser_obj.save_progress(str(input_file), datetime.now().isoformat())
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

