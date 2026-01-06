#!/usr/bin/env python3
"""
Data validation functions.
"""

import re
import logging
from typing import Dict, Any, List, Optional


class EntryValidator:
    """Validate dictionary entry structure and data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_entry_structure(self, entry: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate entry structure.
        
        Args:
            entry: Entry dictionary
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        if 'headword' not in entry:
            errors.append("Missing required field: headword")
        elif not entry['headword'] or not isinstance(entry['headword'], str):
            errors.append("headword must be a non-empty string")
        
        if 'page_number' not in entry:
            errors.append("Missing required field: page_number")
        elif not isinstance(entry['page_number'], int) or entry['page_number'] < 1:
            errors.append("page_number must be a positive integer")
        
        if 'meanings' not in entry:
            errors.append("Missing required field: meanings")
        elif not isinstance(entry['meanings'], list):
            errors.append("meanings must be a list")
        elif len(entry['meanings']) == 0:
            errors.append("meanings list cannot be empty")
        else:
            # Validate each meaning
            for i, meaning in enumerate(entry['meanings']):
                if not isinstance(meaning, dict):
                    errors.append(f"meaning {i} must be a dictionary")
                    continue
                
                if 'definition' not in meaning:
                    errors.append(f"meaning {i} missing required field: definition")
                if 'sense_number' not in meaning:
                    errors.append(f"meaning {i} missing required field: sense_number")
        
        return len(errors) == 0, errors
    
    def validate_edit(self, entry: Dict[str, Any], field: str, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate an edit to an entry field.
        
        Args:
            entry: Entry dictionary
            field: Field name
            value: New value
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if field == 'headword':
            if not value or not isinstance(value, str) or len(value.strip()) == 0:
                return False, "Headword cannot be empty"
        
        elif field == 'page_number':
            if not isinstance(value, int) or value < 1:
                return False, "Page number must be a positive integer"
        
        elif field in ('definition', 'ambonese_example', 'indonesian_translation'):
            if not isinstance(value, str):
                return False, f"{field} must be a string"
        
        elif field == 'sense_number':
            if not isinstance(value, int) or value < 1:
                return False, "Sense number must be a positive integer"
        
        return True, None
    
    def check_duplicate_entries(self, entries: List[Dict[str, Any]]) -> List[tuple[int, int]]:
        """
        Check for duplicate entries (same headword + page).
        
        Args:
            entries: List of entries
        
        Returns:
            List of tuples (index1, index2) for duplicate pairs
        """
        duplicates = []
        seen = {}
        
        for i, entry in enumerate(entries):
            headword = entry.get('headword', '')
            page = entry.get('page_number', 0)
            key = (headword, page)
            
            if key in seen:
                duplicates.append((seen[key], i))
            else:
                seen[key] = i
        
        return duplicates
    
    def validate_page_range(self, entries: List[Dict[str, Any]], max_page: int) -> List[int]:
        """
        Validate page numbers are within range.
        
        Args:
            entries: List of entries
            max_page: Maximum valid page number
        
        Returns:
            List of entry indices with invalid page numbers
        """
        invalid = []
        
        for i, entry in enumerate(entries):
            page = entry.get('page_number', 0)
            if page < 1 or page > max_page:
                invalid.append(i)
        
        return invalid

