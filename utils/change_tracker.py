#!/usr/bin/env python3
"""
Change tracking and logging module.
Tracks all edits, deletions, additions, and validations to entries.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ChangeTracker:
    """Track and log all changes made to entries."""
    
    def __init__(self, log_file: Path = None):
        """
        Initialize change tracker.
        
        Args:
            log_file: Path to change log file (default: viewer_changes.log in current directory)
        """
        if log_file is None:
            log_file = Path.cwd() / "viewer_changes.log"
        else:
            log_file = Path(log_file)
        
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
        # Ensure log file exists
        if not log_file.exists():
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Change Log - Started {datetime.now().isoformat()}\n")
                f.write("# Format: [TIMESTAMP] [ENTRY_ID] [CHANGE_TYPE] [FIELD] [OLD_VALUE] -> [NEW_VALUE]\n\n")
    
    def _get_entry_id(self, entry: Dict[str, Any]) -> str:
        """Generate entry ID from entry data."""
        headword = entry.get('headword', 'unknown')
        page = entry.get('page_number', '?')
        # Use index if available
        if 'entry_index' in entry:
            return f"entry_{entry['entry_index']}"
        return f"{headword}_p{page}"
    
    def log_change(
        self,
        change_type: str,
        entry: Dict[str, Any],
        field: Optional[str] = None,
        old_value: Any = None,
        new_value: Any = None,
        entry_index: Optional[int] = None
    ):
        """
        Log a change to the change log file.
        
        Args:
            change_type: Type of change (EDIT, DELETE, ADD, VALIDATE, FLAG, UNFLAG)
            entry: Entry dictionary
            field: Field name that changed (for EDIT)
            old_value: Old value (for EDIT)
            new_value: New value (for EDIT)
            entry_index: Optional entry index
        """
        timestamp = datetime.now().isoformat()
        entry_id = self._get_entry_id(entry)
        
        if entry_index is not None:
            entry_id = f"entry_{entry_index}"
        
        # Format the log line
        if change_type == 'EDIT' and field:
            old_str = str(old_value)[:100] if old_value is not None else 'None'
            new_str = str(new_value)[:100] if new_value is not None else 'None'
            log_line = f"[{timestamp}] [{entry_id}] [{change_type}] [{field}] [{old_str}] -> [{new_str}]\n"
        elif change_type in ('DELETE', 'ADD'):
            headword = entry.get('headword', 'unknown')
            page = entry.get('page_number', '?')
            log_line = f"[{timestamp}] [{entry_id}] [{change_type}] headword={headword} page={page}\n"
        elif change_type in ('VALIDATE', 'FLAG', 'UNFLAG'):
            status = entry.get('validation_status', 'unknown')
            log_line = f"[{timestamp}] [{entry_id}] [{change_type}] status={status}\n"
        else:
            log_line = f"[{timestamp}] [{entry_id}] [{change_type}]\n"
        
        # Write to log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
            self.logger.debug(f"Logged change: {change_type} for {entry_id}")
        except Exception as e:
            self.logger.error(f"Failed to write to change log: {e}")
    
    def log_edit(self, entry: Dict[str, Any], field: str, old_value: Any, new_value: Any, entry_index: Optional[int] = None):
        """Log an edit to a field."""
        self.log_change('EDIT', entry, field, old_value, new_value, entry_index)
    
    def log_delete(self, entry: Dict[str, Any], entry_index: Optional[int] = None):
        """Log entry deletion."""
        self.log_change('DELETE', entry, entry_index=entry_index)
    
    def log_add(self, entry: Dict[str, Any], entry_index: Optional[int] = None):
        """Log entry addition."""
        self.log_change('ADD', entry, entry_index=entry_index)
    
    def log_validate(self, entry: Dict[str, Any], entry_index: Optional[int] = None):
        """Log entry validation."""
        self.log_change('VALIDATE', entry, entry_index=entry_index)
    
    def log_flag(self, entry: Dict[str, Any], entry_index: Optional[int] = None):
        """Log entry flagging."""
        self.log_change('FLAG', entry, entry_index=entry_index)
    
    def log_unflag(self, entry: Dict[str, Any], entry_index: Optional[int] = None):
        """Log entry unflagging."""
        self.log_change('UNFLAG', entry, entry_index=entry_index)

