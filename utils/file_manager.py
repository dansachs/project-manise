#!/usr/bin/env python3
"""
File operations with failsafes: backup, save, restore.
"""

import json
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any


class FileManager:
    """Manage file operations with safety checks and backups."""
    
    def __init__(self, working_file: Path, backup_suffix: str = "_original.json"):
        """
        Initialize file manager.
        
        Args:
            working_file: Path to working file
            backup_suffix: Suffix for backup file
        """
        self.working_file = Path(working_file)
        self.backup_file = self.working_file.parent / f"{self.working_file.stem}{backup_suffix}{self.working_file.suffix}"
        self.logger = logging.getLogger(__name__)
    
    def ensure_backup(self) -> bool:
        """
        Ensure backup file exists. Create it if it doesn't.
        
        Returns:
            True if backup exists or was created, False otherwise
        """
        if self.backup_file.exists():
            # Verify backup is valid JSON
            try:
                with open(self.backup_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                return True
            except json.JSONDecodeError:
                self.logger.warning(f"Backup file {self.backup_file} is corrupted")
                # Try to create new backup from working file
                if self.working_file.exists():
                    return self.create_backup()
                return False
        
        # Create backup from working file
        if self.working_file.exists():
            return self.create_backup()
        
        return False
    
    def create_backup(self) -> bool:
        """
        Create backup from working file.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.working_file.exists():
            self.logger.warning(f"Working file {self.working_file} does not exist, cannot create backup")
            return False
        
        try:
            # Verify working file is valid JSON before backing up
            with open(self.working_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create backup
            shutil.copy2(self.working_file, self.backup_file)
            self.logger.info(f"Created backup: {self.backup_file}")
            return True
        except json.JSONDecodeError as e:
            self.logger.error(f"Working file is corrupted, cannot create backup: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def check_disk_space(self, required_bytes: int = 10 * 1024 * 1024) -> tuple:
        """
        Check available disk space.
        
        Args:
            required_bytes: Minimum required bytes (default: 10MB)
        
        Returns:
            Tuple of (has_space, available_bytes)
        """
        try:
            import shutil
            stat = shutil.disk_usage(self.working_file.parent)
            available = stat.free
            has_space = available >= required_bytes
            return has_space, available
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            return True, 0  # Assume we have space if we can't check
    
    def save_file(self, data: Dict[str, Any]) -> bool:
        """
        Save data to working file atomically.
        
        Args:
            data: Data dictionary to save
        
        Returns:
            True if successful, False otherwise
        """
        # Check disk space
        # Estimate file size (rough estimate: JSON string length * 1.5)
        estimated_size = len(json.dumps(data, ensure_ascii=False)) * 2
        has_space, available = self.check_disk_space(estimated_size)
        
        if not has_space:
            self.logger.error(f"Insufficient disk space. Available: {available / 1024 / 1024:.2f} MB")
            return False
        
        # Validate data structure
        if not isinstance(data, dict):
            self.logger.error("Data must be a dictionary")
            return False
        
        # Atomic write: write to temp file, then rename
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            delete=False,
            suffix='.json',
            dir=self.working_file.parent
        )
        temp_path = Path(temp_file.name)
        
        try:
            # Write to temp file
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.close()
            
            # Verify temp file is valid JSON
            with open(temp_path, 'r', encoding='utf-8') as f:
                json.load(f)
            
            # Backup current file if it exists
            if self.working_file.exists():
                backup_temp = self.backup_file.with_suffix('.json.backup')
                try:
                    shutil.copy2(self.working_file, backup_temp)
                except Exception as e:
                    self.logger.warning(f"Could not create temporary backup: {e}")
            
            # Atomic move
            shutil.move(str(temp_path), str(self.working_file))
            self.logger.info(f"Saved file: {self.working_file}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON data: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
        except Exception as e:
            self.logger.error(f"Failed to save file: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def load_file(self) -> Optional[Dict[str, Any]]:
        """
        Load data from working file.
        
        Returns:
            Loaded data dictionary or None if failed
        """
        if not self.working_file.exists():
            self.logger.error(f"Working file does not exist: {self.working_file}")
            return None
        
        try:
            with open(self.working_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded file: {self.working_file}")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            # Try to load partial data
            try:
                with open(self.working_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Try to find where error occurs
                    self.logger.error(f"Corrupted JSON at approximately line {content[:1000].count(chr(10))}")
            except:
                pass
            return None
        except Exception as e:
            self.logger.error(f"Failed to load file: {e}")
            return None
    
    def restore_from_backup(self) -> bool:
        """
        Restore working file from backup.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.backup_file.exists():
            self.logger.error(f"Backup file does not exist: {self.backup_file}")
            return False
        
        try:
            # Verify backup is valid
            with open(self.backup_file, 'r', encoding='utf-8') as f:
                json.load(f)
            
            # Restore
            shutil.copy2(self.backup_file, self.working_file)
            self.logger.info(f"Restored from backup: {self.backup_file}")
            return True
        except json.JSONDecodeError as e:
            self.logger.error(f"Backup file is corrupted: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def find_most_recent_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """
        Find most recent file matching pattern in directory.
        
        Args:
            directory: Directory to search
            pattern: Filename pattern (e.g., "progress_*.json")
        
        Returns:
            Path to most recent file or None
        """
        if not directory.exists():
            return None
        
        files = list(directory.glob(pattern))
        if not files:
            return None
        
        # Sort by modification time (most recent first)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0]

