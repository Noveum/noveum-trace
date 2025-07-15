"""
File-based sink for the Noveum Trace SDK.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .base import BaseSink, SinkConfig
from ..types import SpanData
from ..utils.exceptions import SinkError

logger = logging.getLogger(__name__)


class FileSinkConfig(SinkConfig):
    """Configuration for file sink."""
    
    def __init__(
        self,
        name: str = "file-sink",
        directory: str = "./traces",
        file_format: str = "jsonl",  # jsonl, json, or csv
        max_file_size_mb: int = 100,
        max_files: int = 10,
        compress_old_files: bool = True,
        include_timestamp: bool = True,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.directory = directory
        self.file_format = file_format
        self.max_file_size_mb = max_file_size_mb
        self.max_files = max_files
        self.compress_old_files = compress_old_files
        self.include_timestamp = include_timestamp


class FileSink(BaseSink):
    """File-based sink that writes spans to local files."""
    
    def __init__(self, config: Optional[FileSinkConfig] = None):
        """Initialize the file sink."""
        if config is None:
            config = FileSinkConfig()
        
        # Initialize attributes before calling parent constructor
        self._config: FileSinkConfig = config
        self._current_file = None
        self._current_file_size = 0
        self._file_counter = 0
        
        super().__init__(config)
    
    def _initialize(self) -> None:
        """Initialize the file sink."""
        # Create directory if it doesn't exist
        Path(self._config.directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize first file
        self._rotate_file()
        
        logger.info(f"FileSink initialized - directory: {self._config.directory}")
    
    def _send_batch(self, spans: List[SpanData]) -> None:
        """Send a batch of spans to file."""
        if not spans:
            return
        
        try:
            for span in spans:
                self._write_span(span)
            
            # Flush to ensure data is written
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
                
        except Exception as e:
            logger.error(f"Failed to write spans to file: {e}")
            raise SinkError(f"File write failed: {e}")
    
    def _write_span(self, span: SpanData) -> None:
        """Write a single span to the current file."""
        # Check if we need to rotate the file
        if self._should_rotate_file():
            self._rotate_file()
        
        # Convert span to dictionary
        span_dict = span.to_dict()
        
        # Add metadata
        span_dict["_sink_metadata"] = {
            "written_at": datetime.utcnow().isoformat(),
            "sink_name": self._config.name,
            "file_format": self._config.file_format,
        }
        
        # Write based on format
        if self._config.file_format == "jsonl":
            line = json.dumps(span_dict, default=str) + "\n"
            self._current_file.write(line)
            self._current_file_size += len(line.encode('utf-8'))
        
        elif self._config.file_format == "json":
            # For JSON format, we'll append to a list (less efficient but valid JSON)
            json.dump(span_dict, self._current_file, default=str)
            self._current_file.write("\n")
            self._current_file_size += len(json.dumps(span_dict, default=str).encode('utf-8'))
        
        else:
            raise SinkError(f"Unsupported file format: {self._config.file_format}")
    
    def _should_rotate_file(self) -> bool:
        """Check if the current file should be rotated."""
        if not self._current_file:
            return True
        
        # Check file size
        max_size_bytes = self._config.max_file_size_mb * 1024 * 1024
        return self._current_file_size >= max_size_bytes
    
    def _rotate_file(self) -> None:
        """Rotate to a new file."""
        # Close current file
        if self._current_file:
            self._current_file.close()
        
        # Generate new filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S") if self._config.include_timestamp else ""
        filename_parts = ["traces"]
        
        if timestamp:
            filename_parts.append(timestamp)
        
        filename_parts.append(f"{self._file_counter:04d}")
        filename = "_".join(filename_parts) + f".{self._config.file_format}"
        
        file_path = Path(self._config.directory) / filename
        
        # Open new file
        self._current_file = open(file_path, 'w', encoding='utf-8')
        self._current_file_size = 0
        self._file_counter += 1
        
        logger.info(f"Rotated to new file: {file_path}")
        
        # Clean up old files if needed
        self._cleanup_old_files()
    
    def _cleanup_old_files(self) -> None:
        """Clean up old files if we exceed the maximum number."""
        try:
            # Get all trace files in the directory
            trace_files = list(Path(self._config.directory).glob(f"traces_*.{self._config.file_format}"))
            
            # Sort by modification time (oldest first)
            trace_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove excess files
            while len(trace_files) > self._config.max_files:
                old_file = trace_files.pop(0)
                
                if self._config.compress_old_files:
                    # Compress the file before deletion
                    self._compress_file(old_file)
                else:
                    # Just delete the file
                    old_file.unlink()
                    logger.info(f"Deleted old trace file: {old_file}")
        
        except Exception as e:
            logger.warning(f"Failed to cleanup old files: {e}")
    
    def _compress_file(self, file_path: Path) -> None:
        """Compress a file using gzip."""
        try:
            import gzip
            
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            file_path.unlink()
            logger.info(f"Compressed and removed: {file_path} -> {compressed_path}")
            
        except Exception as e:
            logger.warning(f"Failed to compress file {file_path}: {e}")
            # If compression fails, just delete the original
            file_path.unlink()
    
    def _health_check(self) -> bool:
        """Check if the file sink is healthy."""
        try:
            # Check if directory exists and is writable
            directory = Path(self._config.directory)
            if not directory.exists():
                return False
            
            # Try to create a test file
            test_file = directory / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"File sink health check failed: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the file sink."""
        if self._current_file:
            self._current_file.close()
            self._current_file = None
        
        logger.info("FileSink shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sink statistics."""
        stats = {
            "name": self._config.name,
            "status": "active" if self._current_file else "inactive",
        }
        
        try:
            directory = Path(self._config.directory)
            trace_files = list(directory.glob(f"traces_*.{self._config.file_format}"))
            
            total_size = sum(f.stat().st_size for f in trace_files)
            
            stats.update({
                "directory": str(directory.absolute()),
                "file_count": len(trace_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "current_file_size_bytes": self._current_file_size,
                "file_format": self._config.file_format,
            })
        
        except Exception as e:
            logger.warning(f"Failed to get file sink stats: {e}")
        
        return stats

