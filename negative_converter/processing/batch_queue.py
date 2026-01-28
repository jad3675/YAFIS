# Batch queue with per-image settings
"""
Advanced batch processing queue with individual image settings.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import time
from copy import deepcopy

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BatchItemStatus(Enum):
    """Status of a batch item."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchItemSettings:
    """Settings for a single batch item."""
    # Adjustments (None = use global)
    adjustments: Optional[Dict[str, Any]] = None
    
    # Presets (None = use global)
    film_preset: Optional[str] = None
    photo_preset: Optional[str] = None
    
    # Override negative type (None = auto-detect)
    negative_type: Optional[str] = None
    
    # Export settings (None = use global)
    export_format: Optional[str] = None
    export_quality: Optional[int] = None
    
    # Custom output filename (None = auto-generate)
    output_filename: Optional[str] = None
    
    # Skip this item
    skip: bool = False
    
    def has_custom_settings(self) -> bool:
        """Check if this item has any custom settings."""
        return (
            self.adjustments is not None or
            self.film_preset is not None or
            self.photo_preset is not None or
            self.negative_type is not None or
            self.export_format is not None or
            self.export_quality is not None or
            self.output_filename is not None
        )
    
    def merge_with_global(self, global_settings: 'BatchItemSettings') -> 'BatchItemSettings':
        """Merge with global settings, preferring local values."""
        return BatchItemSettings(
            adjustments=self.adjustments if self.adjustments is not None else global_settings.adjustments,
            film_preset=self.film_preset if self.film_preset is not None else global_settings.film_preset,
            photo_preset=self.photo_preset if self.photo_preset is not None else global_settings.photo_preset,
            negative_type=self.negative_type if self.negative_type is not None else global_settings.negative_type,
            export_format=self.export_format if self.export_format is not None else global_settings.export_format,
            export_quality=self.export_quality if self.export_quality is not None else global_settings.export_quality,
            output_filename=self.output_filename,
            skip=self.skip
        )


@dataclass
class BatchItem:
    """A single item in the batch queue."""
    file_path: str
    settings: BatchItemSettings = field(default_factory=BatchItemSettings)
    status: BatchItemStatus = BatchItemStatus.PENDING
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    processing_time: float = 0.0
    
    @property
    def filename(self) -> str:
        return os.path.basename(self.file_path)
    
    @property
    def is_pending(self) -> bool:
        return self.status == BatchItemStatus.PENDING
    
    @property
    def is_completed(self) -> bool:
        return self.status == BatchItemStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        return self.status == BatchItemStatus.FAILED


@dataclass
class BatchQueueStats:
    """Statistics for batch processing."""
    total_items: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_time: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.completed + self.failed + self.skipped) / self.total_items * 100
    
    @property
    def success_rate(self) -> float:
        processed = self.completed + self.failed
        if processed == 0:
            return 0.0
        return self.completed / processed * 100
    
    @property
    def average_time(self) -> float:
        if self.completed == 0:
            return 0.0
        return self.total_time / self.completed


class BatchQueue:
    """
    Advanced batch processing queue with per-image settings.
    """
    
    def __init__(self):
        self._items: List[BatchItem] = []
        self._global_settings = BatchItemSettings()
        self._output_dir: Optional[str] = None
        self._is_processing = False
        self._should_stop = False
        
        # Callbacks
        self._on_item_start: Optional[Callable[[BatchItem], None]] = None
        self._on_item_complete: Optional[Callable[[BatchItem], None]] = None
        self._on_progress: Optional[Callable[[BatchQueueStats], None]] = None
    
    def add_item(self, file_path: str, settings: BatchItemSettings = None) -> BatchItem:
        """Add an item to the queue."""
        item = BatchItem(
            file_path=file_path,
            settings=settings or BatchItemSettings()
        )
        self._items.append(item)
        return item
    
    def add_items(self, file_paths: List[str]) -> List[BatchItem]:
        """Add multiple items to the queue."""
        return [self.add_item(path) for path in file_paths]
    
    def remove_item(self, index: int) -> bool:
        """Remove an item from the queue."""
        if 0 <= index < len(self._items):
            del self._items[index]
            return True
        return False
    
    def clear(self):
        """Clear all items from the queue."""
        self._items.clear()
    
    def get_item(self, index: int) -> Optional[BatchItem]:
        """Get an item by index."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None
    
    def get_items(self) -> List[BatchItem]:
        """Get all items."""
        return self._items.copy()
    
    def get_pending_items(self) -> List[BatchItem]:
        """Get all pending items."""
        return [item for item in self._items if item.is_pending and not item.settings.skip]
    
    def set_item_settings(self, index: int, settings: BatchItemSettings) -> bool:
        """Set settings for a specific item."""
        if 0 <= index < len(self._items):
            self._items[index].settings = settings
            return True
        return False
    
    def set_global_settings(self, settings: BatchItemSettings):
        """Set global settings applied to all items without custom settings."""
        self._global_settings = settings
    
    def get_global_settings(self) -> BatchItemSettings:
        """Get global settings."""
        return self._global_settings
    
    def set_output_dir(self, path: str):
        """Set output directory."""
        self._output_dir = path
    
    def get_output_dir(self) -> Optional[str]:
        """Get output directory."""
        return self._output_dir
    
    def get_effective_settings(self, index: int) -> BatchItemSettings:
        """Get effective settings for an item (merged with global)."""
        item = self.get_item(index)
        if item is None:
            return self._global_settings
        return item.settings.merge_with_global(self._global_settings)
    
    def get_stats(self) -> BatchQueueStats:
        """Get current queue statistics."""
        stats = BatchQueueStats(total_items=len(self._items))
        
        for item in self._items:
            if item.status == BatchItemStatus.PENDING:
                stats.pending += 1
            elif item.status == BatchItemStatus.PROCESSING:
                stats.processing += 1
            elif item.status == BatchItemStatus.COMPLETED:
                stats.completed += 1
                stats.total_time += item.processing_time
            elif item.status == BatchItemStatus.FAILED:
                stats.failed += 1
            elif item.status == BatchItemStatus.SKIPPED:
                stats.skipped += 1
        
        return stats
    
    def reset_status(self):
        """Reset all items to pending status."""
        for item in self._items:
            item.status = BatchItemStatus.PENDING
            item.error_message = None
            item.output_path = None
            item.processing_time = 0.0
    
    def set_callbacks(
        self,
        on_item_start: Callable[[BatchItem], None] = None,
        on_item_complete: Callable[[BatchItem], None] = None,
        on_progress: Callable[[BatchQueueStats], None] = None
    ):
        """Set processing callbacks."""
        self._on_item_start = on_item_start
        self._on_item_complete = on_item_complete
        self._on_progress = on_progress
    
    def request_stop(self):
        """Request processing to stop after current item."""
        self._should_stop = True
    
    def is_processing(self) -> bool:
        """Check if queue is currently processing."""
        return self._is_processing
    
    def process_item(
        self,
        item: BatchItem,
        process_func: Callable[[str, BatchItemSettings], Optional[str]]
    ) -> bool:
        """
        Process a single item.
        
        Args:
            item: The batch item to process.
            process_func: Function that takes (file_path, settings) and returns output_path or None.
            
        Returns:
            True if successful.
        """
        if item.settings.skip:
            item.status = BatchItemStatus.SKIPPED
            return True
        
        item.status = BatchItemStatus.PROCESSING
        
        if self._on_item_start:
            self._on_item_start(item)
        
        start_time = time.time()
        
        try:
            effective_settings = item.settings.merge_with_global(self._global_settings)
            output_path = process_func(item.file_path, effective_settings)
            
            item.processing_time = time.time() - start_time
            
            if output_path:
                item.output_path = output_path
                item.status = BatchItemStatus.COMPLETED
                logger.info("Processed: %s -> %s (%.2fs)", 
                           item.filename, os.path.basename(output_path), item.processing_time)
                return True
            else:
                item.status = BatchItemStatus.FAILED
                item.error_message = "Processing returned no output"
                return False
                
        except Exception as e:
            item.processing_time = time.time() - start_time
            item.status = BatchItemStatus.FAILED
            item.error_message = str(e)
            logger.exception("Failed to process: %s", item.filename)
            return False
        
        finally:
            if self._on_item_complete:
                self._on_item_complete(item)
            if self._on_progress:
                self._on_progress(self.get_stats())
    
    def process_all(
        self,
        process_func: Callable[[str, BatchItemSettings], Optional[str]]
    ) -> BatchQueueStats:
        """
        Process all pending items in the queue.
        
        Args:
            process_func: Function that takes (file_path, settings) and returns output_path.
            
        Returns:
            Final queue statistics.
        """
        self._is_processing = True
        self._should_stop = False
        
        try:
            for item in self._items:
                if self._should_stop:
                    logger.info("Batch processing stopped by user")
                    break
                
                if item.is_pending and not item.settings.skip:
                    self.process_item(item, process_func)
        
        finally:
            self._is_processing = False
        
        return self.get_stats()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize queue to dictionary."""
        return {
            "items": [
                {
                    "file_path": item.file_path,
                    "settings": {
                        "adjustments": item.settings.adjustments,
                        "film_preset": item.settings.film_preset,
                        "photo_preset": item.settings.photo_preset,
                        "negative_type": item.settings.negative_type,
                        "export_format": item.settings.export_format,
                        "export_quality": item.settings.export_quality,
                        "output_filename": item.settings.output_filename,
                        "skip": item.settings.skip,
                    },
                    "status": item.status.value,
                }
                for item in self._items
            ],
            "global_settings": {
                "adjustments": self._global_settings.adjustments,
                "film_preset": self._global_settings.film_preset,
                "photo_preset": self._global_settings.photo_preset,
                "negative_type": self._global_settings.negative_type,
                "export_format": self._global_settings.export_format,
                "export_quality": self._global_settings.export_quality,
            },
            "output_dir": self._output_dir,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchQueue':
        """Deserialize queue from dictionary."""
        queue = cls()
        
        # Load global settings
        gs = data.get("global_settings", {})
        queue._global_settings = BatchItemSettings(
            adjustments=gs.get("adjustments"),
            film_preset=gs.get("film_preset"),
            photo_preset=gs.get("photo_preset"),
            negative_type=gs.get("negative_type"),
            export_format=gs.get("export_format"),
            export_quality=gs.get("export_quality"),
        )
        
        queue._output_dir = data.get("output_dir")
        
        # Load items
        for item_data in data.get("items", []):
            settings_data = item_data.get("settings", {})
            settings = BatchItemSettings(
                adjustments=settings_data.get("adjustments"),
                film_preset=settings_data.get("film_preset"),
                photo_preset=settings_data.get("photo_preset"),
                negative_type=settings_data.get("negative_type"),
                export_format=settings_data.get("export_format"),
                export_quality=settings_data.get("export_quality"),
                output_filename=settings_data.get("output_filename"),
                skip=settings_data.get("skip", False),
            )
            
            item = BatchItem(
                file_path=item_data["file_path"],
                settings=settings,
            )
            
            # Restore status if available
            status_str = item_data.get("status", "pending")
            try:
                item.status = BatchItemStatus(status_str)
            except ValueError:
                item.status = BatchItemStatus.PENDING
            
            queue._items.append(item)
        
        return queue
