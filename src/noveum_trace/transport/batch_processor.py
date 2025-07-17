"""
Batch processor for Noveum Trace SDK.

This module handles batching of traces for efficient transport
to the Noveum platform.
"""

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

from noveum_trace.core.config import get_config
from noveum_trace.utils.exceptions import TransportError

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for efficient trace export.

    This class batches traces and sends them in configurable intervals
    or when batch size limits are reached.
    """

    def __init__(self, send_callback: Callable[[list[dict[str, Any]]], None]):
        """
        Initialize the batch processor.

        Args:
            send_callback: Function to call when sending batches
        """
        self.config = get_config()
        self.send_callback = send_callback

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=self.config.transport.max_queue_size
        )
        self._batch: list[dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._shutdown = False

        # Start background thread
        self._thread = threading.Thread(target=self._process_batches, daemon=True)
        self._thread.start()

        logger.info(
            f"Batch processor started with batch_size={self.config.transport.batch_size}"
        )

    def add_trace(self, trace_data: dict[str, Any]) -> None:
        """
        Add a trace to the batch.

        Args:
            trace_data: Trace data to add

        Raises:
            TransportError: If processor is shutdown or queue is full
        """
        if self._shutdown:
            raise TransportError("Batch processor has been shutdown")

        try:
            self._queue.put(trace_data, timeout=1.0)
        except queue.Full as e:
            logger.warning("Trace queue is full, dropping trace")
            raise TransportError("Trace queue is full") from e

    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush all pending traces.

        Args:
            timeout: Maximum time to wait for flush completion
        """
        if self._shutdown:
            return

        # Send current batch
        with self._batch_lock:
            if self._batch:
                self._send_current_batch()

        # Wait for queue to empty
        start_time = time.time()
        while not self._queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("Flush timeout reached, some traces may be lost")
                break
            time.sleep(0.1)

    def shutdown(self) -> None:
        """Shutdown the batch processor."""
        if self._shutdown:
            return

        logger.info("Shutting down batch processor")
        self._shutdown = True

        # Flush remaining traces
        self.flush(timeout=10.0)

        # Wait for thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info("Batch processor shutdown completed")

    def _process_batches(self) -> None:
        """Background thread to process batches."""
        last_send_time = time.time()

        while not self._shutdown:
            try:
                # Get trace from queue with timeout
                try:
                    trace_data = self._queue.get(timeout=0.5)
                except queue.Empty:
                    # Check if we should send current batch due to timeout
                    current_time = time.time()
                    if (
                        current_time - last_send_time
                    ) >= self.config.transport.batch_timeout:
                        with self._batch_lock:
                            if self._batch:
                                self._send_current_batch()
                                last_send_time = current_time
                    continue

                # Add to current batch
                with self._batch_lock:
                    self._batch.append(trace_data)

                    # Send batch if size limit reached
                    if len(self._batch) >= self.config.transport.batch_size:
                        self._send_current_batch()
                        last_send_time = time.time()

                # Mark task as done
                self._queue.task_done()

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")

    def _send_current_batch(self) -> None:
        """Send the current batch (must be called with batch_lock held)."""
        if not self._batch:
            return

        batch_to_send = self._batch.copy()
        self._batch.clear()

        try:
            self.send_callback(batch_to_send)
            logger.debug(f"Sent batch of {len(batch_to_send)} traces")
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            # In a production implementation, we might want to retry or
            # implement a dead letter queue here

    def get_stats(self) -> dict[str, Any]:
        """
        Get batch processor statistics.

        Returns:
            Dictionary of statistics
        """
        with self._batch_lock:
            return {
                "queue_size": self._queue.qsize(),
                "current_batch_size": len(self._batch),
                "is_shutdown": self._shutdown,
                "thread_alive": self._thread.is_alive(),
            }
