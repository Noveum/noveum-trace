"""
Batch processor for Noveum Trace SDK.

This module handles batching of traces for efficient transport
to the Noveum platform.
"""

import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from noveum_trace.core.config import Config

from noveum_trace.core.config import get_config
from noveum_trace.utils.exceptions import TransportError
from noveum_trace.utils.logging import (
    get_sdk_logger,
    log_debug_enabled,
    log_error_always,
    log_trace_flow,
)

logger = get_sdk_logger("transport.batch_processor")


class BatchProcessor:
    """
    Batch processor for efficient trace export.

    This class batches traces and sends them in configurable intervals
    or when batch size limits are reached.
    """

    def __init__(
        self,
        send_callback: Callable[[dict[str, Any]], None],
        config: Optional["Config"] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            send_callback: Function to call when sending batches (receives dict with 'type' and 'data')
            config: Optional configuration instance
        """
        self.config = config if config is not None else get_config()
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

        if log_debug_enabled():
            logger.debug("🔧 Batch processor configuration:")
            logger.debug(f"    batch_size: {self.config.transport.batch_size}")
            logger.debug(f"    batch_timeout: {self.config.transport.batch_timeout}s")
            logger.debug(f"    max_queue_size: {self.config.transport.max_queue_size}")

    def add_trace(self, trace_data: dict[str, Any]) -> None:
        """
        Add a trace to the batch.

        Args:
            trace_data: Trace data to add

        Raises:
            TransportError: If processor is shutdown or queue is full
        """
        if self._shutdown:
            log_error_always(
                logger,
                "Cannot add trace - batch processor has been shutdown",
                trace_id=trace_data.get("trace_id", "unknown"),
            )
            raise TransportError("Batch processor has been shutdown")

        # Log trace addition details
        trace_id = trace_data.get("trace_id", "unknown")
        trace_name = trace_data.get("name", "unnamed")
        span_count = len(trace_data.get("spans", []))
        queue_size = self._queue.qsize()

        logger.info(
            f"📥 ADDING TRACE TO QUEUE: {trace_name} (ID: {trace_id}) - {span_count} spans"
        )

        if log_debug_enabled():
            log_trace_flow(
                logger,
                "Adding trace to batch queue",
                trace_id=trace_id,
                trace_name=trace_name,
                span_count=span_count,
                queue_size_before=queue_size,
                max_queue_size=self.config.transport.max_queue_size,
                trace_data_keys=list(trace_data.keys()),
            )

        # Mark as trace type
        trace_data["_item_type"] = "trace"

        try:
            self._queue.put(trace_data, block=False)
            new_queue_size = self._queue.qsize()

            if log_debug_enabled():
                logger.debug(
                    f"📊 Queue size: {new_queue_size}/{self.config.transport.max_queue_size}"
                )

            logger.info(f"✅ Successfully queued trace {trace_id}")
        except queue.Full as e:
            log_error_always(
                logger,
                f"Trace queue is full, dropping trace {trace_id}, please increase the max queue size",
                trace_id=trace_id,
                queue_size=queue_size,
                max_queue_size=self.config.transport.max_queue_size,
            )
            raise TransportError("Trace queue is full") from e

    def add_audio(self, audio_data: dict[str, Any]) -> None:
        """
        Add audio to the batch queue.

        Args:
            audio_data: Audio data with metadata

        Raises:
            TransportError: If processor is shutdown, queue is full, or audio_uuid is missing
        """
        # Validate audio_uuid is present
        audio_uuid = audio_data.get("audio_uuid")
        if not audio_uuid:
            audio_data_keys = list(audio_data.keys())
            log_error_always(
                logger,
                "Cannot add audio - missing audio_uuid, dropping audio file",
                audio_data_keys=audio_data_keys,
            )
            raise TransportError(
                f"Cannot add audio - missing audio_uuid, dropping audio file. "
                f"Available keys: {audio_data_keys}"
            )

        if self._shutdown:
            log_error_always(
                logger,
                "Cannot add audio - batch processor has been shutdown",
                audio_uuid=audio_uuid,
            )
            raise TransportError("Batch processor has been shutdown")

        # Mark as audio type
        audio_data["_item_type"] = "audio"

        # Log audio addition
        trace_id = audio_data.get("trace_id", "unknown")
        logger.info(f"📥 ADDING AUDIO TO QUEUE: {audio_uuid} for trace {trace_id}")

        try:
            self._queue.put(audio_data, block=False)
            logger.info(f"✅ Successfully queued audio {audio_uuid}")
        except queue.Full as e:
            log_error_always(logger, f"Queue is full, dropping audio {audio_uuid}")
            raise TransportError("Audio queue is full") from e

    def add_image(self, image_data: dict[str, Any]) -> None:
        """
        Add image to the batch queue.

        Args:
            image_data: Image data with metadata

        Raises:
            TransportError: If processor is shutdown, queue is full, or image_uuid is missing
        """
        # Validate image_uuid is present
        image_uuid = image_data.get("image_uuid")
        if not image_uuid:
            image_data_keys = list(image_data.keys())
            log_error_always(
                logger,
                "Cannot add image - missing image_uuid, dropping image file",
                image_data_keys=image_data_keys,
            )
            raise TransportError(
                f"Cannot add image - missing image_uuid, dropping image file. "
                f"Available keys: {image_data_keys}"
            )

        if self._shutdown:
            log_error_always(
                logger,
                "Cannot add image - batch processor has been shutdown",
                image_uuid=image_uuid,
            )
            raise TransportError("Batch processor has been shutdown")

        # Mark as image type
        image_data["_item_type"] = "image"

        # Log image addition
        trace_id = image_data.get("trace_id", "unknown")
        logger.info(f"📥 ADDING IMAGE TO QUEUE: {image_uuid} for trace {trace_id}")

        try:
            self._queue.put(image_data, block=False)
            logger.info(f"✅ Successfully queued image {image_uuid}")
        except queue.Full as e:
            log_error_always(logger, f"Queue is full, dropping image {image_uuid}")
            raise TransportError("Image queue is full") from e

    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush all pending traces.

        Args:
            timeout: Maximum time to wait for flush completion
        """
        if self._shutdown:
            logger.debug("Batch processor already shutdown, skipping flush")
            return

        log_trace_flow(logger, "Starting batch processor flush", timeout=timeout)

        # Send current batch
        with self._batch_lock:
            if self._batch:
                logger.info(
                    f"🔄 FLUSH: Sending current batch of {len(self._batch)} traces"
                )
                self._send_current_batch()
            else:
                logger.debug("🔄 FLUSH: No current batch to send")

        # Wait for queue to empty
        start_time = time.time()
        initial_queue_size = self._queue.qsize()

        if initial_queue_size > 0:
            logger.info(
                f"🔄 FLUSH: Waiting for {initial_queue_size} queued traces to process..."
            )

        while not self._queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                remaining_traces = self._queue.qsize()
                log_error_always(
                    logger,
                    f"Flush timeout reached, {remaining_traces} traces may be lost",
                    timeout=timeout,
                    remaining_traces=remaining_traces,
                    elapsed_time=time.time() - start_time,
                )
                break
            time.sleep(0.1)

        elapsed_time = time.time() - start_time
        try:
            logger.info(f"✅ Batch processor flush completed in {elapsed_time:.2f}s")
        except (ValueError, OSError, RuntimeError, Exception):
            # Logger may be closed during shutdown
            pass

    def shutdown(self) -> None:
        """Shutdown the batch processor."""
        if self._shutdown:
            try:
                logger.debug("Batch processor already shutdown")
            except (ValueError, OSError, RuntimeError, Exception):
                pass
            return

        try:
            logger.info("Shutting down batch processor")
        except (ValueError, OSError, RuntimeError, Exception):
            # Logger may be closed during shutdown
            pass
        self._shutdown = True

        try:
            # Flush any remaining traces
            self.flush(timeout=5.0)

            # Wait for background thread to stop
            if self._thread.is_alive():
                try:
                    logger.debug("Waiting for background thread to stop...")
                except (ValueError, OSError, RuntimeError, Exception):
                    pass
                self._thread.join(timeout=10.0)
                if self._thread.is_alive():
                    try:
                        logger.warning(
                            "⚠️  Background thread did not stop within timeout"
                        )
                    except (ValueError, OSError, RuntimeError, Exception):
                        pass

            try:
                logger.info("Batch processor shutdown completed")
            except (ValueError, OSError, RuntimeError, Exception):
                # Logger may be closed during shutdown
                pass
        except Exception as e:
            log_error_always(
                logger,
                "Error during batch processor shutdown",
                exc_info=True,
                error=str(e),
            )

    def _process_batches(self) -> None:
        """Background thread to process batches."""
        last_send_time = time.time()
        logger.info(
            f"🔄 Batch processor background thread started "
            f"(batch_size={self.config.transport.batch_size}, "
            f"timeout={self.config.transport.batch_timeout}s)"
        )

        while not self._shutdown:
            try:
                # Get item from queue with timeout (could be trace or audio)
                try:
                    item_data = self._queue.get(timeout=0.5)
                    item_type = item_data.get("_item_type", "trace")
                    # Get appropriate ID based on item type
                    # Note: audio_uuid and image_uuid validation happens in add_audio() and add_image()
                    item_id = (
                        item_data.get("trace_id")
                        or item_data.get("audio_uuid")
                        or item_data.get("image_uuid")
                        or "missing-id"
                    )

                    if log_debug_enabled():
                        log_trace_flow(
                            logger,
                            f"Background thread got {item_type} from queue",
                            item_id=item_id,
                            queue_size=self._queue.qsize(),
                        )

                except queue.Empty:
                    # Check if we should send current batch due to timeout
                    current_time = time.time()
                    time_since_last_send = current_time - last_send_time

                    if time_since_last_send >= self.config.transport.batch_timeout:
                        with self._batch_lock:
                            if self._batch:
                                try:
                                    logger.info(
                                        f"⏰ TIMEOUT TRIGGER: Sending batch due to timeout "
                                        f"({time_since_last_send:.1f}s >= {self.config.transport.batch_timeout}s)"
                                    )
                                except (ValueError, OSError, RuntimeError, Exception):
                                    # Logger may be closed during shutdown
                                    pass
                                if log_debug_enabled():
                                    logger.debug(
                                        f"    Batch size: {len(self._batch)} traces"
                                    )
                                self._send_current_batch()
                                last_send_time = current_time
                            else:
                                if log_debug_enabled():
                                    logger.debug(
                                        f"⏰ Timeout reached but batch is empty "
                                        f"(waited {time_since_last_send:.1f}s)"
                                    )
                    continue

                # Add to current batch (both audio and traces)
                with self._batch_lock:
                    self._batch.append(item_data)
                    batch_size = len(self._batch)

                    if log_debug_enabled():
                        log_trace_flow(
                            logger,
                            f"Added {item_type} to batch",
                            item_id=item_id,
                            batch_size=batch_size,
                            max_batch_size=self.config.transport.batch_size,
                        )

                    # Send batch if size limit reached
                    if batch_size >= self.config.transport.batch_size:
                        logger.info(
                            f"📦 SIZE TRIGGER: Sending batch due to size limit "
                            f"({batch_size} >= {self.config.transport.batch_size})"
                        )
                        self._send_current_batch()
                        last_send_time = time.time()

                # Mark task as done
                self._queue.task_done()

            except Exception as e:
                log_error_always(
                    logger,
                    "Error in batch processor background thread",
                    exc_info=True,
                    error=str(e),
                )

        try:
            logger.info("🔄 Batch processor background thread stopped")
        except (ValueError, OSError, RuntimeError, Exception):
            # Logger may be closed during shutdown
            pass

    def _send_current_batch(self) -> None:
        """
        Send the current batch, splitting audio, images, and traces.
        Audio and images are sent first (individually), then traces.
        """
        if not self._batch:
            try:
                logger.debug("_send_current_batch called but batch is empty")
            except (ValueError, OSError, RuntimeError, Exception):
                pass
            return

        batch_to_process = self._batch.copy()
        self._batch.clear()

        # Split into audio, images, and traces
        audio_items = [
            item for item in batch_to_process if item.get("_item_type") == "audio"
        ]
        image_items = [
            item for item in batch_to_process if item.get("_item_type") == "image"
        ]
        trace_items = [
            item for item in batch_to_process if item.get("_item_type") == "trace"
        ]

        try:
            logger.info(
                f"📤 PROCESSING BATCH: {len(audio_items)} audio, {len(image_items)} images, {len(trace_items)} traces"
            )
        except (ValueError, OSError, RuntimeError, Exception):
            pass

        # Send ALL audio first (individually)
        if audio_items:
            try:
                logger.info(f"🎵 Sending {len(audio_items)} audio files first...")
            except (ValueError, OSError, RuntimeError, Exception):
                pass

            for audio_item in audio_items:
                # Remove internal marker before sending
                audio_item.pop("_item_type", None)

                # Validate audio_uuid exists (should always exist due to add_audio validation)
                audio_uuid = audio_item.get("audio_uuid")
                if not audio_uuid:
                    log_error_always(
                        logger,
                        "Skipping audio in batch - missing audio_uuid",
                        audio_item_keys=list(audio_item.keys()),
                    )
                    continue

                try:
                    self.send_callback({"type": "audio", "data": audio_item})
                except Exception as e:
                    log_error_always(
                        logger,
                        f"Failed to send audio {audio_uuid}",
                        exc_info=True,
                        error=str(e),
                        audio_uuid=audio_uuid,
                    )

        # Send ALL images (individually)
        if image_items:
            try:
                logger.info(f"🖼️  Sending {len(image_items)} image files...")
            except (ValueError, OSError, RuntimeError, Exception):
                pass

            for image_item in image_items:
                # Remove internal marker before sending
                image_item.pop("_item_type", None)

                # Validate image_uuid exists (should always exist due to add_image validation)
                image_uuid = image_item.get("image_uuid")
                if not image_uuid:
                    log_error_always(
                        logger,
                        "Skipping image in batch - missing image_uuid",
                        image_item_keys=list(image_item.keys()),
                    )
                    continue

                try:
                    self.send_callback({"type": "image", "data": image_item})
                except Exception as e:
                    log_error_always(
                        logger,
                        f"Failed to send image {image_uuid}",
                        exc_info=True,
                        error=str(e),
                        image_uuid=image_uuid,
                    )

        # Then send traces (batched)
        if trace_items:
            try:
                logger.info(f"📦 Sending {len(trace_items)} traces...")
            except (ValueError, OSError, RuntimeError, Exception):
                pass

            # Remove internal marker from traces
            for item in trace_items:
                item.pop("_item_type", None)

            if log_debug_enabled():
                # Log trace IDs in the batch
                for i, trace in enumerate(trace_items):
                    trace_id = trace.get("trace_id", "unknown")
                    trace_name = trace.get("name", "unnamed")
                    logger.debug(f"    [{i+1}] {trace_name} (ID: {trace_id})")

            try:
                self.send_callback({"type": "traces", "data": trace_items})
                try:
                    logger.info(
                        f"✅ Successfully sent batch of {len(trace_items)} traces via callback"
                    )
                except (ValueError, OSError, RuntimeError, Exception):
                    pass
            except Exception as e:
                log_error_always(
                    logger,
                    f"Failed to send batch of {len(trace_items)} traces",
                    exc_info=True,
                    batch_size=len(trace_items),
                    error=str(e),
                )
                # In a production implementation, we might want to retry or
                # implement a dead letter queue here
