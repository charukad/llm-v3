"""
Resource management system for optimizing CPU, memory, and GPU utilization.
Provides dynamic resource allocation, request scheduling, and overload protection.
"""
import time
import threading
import queue
import logging
import os
import psutil
import signal
import atexit
import json
from typing import Dict, Any, List, Callable, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from math_llm_system.orchestration.monitoring.logger import get_logger

logger = get_logger("orchestration.performance.resource_manager")

# Try to import GPU libraries, fall back to CPU-only mode if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available. GPU monitoring and optimization disabled.")

# Try to import NVIDIA management library for detailed GPU stats
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except (ImportError, pynvml.NVMLError):
    HAS_NVML = False
    logger.warning("NVIDIA Management Library not available. Detailed GPU stats disabled.")

class ResourceManager:
    """
    Resource management system for optimizing system resource utilization.
    Manages CPU cores, memory, GPU resources, and request scheduling.
    """
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the resource manager.
        
        Args:
            config: Configuration dictionary for resource manager
        """
        # Only initialize once for singleton
        if self._initialized:
            return
            
        # Default configuration
        self.default_config = {
            "cpu": {
                "max_workers": os.cpu_count(),
                "reserved_cores": 1,  # Keep at least 1 core free
                "thread_pool_size": os.cpu_count() * 2,
                "process_pool_size": max(1, os.cpu_count() - 1)
            },
            "memory": {
                "max_usage_percent": 85.0,  # Max memory usage percentage
                "critical_threshold": 95.0,  # Critical threshold for emergency action
                "reserve_gb": 1.0  # Reserve at least 1GB
            },
            "gpu": {
                "enabled": HAS_TORCH,
                "max_usage_percent": 85.0,
                "devices": None,  # None = use all available
                "reserved_memory_mb": 512  # Reserve 512MB per GPU
            },
            "scheduling": {
                "priority_levels": 3,
                "max_queue_size": 1000,
                "default_timeout": 30.0,
                "high_priority_weight": 3.0
            },
            "monitoring": {
                "interval": 1.0,  # Monitoring interval in seconds
                "history_size": 60,  # Keep 60 data points (1 minute at 1s interval)
                "log_interval": 10,  # Log resource usage every 10 seconds
                "alert_threshold": 90.0  # Alert threshold percentage
            }
        }
        
        # Apply custom configuration
        self.config = self.default_config.copy()
        if config:
            self._deep_update(self.config, config)
        
        # Initialize resource tracking
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        
        # Initialize thread and process pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config["cpu"]["thread_pool_size"]
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config["cpu"]["process_pool_size"]
        )
        
        # Initialize task queues by priority
        self.task_queues = [
            queue.PriorityQueue(self.config["scheduling"]["max_queue_size"])
            for _ in range(self.config["scheduling"]["priority_levels"])
        ]
        
        # Active tasks tracking
        self.active_tasks = []
        self.active_tasks_lock = threading.Lock()
        
        # Initialize GPU resources if available
        self.gpu_devices = []
        if self.config["gpu"]["enabled"] and HAS_TORCH:
            self._initialize_gpu_resources()
        
        # Start resource monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start task scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._task_scheduler,
            daemon=True
        )
        self.scheduler_thread.start()
        
        # Flag to indicate running state
        self.running = True
        
        # Register cleanup handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self._initialized = True
        logger.info("Resource Manager initialized")
    
    def _deep_update(self, original: Dict[str, Any], update: Dict[str, Any]):
        """
        Deep update a nested dictionary.
        
        Args:
            original: Original dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def _initialize_gpu_resources(self):
        """Initialize GPU resources if available."""
        if not HAS_TORCH:
            return
            
        # Get available GPU devices
        gpu_count = torch.cuda.device_count()
        
        if gpu_count == 0:
            logger.warning("No GPU devices found despite PyTorch CUDA support.")
            self.config["gpu"]["enabled"] = False
            return
        
        # Use all devices or specified subset
        device_indices = self.config["gpu"]["devices"] or list(range(gpu_count))
        
        # Initialize device information
        for idx in device_indices:
            if idx < gpu_count:
                device_info = {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory": torch.cuda.get_device_properties(idx).total_memory,
                    "allocated_memory": 0,
                    "reserved_memory": self.config["gpu"]["reserved_memory_mb"] * 1024 * 1024
                }
                self.gpu_devices.append(device_info)
        
        logger.info(f"Initialized {len(self.gpu_devices)} GPU devices")
        
        # Get detailed GPU info if NVML is available
        if HAS_NVML:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i, device_info in enumerate(self.gpu_devices):
                    idx = device_info["index"]
                    if idx < device_count:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        device_info["total_memory"] = memory_info.total
                        device_info["nvml_handle"] = handle
            except pynvml.NVMLError as e:
                logger.error(f"NVML error in GPU initialization: {e}")
    
    def _monitor_resources(self):
        """Background thread for monitoring system resources."""
        last_log_time = time.time()
        
        while self.running:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Update usage history (limit to history_size)
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                if len(self.cpu_usage) > self.config["monitoring"]["history_size"]:
                    self.cpu_usage = self.cpu_usage[-self.config["monitoring"]["history_size"]:]
                if len(self.memory_usage) > self.config["monitoring"]["history_size"]:
                    self.memory_usage = self.memory_usage[-self.config["monitoring"]["history_size"]:]
                
                # Monitor GPU usage if available
                if self.config["gpu"]["enabled"] and self.gpu_devices:
                    self._update_gpu_usage()
                
                # Check for resource overload
                self._check_resource_thresholds()
                
                # Periodically log resource usage
                current_time = time.time()
                if current_time - last_log_time > self.config["monitoring"]["log_interval"]:
                    self._log_resource_usage()
                    last_log_time = current_time
                
                # Sleep for monitoring interval
                time.sleep(self.config["monitoring"]["interval"])
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)  # Sleep longer on error
    
    def _update_gpu_usage(self):
        """Update GPU usage statistics."""
        if not self.config["gpu"]["enabled"] or not self.gpu_devices:
            return
            
        gpu_usage_data = []
        
        try:
            # Update GPU memory usage
            if HAS_NVML:
                # Use NVML for detailed stats
                for device_info in self.gpu_devices:
                    try:
                        handle = device_info.get("nvml_handle")
                        if handle:
                            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            
                            usage = {
                                "index": device_info["index"],
                                "memory_used": memory_info.used,
                                "memory_total": memory_info.total,
                                "memory_percent": (memory_info.used / memory_info.total) * 100,
                                "gpu_utilization": utilization.gpu,
                                "memory_utilization": utilization.memory
                            }
                            gpu_usage_data.append(usage)
                    except pynvml.NVMLError as e:
                        logger.error(f"NVML error updating GPU {device_info['index']}: {e}")
            elif HAS_TORCH:
                # Fall back to PyTorch stats
                for device_info in self.gpu_devices:
                    idx = device_info["index"]
                    
                    # Get allocated memory from PyTorch
                    allocated = torch.cuda.memory_allocated(idx)
                    reserved = torch.cuda.memory_reserved(idx)
                    total = device_info["total_memory"]
                    
                    usage = {
                        "index": idx,
                        "memory_used": allocated,
                        "memory_reserved": reserved,
                        "memory_total": total,
                        "memory_percent": (allocated / total) * 100
                    }
                    gpu_usage_data.append(usage)
            
            # Update usage history
            self.gpu_usage.append(gpu_usage_data)
            if len(self.gpu_usage) > self.config["monitoring"]["history_size"]:
                self.gpu_usage = self.gpu_usage[-self.config["monitoring"]["history_size"]:]
                
        except Exception as e:
            logger.error(f"Error updating GPU usage: {e}")
    
    def _check_resource_thresholds(self):
        """Check if resource usage exceeds thresholds and take action if needed."""
        # Check memory usage
        if self.memory_usage and self.memory_usage[-1] > self.config["memory"]["critical_threshold"]:
            logger.warning(f"Critical memory usage: {self.memory_usage[-1]}%")
            self._handle_memory_overload()
        
        # Check CPU usage
        if self.cpu_usage and np.mean(self.cpu_usage[-5:]) > self.config["monitoring"]["alert_threshold"]:
            logger.warning(f"High CPU usage: {np.mean(self.cpu_usage[-5:])}%")
        
        # Check GPU memory usage
        if self.gpu_usage and self.config["gpu"]["enabled"]:
            for device_data in self.gpu_usage[-1]:
                if device_data.get("memory_percent", 0) > self.config["gpu"]["max_usage_percent"]:
                    logger.warning(f"High GPU memory usage on device {device_data['index']}: "
                                  f"{device_data['memory_percent']:.1f}%")
    
    def _handle_memory_overload(self):
        """Handle critical memory overload situation."""
        logger.critical("Handling memory overload - emergency measures activated")
        
        # Cancel low priority tasks
        with self.active_tasks_lock:
            low_priority_tasks = [task for task in self.active_tasks 
                                if task.get("priority", 1) == 0]
            
            for task in low_priority_tasks:
                if "future" in task and not task["future"].done():
                    logger.warning(f"Cancelling low priority task {task.get('id', 'unknown')}")
                    task["future"].cancel()
        
        # Clear all pending tasks from low priority queue
        while not self.task_queues[0].empty():
            try:
                self.task_queues[0].get_nowait()
                self.task_queues[0].task_done()
            except queue.Empty:
                break
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # If PyTorch is available, clear CUDA cache
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory overload handling complete")
    
    def _log_resource_usage(self):
        """Log current resource usage."""
        # Log CPU and memory usage
        if self.cpu_usage and self.memory_usage:
            logger.info(f"Resource usage - CPU: {self.cpu_usage[-1]:.1f}%, "
                       f"Memory: {self.memory_usage[-1]:.1f}%")
        
        # Log GPU usage if available
        if self.gpu_usage and self.config["gpu"]["enabled"]:
            for device_data in self.gpu_usage[-1]:
                logger.info(f"GPU {device_data['index']} ({device_data.get('name', 'unknown')}): "
                          f"Memory {device_data.get('memory_percent', 0):.1f}%, "
                          f"Utilization {device_data.get('gpu_utilization', 0)}%")
    
    def _task_scheduler(self):
        """Background thread for scheduling and executing tasks."""
        while self.running:
            try:
                # Check resource availability before processing more tasks
                if self._can_accept_more_tasks():
                    # Try to get task from queues, starting with highest priority
                    task = None
                    for priority in range(self.config["scheduling"]["priority_levels"] - 1, -1, -1):
                        try:
                            # Get task from queue (non-blocking)
                            # Queue items are (timestamp, task_info) tuples
                            _, task = self.task_queues[priority].get_nowait()
                            self.task_queues[priority].task_done()
                            break
                        except queue.Empty:
                            continue
                    
                    # If we found a task, execute it
                    if task:
                        self._execute_task(task)
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in task scheduler: {e}")
                time.sleep(1)
    
    def _can_accept_more_tasks(self) -> bool:
        """
        Check if system can accept more tasks based on resource usage.
        
        Returns:
            Boolean indicating if more tasks can be accepted
        """
        # Check active tasks count against CPU count
        with self.active_tasks_lock:
            active_count = len([t for t in self.active_tasks if "future" in t and not t["future"].done()])
        
        if active_count >= self.config["cpu"]["max_workers"]:
            return False
        
        # Check memory usage
        if self.memory_usage and self.memory_usage[-1] > self.config["memory"]["max_usage_percent"]:
            return False
        
        # Check CPU usage
        if self.cpu_usage and np.mean(self.cpu_usage[-5:]) > self.config["cpu"]["max_workers"] * 100 / os.cpu_count():
            return False
        
        # We can accept more tasks
        return True
    
    def _execute_task(self, task: Dict[str, Any]):
        """
        Execute a task with appropriate resources.
        
        Args:
            task: Task information dictionary
        """
        # Determine correct executor based on task type
        if task.get("use_process_pool", False):
            executor = self.process_pool
        else:
            executor = self.thread_pool
        
        # Submit task to executor
        future = executor.submit(
            self._wrapped_task_execution,
            task.get("func"),
            task.get("args", ()),
            task.get("kwargs", {})
        )
        
        # Add completion callback
        future.add_done_callback(
            lambda f: self._task_completed_callback(f, task)
        )
        
        # Add to active tasks
        task["future"] = future
        task["start_time"] = time.time()
        
        with self.active_tasks_lock:
            self.active_tasks.append(task)
        
        logger.debug(f"Task {task.get('id', 'unknown')} started with "
                   f"priority {task.get('priority', 1)}")
    
    def _wrapped_task_execution(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """
        Wrapper for task execution to handle exceptions and resource tracking.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        # Record start time
        start_time = time.time()
        
        try:
            # Execute the task
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            # Log the error
            logger.error(f"Task execution error: {e}")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }
    
    def _task_completed_callback(self, future: Future, task: Dict[str, Any]):
        """
        Callback for task completion.
        
        Args:
            future: Future object
            task: Task information dictionary
        """
        # Calculate task duration
        end_time = time.time()
        duration = end_time - task.get("start_time", end_time)
        
        # Handle different completion states
        if future.cancelled():
            logger.info(f"Task {task.get('id', 'unknown')} was cancelled after {duration:.2f}s")
            completion_status = "cancelled"
        else:
            try:
                result = future.result()
                if result.get("success", False):
                    logger.debug(f"Task {task.get('id', 'unknown')} completed successfully "
                               f"in {duration:.2f}s")
                    completion_status = "success"
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"Task {task.get('id', 'unknown')} failed: {error_msg} "
                               f"after {duration:.2f}s")
                    completion_status = "error"
            except Exception as e:
                logger.error(f"Error retrieving task result: {e}")
                completion_status = "error"
        
        # Execute completion callback if provided
        if "completion_callback" in task:
            try:
                callback_func = task["completion_callback"]
                callback_func(task, future, completion_status)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
        
        # Remove from active tasks
        with self.active_tasks_lock:
            self.active_tasks = [t for t in self.active_tasks if t.get("id") != task.get("id")]
    
    def submit_task(self, func: Callable, args: tuple = None, kwargs: dict = None,
                  priority: int = 1, task_id: str = None, use_process_pool: bool = False,
                  completion_callback: Callable = None) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Task priority (0-2, higher is more important)
            task_id: Optional task identifier
            use_process_pool: Whether to use process pool instead of thread pool
            completion_callback: Optional callback for task completion
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{int(time.time())}_{id(func)}"
        
        # Normalize arguments
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        
        # Ensure priority is valid
        priority = max(0, min(priority, self.config["scheduling"]["priority_levels"] - 1))
        
        # Create task info
        task = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "priority": priority,
            "use_process_pool": use_process_pool,
            "submit_time": time.time(),
            "completion_callback": completion_callback
        }
        
        # Add to appropriate queue with timestamp for FIFO within same priority
        submit_timestamp = time.time()
        try:
            self.task_queues[priority].put((submit_timestamp, task))
            logger.debug(f"Task {task_id} submitted with priority {priority}")
            return task_id
        except queue.Full:
            logger.error(f"Task queue {priority} is full, rejecting task {task_id}")
            raise RuntimeError(f"Task queue {priority} is full")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary with task status information
        """
        # Check active tasks
        with self.active_tasks_lock:
            active_task = next((t for t in self.active_tasks if t.get("id") == task_id), None)
        
        if active_task:
            # Task is active, check future status
            future = active_task.get("future")
            if future:
                if future.cancelled():
                    status = "cancelled"
                elif future.done():
                    try:
                        result = future.result()
                        status = "completed" if result.get("success", False) else "failed"
                    except Exception:
                        status = "failed"
                else:
                    status = "running"
            else:
                status = "pending"
            
            return {
                "id": task_id,
                "status": status,
                "priority": active_task.get("priority", 1),
                "submit_time": active_task.get("submit_time"),
                "start_time": active_task.get("start_time"),
                "elapsed": time.time() - active_task.get("start_time", time.time())
            }
        
        # Check pending tasks in queues
        for priority, queue_obj in enumerate(self.task_queues):
            # Create a temporary copy of the queue to avoid modifying it
            temp_queue = []
            found_task = None
            
            # Search through queue
            try:
                while not queue_obj.empty():
                    item = queue_obj.get_nowait()
                    temp_queue.append(item)
                    
                    if item[1].get("id") == task_id:
                        found_task = item[1]
            except queue.Empty:
                pass
            
            # Restore queue
            for item in temp_queue:
                queue_obj.put(item)
            
            # If task found, return status
            if found_task:
                return {
                    "id": task_id,
                    "status": "queued",
                    "priority": priority,
                    "submit_time": found_task.get("submit_time"),
                    "queue_time": time.time() - found_task.get("submit_time", time.time())
                }
        
        # Task not found
        return {
            "id": task_id,
            "status": "not_found"
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if possible.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Boolean indicating if task was cancelled
        """
        # Check active tasks
        with self.active_tasks_lock:
            active_task = next((t for t in self.active_tasks if t.get("id") == task_id), None)
        
        if active_task:
            # Task is active, try to cancel its future
            future = active_task.get("future")
            if future and not future.done():
                return future.cancel()
            return False
        
        # Check pending tasks in queues
        for priority, queue_obj in enumerate(self.task_queues):
            # Create a temporary copy of the queue to avoid losing data
            temp_queue = []
            found_task = False
            
            # Search through queue
            try:
                while not queue_obj.empty():
                    timestamp, task = queue_obj.get_nowait()
                    
                    if task.get("id") == task_id:
                        found_task = True
                        logger.info(f"Cancelled queued task {task_id} with priority {priority}")
                    else:
                        temp_queue.append((timestamp, task))
            except queue.Empty:
                pass
            
            # Restore queue without the cancelled task
            for item in temp_queue:
                queue_obj.put(item)
            
            if found_task:
                return True
        
        # Task not found
        return False
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage statistics.
        
        Returns:
            Dictionary with resource usage information
        """
        # Get current CPU and memory usage
        cpu_percent = np.mean(self.cpu_usage[-5:]) if self.cpu_usage else 0
        memory_percent = self.memory_usage[-1] if self.memory_usage else 0
        
        # Get memory details
        memory = psutil.virtual_memory()
        
        # Get GPU usage if available
        gpu_info = []
        if self.gpu_usage and self.config["gpu"]["enabled"]:
            gpu_info = self.gpu_usage[-1] if self.gpu_usage else []
        
        # Get task queue sizes
        queue_sizes = [q.qsize() for q in self.task_queues]
        
        # Get active task count
        with self.active_tasks_lock:
            active_count = len(self.active_tasks)
            active_by_priority = {}
            for task in self.active_tasks:
                priority = task.get("priority", 1)
                active_by_priority[priority] = active_by_priority.get(priority, 0) + 1
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": os.cpu_count(),
                "available_workers": self.config["cpu"]["max_workers"] - active_count
            },
            "memory": {
                "percent": memory_percent,
                "total_gb": memory.total / (1024 ** 3),
                "used_gb": memory.used / (1024 ** 3),
                "available_gb": memory.available / (1024 ** 3)
            },
            "gpu": gpu_info,
            "tasks": {
                "active_count": active_count,
                "active_by_priority": active_by_priority,
                "queue_sizes": queue_sizes,
                "total_queued": sum(queue_sizes)
            },
            "timestamp": time.time()
        }
    
    def get_resource_history(self) -> Dict[str, Any]:
        """
        Get historical resource usage data.
        
        Returns:
            Dictionary with resource usage history
        """
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "history_size": len(self.cpu_usage),
            "monitoring_interval": self.config["monitoring"]["interval"]
        }
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get information about active tasks.
        
        Returns:
            List of active task information dictionaries
        """
        with self.active_tasks_lock:
            # Create a copy with only safe information
            active_task_info = []
            for task in self.active_tasks:
                task_info = {
                    "id": task.get("id"),
                    "priority": task.get("priority", 1),
                    "submit_time": task.get("submit_time"),
                    "start_time": task.get("start_time")
                }
                
                # Add status information
                if "future" in task:
                    future = task["future"]
                    if future.cancelled():
                        task_info["status"] = "cancelled"
                    elif future.done():
                        try:
                            result = future.result()
                            task_info["status"] = "completed" if result.get("success", False) else "failed"
                        except Exception:
                            task_info["status"] = "failed"
                    else:
                        task_info["status"] = "running"
                        task_info["elapsed"] = time.time() - task.get("start_time", time.time())
                else:
                    task_info["status"] = "pending"
                
                active_task_info.append(task_info)
        
        return active_task_info
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update resource manager configuration.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            Boolean indicating success
        """
        try:
            # Apply new configuration
            self._deep_update(self.config, new_config)
            
            # Update thread and process pools if needed
            if "cpu" in new_config:
                if "thread_pool_size" in new_config["cpu"]:
                    # Create new thread pool with updated size
                    old_pool = self.thread_pool
                    self.thread_pool = ThreadPoolExecutor(
                        max_workers=self.config["cpu"]["thread_pool_size"]
                    )
                    # Shutdown old pool after a delay to allow tasks to complete
                    threading.Thread(target=lambda: (time.sleep(60), old_pool.shutdown())).start()
                
                if "process_pool_size" in new_config["cpu"]:
                    # Create new process pool with updated size
                    old_pool = self.process_pool
                    self.process_pool = ProcessPoolExecutor(
                        max_workers=self.config["cpu"]["process_pool_size"]
                    )
                    # Shutdown old pool after a delay to allow tasks to complete
                    threading.Thread(target=lambda: (time.sleep(60), old_pool.shutdown())).start()
            
            logger.info("Resource manager configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Error updating resource manager configuration: {e}")
            return False
    
    def shutdown(self):
        """Shut down the resource manager and release resources."""
        if not self.running:
            return
            
        logger.info("Shutting down resource manager")
        self.running = False
        
        # Shutdown thread pool
        logger.debug("Shutting down thread pool")
        self.thread_pool.shutdown(wait=False)
        
        # Shutdown process pool
        logger.debug("Shutting down process pool")
        self.process_pool.shutdown(wait=False)
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2)
        
        logger.info("Resource manager shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """
        Handle termination signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down")
        self.shutdown()


# Decorator for resource-managed execution
def resource_managed(priority: int = 1, use_process_pool: bool = False):
    """
    Decorator for executing functions with resource management.
    
    Args:
        priority: Task priority (0-2, higher is more important)
        use_process_pool: Whether to use process pool instead of thread pool
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get resource manager
            resource_manager = ResourceManager()
            
            # Submit task and wait for result
            task_id = resource_manager.submit_task(
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                use_process_pool=use_process_pool
            )
            
            # Wait for task to complete
            while True:
                status = resource_manager.get_task_status(task_id)
                if status["status"] in ["completed", "failed", "cancelled"]:
                    break
                time.sleep(0.1)
            
            # Check for active task with this ID
            with resource_manager.active_tasks_lock:
                active_task = next((t for t in resource_manager.active_tasks if t.get("id") == task_id), None)
            
            if active_task and "future" in active_task:
                future = active_task["future"]
                try:
                    # Get result (will raise exception if task failed)
                    result_data = future.result()
                    if not result_data.get("success", False):
                        raise RuntimeError(f"Task failed: {result_data.get('error', 'Unknown error')}")
                    return result_data.get("result")
                except Exception as e:
                    raise RuntimeError(f"Task execution failed: {str(e)}")
            
            raise RuntimeError(f"Task {task_id} not found or did not complete")
        
        return wrapper
    
    return decorator


# Async context manager for GPU memory management
class GPUMemoryManager:
    """Context manager for optimized GPU memory management."""
    
    def __init__(self, device_index: int = 0, reserve_memory_mb: int = 0):
        """
        Initialize GPU memory manager.
        
        Args:
            device_index: GPU device index
            reserve_memory_mb: Memory to reserve in MB
        """
        self.device_index = device_index
        self.reserve_memory_mb = reserve_memory_mb
        self.device = None
        self.reserved_memory = None
    
    async def __aenter__(self):
        """Acquire GPU memory."""
        if not HAS_TORCH or not torch.cuda.is_available():
            logger.warning("PyTorch or CUDA not available, GPU memory management disabled")
            return self
        
        # Select device
        self.device = torch.device(f"cuda:{self.device_index}")
        
        # Reserve memory if requested
        if self.reserve_memory_mb > 0:
            size = self.reserve_memory_mb * 1024 * 1024  # Convert to bytes
            try:
                self.reserved_memory = torch.empty(size, 
                                                  device=self.device, 
                                                  dtype=torch.uint8)
            except Exception as e:
                logger.error(f"Failed to reserve GPU memory: {e}")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release GPU memory."""
        if self.reserved_memory is not None:
            # Release reserved memory
            del self.reserved_memory
            self.reserved_memory = None
        
        # Clear CUDA cache for this device
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")
