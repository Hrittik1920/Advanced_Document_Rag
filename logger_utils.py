# logger_utils.py
import os
import time
import functools
import inspect
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_today_log_file() -> str:
    today = datetime.now().strftime("%d%m%Y")
    return os.path.join(LOG_DIR, f"log{today}.txt")

def log_timing(message: str):
    """Writes a message to the daily log file with a timestamp."""
    log_file = get_today_log_file()
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_debug(message: str):
    """Wrapper for log_timing to specifically mark debug messages."""
    log_timing(f"[DEBUG] {message}")

def timed(func):
    """
    Decorator that calculates and logs the execution time of a function.
    Automatically detects and handles both synchronous and asynchronous functions.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        log_timing(f"[TOTAL] {func.__name__} took {duration:.2f} ms")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        log_timing(f"[TOTAL] {func.__name__} took {duration:.2f} ms")
        return result

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper