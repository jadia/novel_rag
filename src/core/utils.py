"""
=============================================================================
Utility Module — Logging & Performance Measurement
=============================================================================
This module provides two simple but essential tools used throughout the app:

1. get_logger() — Creates a formatted logger for any module. Using Python's
   built-in logging module instead of raw print() lets us categorize messages
   by severity (INFO, WARNING, ERROR) and trace them to their source module.

2. time_it — A decorator that wraps functions to measure execution time.
   Decorators are a Python pattern where you "wrap" a function with another
   function to add behavior (like timing) without modifying the original code.
=============================================================================
"""

import time
import logging
from functools import wraps


def get_logger(name: str) -> logging.Logger:
    """
    Creates a customized logger with a consistent format.

    WHY use logging instead of print()?
    - Messages are categorized (INFO, WARNING, ERROR) for filtering
    - Each message shows which module produced it (the 'name' parameter)
    - In production, you can redirect logs to files without changing code
    - Libraries like ChromaDB also use logging, so everything stays consistent

    Args:
        name: The name tag for this logger (usually the module or class name).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure handlers once to avoid duplicate messages when
    # get_logger() is called multiple times with the same name.
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create a console handler that formats the output nicely
        console_handler = logging.StreamHandler()

        # Format: [LEVEL] - ModuleName - The actual message
        # This makes it easy to grep/filter logs by module or severity.
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def time_it(func):
    """
    A 'Decorator' that automatically measures how long a function takes.

    HOW DECORATORS WORK:
    When you write @time_it above a function definition, Python replaces
    that function with a new 'wrapper' function. The wrapper:
      1. Records the start time
      2. Calls the original function
      3. Records the end time
      4. Prints the difference
      5. Returns whatever the original function returned

    This is the 'Aspect-Oriented Programming' pattern — adding cross-cutting
    concerns (timing, logging, caching) without cluttering your business logic.

    Usage:
        @time_it
        def my_slow_function():
            ...
    """
    @wraps(func)  # Preserves the original function's name and docstring
    def wrapper(*args, **kwargs):
        # Record the exact moment before the function runs
        start_time = time.perf_counter()

        # Run the actual function
        result = func(*args, **kwargs)

        # Record the moment it finishes
        end_time = time.perf_counter()

        # Calculate the difference to find out how long it took in seconds
        execution_time = end_time - start_time

        # We print directly here so the metrics always stand out clearly
        print(f"⏱️  METRIC: {func.__name__!r} executed in {execution_time:.4f} seconds.")

        # Return whatever the original function was supposed to return
        return result

    return wrapper
