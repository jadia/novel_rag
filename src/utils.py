import time
import logging
from functools import wraps

def get_logger(name: str) -> logging.Logger:
    """
    Creates a customized logger.
    Instead of using standard print statements, a logger helps us categorize
    whether a message is just info, a warning, or a critical error.
    """
    logger = logging.getLogger(name)
    
    # Only configure it once to avoid duplicate messages
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create a console handler that formats the output nicely
        console_handler = logging.StreamHandler()
        # Look how we format: [Info/Warning] - NameOfFile - The actual message
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger

def time_it(func):
    """
    This is a 'Decorator'. It wraps around other functions to automatically
    measure how long they take to execute without adding clutter directly
    into those functions.
    
    Usage:
    @time_it
    def my_slow_function(): ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Record the exact moment before the function runs
        start_time = time.perf_counter()
        
        # Run the actual function
        result = func(*args, **kwargs)
        
        # Record the moment it finishes
        end_time = time.perf_counter()
        
        # Calculate the difference to find out how long it took in seconds
        execution_time = end_time - start_time
        
        # We print directly here so the metrics always stand out clearly in the console
        print(f"⏱️  METRIC: {func.__name__!r} executed in {execution_time:.4f} seconds.")
        
        # Return whatever the original function was supposed to return
        return result
        
    return wrapper
