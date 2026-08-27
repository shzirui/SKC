import logging
import os
from pathlib import Path

_logging_initialized = False

def setup_logging():
    """Set up unified logging configuration: output to both console and file"""
    global _logging_initialized
    
    # Avoid duplicate initialization
    if _logging_initialized:
        return logging.getLogger()
    
    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear previous handlers (avoid duplicates)
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. File handler
    log_file = log_dir / "computer_use_rollout.log"
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # More detailed logs in file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 3. Configure logging levels for specific modules
        # Asyncio module logging
        asyncio_logger = logging.getLogger('asyncio')
        asyncio_logger.setLevel(logging.ERROR)
        asyncio_logger.addHandler(file_handler)
        
        # Ray module logging 
        ray_logger = logging.getLogger('ray')
        ray_logger.setLevel(logging.ERROR)
        ray_logger.addHandler(file_handler)
        
        # Ensure all submodule logs are recorded
        root_logger.propagate = True
        
        _logging_initialized = True
        logging.info(f"Logging system initialized - Log file: {log_file.absolute()}")
        
        # 4. Redirect standard error output to logs (simplified)
        import sys
        
        class SimpleLogCapture:
            def __init__(self, original_stream, logger):
                self.original_stream = original_stream
                self.logger = logger
                self.line_buffer = []
                
            def write(self, message):
                # Output to both original stream and logs
                if hasattr(self.original_stream, 'write'):
                    self.original_stream.write(message)
                
                # Log to file, choose appropriate log level based on content
                if message and message.strip():
                    lines = message.split('\n')
                    for line in lines:
                        stripped_line = line.strip()
                        if stripped_line:
                            self._log_line_appropriately(stripped_line)
                            
            def _log_line_appropriately(self, line):
                """Choose appropriate log level based on content"""
                # Filter Ray Actor internal identification info
                if line.startswith(':actor_name:'):
                    return
                    
                # Check if it's Ray Actor output
                if '(TrajectoryRunnerActor pid=' in line or '(ModelServicePool pid=' in line:
                    # Determine if it's a real error message
                    if self._is_real_error(line):
                        self.logger.error(f"STDERR: {line}")
                    else:
                        # Normal program output, use INFO level instead of DEBUG
                        self.logger.info(f"Ray Actor: {line}")
                    return
                    
                # Filter other Ray internal info
                if 'repeated' in line and 'across cluster' in line:
                    return
                    
                # Filter Ray's stderr wrapper output
                if line.startswith('[stderr][ERROR] - STDERR:'):
                    # Extract actual content
                    actual_content = line.replace('[stderr][ERROR] - STDERR:', '').strip()
                    if 'Logging system initialized' in actual_content:
                        # This is normal initialization message, use INFO level
                        self.logger.info(f"Ray Process: {actual_content}")
                    elif self._is_real_error(actual_content):
                        self.logger.error(f"Ray Process Error: {actual_content}")
                    else:
                        self.logger.info(f"Ray Process: {actual_content}")
                    return
                    
                # Other stderr output still recorded as errors
                self.logger.error(f"STDERR: {line}")
                
            def _is_real_error(self, line):
                """Determine if it's a real error message"""
                error_indicators = [
                    'Traceback', 'Exception', 'Error:', 'ERROR:',
                    'Failed', 'failed', 'AttributeError', 'ValueError', 
                    'TypeError', 'RuntimeError', 'KeyError'
                ]
                return any(indicator in line for indicator in error_indicators)
            
            def flush(self):
                if hasattr(self.original_stream, 'flush'):
                    self.original_stream.flush()
                    
            def __getattr__(self, name):
                return getattr(self.original_stream, name)
        
        # Save original stderr and redirect
        stderr_logger = logging.getLogger('stderr')
        original_stderr = sys.stderr
        sys.stderr = SimpleLogCapture(original_stderr, stderr_logger)
        
    except Exception as e:
        logging.error(f"Unable to create log file handler: {e}")
        print(f"Warning: Unable to create log file {log_file}: {e}")
    
    return root_logger