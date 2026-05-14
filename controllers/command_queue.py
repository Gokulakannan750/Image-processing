"""
controllers/command_queue.py
============================
Priority queue for hardware commands.
Decouples navigation decisions from hardware execution.
"""
import time
import queue
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict

class CommandPriority(IntEnum):
    CRITICAL = 0   # E-STOP, Safety Overrides
    HIGH = 1       # Alignment corrections
    NORMAL = 2     # U-Turns
    LOW = 3        # Informational / Status queries

@dataclass(order=True)
class HardwareCommand:
    priority: CommandPriority
    timestamp: float = field(compare=False)
    command_type: str = field(compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class CommandQueue:
    """Thread-safe priority queue for hardware commands."""
    
    def __init__(self, maxsize: int = 100):
        self._queue = queue.PriorityQueue(maxsize=maxsize)
        
    def push(self, command: HardwareCommand) -> bool:
        """Pushes a command onto the queue. Returns True if successful."""
        try:
            self._queue.put_nowait(command)
            return True
        except queue.Full:
            return False
            
    def pop(self, timeout: float = None) -> HardwareCommand:
        """Pops the highest priority command from the queue. Blocks up to timeout seconds."""
        try:
            return self._queue.get(block=timeout is not None, timeout=timeout)
        except queue.Empty:
            return None
            
    def clear(self) -> None:
        """Clears all pending commands (e.g. upon E-STOP)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
