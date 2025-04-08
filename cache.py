import psutil
import numpy as np
from collections import OrderedDict
import sys

def get_array_size_bytes(arr):
    return arr.nbytes if isinstance(arr, np.ndarray) else sys.getsizeof(arr)

class MemoryAwareLRUCache:
    def __init__(self, max_memory_fraction=0.5):
        self.cache = OrderedDict()
        self.total_size = 0
        self.max_memory = psutil.virtual_memory().total * max_memory_fraction

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key][0]

    def put(self, key, value):
        size = get_array_size_bytes(value)
        # Evict if needed
        while self.total_size + size > self.max_memory and len(self.cache) > 0:
            _, (evicted_val, evicted_size) = self.cache.popitem(last=False)
            self.total_size -= evicted_size

        self.cache[key] = (value, size)
        self.total_size += size
        self.cache.move_to_end(key)

    def clear(self):
        self.cache.clear()
        self.total_size = 0
