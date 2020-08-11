import collections

class cache():
    def __init__(self, size):
        self.capacity = size
        self.cache = collections.OrderedDict()

    def get_capacity(self):
        return self.capacity

    def set_cache(self, key: int, value: int, virtual_time: int):
        self.cache[key] = [value, virtual_time]

    def move_key(self, key: int):
        self.cache.move_to_end(key)

    def pop(self, key):
        self.cache.popitem(key)

    def get_items(self):
        return self.cache.items()