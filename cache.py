import collections

class cache():
    def __init__(self, size):
        self.capacity = size
        self.cache = collections.OrderedDict()

    def get_capacity(self):
        return self.capacity

    def set_cache(self, key: int, value: int,reuse: int, recency: int, time: int):
        self.cache[key] = [value, [reuse, recency, max(reuse, recency)], time]

    def move_key(self, key):
        self.cache.move_to_end(key)

    def pop(self, key: int):
        self.cache.popitem(key)

    def get_access(self, key):
        return self.cache[key][1][1]

    def get_items(self):
        return self.cache.items()