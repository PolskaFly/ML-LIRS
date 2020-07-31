import collections

class cache():
    def __init__(self, size):
        self.capacity = size
        self.cache = collections.OrderedDict()

    def get_capacity(self):
        return self.capacity

    def set_cache(self, key: int, value: int, access_count: int):
        self.cache[key] = [value, access_count]

    def move_key(self, key: int):
        self.cache.move_to_end(key)