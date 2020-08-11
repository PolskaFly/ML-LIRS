class lru_cache:
    def __init__(self):
        self.faults = 0
        self.hits = 0

    def set(self, key: int, value: int, cache, virtual_time: int, features) -> None:
        if key in cache.cache:
            self.hits += 1
            cache.set_cache(key, value, virtual_time)
            cache.move_key(key)
        elif len(cache.cache) < cache.get_capacity():
            self.faults += 1
            cache.set_cache(key, value, virtual_time)
        elif len(cache.cache) >= cache.get_capacity():
            cache.cache.popitem(last=False)
            cache.set_cache(key, value, virtual_time)
            self.faults += 1

    def get_hits(self) -> int:
        return self.hits

    def get_faults(self) -> int:
        return self.faults
