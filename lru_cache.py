class lru_cache:
    def __init__(self):
        self.faults = 0
        self.hits = 0

    def set(self, key: int, value: int, cache, access_amount: int, access_time: int, labels, features) -> None:
        if key in cache.cache:
            self.hits += 1
            cache.cache[key][1] += 1
            cache.move_key(key)
        elif len(cache.cache) < cache.get_capacity():
            self.faults += 1
            cache.set_cache(key, value, access_amount)
        elif len(cache.cache) >= cache.get_capacity():
            cache.cache.popitem(last=False)
            cache.set_cache(key, value, access_amount)
            self.faults += 1

        if key in labels:
            labels[key][1] = access_time
            features[key][0] += 1
        else:
            labels[key] = [access_time, -1]
            features[key] = [access_amount]

    def get_hits(self) -> int:
        return self.hits

    def get_faults(self) -> int:
        return self.faults
