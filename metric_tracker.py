from collections import defaultdict


class MetricTracker:
    def __init__(self):
        self.values = defaultdict(list)
        # self.counts = defaultdict(int)
        self.aggregators = defaultdict(lambda: "avg")

    def _avg(self, values):
        if len(values) == 0:
            return 0
        return sum(values) / len(values)

    def _sum(self, values):
        return sum(values)

    def _last(self, values):
        if len(values) == 0:
            return 0
        return values[-1]

    def update_if_nonzero(self, key: str, value: float, aggregator: str = "avg"):
        if value == 0:
            return
        self.update(key, value, aggregator)

    def update(self, key: str, value: float, aggregator: str = "avg"):
        self.values[key].append(value)
        # self.counts[key] += 1
        if key in self.aggregators and self.aggregators[key] != aggregator:
            raise ValueError(f"Aggregator for {key} already set to {self.aggregators[key]}")
        self.aggregators[key] = aggregator

    def aggregate(self, key: str) -> float:
        if key not in self.values:
            return 0
        if self.aggregators[key] == "avg":
            return self._avg(self.values[key])
        elif self.aggregators[key] == "sum":
            return self._sum(self.values[key])
        elif self.aggregators[key] == "last":
            return self._last(self.values[key])
        return 0

    def reset(self):
        self.values = defaultdict(list)
        self.counts = defaultdict(int)

    def get_all_results(self):
        return {key: self.aggregate(key) for key in self.values}

    def get_formatted_result(self, sep: str = " | ", prefix: str = ""):
        body = sep.join([f"{key}: {self.aggregate(key):.4f}" for key in self.values])
        if prefix:
            return f"{prefix}{sep}{body}"
        return body
