import numpy as np


class TinyStatistician():

    @staticmethod
    def mean(x):
        if len(x) == 0:
            return None
        return float(sum(x))/float(len(x))

    @staticmethod
    def median(x):
        if len(x) == 0:
            return None
        # return min(x) + (max(x) - min(x)) / 2 # mathemeatical median
        return TinyStatistician.quartile(x, 50)  # Median value

    @staticmethod
    def quartile(x, percentile):
        if len(x) == 0 or percentile <= 0.0 or percentile > 100.0:
            return None
        percentile /= 100.0
        cp = np.sort(x)

        idx = (len(x) - 1) * percentile
        print(idx, idx.is_integer())
        if idx.is_integer():
            return float(cp[int(idx)])
        else:
            return TinyStatistician.mean([float(cp[int(idx)]), float(cp[int(idx + 1)])])

    @staticmethod
    def var(x):
        if len(x) == 0:
            return None
        mean = TinyStatistician.mean(x)
        return sum((float(elem) - mean)**2 for elem in x) / len(x)

    @staticmethod
    def std(x):
        if len(x) == 0:
            return None
        return np.sqrt(TinyStatistician.var(x))
