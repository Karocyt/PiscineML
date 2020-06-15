class Vector():
    def __init__(self, values):
        if type(values) is list:
            self.values = values
            self.size = len(values)
        if type(values) is tuple:
            values = range(values[0], values[1])
            self.size = values[1] - values[0]
        if type(values) is int:
            self.size = values
            values = range(0, values)
        if type(values) is range:
            self.values = [float(i) for i in values]

    def __add__(self, v2):
        if type(v2) is Vector:
            if self.size == v2.size:
                v3 = Vector(self.size)
                for i, f in enumerate(self.values):
                    v3.values[i] = self.values[i] + v2.values[i]
                return v3
            raise ValueError("vector with not same size")
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __sub__(self, v2):
        if type(v2) is Vector:
            if self.size == v2.size:
                v3 = Vector(self.size)
                for i, f in enumerate(self.values):
                    v3.values[i] = self.values[i] - v2.values[i]
                return v3
            raise ValueError("vector with not same size")
        return NotImplemented

    def __rsub__(self, v1):
        return NotImplemented

    def __mul__(self, v2):
        if type(v2) is Vector:
            if self.size == v2.size:
                v3 = Vector(self.size)
                for i, f in enumerate(self.values):
                    v3.values[i] = self.values[i] * v2.values[i]
                return sum(v3.values)
            raise ValueError("vector with not same size")
        if isinstance(v2, (int, float)):
            v3 = Vector(self.size)
            for i, f in enumerate(self.values):
                v3.values[i] = self.values[i] * v2
            return v3
        return NotImplemented

    def __rmul__(self, v1):
        return self * v1

    def __truediv__(self, v2):
        if isinstance(v2, (int, float)):
            v3 = Vector(self.size)
            for i, f in enumerate(self.values):
                if v2 == 0:
                    raise ValueError("no zero div sorry")
                v3.values[i] = self.values[i] / v2
            return v3
        return NotImplemented

    def __str__(self):
        txt = "(Vector [ "
        for f in self.values:
            txt += f"{f:.1f} "
        txt += "])"
        return txt

    def __repr__(self):
        return str(self)