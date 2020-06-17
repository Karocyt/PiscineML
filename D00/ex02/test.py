from TinyStatistician import TinyStatistician


def test_mean():
    assert TinyStatistician.mean([2.0, 5.0]) == 3.5


def test_median():
    # assert TinyStatistician.median([2.0, 18.0, 19.0]) == 10.5 # Mathematical median
    assert TinyStatistician.median([2.0, 18.0, 19.0]) == 18.0  # median value


def test_quartiles():
    # 1D array
    arr = [20, 2, 7, 1, 34]

    assert TinyStatistician.quartile(arr, 50) == 7.0
    assert TinyStatistician.quartile(arr, 25) == 2.0
    assert TinyStatistician.quartile(arr, 75) == 20.0
    assert TinyStatistician.quartile(arr, 1) <= 2.0
    assert TinyStatistician.quartile(arr, 1) > 1.0


def test_var():
    assert TinyStatistician.var([2.0, 6.0]) == 4.0


def test_std():
    assert TinyStatistician.std([2.0, 6.0]) == 2.0


def test_correct():
    a = [1, 42, 300, 10, 59]
    # a = [1, 10, 42, 59, 300]
    assert TinyStatistician.mean(a) == 82.4
    assert TinyStatistician.median(a) == 42.0
    assert TinyStatistician.quartile(a, 25) == 10.0
    assert TinyStatistician.quartile(a, 75) == 59.0
    assert TinyStatistician.var(a) == 12279.439999999999
    assert TinyStatistician.std(a) == 110.81263465868862
