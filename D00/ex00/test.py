from Vector import Vector


def test_add():  # basic creation
    op = (Vector([3, 4.0, 5.0]) + Vector([1, 2, 3]))
    res = Vector([4.0, 6.0, 8.0])
    assert op == res


def test_sub():  # basic creation
    op = (Vector([3.0, 4.0, 5.0]) - Vector([1, 2, 3]))
    res = Vector([2.0, 2.0, 2.0])
    assert op == res


def test_dotmul_v():  # basic creation
    op = (Vector([3.0, 4.0, 5.0]) * Vector([1.0, 2.0, 3.0]))
    res = 26
    assert op == res


def test_mul_f():  # basic creation
    op = (2 * Vector([3.0, 4.0, 5.0]))
    res = Vector([6.0, 8.0, 10.0])
    assert op == res


def test_div():  # basic creation
    op = (Vector([3.0, 4.0, 5.0]) / 2)
    res = Vector([1.5, 2.0, 2.5])
    assert op == res
