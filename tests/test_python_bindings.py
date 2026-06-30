import miniNumpy as mnp


def test_array1d_basic():
    a = mnp.arange(0, 5)

    assert len(a) == 5
    assert a.shape == (5,)

    assert a[0] == 0.0
    assert a[1] == 1.0
    assert a[-1] == 4.0

    a[1] = 10.0
    assert a[1] == 10.0


def test_array1d_arithmetic():
    a = mnp.arange(0, 5)
    b = mnp.ones(5)

    assert str(a + b) == "[1,2,3,4,5]"
    assert str(a * 2.0) == "[0,2,4,6,8]"
    assert str(2.0 * a) == "[0,2,4,6,8]"
    assert str(10.0 - a) == "[10,9,8,7,6]"


def test_array1d_reductions():
    a = mnp.arange(0, 5)

    assert a.sum() == 10.0
    assert a.mean() == 2.0
    assert a.min() == 0.0
    assert a.max() == 4.0


def test_array1d_index_error():
    a = mnp.arange(0, 5)

    try:
        _ = a[100]
        assert False
    except IndexError:
        pass


def test_array2d_basic():
    x = mnp.zeros2d(2, 3)

    assert x.shape == (2, 3)
    assert x.rows() == 2
    assert x.cols() == 3
    assert x.size() == 6

    x[0, 1] = 5.0
    x[-1, -1] = 9.0

    assert x[0, 1] == 5.0
    assert x[-1, -1] == 9.0


def test_array2d_reductions():
    x = mnp.ones2d(2, 3)

    assert x.sum() == 6.0
    assert x.mean() == 1.0
    assert x.min() == 1.0
    assert x.max() == 1.0


def test_array2d_transpose_flatten():
    a = mnp.reshape(mnp.arange(0, 6), 2, 3)

    assert a.shape == (2, 3)

    t = a.transpose()
    assert t.shape == (3, 2)

    flat = a.flatten()
    assert flat.shape == (6,)
    assert str(flat) == "[0,1,2,3,4,5]"


def test_array2d_index_error():
    x = mnp.zeros2d(2, 3)

    try:
        _ = x[10, 0]
        assert False
    except IndexError:
        pass

def test_array2d_row_col():
    a = mnp.reshape(mnp.arange(0, 6), 2, 3)

    r0 = a.row(0)
    assert str(r0) == "[0,1,2]"

    r1 = a.row(1)
    assert str(r1) == "[3,4,5]"

    c0 = a.col(0)
    assert str(c0) == "[0,3]"

    c1 = a.col(1)
    assert str(c1) == "[1,4]"

    c2 = a.col(2)
    assert str(c2) == "[2,5]"


if __name__ == "__main__":
    test_array1d_basic()
    test_array1d_arithmetic()
    test_array1d_reductions()
    test_array1d_index_error()
    test_array2d_basic()
    test_array2d_reductions()
    test_array2d_transpose_flatten()
    test_array2d_index_error()
    test_array2d_row_col()

    print("All Python binding tests passed!")