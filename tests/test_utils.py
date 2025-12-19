import numpy as np

from amsterdm.utils import symlog, symlog10


def test_symlog():
    array = np.linspace(-np.exp(5), np.exp(5), 20)
    larray = symlog(array)
    # Note how the end points are not exactly 5, sine there is a +1 offset in symlog
    np.testing.assert_allclose(
        [
            -5.00671535,
            -4.8962768,
            -4.77210974,
            -4.63030997,
            -4.46502736,
            -4.26690993,
            -4.01959466,
            -3.69028083,
            -3.19596155,
            -2.17602578,
            2.17602578,
            3.19596155,
            3.69028083,
            4.01959466,
            4.26690993,
            4.46502736,
            4.63030997,
            4.77210974,
            4.8962768,
            5.00671535,
        ],
        larray,
    )

    array = np.logspace(1, 10)
    larray = symlog(array)
    np.testing.assert_allclose(np.log(1 + array), larray)


def test_symlog10():
    array = np.linspace(-1e5, 1e5, 20)
    larray = symlog10(array)
    # Note how the end points are not exactly 5, sine there is a +1 offset in symlog10
    np.testing.assert_allclose(
        [
            -5.36221757,
            -5.31391312,
            -5.25955574,
            -5.1974082,
            -5.12485803,
            -5.03770858,
            -4.92856525,
            -4.78243926,
            -4.56059529,
            -4.08349792,
            4.08349792,
            4.56059529,
            4.78243926,
            4.92856525,
            5.03770858,
            5.12485803,
            5.1974082,
            5.25955574,
            5.31391312,
            5.36221757,
        ],
        larray,
    )
