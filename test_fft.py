"""
Checks that fft_interpolation.py gives the right answers.

Run it with:
    python3 test_fft.py

Run make_sample_data.py first so the data files exist.
"""

import cmath

from fft_interpolation import (
    check_data,
    evaluate,
    fft,
    get_coefficients,
    get_interval,
    read_data,
)


def run(filename):
    """Run the whole program on one file and return everything it found."""
    x, y = read_data(filename)
    m, p = check_data(x, y)
    c = fft(y, m, p)
    a, b = get_coefficients(c, m)
    return x, y, m, c, a, b


def slow_transform(y, m):
    """The slow, obvious way to get the c's: add everything up directly.

    The FFT must give the same numbers as this, only faster.
    """
    c = []
    for k in range(2 * m):
        total = 0
        for j in range(2 * m):
            total = total + y[j] * cmath.exp(1j * k * cmath.pi * j / m)
        c.append(total)
    return c


def close(first, second, tolerance):
    return abs(first - second) < tolerance


def test_book_example_1():
    x, y, m, c, a, b = run("data/example1.csv")
    assert close(a[0], -3.19559339, 1e-8)
    assert close(a[1], -9.86960441, 1e-8)
    assert close(a[2], 4.93480220, 1e-8)
    assert close(b[1], 0.0, 1e-8)


def test_book_example_2():
    x, y, m, c, a, b = run("data/example2.csv")
    # The book prints 6 significant digits.
    # It writes a_0/2 and a_4/2 in its formula, so those two are halved here.
    assert close(a[0] / 2, 0.761979, 1e-6)
    assert close(a[1], 0.771841, 1e-6)
    assert close(a[2], 0.0173037, 1e-7)
    assert close(a[3], 0.00686304, 1e-8)
    assert close(a[4] / 2, -0.000578545, 1e-9)
    assert close(b[1], -0.386374, 1e-6)
    assert close(b[2], 0.0468750, 1e-7)
    assert close(b[3], -0.0113738, 1e-7)


def test_known_waves():
    # The data is 2 + 3*cos(x) + sin(2x) + 0.5*cos(5x), so we know exactly
    # how much of each wave the program should find.
    x, y, m, c, a, b = run("data/known_waves_256.csv")
    expected_a = [0.0] * (m + 1)
    expected_b = [0.0] * m
    expected_a[0] = 4.0   # the curve uses a_0/2, and 4/2 = 2
    expected_a[1] = 3.0
    expected_b[2] = 1.0
    expected_a[5] = 0.5
    for k in range(m + 1):
        assert close(a[k], expected_a[k], 1e-9), k
    for k in range(1, m):
        assert close(b[k], expected_b[k], 1e-9), k


def test_fft_matches_slow_way():
    x, y, m, c, a, b = run("data/sample_256.csv")
    slow_c = slow_transform(y, m)
    for k in range(2 * m):
        assert close(c[k], slow_c[k], 1e-8), k


def test_curve_goes_through_every_point():
    x, y, m, c, a, b = run("data/sample_256.csv")
    x_start, x_end = get_interval(x)
    for j in range(2 * m):
        height = evaluate(x[j], a, b, m, x_start, x_end)
        assert close(height, y[j], 1e-9), j


def test_fft_matches_numpy():
    try:
        import numpy
    except ImportError:
        print("  (numpy is not installed, skipping this check)")
        return
    x, y, m, c, a, b = run("data/sample_256.csv")
    # numpy's ifft divides by the number of points, so we multiply it back.
    numpy_c = 2 * m * numpy.fft.ifft(y)
    for k in range(2 * m):
        assert close(c[k], numpy_c[k], 1e-8), k


def test_bad_point_count_is_rejected():
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    try:
        check_data(x, y)
    except ValueError:
        return
    assert False, "6 points should have been rejected"


def test_uneven_spacing_is_rejected():
    x = [0.0, 1.0, 2.0, 3.5]
    y = [1.0, 2.0, 3.0, 4.0]
    try:
        check_data(x, y)
    except ValueError:
        return
    assert False, "uneven spacing should have been rejected"


tests = [
    test_book_example_1,
    test_book_example_2,
    test_known_waves,
    test_fft_matches_slow_way,
    test_curve_goes_through_every_point,
    test_fft_matches_numpy,
    test_bad_point_count_is_rejected,
    test_uneven_spacing_is_rejected,
]

for test in tests:
    print("Running", test.__name__)
    test()
print("All checks passed.")
