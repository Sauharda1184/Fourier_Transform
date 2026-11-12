"""
Creates the example data files in the data/ folder.

Run it once with:
    python3 make_sample_data.py
"""

import csv
import math
import os


def write_file(filename, x_start, x_end, number_of_points, f):
    """Write evenly spaced points (x, f(x)) to a file.

    The right-hand end x_end is left out on purpose, the same way the
    book does it.
    """
    gap = (x_end - x_start) / number_of_points
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        for j in range(number_of_points):
            x = x_start + j * gap
            writer.writerow([repr(x), repr(f(x))])
    print("Wrote", filename)


def example1(x):
    # Book Example 1
    return 2 * x ** 2 - 9


def example2(x):
    # Book Example 2. The book's "tan x(x - 2)" means tan of x*(x - 2).
    return x ** 4 - 3 * x ** 3 + 2 * x ** 2 - math.tan(x * (x - 2))


def known_waves(x):
    # Built from waves we chose ourselves, so we know the right answer:
    # a_0 = 4 (because the curve uses a_0/2), a_1 = 3, b_2 = 1, a_5 = 0.5,
    # and every other coefficient is 0.
    return 2 + 3 * math.cos(x) + math.sin(2 * x) + 0.5 * math.cos(5 * x)


def sample(x):
    # A wiggly made-up function for a big dataset
    return math.exp(-x) * math.sin(5 * x) + 0.5 * x


os.makedirs("data", exist_ok=True)
write_file("data/example1.csv", -math.pi, math.pi, 4, example1)
write_file("data/example2.csv", 0, 2, 8, example2)
write_file("data/known_waves_256.csv", -math.pi, math.pi, 256, known_waves)
write_file("data/sample_256.csv", 0, 4, 256, sample)
