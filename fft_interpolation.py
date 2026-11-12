"""
Fast Fourier Transform for trigonometric interpolation.

This program follows the Fast Fourier Transform algorithm from Burden & Faires,
"Numerical Analysis", Chapter 8. The comments say "Step 1", "Step 2", ... so
you can read the code side by side with the book.

What it does, in plain words:
    1. Reads a list of points (x, y) from a file.
    2. Checks that the points are usable.
    3. Runs the Fast Fourier Transform on the y-values.
    4. Turns the result into the numbers a_0, a_1, ... and b_1, b_2, ...
    5. Prints those numbers and saves them to a file.

Those numbers build a wavy curve S that passes through every data point:

    S(z) = a_0/2 + a_m*cos(m*z)/2
           + [a_1*cos(1*z) + b_1*sin(1*z)]
           + [a_2*cos(2*z) + b_2*sin(2*z)]
           + ...
           + [a_(m-1)*cos((m-1)*z) + b_(m-1)*sin((m-1)*z)]

How to run it:
    python3 fft_interpolation.py data/sample_256.csv
"""

import cmath
import csv
import math
import sys


def read_data(filename):
    """Read the points from a file and return two lists: the x's and the y's.

    The file has one point per line, written as "x,y". The first line may be
    a title line such as "x,y"; it is skipped.
    """
    x = []
    y = []
    with open(filename) as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            # Skip empty lines.
            if len(row) == 0 or "".join(row).strip() == "":
                continue
            if len(row) != 2:
                raise ValueError(
                    f"Line {line_number} of {filename} should have exactly "
                    f"two numbers separated by a comma, like:  0.25,1.5"
                )
            try:
                x_value = float(row[0])
                y_value = float(row[1])
            except ValueError:
                # Words instead of numbers are fine on the first line only,
                # because that is the title line.
                if len(x) == 0:
                    continue
                raise ValueError(
                    f"Line {line_number} of {filename} has something that "
                    f"is not a number: {row}"
                )
            x.append(x_value)
            y.append(y_value)
    return x, y


def check_data(x, y):
    """Make sure the data works with the FFT, and return m and p.

    The book calls the number of points 2m, and needs m = 2^p.
    So the number of points must be 2, 4, 8, 16, 32, 64, 128, 256, 512, ...
    The x-values must also be evenly spaced, like marks on a ruler.
    """
    number_of_points = len(y)
    if number_of_points < 2:
        raise ValueError("The file needs at least 2 data points.")

    # Find m and p so that number_of_points = 2m and m = 2^p.
    m = number_of_points // 2
    p = 0
    while 2 ** p < m:
        p = p + 1
    if 2 * m != number_of_points or 2 ** p != m:
        # Find the closest power of 2 below, to suggest it in the message.
        smaller = 2
        while smaller * 2 <= number_of_points:
            smaller = smaller * 2
        raise ValueError(
            f"The file has {number_of_points} points, but the FFT needs the "
            f"number of points to be a power of 2 "
            f"(for example {smaller} or {2 * smaller})."
        )

    # Check the gap between neighbouring x-values is always the same.
    gap = x[1] - x[0]
    if gap <= 0:
        raise ValueError("The x-values must go from smallest to largest.")
    for j in range(1, number_of_points):
        if abs((x[j] - x[j - 1]) - gap) > 1e-6 * gap:
            raise ValueError(
                f"The x-values are not evenly spaced: the gap between point "
                f"{j - 1} and point {j} is {x[j] - x[j - 1]}, but the first "
                f"gap is {gap}."
            )
    return m, p


def reverse_bits(number, number_of_bits):
    """Write a number in binary, flip the digits back to front, and return it.

    Example with 3 bits:  6 is 110 in binary.  Backwards that is 011, which
    is 3.  So reverse_bits(6, 3) gives 3.

    This is the "Decompose K" part of Steps 8 and 14 in the book.
    """
    result = 0
    for _ in range(number_of_bits):
        last_bit = number % 2          # take the last binary digit
        result = result * 2 + last_bit  # stick it on the end of the result
        number = number // 2           # throw that digit away
    return result


def fft(y, m, p):
    """Run the Fast Fourier Transform from the book (Steps 1 to 16).

    y is the list of 2m data values.  Returns the list c_0, ..., c_(2m-1),
    which are complex numbers.
    """
    # Step 1
    M = m
    q = p
    zeta = cmath.exp(cmath.pi * 1j / m)

    # Step 2: start with c equal to the data.
    c = []
    for j in range(2 * m):
        c.append(complex(y[j]))

    # Step 3: xi holds points spaced evenly around a circle.
    # We work them out once here so we never have to redo the work later.
    xi = [0] * (2 * m + 1)
    for j in range(1, M + 1):
        xi[j] = zeta ** j
        xi[j + M] = -xi[j]

    # Step 4
    K = 0
    xi[0] = 1

    # Step 5: each pass mixes the numbers in pairs. The pairs start far
    # apart (M apart) and get closer each pass, because M is halved.
    for L in range(1, p + 2):

        # Step 6
        while K < 2 * m - 1:

            # Step 7
            for j in range(1, M + 1):

                # Step 8: chop the last q binary digits off K, then
                # flip what is left back to front.
                K1 = K // 2 ** q
                K2 = reverse_bits(K1, p + 1)

                # Step 9: mix c[K] with its partner c[K + M].
                eta = c[K + M] * xi[K2]
                c[K + M] = c[K] - eta
                c[K] = c[K] + eta

                # Step 10
                K = K + 1

            # Step 11: jump over the partners we just handled.
            K = K + M

        # Step 12 (// keeps M a whole number)
        K = 0
        M = M // 2
        q = q - 1

    # Step 13: the answers are all there, but in a scrambled order.
    # Steps 14-16 put them back in the right order.
    while K < 2 * m - 1:

        # Step 14
        j = reverse_bits(K, p + 1)

        # Step 15
        if j > K:
            c[j], c[K] = c[K], c[j]

        # Step 16
        K = K + 1

    return c


def get_coefficients(c, m):
    """Turn the c's into the a's and b's (Steps 17 and 18).

    Returns two lists.  a[k] is a_k for k = 0, ..., m.  b[k] is b_k for
    k = 1, ..., m-1  (b[0] is not used and is left as 0).
    """
    a = [0.0] * (m + 1)
    b = [0.0] * m

    # Step 17
    a[0] = (c[0] / m).real
    a[m] = (cmath.exp(-1j * cmath.pi * m) * c[m] / m).real

    # Step 18  (e^(-i*pi*j) is just +1 when j is even and -1 when j is odd)
    for j in range(1, m):
        value = cmath.exp(-1j * cmath.pi * j) * c[j] / m
        a[j] = value.real
        b[j] = value.imag

    return a, b


def evaluate(x_value, a, b, m, x_start, x_end):
    """Work out the height of the wavy curve S at one x-value.

    The book's formulas live on the interval from -pi to pi, so first we
    slide and stretch x into that interval and call the result z.
    """
    z = math.pi * (2 * (x_value - x_start) / (x_end - x_start) - 1)

    total = a[0] / 2 + a[m] * math.cos(m * z) / 2
    for k in range(1, m):
        total = total + a[k] * math.cos(k * z) + b[k] * math.sin(k * z)
    return total


def get_interval(x):
    """Return the start and end of the interval the data lives on.

    The end is one gap past the last point, because the book's data stops
    one step short of the right-hand end.
    """
    gap = x[1] - x[0]
    return x[0], x[0] + len(x) * gap


def save_results(filename, c, a, b, m):
    """Save all the c's, a's and b's to a file that opens in Excel."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["k", "c_k real part", "c_k imaginary part", "a_k", "b_k"])
        for k in range(2 * m):
            a_text = repr(a[k]) if k <= m else ""
            b_text = repr(b[k]) if 1 <= k <= m - 1 else ""
            writer.writerow([k, repr(c[k].real), repr(c[k].imag), a_text, b_text])


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("Name of the data file: ").strip()

    try:
        x, y = read_data(filename)
        m, p = check_data(x, y)
    except (OSError, ValueError) as problem:
        print("Problem with the data file:", problem)
        sys.exit(1)

    c = fft(y, m, p)
    a, b = get_coefficients(c, m)
    x_start, x_end = get_interval(x)

    # Step 19: OUTPUT
    print(f"Read {2 * m} points from {filename}   (m = {m}, p = {p})")
    print(f"The data covers the interval from {x_start:.6g} to {x_end:.6g}.")
    print()
    print("The curve through the points is")
    print(f"  S(z) = a_0/2 + a_{m}*cos({m}z)/2 + sum of [a_k*cos(kz) + b_k*sin(kz)]"
          f" for k = 1 to {m - 1}")
    print(f"  where z = pi*(2*(x - {x_start:.6g})/{x_end - x_start:.6g} - 1)")
    print()

    rows_to_show = min(m, 10)
    print(f"{'k':>4} {'a_k':>18} {'b_k':>18}")
    for k in range(rows_to_show + 1):
        b_text = f"{b[k]:18.10f}" if 1 <= k <= m - 1 else f"{'-':>18}"
        print(f"{k:>4} {a[k]:18.10f} {b_text}")
    if m > rows_to_show:
        print(f"  ... ({m - rows_to_show} more rows are in the output file)")
    print()

    # Safety check: the curve should hit every data point.
    biggest_miss = 0.0
    for j in range(2 * m):
        miss = abs(evaluate(x[j], a, b, m, x_start, x_end) - y[j])
        biggest_miss = max(biggest_miss, miss)
    print(f"Check: the curve misses the data points by at most {biggest_miss:.2e}")

    if filename.endswith(".csv"):
        output_filename = filename[:-4] + "_coefficients.csv"
    else:
        output_filename = filename + "_coefficients.csv"
    save_results(output_filename, c, a, b, m)
    print(f"All coefficients were saved to {output_filename}")


if __name__ == "__main__":
    main()
