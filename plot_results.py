"""
Draws a picture of what the FFT found for a data file: the data points,
with the wavy curve S(x) drawn through them.

How to run it:
    python3 plot_results.py data/sawtooth_256.csv     (one file)
    python3 plot_results.py                           (every data file)

The pictures are saved in the plots/ folder.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")   # draw straight to files; no window needed
import matplotlib.pyplot as plt

from fft_interpolation import (
    check_data,
    evaluate,
    fft,
    get_coefficients,
    get_interval,
    read_data,
)


def plot_file(filename):
    x, y = read_data(filename)
    m, p = check_data(x, y)
    c = fft(y, m, p)
    a, b = get_coefficients(c, m)
    x_start, x_end = get_interval(x)

    # 2000 x-values close together, so the curve looks smooth.
    fine_x = [x_start + (x_end - x_start) * i / 1999 for i in range(2000)]
    curve = [evaluate(t, a, b, m, x_start, x_end) for t in fine_x]

    name = os.path.splitext(os.path.basename(filename))[0]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(fine_x, curve, label="Curve S(x) from the FFT")
    axis.plot(x, y, "k.", label="Data points")
    axis.set_title(f"{name}  ({2 * m} points)")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()

    figure.tight_layout()
    os.makedirs("plots", exist_ok=True)
    output_filename = os.path.join("plots", name + ".png")
    figure.savefig(output_filename)
    plt.close(figure)
    print("Saved", output_filename)


def main():
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        filenames = [f for f in sorted(glob.glob("data/*.csv"))
                     if not f.endswith("_coefficients.csv")]
    for filename in filenames:
        plot_file(filename)


if __name__ == "__main__":
    main()
