"""
Animates the Fast Fourier Transform on book Example 1, step by step.

It runs the same steps as fft() in fft_interpolation.py on the 4 data points
and shows:
    1. the starting numbers (c = y),
    2. each pairing round as a "butterfly" picture,
    3. the reordering (Steps 13-16),
    4. the a's and b's (Steps 17-18),
    5. the curve through the data points.

How to run it (inside the virtual environment):
    manim -pql animate_fft.py FFTExample1     quick, low quality, for checking
    manim -pqh animate_fft.py FFTExample1     slower, high quality, for the presentation

FFTExample2 does the same for book Example 2 (8 points).
The videos are saved in the media/ folder.
"""

import cmath
import math

from manim import *

from fft_interpolation import (
    check_data,
    evaluate,
    fft,
    get_coefficients,
    get_interval,
    read_data,
    reverse_bits,
)

SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(number):
    """Turn 12 into ₁₂, for labels like c₁₂."""
    return str(number).translate(SUBSCRIPTS)


def short(value):
    """Write a real number with 2 decimals, dropping ".00": 1.50, -9, 0."""
    if abs(value) < 0.005:
        return "0"
    text = f"{value:.2f}"
    if text.endswith(".00"):
        text = text[:-3]
    return text.replace("-", "−")


def show(value):
    """Write a possibly complex number, like 10.74, i, or −2+2i."""
    real = short(value.real)
    imag = short(value.imag)
    if imag == "0":
        return real
    if imag in ("1", "−1"):
        imag = imag[:-1]          # write i instead of 1i
    if real == "0":
        return imag + "i"
    sign = "" if imag.startswith("−") else "+"
    return real + sign + imag + "i"


def fft_with_history(y, m, p):
    """The same steps as fft() in fft_interpolation.py, but it also writes
    down every pairing so the animation can show it.

    Returns (c, rounds, swaps):
        rounds: one list per round (Step 5), with one entry per pairing:
                (K, K + M, K2, xi[K2], old c[K], old c[K+M], eta, new c[K], new c[K+M])
        swaps:  the pairs of positions swapped in Steps 13-16.
    """
    # Steps 1-4
    M = m
    q = p
    zeta = cmath.exp(cmath.pi * 1j / m)
    c = [complex(value) for value in y]
    xi = [0] * (2 * m + 1)
    for j in range(1, M + 1):
        xi[j] = zeta ** j
        xi[j + M] = -xi[j]
    K = 0
    xi[0] = 1

    # Steps 5-12, writing down each pairing
    rounds = []
    for L in range(1, p + 2):
        pairings = []
        while K < 2 * m - 1:
            for j in range(1, M + 1):
                K1 = K // 2 ** q
                K2 = reverse_bits(K1, p + 1)
                top, bottom = c[K], c[K + M]
                eta = c[K + M] * xi[K2]
                c[K + M] = c[K] - eta
                c[K] = c[K] + eta
                pairings.append((K, K + M, K2, xi[K2], top, bottom, eta, c[K], c[K + M]))
                K = K + 1
            K = K + M
        rounds.append((M, pairings))
        K = 0
        M = M // 2
        q = q - 1

    # Steps 13-16, writing down each swap
    swaps = []
    while K < 2 * m - 1:
        j = reverse_bits(K, p + 1)
        if j > K:
            c[j], c[K] = c[K], c[j]
            swaps.append((K, j))
        K = K + 1
    return c, rounds, swaps


class FFTExample1(Scene):
    data_file = "data/example1.csv"
    title = "The FFT on book Example 1:  y = 2x² − 9"
    # The book's answers, shown next to ours at the end.
    book_values = {"a₀": "−3.19559339", "a₁": "−9.86960441",
                   "b₁": "0", "a₂": "4.93480220"}
    pause = 1.5      # seconds to wait so people can read each step

    def construct(self):
        x, y = read_data(self.data_file)
        m, p = check_data(x, y)
        n = 2 * m
        c, rounds, swaps = fft_with_history(y, m, p)

        # Safety check: the recorded steps must give our real fft's answer.
        real_answer = fft(y, m, p)
        assert all(abs(c[k] - real_answer[k]) < 1e-9 for k in range(n))

        self.caption = None
        self.heading = Text(self.title, font_size=32).to_edge(UP)
        self.add(self.heading)

        # Layout: one column per stage, one row per position.
        columns = len(rounds) + 2          # start, each round, reordered
        column_x = [-4.2 + i * 9.4 / (columns - 1) for i in range(columns)]
        row_y = [1.6 - i * 3.6 / (n - 1) for i in range(n)]
        box_height = min(0.6, 0.75 * 3.6 / (n - 1))
        font = 26 if n <= 4 else 18
        bits = p + 1

        def box(value, column, row, color=BLUE):
            rectangle = RoundedRectangle(corner_radius=0.08, width=1.7,
                                         height=box_height, color=color)
            label = Text(show(value), font_size=font)
            return VGroup(rectangle, label).move_to([column_x[column], row_y[row], 0])

        def header(text, column):
            return Text(text, font_size=22).move_to([column_x[column], 2.6, 0])

        # 1. Start: c = y
        self.say(f"{n} data points. Only the y-values go into the FFT.")
        labels = VGroup(*[
            Text(f"{k}  ({k:0{bits}b})", font_size=font, color=GREY)
            .move_to([column_x[0] - 1.9, row_y[k], 0])
            for k in range(n)
        ])
        boxes = [box(y[k], 0, k) for k in range(n)]
        self.play(Write(header("Start\nc = y", 0)), FadeIn(labels),
                  *[FadeIn(b) for b in boxes])
        self.say("Step 2: start with c equal to the y-values")
        self.wait(self.pause)

        # 2. The pairing rounds
        for number, (M, pairings) in enumerate(rounds, start=1):
            self.play(Write(header(f"Round {number}\n({M} apart)", number)))
            self.say(f"Round {number}: each number is paired with the one {M} place(s) below it")
            self.wait(self.pause / 2)
            new_boxes = [None] * n
            lines = VGroup()
            for K, KM, K2, xi, top, bottom, eta, new_top, new_bottom in pairings:
                old_top, old_bottom = boxes[K], boxes[KM]
                new_boxes[K] = box(new_top, number, K)
                new_boxes[KM] = box(new_bottom, number, KM)
                butterfly = VGroup(*[
                    Line(start.get_right(), end.get_left(), stroke_width=2, color=YELLOW)
                    for start in (old_top, old_bottom)
                    for end in (new_boxes[K], new_boxes[KM])
                ])
                lines.add(butterfly)
                self.play(old_top[0].animate.set_color(YELLOW),
                          old_bottom[0].animate.set_color(YELLOW), Create(butterfly))
                self.say(f"η = c{sub(KM)} × ξ{sub(K2)} = {show(bottom)} × {show(xi)} = {show(eta)}")
                self.wait(self.pause)
                self.say(f"top: {show(top)} + η = {show(new_top)}"
                         f"        bottom: {show(top)} − η = {show(new_bottom)}")
                self.play(TransformFromCopy(old_top, new_boxes[K]),
                          TransformFromCopy(old_bottom, new_boxes[KM]))
                self.wait(self.pause)
                self.play(old_top[0].animate.set_color(BLUE),
                          old_bottom[0].animate.set_color(BLUE),
                          butterfly.animate.set_color(GREY).set_stroke(width=1))
            boxes = new_boxes

        # 3. Reordering (Steps 13-16)
        last = columns - 1
        self.play(Write(header("Reordered", last)))
        swap_text = ",  ".join(f"{a} ({a:0{bits}b}) ↔ {b} ({b:0{bits}b})" for a, b in swaps)
        self.say("Steps 13–16: read each position's binary label backwards.  Swap "
                 + swap_text)
        self.wait(self.pause)
        final_boxes = [None] * n
        moves = []
        for row in range(n):
            target = reverse_bits(row, bits)
            final_boxes[target] = boxes[row].copy()
            arc = 0 if target == row else -PI / 3
            moves.append(final_boxes[target].animate(path_arc=arc)
                         .move_to([column_x[last], row_y[target], 0]))
        self.play(*moves, run_time=2)
        names = VGroup(*[
            Text(f"c{sub(k)}", font_size=font, color=YELLOW).next_to(final_boxes[k], RIGHT)
            for k in range(n)
        ])
        self.play(FadeIn(names))
        self.say("These are the c's from the book's algorithm, in the right order")
        self.wait(self.pause * 2)

        # 4. The a's and b's (Steps 17-18)
        a, b = get_coefficients(c, m)
        self.clear_diagram()
        self.say(f"Steps 17–18: divide by m = {m}, and flip the sign when k is odd")
        rows = [f"a₀ = c₀ / {m} = {self.six(a[0])}"]
        for k in range(1, m):
            sign = "−" if k % 2 else ""
            rows.append(f"a{sub(k)} + i·b{sub(k)} = {sign}c{sub(k)} / {m}"
                        f"   →   a{sub(k)} = {self.six(a[k])},  b{sub(k)} = {self.six(b[k])}")
        sign = "−" if m % 2 else ""
        rows.append(f"a{sub(m)} = {sign}c{sub(m)} / {m} = {self.six(a[m])}")
        table = VGroup(*[Text(row, font_size=26) for row in rows])
        table.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.6)
        if table.width > 13:
            table.scale_to_fit_width(13)
        for row in table:
            self.play(Write(row), run_time=1)
        if self.book_values:
            book = Text("Book:  " + ",  ".join(f"{name} = {value}" for name, value
                                               in self.book_values.items()),
                        font_size=24, color=GREEN).next_to(table, DOWN, buff=0.5)
            self.play(FadeIn(book))
        self.wait(self.pause * 4)

        # 5. The curve through the data points
        self.clear_diagram()
        x_start, x_end = get_interval(x)

        def S(t):
            return evaluate(t, a, b, m, x_start, x_end)

        samples = [S(x_start + (x_end - x_start) * i / 200) for i in range(201)]
        low = min(samples + y)
        high = max(samples + y)
        padding = 0.1 * (high - low)
        axes = Axes(x_range=[x_start, x_end, (x_end - x_start) / 4],
                    y_range=[low - padding, high + padding, (high - low) / 4],
                    x_length=10, y_length=4.5, tips=False).move_to(DOWN * 0.2)
        ends = VGroup(
            Text(self.place(x_start), font_size=24).next_to(axes.x_axis.get_start(), DOWN),
            Text(self.place(x_end), font_size=24).next_to(axes.x_axis.get_end(), DOWN),
        )
        dots = VGroup(*[Dot(axes.c2p(x[j], y[j]), color=YELLOW, radius=0.08)
                        for j in range(n)])
        self.play(Create(axes), FadeIn(ends), FadeIn(dots))
        self.say("Put the a's and b's into the curve S(x)…")
        curve = axes.plot(S, x_range=[x_start, x_end], color=BLUE)
        self.play(Create(curve), run_time=3)
        self.say(f"…and it passes through all {n} data points")
        self.wait(self.pause * 2)

    def clear_diagram(self):
        """Fade out everything except the title and the caption."""
        keep = (self.heading, self.caption)
        self.play(*[FadeOut(thing) for thing in self.mobjects if thing not in keep])

    def say(self, text):
        """Show a caption at the bottom, replacing the previous one."""
        new = Text(text, font_size=24).move_to(DOWN * 3.3)
        if new.width > 13:
            new.scale_to_fit_width(13)
        if self.caption is None:
            self.play(FadeIn(new), run_time=0.5)
        else:
            self.play(FadeOut(self.caption), FadeIn(new), run_time=0.5)
        self.caption = new

    @staticmethod
    def six(value):
        """Write a number with 6 decimals, without showing −0.000000."""
        if abs(value) < 5e-7:
            value = 0.0
        return f"{value:.6f}".replace("-", "−")

    @staticmethod
    def place(value):
        """Label an end of the x-axis, writing ±π when it is ±π."""
        if abs(value - math.pi) < 1e-9:
            return "π"
        if abs(value + math.pi) < 1e-9:
            return "−π"
        return f"{value:g}".replace("-", "−")


class FFTExample2(FFTExample1):
    data_file = "data/example2.csv"
    title = "The FFT on book Example 2 (8 points)"
    book_values = None
    pause = 0.6
