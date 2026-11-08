# Fast Fourier Transform for Trigonometric Interpolation

**MATH 470/570 — Numerical Analysis | Fall 2026**

## Overview

This project focuses on implementing the **Fast Fourier Transform (FFT)** algorithm to compute the coefficients of an interpolatory trigonometric polynomial for a given dataset.

The FFT is an efficient algorithm for computing the Discrete Fourier Transform (DFT). By using the FFT, we aim to construct trigonometric interpolating polynomials from numerical data while reducing computational complexity compared to directly evaluating the DFT.

This project is developed as part of the MATH 470/570 Numerical Analysis course.

## Project Objectives

The primary objectives of this project are to:

- Understand the mathematical foundations of trigonometric interpolation and the Discrete Fourier Transform.
- Implement the Fast Fourier Transform algorithm using Python or C#.
- Develop a program capable of reading numerical datasets from external files.
- Compute Fourier coefficients required to construct interpolatory trigonometric polynomials.
- Support datasets containing hundreds of data points.
- Test and verify the correctness and efficiency of the implementation.

## Mathematical Background

### Trigonometric Interpolation

Trigonometric interpolation involves constructing a trigonometric polynomial that passes through a given collection of data points.

A general trigonometric polynomial can be expressed as

$$
T_n(x)=a_0+\sum_{k=1}^{n}
\left[a_k\cos(kx)+b_k\sin(kx)\right],
$$

where $a_k$ and $b_k$ are coefficients determined from the dataset.

For equally spaced data points $(x_j,y_j)$, the goal is to construct a polynomial satisfying

$$
T_n(x_j)=y_j.
$$

### Discrete Fourier Transform (DFT)

The Discrete Fourier Transform converts a finite sequence of numerical values into its frequency-domain representation.

For a sequence $x_0,x_1,\ldots,x_{N-1}$, the DFT is defined by

$$
X_k=\sum_{j=0}^{N-1}x_j e^{-2\pi i jk/N},
\qquad k=0,1,\ldots,N-1.
$$

The resulting Fourier coefficients can be used to construct a trigonometric interpolating polynomial.

Direct computation of the DFT requires $O(N^2)$ operations.

### Fast Fourier Transform (FFT)

The Fast Fourier Transform is a family of algorithms designed to compute the DFT more efficiently.

A standard radix-2 FFT uses a divide-and-conquer approach, separating the input sequence into even-indexed and odd-indexed elements.

For datasets containing $N=2^m$ samples, this reduces computational complexity to

$$
\boxed{O(N\log N)}.
$$

This efficiency makes the FFT suitable for processing larger numerical datasets.

## Planned Implementation

The program is intended to follow this workflow:

1. Read a numerical dataset from an external file.
2. Validate and prepare the input data.
3. Apply the Fast Fourier Transform algorithm.
4. Compute the Fourier coefficients.
5. Use the coefficients to construct an interpolatory trigonometric polynomial.
6. Output the resulting coefficients for further analysis.

The initial implementation will follow the FFT algorithm presented in the course textbook.

## Technologies

- **Programming Language:** Python or C# (to be finalized)
- **Version Control:** Git and GitHub
- **Numerical Methods:** Discrete Fourier Transform, Fast Fourier Transform, and Trigonometric Interpolation

## Project Team

| Team Member | Primary Responsibilities |
|---|---|
| Ike Halderan | Algorithm implementation, debugging, and testing |
| Josiah Hoppe | Technical report, presentation planning, and group spokesperson |
| Jacari Jellum | Technical report and presentation planning |
| Sauharda Shrestha | Algorithm implementation, debugging, and testing |

All team members will collaborate on understanding the mathematical concepts, reviewing project progress, and completing the final deliverables.

## Project Timeline

| Milestone | Deadline |
|---|---|
| First Progress Report | November 10, 2026 |
| Second Progress Report | November 24, 2026 |
| Project Presentation | December 4, 2026 |
| Final Project Report | December 8, 2026 |

## Expected Outcomes

By the completion of this project, we aim to deliver:

- A working implementation of the FFT algorithm.
- Support for reading numerical datasets from files.
- Computation of Fourier coefficients for trigonometric interpolation.
- Verification of the algorithm using test datasets.
- A technical report explaining the mathematical foundations, implementation, and results.
- A presentation demonstrating the algorithm and its applications.

## References

1. Burden, R. L., Faires, J. D., and Burden, A. M. *Numerical Analysis*, 10th Edition.
2. MATH 470/570 Course Project 2 Guidelines, Fall 2026.

## Project Status

**In Development**

The project is currently in its initial development stage. The team is working on understanding the FFT algorithm, establishing the implementation, and preparing the required progress reports.

---

**Minnesota State University, Mankato**

MATH 470/570 — Numerical Analysis, Fall 2026
