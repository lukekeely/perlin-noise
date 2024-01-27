# Perlin Noise Generation

## Overview

This project focuses on the implementation and visualization of multi-dimensional Perlin noise, a technique commonly used in procedural texture generation and modeling of complex, natural phenomena. Initially conceptualized in 2D and 3D spaces, this project extends Perlin noise generation to 4D and explores its generalization to N dimensions.

<img src="perlin_noise_4d.gif" alt="Perlin Noise Animation" width="60%"/>

## Dependencies

To run this project, the following Python libraries are required:

- **Python**: Version 3.7 or newer.
- **NumPy**: For array and numerical operations. Install with `pip install numpy`.
- **Matplotlib**: For plotting and visualizing the Perlin noise. Install with `pip install matplotlib`.
- **Pillow**: For image processing and GIF creation. Install with `pip install pillow`.

Run `pip install numpy matplotlib pillow` to install all dependencies.

## Multi-Dimensional Perlin Noise: The Mathematics

Perlin noise, developed by Ken Perlin, is a gradient noise function that produces a more natural appearance compared to purely random noise. The mathematical foundation of Perlin noise can be generalized to N dimensions.

### Basic Concept

In its simplest form, Perlin noise is generated using a lattice of points in N-dimensional space. Each point on this lattice is assigned a random gradient vector. The noise value at any given point in the space is calculated by interpolating these gradient vectors.

### Mathematical Representation

For a point \( P(x_1, x_2, ..., x_n) \) in N-dimensional space, the noise value is determined as follows:

1. **Identify Surrounding Points**: Find the lattice points surrounding \( P \), denoted as \( L_i \), where \( i \) ranges over the \( 2^n \) corners of the hypercube surrounding \( P \).

2. **Dot Product**: Compute the dot product of the gradient vector at each \( L_i \) with the vector from \( L_i \) to \( P \). Let this be \( D_i(P) \).

3. **Interpolation**: Apply a smoothing (fade) function \( f(t) \) to each coordinate of \( P \), and interpolate these dot products \( D_i \) using \( f(t) \). The fade function is typically \( f(t) = 6t^5 - 15t^4 + 10t^3 \).

4. **Noise Value**: The resulting value from this interpolation is the Perlin noise value at \( P \).

### Generalization to N Dimensions

The process described above can be generalized to an N-dimensional space by considering an N-dimensional hypercube surrounding the point \( P \). The computation involves \( 2^n \) dot products and a series of N-dimensional linear interpolations.

## License

This project is licensed under the terms of the GNU General Public License, version 3 (GPL-3.0). You may obtain a copy of the license [here](https://www.gnu.org/licenses/gpl-3.0.en.html).
