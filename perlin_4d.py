import numpy as np
import matplotlib.pyplot as plt
import logging
from PIL import Image
import os
import shutil

def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def lerp(a, b, x):
    return a + x * (b - a)

def gradient(h, x, y, z, t):
    gradients = np.array([[0, 1, 1, 1], [0, -1, -1, -1], [1, 0, 1, -1], [-1, 0, -1, 1],
                          [1, 1, 0, -1], [-1, -1, 0, 1], [1, -1, 0, -1], [-1, 1, 0, 1],
                          [0, -1, 1, -1], [0, 1, -1, 1], [1, 0, -1, 1], [-1, 0, 1, -1]])
    g = gradients[h % 12]
    return g[0] * x + g[1] * y + g[2] * z + g[3] * t

def perlin_4d(x, y, z, t, seed=None):
    if seed:
        np.random.seed(seed)

    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    xi, yi, zi, ti = int(x), int(y), int(z), int(t)
    xf, yf, zf, tf = x - xi, y - yi, z - zi, t - ti
    u, v, w, s = fade(xf), fade(yf), fade(zf), fade(tf)

    # 16 corners of the 4D hypercube
    n0000 = gradient(p[p[p[p[xi] + yi] + zi] + ti], xf, yf, zf, tf)
    n0001 = gradient(p[p[p[p[xi] + yi] + zi] + ti + 1], xf, yf, zf, tf - 1)
    n0010 = gradient(p[p[p[p[xi] + yi] + zi + 1] + ti], xf, yf, zf - 1, tf)
    n0011 = gradient(p[p[p[p[xi] + yi] + zi + 1] + ti + 1], xf, yf, zf - 1, tf - 1)
    n0100 = gradient(p[p[p[p[xi] + yi + 1] + zi] + ti], xf, yf - 1, zf, tf)
    n0101 = gradient(p[p[p[p[xi] + yi + 1] + zi] + ti + 1], xf, yf - 1, zf, tf - 1)
    n0110 = gradient(p[p[p[p[xi] + yi + 1] + zi + 1] + ti], xf, yf - 1, zf - 1, tf)
    n0111 = gradient(p[p[p[p[xi] + yi + 1] + zi + 1] + ti + 1], xf, yf - 1, zf - 1, tf - 1)
    n1000 = gradient(p[p[p[p[xi + 1] + yi] + zi] + ti], xf - 1, yf, zf, tf)
    n1001 = gradient(p[p[p[p[xi + 1] + yi] + zi] + ti + 1], xf - 1, yf, zf, tf - 1)
    n1010 = gradient(p[p[p[p[xi + 1] + yi] + zi + 1] + ti], xf - 1, yf, zf - 1, tf)
    n1011 = gradient(p[p[p[p[xi + 1] + yi] + zi + 1] + ti + 1], xf - 1, yf, zf - 1, tf - 1)
    n1100 = gradient(p[p[p[p[xi + 1] + yi + 1] + zi] + ti], xf - 1, yf - 1, zf, tf)
    n1101 = gradient(p[p[p[p[xi + 1] + yi + 1] + zi] + ti + 1], xf - 1, yf - 1, zf, tf - 1)
    n1110 = gradient(p[p[p[p[xi + 1] + yi + 1] + zi + 1] + ti], xf - 1, yf - 1, zf - 1, tf)
    n1111 = gradient(p[p[p[p[xi + 1] + yi + 1] + zi + 1] + ti + 1], xf - 1, yf - 1, zf - 1, tf - 1)

    # Interpolate along t
    x1 = lerp(n0000, n0001, s)
    x2 = lerp(n0010, n0011, s)
    x3 = lerp(n0100, n0101, s)
    x4 = lerp(n0110, n0111, s)
    x5 = lerp(n1000, n1001, s)
    x6 = lerp(n1010, n1011, s)
    x7 = lerp(n1100, n1101, s)
    x8 = lerp(n1110, n1111, s)

    # Interpolate along z
    y1 = lerp(x1, x2, w)
    y2 = lerp(x3, x4, w)
    y3 = lerp(x5, x6, w)
    y4 = lerp(x7, x8, w)

    # Interpolate along y
    z1 = lerp(y1, y2, v)
    z2 = lerp(y3, y4, v)

    # Interpolate along x
    return lerp(z1, z2, u)

def generate_perlin_slice_4d(x_dim, y_dim, z_val, t_val, scale, seed=None):
    slice = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            x_val, y_val = i / scale, j / scale
            slice[i, j] = perlin_4d(x_val, y_val, z_val, t_val, seed)
    return slice

def save_frame(frame_number, t_pos):
    fig = plt.figure(figsize=(6, 6), dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, zoom_scale * slice_spacing)

    for z in range(num_slices):
        z_val = z / zoom_scale
        slice = generate_perlin_slice_4d(slice_dim, slice_dim, z_val, t_pos, zoom_scale, seed)
        x, y = np.meshgrid(np.linspace(0, 1, slice_dim), np.linspace(0, 1, slice_dim))
        ax.plot_surface(x, y, slice + z * slice_spacing, cmap='magma')

    frame_file = os.path.join('chunks', f'frame_{frame_number}.png')
    plt.savefig(frame_file)
    plt.close(fig)
    return frame_file

seed = np.random.randint(0, 1000)
slice_dim = 25
zoom_scale = 10
t_pos = 0.0
t_step = 0.01
num_slices = 10
slice_spacing = 1

logging.basicConfig(level=logging.INFO)

# Create 'chunks' folder if it doesn't exist
if not os.path.exists('chunks'):
    os.makedirs('chunks')

# Save frames individually in 'chunks' folder
frame_count = 200
frame_files = []
for frame_number in range(frame_count):
    t_pos = frame_number * t_step
    frame_file = save_frame(frame_number, t_pos)
    frame_files.append(frame_file)
    logging.info(f"Frame {frame_number} saved")

# Create and save GIF from individual frames
with Image.open(frame_files[0]) as first_frame:
    first_frame.save('perlin_noise_4d.gif', save_all=True, append_images=[Image.open(f) for f in frame_files[1:]], optimize=False, duration=150, loop=0)

logging.info("GIF saved successfully")

# Clean up: Delete the 'chunks' folder
shutil.rmtree('chunks')