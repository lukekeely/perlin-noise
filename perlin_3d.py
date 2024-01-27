import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def lerp(a, b, x):
    return a + x * (b - a)

def gradient(h, x, y, z):
    gradients = np.array([[0, 1, 1], [0, -1, -1], [1, 0, 1], [-1, 0, -1], [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]])
    g = gradients[h % 8]
    return g[0] * x + g[1] * y + g[2] * z

def perlin_3d(x, y, z, seed=None):
    if seed:
        np.random.seed(seed)

    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()

    xi, yi, zi = int(x), int(y), int(z)
    xf, yf, zf = x - xi, y - yi, z - zi
    u, v, w = fade(xf), fade(yf), fade(zf)

    n000 = gradient(p[p[p[xi] + yi] + zi], xf, yf, zf)
    n001 = gradient(p[p[p[xi] + yi] + zi + 1], xf, yf, zf - 1)
    n011 = gradient(p[p[p[xi] + yi + 1] + zi + 1], xf, yf - 1, zf - 1)
    n010 = gradient(p[p[p[xi] + yi + 1] + zi], xf, yf - 1, zf)
    n100 = gradient(p[p[p[xi + 1] + yi] + zi], xf - 1, yf, zf)
    n101 = gradient(p[p[p[xi + 1] + yi] + zi + 1], xf - 1, yf, zf - 1)
    n111 = gradient(p[p[p[xi + 1] + yi + 1] + zi + 1], xf - 1, yf - 1, zf - 1)
    n110 = gradient(p[p[p[xi + 1] + yi + 1] + zi], xf - 1, yf - 1, zf)

    x1 = lerp(lerp(n000, n100, u), lerp(n010, n110, u), v)
    x2 = lerp(lerp(n001, n101, u), lerp(n011, n111, u), v)

    return lerp(x1, x2, w)

def generate_perlin_slice(x_dim, y_dim, z_val, scale, seed=None):
    slice = np.zeros((x_dim, y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            x_val, y_val = i / scale, j / scale
            slice[i, j] = perlin_3d(x_val, y_val, z_val, seed)
    return slice

def update(frame):
    global z_pos
    slice = generate_perlin_slice(scale, scale, z_pos, zoom_scale, seed)
    im.set_data(slice)
    z_pos += z_step  # Move to the next slice
    return [im]

seed = np.random.randint(0, 1000)
scale = 25  # Dimensions of each slice
zoom_scale = 10  # Determines the 'zoomed-out' effect
z_pos = 0.0  # Starting position in the z-axis
z_step = 0.05  # Step size along the z-axis

fig, ax = plt.subplots()
first_slice = generate_perlin_slice(scale, scale, z_pos, zoom_scale, seed)
im = ax.imshow(first_slice, origin='upper', cmap='magma', animated=True)
plt.colorbar(im, ax=ax)

ani = FuncAnimation(fig, update, interval=10, blit=True)
plt.show()
