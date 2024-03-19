import matplotlib.pyplot as plt
import numpy as np


def laplacian_5_operator(size_z, size_x, size_y, delta_z, delta_x, delta_y, p):
    pzz = np.zeros((size_z, size_x, size_y))
    pxx = np.zeros((size_z, size_x, size_y))
    pyy = np.zeros((size_z, size_x, size_y))

    for z in range(2, size_z - 2):
        pzz[z, :, :] = ((-1/12) * p[z + 2, :, :] + (4/3) * p[z + 1, :, :] - (5/2) * p[z, :, :] + (4/3) * p[z - 1, :, :]
                        - (1/12) * p[z - 2, :, :]) / (delta_z ** 2)

    for x in range(2, size_x - 2):
        pxx[:, x, :] = ((-1/12) * p[:, x + 2, :] + (4/3) * p[:, x + 1, :] - (5/2) * p[:, x, :] + (4/3) * p[:, x - 1, :]
                        - (1/12) * p[:, x - 2, :]) / (delta_x ** 2)

    for y in range(2, size_y - 2):
        pyy[:, :, y] = ((-1/12) * p[:, :, y + 2] + (4/3) * p[:, :, y + 1] - (5/2) * p[:, :, y] + (4/3) * p[:, :, y - 1]
                        - (1/12) * p[:, :, y - 2]) / (delta_y ** 2)

    return pzz + pxx + pyy


grid_size_z = 20
grid_size_x = 300
grid_size_y = 300

dz = 1
dx = 1
dy = 1

total_time = 500
dt = 0.001

source_z = int(grid_size_z / 2)
source_x = int(grid_size_x / 2)
source_y = int(grid_size_y / 2)

c0 = 450

f0 = 10
t0 = 2 / f0

time = np.linspace(0, total_time * dt, total_time)

source = -8. * (time - t0) * f0 * (np.exp(-1. * (time - t0) ** 2 * (f0 * 4) ** 2))

p_present = np.zeros((grid_size_z, grid_size_x, grid_size_y))
p_past = np.zeros((grid_size_z, grid_size_x, grid_size_y))
p_future = np.zeros((grid_size_z, grid_size_x, grid_size_y))

c = np.zeros((grid_size_z, grid_size_x, grid_size_y))
c += c0

for t in range(total_time):
    p_future = (c ** 2) * laplacian_5_operator(grid_size_z, grid_size_x, grid_size_y, dz, dx, dy, p_present) * (dt ** 2)
    p_future += 2 * p_present - p_past

    p_past = p_present
    p_present = p_future

    p_future[source_z, source_x, source_y] += source[t]

X = np.arange(0, grid_size_x, 1)
Y = np.arange(0, grid_size_y, 1)
X, Y = np.meshgrid(X, Y)
Z = p_future[grid_size_z // 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dem3d = ax.plot_surface(X, Y, Z, cmap='afmhot', linewidth=0)
ax.set_title(f'Future Pressure Field at Z = {grid_size_z // 2}')
plt.show()
