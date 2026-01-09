import numpy as np
import struct
import sys

N = 256
N_PEAKS = 10
dim = np.linspace(0, 1, N)
X, Y = np.meshgrid(dim, dim)

peak_means_x = np.random.uniform(0.3, 0.7, N_PEAKS)
peak_means_y = np.random.uniform(0.3, 0.7, N_PEAKS)
peak_std = np.random.uniform(0.01, 0.001, N_PEAKS)

output = np.zeros_like(X)
for px, py, std in zip(peak_means_x, peak_means_y, peak_std):
    output += np.exp(-((X - px) ** 2 + (Y - py) ** 2) / std)

# store
header = struct.pack(f"<{'Q' * (1 + len(output.shape))}", 2, N, N)
with open(sys.argv[1], "wb") as outf:
    outf.write(header)
    output.astype(np.float64).tofile(outf)
