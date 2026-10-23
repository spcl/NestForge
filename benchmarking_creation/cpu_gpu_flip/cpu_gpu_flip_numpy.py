"""cpu_gpu_flip: a column kernel whose best schedule differs between CPU and GPU (numpy reference).

Three stages on fields of shape (nlev, ncol), ncol contiguous (row-major):

  A. producer   zqs = saturation mixing ratio, one exp and one division per cell.
  B. stencil    zsm = 3-point smoothing of zqs ACROSS columns (horizontal neighbours).
  C. solver     tridiagonal system ALONG levels per column, g = KAPPA * zsm, rhs = q - zsm:
                    -g[k] x[k-1] + (1 + g[k] + g[k+1]) x[k] - g[k+1] x[k+1] = q[k] - zsm[k]
                diagonally dominant for g >= 0, so the Thomas algorithm needs no pivoting.

The starting schedule is the one numpy gives: three separate passes with full-size transients,
levels outermost, columns as the inner (vector) dimension. The two device-dependent questions:

  1. Halo of the stencil: recompute the producer at the neighbour columns, or keep zqs in a
     buffer (tile-local on CPU, global on GPU).
  2. Mapping of the column loop: a block of columns as the inner vector loop with levels outside
     (CPU), or one thread per column with the level loop inside and the sweep state in registers
     (GPU).
"""

import numpy as np

#: Tetens saturation constants (as in IFS YOETHF) and the reference temperature.
R2ES = 380.1608703442847
R3LES = 17.502
R4LES = 32.19
RTT = 273.16
#: Coupling strength of the vertical solver.
KAPPA = 10.0


def cpu_gpu_flip(t, q, p, x, nlev, ncol):
    # array shapes (numpy->dace): t, q, p, x = (nlev, ncol)
    # zqs, zsm, cp, dp are transients: only x is an output, so a schedule may fuse, recompute or
    # shrink them freely.
    zqs = np.empty((nlev, ncol), dtype=t.dtype)
    zsm = np.empty((nlev, ncol), dtype=t.dtype)
    cp = np.empty((nlev, ncol), dtype=t.dtype)
    dp = np.empty((nlev, ncol), dtype=t.dtype)
    # A. producer
    zqs[:, :] = R2ES * np.exp(R3LES * (t - RTT) / (t - R4LES)) / p
    # B. horizontal 3-point stencil, edge columns copied
    zsm[:, 1:ncol - 1] = 0.25 * zqs[:, 0:ncol - 2] + 0.5 * zqs[:, 1:ncol - 1] + 0.25 * zqs[:, 2:ncol]
    zsm[:, 0] = zqs[:, 0]
    zsm[:, ncol - 1] = zqs[:, ncol - 1]
    # C. vertical Thomas solve, forward elimination
    cp[0, :] = -KAPPA * zsm[1, :] / (1.0 + KAPPA * zsm[1, :])
    dp[0, :] = (q[0, :] - zsm[0, :]) / (1.0 + KAPPA * zsm[1, :])
    for k in range(1, nlev - 1):
        m = 1.0 + KAPPA * zsm[k, :] + KAPPA * zsm[k + 1, :] + KAPPA * zsm[k, :] * cp[k - 1, :]
        cp[k, :] = -KAPPA * zsm[k + 1, :] / m
        dp[k, :] = ((q[k, :] - zsm[k, :]) + KAPPA * zsm[k, :] * dp[k - 1, :]) / m
    m = 1.0 + KAPPA * zsm[nlev - 1, :] + KAPPA * zsm[nlev - 1, :] * cp[nlev - 2, :]
    dp[nlev - 1, :] = ((q[nlev - 1, :] - zsm[nlev - 1, :]) + KAPPA * zsm[nlev - 1, :] * dp[nlev - 2, :]) / m
    # C. back substitution
    x[nlev - 1, :] = dp[nlev - 1, :]
    for k in range(nlev - 2, -1, -1):
        x[k, :] = dp[k, :] - cp[k, :] * x[k + 1, :]
