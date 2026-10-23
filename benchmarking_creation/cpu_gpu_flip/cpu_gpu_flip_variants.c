/* Three schedules of cpu_gpu_flip (see cpu_gpu_flip_numpy.py), written as plain C so each can be
 * checked against the numpy oracle on any machine. Row-major (nlev, ncol), column index contiguous.
 *
 *   v0_naive   what numpy gives: three passes, full-size transients, levels outer, columns inner.
 *   v1_cpu     column blocks. zqs of one block (plus a one-column halo each side) lives in a
 *              block-local buffer, computed once, so the exp is never recomputed (halo overhead is
 *              2/BC). Levels outer, the block's columns inner and vectorizable. The solver state
 *              (cp, dp) is block-local too.
 *   v2_gpu     one "thread" per column (the outer loop), level loop sequential inside, sweep state
 *              in scalars. The stencil is fused in and the producer is RECOMPUTED at the three
 *              neighbour columns instead of being stored: 3 exps per cell. cp and dp are kept in
 *              per-column arrays because the back substitution needs them.
 *
 * On a CPU v2 is the "wrong" schedule: the level loop is the inner one, so each column walks the
 * arrays with stride ncol, and it pays three exps per cell. Its structure is what a GPU wants.
 * No CUDA here: this file only fixes the schedules and their results.
 */
#include <math.h>
#include <stdlib.h>

#define R2ES 380.1608703442847
#define R3LES 17.502
#define R4LES 32.19
#define RTT 273.16
#define KAPPA 10.0
#define BC 256

static inline double qs(double t, double p) { return R2ES * exp(R3LES * (t - RTT) / (t - R4LES)) / p; }

void v0_naive(const double *t, const double *q, const double *p, double *x, long nlev, long ncol) {
  double *zqs = malloc(sizeof(double) * nlev * ncol), *zsm = malloc(sizeof(double) * nlev * ncol);
  double *cp = malloc(sizeof(double) * nlev * ncol), *dp = malloc(sizeof(double) * nlev * ncol);
  for (long k = 0; k < nlev; k++)
    for (long c = 0; c < ncol; c++) zqs[k * ncol + c] = qs(t[k * ncol + c], p[k * ncol + c]);
  for (long k = 0; k < nlev; k++) {
    for (long c = 1; c < ncol - 1; c++)
      zsm[k * ncol + c] = 0.25 * zqs[k * ncol + c - 1] + 0.5 * zqs[k * ncol + c] + 0.25 * zqs[k * ncol + c + 1];
    zsm[k * ncol] = zqs[k * ncol];
    zsm[k * ncol + ncol - 1] = zqs[k * ncol + ncol - 1];
  }
  for (long c = 0; c < ncol; c++) {
    cp[c] = -KAPPA * zsm[ncol + c] / (1.0 + KAPPA * zsm[ncol + c]);
    dp[c] = (q[c] - zsm[c]) / (1.0 + KAPPA * zsm[ncol + c]);
  }
  for (long k = 1; k < nlev - 1; k++)
    for (long c = 0; c < ncol; c++) {
      double g = KAPPA * zsm[k * ncol + c], gn = KAPPA * zsm[(k + 1) * ncol + c];
      double m = 1.0 + g + gn + g * cp[(k - 1) * ncol + c];
      cp[k * ncol + c] = -gn / m;
      dp[k * ncol + c] = ((q[k * ncol + c] - zsm[k * ncol + c]) + g * dp[(k - 1) * ncol + c]) / m;
    }
  for (long c = 0; c < ncol; c++) {
    long k = nlev - 1;
    double g = KAPPA * zsm[k * ncol + c];
    double m = 1.0 + g + g * cp[(k - 1) * ncol + c];
    dp[k * ncol + c] = ((q[k * ncol + c] - zsm[k * ncol + c]) + g * dp[(k - 1) * ncol + c]) / m;
    x[k * ncol + c] = dp[k * ncol + c];
  }
  for (long k = nlev - 2; k >= 0; k--)
    for (long c = 0; c < ncol; c++) x[k * ncol + c] = dp[k * ncol + c] - cp[k * ncol + c] * x[(k + 1) * ncol + c];
  free(zqs); free(zsm); free(cp); free(dp);
}

void v1_cpu(const double *t, const double *q, const double *p, double *x, long nlev, long ncol) {
  double *zq = malloc(sizeof(double) * nlev * (BC + 2));
  double *zs = malloc(sizeof(double) * nlev * BC);
  double *cp = malloc(sizeof(double) * nlev * BC), *dp = malloc(sizeof(double) * nlev * BC);
  for (long c0 = 0; c0 < ncol; c0 += BC) {
    long n = (ncol - c0 < BC) ? ncol - c0 : BC;
    /* zq column j holds global column c0 - 1 + j, clamped; computed once. */
    for (long k = 0; k < nlev; k++)
      for (long j = 0; j < n + 2; j++) {
        long c = c0 - 1 + j;
        if (c < 0) c = 0;
        if (c > ncol - 1) c = ncol - 1;
        zq[k * (BC + 2) + j] = qs(t[k * ncol + c], p[k * ncol + c]);
      }
    for (long k = 0; k < nlev; k++)
      for (long j = 0; j < n; j++) {
        long c = c0 + j;
        zs[k * BC + j] = (c == 0 || c == ncol - 1)
                             ? zq[k * (BC + 2) + j + 1]
                             : 0.25 * zq[k * (BC + 2) + j] + 0.5 * zq[k * (BC + 2) + j + 1] + 0.25 * zq[k * (BC + 2) + j + 2];
      }
    for (long j = 0; j < n; j++) {
      cp[j] = -KAPPA * zs[BC + j] / (1.0 + KAPPA * zs[BC + j]);
      dp[j] = (q[c0 + j] - zs[j]) / (1.0 + KAPPA * zs[BC + j]);
    }
    for (long k = 1; k < nlev - 1; k++)
      for (long j = 0; j < n; j++) {
        double g = KAPPA * zs[k * BC + j], gn = KAPPA * zs[(k + 1) * BC + j];
        double m = 1.0 + g + gn + g * cp[(k - 1) * BC + j];
        cp[k * BC + j] = -gn / m;
        dp[k * BC + j] = ((q[k * ncol + c0 + j] - zs[k * BC + j]) + g * dp[(k - 1) * BC + j]) / m;
      }
    for (long j = 0; j < n; j++) {
      long k = nlev - 1;
      double g = KAPPA * zs[k * BC + j];
      double m = 1.0 + g + g * cp[(k - 1) * BC + j];
      dp[k * BC + j] = ((q[k * ncol + c0 + j] - zs[k * BC + j]) + g * dp[(k - 1) * BC + j]) / m;
      x[k * ncol + c0 + j] = dp[k * BC + j];
    }
    for (long k = nlev - 2; k >= 0; k--)
      for (long j = 0; j < n; j++) x[k * ncol + c0 + j] = dp[k * BC + j] - cp[k * BC + j] * x[(k + 1) * ncol + c0 + j];
  }
  free(zq); free(zs); free(cp); free(dp);
}

/* smoothed producer at (k, c): three recomputed exps, edge columns use their own value */
static inline double zsm_at(const double *t, const double *p, long k, long c, long ncol) {
  long i = k * ncol + c;
  if (c == 0 || c == ncol - 1) return qs(t[i], p[i]);
  return 0.25 * qs(t[i - 1], p[i - 1]) + 0.5 * qs(t[i], p[i]) + 0.25 * qs(t[i + 1], p[i + 1]);
}

void v2_gpu(const double *t, const double *q, const double *p, double *x, long nlev, long ncol) {
  double *cp = malloc(sizeof(double) * nlev * ncol), *dp = malloc(sizeof(double) * nlev * ncol);
  for (long c = 0; c < ncol; c++) { /* one thread per column */
    double z0 = zsm_at(t, p, 0, c, ncol);
    double z1 = zsm_at(t, p, 1, c, ncol);
    cp[c] = -KAPPA * z1 / (1.0 + KAPPA * z1);
    dp[c] = (q[c] - z0) / (1.0 + KAPPA * z1);
    double zk = z1; /* zsm at level k, rolled forward */
    for (long k = 1; k < nlev - 1; k++) {
      double zn = zsm_at(t, p, k + 1, c, ncol);
      double g = KAPPA * zk, gn = KAPPA * zn;
      double m = 1.0 + g + gn + g * cp[(k - 1) * ncol + c];
      cp[k * ncol + c] = -gn / m;
      dp[k * ncol + c] = ((q[k * ncol + c] - zk) + g * dp[(k - 1) * ncol + c]) / m;
      zk = zn;
    }
    long k = nlev - 1;
    double g = KAPPA * zk;
    double m = 1.0 + g + g * cp[(k - 1) * ncol + c];
    dp[k * ncol + c] = ((q[k * ncol + c] - zk) + g * dp[(k - 1) * ncol + c]) / m;
    x[k * ncol + c] = dp[k * ncol + c];
    for (k = nlev - 2; k >= 0; k--) x[k * ncol + c] = dp[k * ncol + c] - cp[k * ncol + c] * x[(k + 1) * ncol + c];
  }
  free(cp); free(dp);
}
