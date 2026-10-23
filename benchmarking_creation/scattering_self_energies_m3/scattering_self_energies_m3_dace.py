"""scattering_self_energies_m3: hand-written DaCe program (no hpcagent_bench-autogen marker, so it is a hand override).

Same sum as scattering_self_energies (HPCAgent-Bench e63f9f1), in the loop order k, E, q, w, i, a, b, j (the atom map a
sits under three reduction loops instead of four), with the 10 x 10 complex products written as explicit orbital loops
o1..o4. Emitting from the oracle would bring back MatMult library nodes in the loop body, which NestForge's stage 3 to 5
emitter refuses (UnsupportedNest) at 4df1830; the explicit loops build. W3 variant m3, 2026-10-21."""
import dace as dc

Nkz = dc.symbol('Nkz', dtype=dc.int64, positive=True)
NE = dc.symbol('NE', dtype=dc.int64, positive=True)
Nqz = dc.symbol('Nqz', dtype=dc.int64, positive=True)
Nw = dc.symbol('Nw', dtype=dc.int64, positive=True)
N3D = dc.symbol('N3D', dtype=dc.int64, positive=True)
NA = dc.symbol('NA', dtype=dc.int64, positive=True)
NB = dc.symbol('NB', dtype=dc.int64, positive=True)


@dc.program
def scattering_self_energies_m3(neigh_idx: dc.int32[NA, NB], dH: dc.complex128[NA, NB, N3D, 10, 10], G: dc.complex128[Nkz, NE, NA, 10, 10], D: dc.complex128[Nqz, Nw, NA, NB, N3D, N3D], Sigma: dc.complex128[Nkz, NE, NA, 10, 10]):
    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for a in range(NA):
                            for b in range(NB):
                                for j in range(N3D):
                                    if E - w >= 0:
                                        for o1 in range(10):
                                            for o2 in range(10):
                                                for o3 in range(10):
                                                    for o4 in range(10):
                                                        Sigma[k, E, a, o1, o2] = Sigma[k, E, a, o1, o2] + G[k, E - w, neigh_idx[a, b], o1, o3] * dH[a, b, i, o3, o4] * dH[a, b, j, o4, o2] * D[q, w, a, b, i, j]
