"""sweep_scale: a carried column sweep followed by a pointwise scaling of its result (numpy reference).

Arrays are row-major with ``NI`` contiguous. The sweep is carried down each column (``j``) and independent across
columns (``i``), and both nests are written column by column, so the inner loop strides by a whole row. The two
nests share ``aa``: the second reads what the first wrote.
"""


def sweep_scale(aa, bb, cc, w, NJ, NI):
    # array shapes (numpy->dace): aa=(NJ,NI), bb=(NJ,NI), cc=(NJ,NI), w=(NI,)
    for i in range(NI):
        for j in range(1, NJ):
            aa[j, i] = aa[j - 1, i] + bb[j, i]
    for i in range(NI):
        for j in range(NJ):
            cc[j, i] = aa[j, i] * w[i]
