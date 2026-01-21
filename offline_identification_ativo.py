# -*- coding: utf-8 -*-
"""
offline_identification_Zonly.py

Offline identification of Z matrix (8x8) using commissioning data (16 tests).

Now includes:
- Rat_meas_commissioning (Table 4) 16x4 (RatA..RatD), with np.inf for open-loop.

Saves:
- Z
- Rat_pred_ref (FFFF)
- Rat_meas_ref (FFFF)
- Rat_meas_commissioning (16x4)
- alpha_ref_cp (FFFF)
- cp_label_normal = "FFFF"
- electrodes

Author: Alexandre Giacomelli Leal
"""

import numpy as np


# =====================================================
# 1) DADOS DE CAMPO (COMISSIONAMENTO – 16 ENSAIOS)
# =====================================================

electrodes = ["FA", "FB", "FC", "FD", "CA", "CB", "CC", "CD"]

cp_labels = [
    "FFFF","AFFF","FAFF","FFAF",
    "FFFA","AAFF","FAAF","FFAA",
    "AFAF","FAFA","AFFA","AAAF",
    "AFAA","AAFA","FAAA","AAAA"
]

# Corrente total por perna da torre (A)
Itot = np.array([
    0.631, 0.580, 0.715, 0.805,
    0.877, 0.860, 0.767, 0.744,
    0.801, 0.764, 0.748, 0.773,
    0.717, 0.696, 0.699, 0.638
], dtype=float)

# Resistência global medida pelo método passivo (ohm)
R_ref = np.array([
    1.150, 1.195, 1.166, 1.190,
    1.303, 1.267, 1.253, 1.385,
    1.278, 1.333, 1.371, 1.342,
    1.497, 1.454, 1.458, 1.596
], dtype=float)

# Potencial da torre em relação ao terra remoto
V_tower = Itot * R_ref

# Correntes nos 4 cabos contrapeso (A)
I_CP = np.array([
    [0.0946, 0.0505, 0.0686, 0.1400],
    [0.0000, 0.0610, 0.0750, 0.1453],
    [0.1020, 0.0000, 0.0770, 0.1510],
    [0.1205, 0.0653, 0.0000, 0.1943],
    [0.1520, 0.0788, 0.1310, 0.0000],
    [0.0000, 0.0000, 0.1156, 0.2365],
    [0.1380, 0.0000, 0.0000, 0.2203],
    [0.1420, 0.0890, 0.0000, 0.0000],
    [0.0000, 0.0800, 0.0000, 0.2230],
    [0.1401, 0.0000, 0.1190, 0.0000],
    [0.0000, 0.0803, 0.1190, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.2204],
    [0.0000, 0.0872, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.1227, 0.0000],
    [0.1423, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000]
], dtype=float)

# Correntes reais nas 4 fundações (A) – usadas OFFLINE
I_F = np.array([
    [0.0714, 0.0615, 0.0584, 0.0760],
    [0.1070, 0.0720, 0.0630, 0.1067],
    [0.0800, 0.0780, 0.0630, 0.0980],
    [0.0835, 0.0757, 0.0850, 0.1357],
    [0.1050, 0.0832, 0.0880, 0.2510],
    [0.1470, 0.1150, 0.0864, 0.1945],
    [0.0980, 0.1010, 0.0990, 0.1727],
    [0.1040, 0.0820, 0.1120, 0.2510],
    [0.1310, 0.0910, 0.0980, 0.1780],
    [0.1019, 0.1010, 0.0830, 0.2210],
    [0.1370, 0.0897, 0.0850, 0.2350],
    [0.1410, 0.1140, 0.1050, 0.1886],
    [0.1440, 0.0998, 0.1160, 0.2690],
    [0.1460, 0.1130, 0.0923, 0.2430],
    [0.1047, 0.1060, 0.1130, 0.2330],
    [0.1480, 0.1180, 0.1160, 0.2530]
], dtype=float)

I_full = np.hstack([I_F, I_CP])  # 16x8


# =====================================================
# 1.1) Rat_meas (Tabela 4 do artigo) 16x4
# =====================================================
# RatA, RatB, RatC, RatD (ohm); "inf" = circuito aberto/desconectado
Rat_meas_commissioning = np.array([
    [4.540, 6.540, 5.900, 4.020],     # 01 FFFF
    [np.inf, 6.630, 5.930, 4.060],    # 02 AFFF
    [4.590, np.inf, 5.930, 4.030],    # 03 FAFF
    [4.550, 6.580, np.inf, 4.110],    # 04 FFAF
    [4.600, 6.560, 6.030, np.inf],    # 05 FFFA
    [np.inf, np.inf, 5.960, 4.060],   # 06 AAFF
    [4.630, np.inf, np.inf, 4.130],   # 07 FAAF
    [4.620, 6.610, np.inf, np.inf],   # 08 FFAA
    [np.inf, 6.670, np.inf, 4.150],   # 09 AFAF
    [4.700, np.inf, 6.090, np.inf],   # 10 FAFA
    [np.inf, 6.680, 6.080, np.inf],   # 11 AFFA
    [np.inf, np.inf, np.inf, 4.170],  # 12 AAAF
    [np.inf, 6.720, np.inf, np.inf],  # 13 AFAA
    [np.inf, np.inf, 6.110, np.inf],  # 14 AAFA
    [4.670, np.inf, np.inf, np.inf],  # 15 FAAA
    [np.inf, np.inf, np.inf, np.inf], # 16 AAAA
], dtype=float)

Rat_meas_ref = Rat_meas_commissioning[0].copy()


# =====================================================
# 2) IDENTIFICAÇÃO DA MATRIZ Z (8×8)
# =====================================================

def identify_Z(I_full, V_tower, cp_labels, ridge=0.0):
    """
    Identify symmetric Z (8x8) from:
      V_i = sum_j Z_ij * I_j  and V_i = V_tower
    Only for connected electrodes in each test.
    """
    n_tests, n_e = I_full.shape

    idx_map = {}
    pairs = []
    k = 0
    for i in range(n_e):
        for j in range(i, n_e):
            idx_map[(i, j)] = k
            pairs.append((i, j))
            k += 1
    n_vars = k

    A_rows, b = [], []

    for m in range(n_tests):
        lbl = cp_labels[m]
        connected = [True]*4 + [c == "F" for c in lbl]

        for i in range(n_e):
            if not connected[i]:
                continue

            row = np.zeros(n_vars, dtype=float)
            for j in range(n_e):
                ii, jj = (i, j) if i <= j else (j, i)
                row[idx_map[(ii, jj)]] += I_full[m, j]

            A_rows.append(row)
            b.append(V_tower[m])

    A = np.vstack(A_rows)
    b = np.array(b, dtype=float)

    if ridge > 0:
        A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(n_vars)])
        b_aug = np.hstack([b, np.zeros(n_vars)])
        z, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    else:
        z, *_ = np.linalg.lstsq(A, b, rcond=None)

    Z = np.zeros((n_e, n_e), dtype=float)
    for (i, j), val in zip(pairs, z):
        Z[i, j] = val
        Z[j, i] = val

    return Z


Z = identify_Z(I_full, V_tower, cp_labels, ridge=0.0)


# =====================================================
# 3) Rat_pred_ref (FFFF) usando Z
# =====================================================

def loop_impedance_from_Z(Z, cp_label, cable_idx):
    """
    Predict loop impedance (Rat) for cable electrode (CA..CD)
    by eliminating the rest of the connected electrodes under equipotential constraint.
    """
    if cp_label[cable_idx - 4] != "F":
        return np.inf

    active = [0, 1, 2, 3]
    for k, ch in enumerate(cp_label):
        if ch == "F":
            active.append(4 + k)
    active = sorted(active)

    Zsub = Z[np.ix_(active, active)]

    pos = active.index(cable_idx)
    Zrest = np.delete(np.delete(Zsub, pos, axis=0), pos, axis=1)

    m = Zrest.shape[0]
    ones = np.ones((m, 1))
    M = np.block([
        [Zrest, -ones],
        [ones.T, np.zeros((1, 1))]
    ])
    rhs = np.zeros(m + 1)
    rhs[-1] = 1.0

    sol = np.linalg.solve(M, rhs)
    Veq = sol[-1]

    Zself = Z[cable_idx, cable_idx]
    return float(Zself + Veq)


def predict_Rat_all(Z, cp_label="FFFF"):
    Rat = np.zeros(4, dtype=float)
    for k in range(4):
        Rat[k] = loop_impedance_from_Z(Z, cp_label, 4 + k)
    return Rat


cp_label_normal = "FFFF"
Rat_pred_ref = predict_Rat_all(Z, cp_label=cp_label_normal)


# =====================================================
# 4) Baseline alpha_ref_cp (usando ensaio FFFF)
# =====================================================

I_tot_hat_0 = float(np.sum(np.abs(I_full[0]))) + 1e-12
alpha_ref_cp = I_CP[0] / I_tot_hat_0


# =====================================================
# 5) SALVAR MODELO
# =====================================================

np.savez(
    "grounding_model_Zonly.npz",
    Z=Z,
    Rat_pred_ref=Rat_pred_ref,
    Rat_meas_ref=Rat_meas_ref,
    Rat_meas_commissioning=Rat_meas_commissioning,
    alpha_ref_cp=alpha_ref_cp,
    cp_label_normal=cp_label_normal,
    electrodes=np.array(electrodes, dtype=object)
)

print("Modelo salvo em grounding_model_Zonly.npz")
print("Rat_meas_ref (FFFF) =", Rat_meas_ref)
print("Rat_pred_ref (FFFF) =", Rat_pred_ref)
print("alpha_ref_cp (FFFF) =", alpha_ref_cp)

