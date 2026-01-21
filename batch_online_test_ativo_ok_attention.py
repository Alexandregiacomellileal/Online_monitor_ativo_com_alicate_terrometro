# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:48:43 2026

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
batch_online_test_Zonly_commissioning.py

Batch test for Z-only online monitoring using commissioning dataset as if it were online measurements.

ONLINE INPUTS ONLY:
- I_CP (counterpoise currents) 4 values: [CA, CB, CC, CD]
- Rat_meas (loop resistances) 4 values: [RatA, RatB, RatC, RatD]

NO sensor potentials (V0..V4)
NO tower total current measurement (Itot)

WHAT IT DOES:
- Loads Z_off + Rat_pred_ref + alpha_ref_cp from grounding_model_Zonly.npz
- Uses Table-4 Rat_meas (real commissioning loop resistances)
- For each test (15 tests, skipping AAAA):
    * infer topology from I_CP + Rat_meas
    * estimate k from Rat_meas / Rat_pred_ref
    * Z_on = k * Z_off
    * solve equipotential equations to estimate V_tower and I_F
    * compute R_tower_est and compare to R_ref (commissioning)
    * prints full report (same style of online_monitor_Zonly.py)
- Saves:
    * Zonly_online_all15_reports.txt
    * appendix_Zonly_online_all15_reports.tex
    * Zonly_Rtower_error_summary.csv

Author: Alexandre Giacomelli Leal
"""

import numpy as np
import os
import csv


# =====================================================
# 0) CONFIG
# =====================================================
MODEL_FILE = "grounding_model_Zonly.npz"

TH_OK = 0.35
I_CP_MIN = 1e-3

TOL_ALPHA   = 0.05
TOLZ_REL    = 0.05
TOL_RAT_REL = 0.05

# Pesos do HI_TOTAL do contrapeso
W_I   = 0.35   # HI_I (corrente)
W_RAT = 0.55   # HI_Rat (loop)
W_ZCP = 0.10   # healthZ_CP

# Arquivos de saída
OUT_TXT = "Zonly_online_all15_reports.txt"
OUT_TEX = "appendix_Zonly_online_all15_reports.tex"
OUT_CSV = "Zonly_Rtower_error_summary.csv"


# =====================================================
# 1) LOAD MODEL
# =====================================================
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Arquivo {MODEL_FILE} não encontrado.\n"
        f"Rode primeiro: offline_identification_Zonly.py"
    )

model = np.load(MODEL_FILE, allow_pickle=True)

Z_off         = model["Z"]
Rat_pred_ref  = model["Rat_pred_ref"]
alpha_ref_cp  = model["alpha_ref_cp"]
cp_label_norm = str(np.array(model["cp_label_normal"]).reshape(-1)[0])
electrodes    = list(model["electrodes"])

# Rat_meas reais da Tabela 4 (16 x 4)
Rat_meas_commissioning = model["Rat_meas_commissioning"]


# =====================================================
# 2) COMMISSIONING DATA REQUIRED FOR BATCH
# =====================================================

cp_labels = [
    "FFFF","AFFF","FAFF","FFAF",
    "FFFA","AAFF","FAAF","FFAA",
    "AFAF","FAFA","AFFA","AAAF",
    "AFAA","AAFA","FAAA","AAAA"
]

# R_ref do comissionamento (método passivo) - usado somente para comparar erro (%)
R_ref = np.array([
    1.150, 1.195, 1.166, 1.190,
    1.303, 1.267, 1.253, 1.385,
    1.278, 1.333, 1.371, 1.342,
    1.497, 1.454, 1.458, 1.596
], dtype=float)

# Correntes medidas nos contrapesos (A) - usadas como "I_CP online input"
I_CP_all = np.array([
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

cp_names = ["CA", "CB", "CC", "CD"]


# =====================================================
# 3) FUNCTIONS (same logic as online_monitor_Zonly.py)
# =====================================================

def ok_attention(h):
    return "OK" if float(h) > TH_OK else "ATTENTION"


def robust_scale_k(Rat_meas, Rat_pred_ref):
    Rat_meas = np.asarray(Rat_meas, dtype=float).reshape(4)
    Rat_pred_ref = np.asarray(Rat_pred_ref, dtype=float).reshape(4)

    mask = np.isfinite(Rat_meas) & np.isfinite(Rat_pred_ref) & (Rat_pred_ref > 1e-12)
    if np.sum(mask) == 0:
        return 1.0

    ratios = Rat_meas[mask] / Rat_pred_ref[mask]
    return float(np.median(ratios))


def health_Rat(Rat_meas, Rat_pred, tol_rel=TOL_RAT_REL):
    Rat_meas = np.asarray(Rat_meas, dtype=float).reshape(4)
    Rat_pred = np.asarray(Rat_pred, dtype=float).reshape(4)

    out = np.zeros(4, dtype=float)
    for k in range(4):
        if (not np.isfinite(Rat_meas[k])) or (not np.isfinite(Rat_pred[k])) or (abs(Rat_pred[k]) < 1e-12):
            out[k] = 0.0
        else:
            denom = tol_rel * abs(Rat_pred[k]) + 1e-12
            out[k] = 1.0 / (1.0 + abs(Rat_meas[k] - Rat_pred[k]) / denom)
    return out


def health_I_CP(I_CP, I_tot_est, alpha_ref, tol=TOL_ALPHA):
    I_CP = np.asarray(I_CP, dtype=float).reshape(4)
    I_tot_est = max(float(I_tot_est), 1e-12)

    alpha = np.abs(I_CP) / I_tot_est
    return 1.0 / (1.0 + np.abs(alpha - alpha_ref) / tol)


def build_connected_cp(I_CP_meas, Rat_meas):
    """
    Decide se cada CP está conectado (F) ou desconectado (A):
    - corrente ~0  -> aberto
    - Rat_meas = inf -> aberto
    """
    I_CP_meas = np.asarray(I_CP_meas, dtype=float).reshape(4)
    Rat_meas = np.asarray(Rat_meas, dtype=float).reshape(4)

    connected = np.ones(4, dtype=bool)

    for i in range(4):
        if abs(I_CP_meas[i]) < I_CP_MIN:
            connected[i] = False
        if np.isinf(Rat_meas[i]):
            connected[i] = False

    return connected


def cp_label_from_connected(connected_cp):
    return "".join(["F" if c else "A" for c in connected_cp.tolist()])


def loop_impedance_from_Z(Z, cp_label, cable_idx):
    """
    Predict Rat for CA/CB/CC/CD under a given topology cp_label (e.g. FFFF, AFFF,...)
    """
    if cp_label[cable_idx - 4] != "F":
        return np.inf

    active = [0, 1, 2, 3]  # fundações sempre
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


def solve_equipotential_Zonly(Z_on, I_CP_meas, connected_cp):
    """
    Unknowns: I_F (4) and V_tower (1)
    Equations: for each connected electrode i (FA..FD always + connected CPs)
        Vt = Z_iF I_F + Z_iCP I_CP
    -> Z_iF I_F - Vt = -Z_iCP I_CP
    """
    I_CP_meas = np.asarray(I_CP_meas, dtype=float).reshape(4)
    connected_cp = np.asarray(connected_cp, dtype=bool).reshape(4)

    connected_electrodes = [0, 1, 2, 3] + [4 + k for k in range(4) if connected_cp[k]]

    A_rows, b_rows = [], []
    for i in connected_electrodes:
        Z_iF  = Z_on[i, 0:4]
        Z_iCP = Z_on[i, 4:8]

        row = np.zeros(5, dtype=float)
        row[0:4] = Z_iF
        row[4] = -1.0

        rhs = -float(Z_iCP @ I_CP_meas)

        A_rows.append(row)
        b_rows.append(rhs)

    A = np.vstack(A_rows)
    b = np.array(b_rows, dtype=float)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    I_F_est = x[0:4]
    Vt_est = float(x[4])

    return I_F_est, Vt_est


def healthZ_vector(Z_on, I_hat, Vt_est, tol_rel=TOLZ_REL):
    """
    healthZ_i = 1 / (1 + |(Z*I)_i - Vt| / (tol_rel*|Vt|) )
    """
    Vtol = tol_rel * abs(float(Vt_est)) + 1e-12
    V_model = Z_on @ I_hat
    rZ = V_model - float(Vt_est)
    healthZ = 1.0 / (1.0 + np.abs(rZ) / Vtol)
    return V_model, rZ, healthZ


# =====================================================
# 4) BATCH RUN (15 tests: skipping AAAA)
# =====================================================

full_reports = []
summary_rows = []

print(f"\nModelo carregado: {MODEL_FILE}")
print(f"cp_label_normal (npz) = {cp_label_norm}")
print(f"TH_OK = {TH_OK}\n")

for m in range(16):

    if cp_labels[m] == "AAAA":
        # omitindo conforme solicitado (indeterminado)
        continue

    lbl = cp_labels[m]

    # "online inputs"
    I_CP_meas = I_CP_all[m]
    Rat_meas = Rat_meas_commissioning[m]

    # inferred topology
    connected_cp = build_connected_cp(I_CP_meas, Rat_meas)
    topo_inferred = cp_label_from_connected(connected_cp)

    # scale factor
    kscale = robust_scale_k(Rat_meas, Rat_pred_ref)

    # scaled Z
    Z_on = kscale * Z_off

    # predicted Rat for inferred topology
    Rat_pred_top = predict_Rat_all(Z_on, cp_label=topo_inferred)

    # solve Vt + foundation currents
    I_F_est, Vt_est = solve_equipotential_Zonly(Z_on, I_CP_meas, connected_cp)

    # total current estimate
    I_hat = np.hstack([I_F_est, I_CP_meas])
    I_tot_est = float(np.sum(np.abs(I_hat))) + 1e-12

    # R_tower estimate
    R_tower_est = Vt_est / I_tot_est

    # compare with R_ref
    R_true = float(R_ref[m])
    err_pct = 100.0 * (R_tower_est - R_true) / R_true

    # health metrics
    V_model, rZ, healthZ = healthZ_vector(Z_on, I_hat, Vt_est)
    healthZ_cp = healthZ[4:8]

    HI_Rat = health_Rat(Rat_meas, Rat_pred_top)
    HI_I = health_I_CP(I_CP_meas, I_tot_est, alpha_ref_cp)

    HI_total = np.zeros(4, dtype=float)
    status_cp = [""] * 4

    for i in range(4):
        if not connected_cp[i]:
            HI_total[i] = 0.0
            status_cp[i] = "ATTENTION"
        else:
            HI_total[i] = (W_I * HI_I[i]) + (W_RAT * HI_Rat[i]) + (W_ZCP * healthZ_cp[i])
            status_cp[i] = ok_attention(HI_total[i])

    # report block
    lines = []
    lines.append(f"\n\n########## TESTE {m+1:02d}  ({lbl}) ##########")
    lines.append("=== ONLINE ESTIMATION (Z-ONLY: I_CP + Rat) ===")
    lines.append(f"cp_label_normal (npz) = {cp_label_norm}")
    lines.append(f"Inferred topology (from I_CP/Rat) = {topo_inferred}")
    lines.append(f"Scale factor k (Rat calibration) = {kscale:.4f}")
    lines.append(f"V_tower_est = {Vt_est:.6f} V")
    lines.append(f"I_tot_est   = {I_tot_est:.6f} A")
    lines.append(f"R_tower_est = {R_tower_est:.6f} ohm")
    lines.append(f"R_tower_ref = {R_true:.6f} ohm")
    lines.append(f"Erro (R_est - R_ref) = {err_pct:+.2f} %\n")

    lines.append("Inferred foundation currents (I_F):")
    for name, val in zip(["FA", "FB", "FC", "FD"], I_F_est):
        lines.append(f"  {name}: {val:.6f} A")

    lines.append("\nMeasured counterpoise currents (I_CP):")
    for name, val in zip(cp_names, I_CP_meas):
        lines.append(f"  {name}: {val:.6f} A")

    lines.append("\nRat check (meas vs pred):")
    for i, name in enumerate(cp_names):
        meas = Rat_meas[i]
        pred = Rat_pred_top[i]
        if np.isfinite(meas) and np.isfinite(pred) and abs(pred) > 1e-12:
            e = 100.0 * (meas - pred) / pred
            lines.append(f"  {name}: Rat_meas={meas:.6f} | Rat_pred={pred:.6f} | err={e:+.2f}%")
        else:
            lines.append(f"  {name}: Rat_meas={meas} | Rat_pred={pred} | err=NA")

    lines.append("\nElectrode health via Z (healthZ):")
    for n, h in zip(electrodes, healthZ):
        lines.append(f"  {n}: healthZ={h:.3f} -> {ok_attention(h)}")

    lines.append("\nCounterpoise health (HI_I / HI_Rat / healthZ_CP / HI_TOTAL):")
    for i, name in enumerate(cp_names):
        lines.append(
            f"  {name}: HI_I={HI_I[i]:.3f} | HI_Rat={HI_Rat[i]:.3f} | "
            f"healthZ_CP={healthZ_cp[i]:.3f} | HI_TOTAL={HI_total[i]:.3f} -> {status_cp[i]}"
        )

    report = "\n".join(lines)
    full_reports.append(report)

    # summary row
    summary_rows.append([
        m + 1,
        lbl,
        topo_inferred,
        f"{kscale:.4f}",
        f"{R_tower_est:.6f}",
        f"{R_true:.6f}",
        f"{err_pct:+.2f}"
    ])

    # also print on screen
    print(report)


# =====================================================
# 5) SAVE OUTPUTS
# =====================================================

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(full_reports).strip() + "\n")

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED Appendix: Z-only online reports for commissioning configurations\n")
    f.write("% Inputs: I_CP + Rat_meas only (Table 4). No sensors. No Itot.\n")
    f.write(f"% TH_OK = {TH_OK}\n\n")
    f.write("\\section*{Appendix: Z-only Online Estimation Using Counterpoise Currents and Loop Resistances}\n")
    f.write("\\noindent The following logs were generated by feeding the commissioning measurements as if they were online inputs, using only counterpoise currents $\\mathbf{I}_{CP}\\in\\mathbb{R}^4$ and loop resistances $\\mathbf{R}_{at}\\in\\mathbb{R}^4$ (Table~4). The tower potential and grounding resistance were inferred using the impedance matrix $Z$ identified offline. The test ``AAAA'' was omitted due to lack of connected electrodes.\n\n")

    for blk in full_reports:
        hdr = blk.splitlines()[2] if blk.startswith("\n\n") else blk.splitlines()[0]
        title = hdr.replace("#", "").strip()
        f.write(f"\\subsection*{{{title}}}\n")
        f.write("\\begin{verbatim}\n")
        f.write(blk.strip() + "\n")
        f.write("\\end{verbatim}\n\n")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f, delimiter=";")
    wr.writerow(["Test", "Label", "Topology_inferred", "k", "R_tower_est", "R_tower_ref", "Error_%"])
    wr.writerows(summary_rows)

print("\n\n=== FILES GENERATED ===")
print(f"- {OUT_TXT}")
print(f"- {OUT_TEX}")
print(f"- {OUT_CSV}")
