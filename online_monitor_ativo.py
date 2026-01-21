# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:17:18 2026

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
online_monitor_Zonly.py

ONLINE monitoring using ONLY:
- Counterpoise currents: I_CP = [I_CA, I_CB, I_CC, I_CD]
- Counterpoise loop resistances measured: Rat_meas = [RatA, RatB, RatC, RatD]

No sensor potentials (V0..V4)
No tower total current measurement (Itot)

Method:
1) Load Z_off and Rat_pred_ref (FFFF) from .npz
2) Estimate scale factor k using Rat_meas / Rat_pred_ref
3) Z_on = k * Z_off
4) Solve equipotential equations to estimate:
   - Foundation currents I_F
   - Tower potential V_tower
5) Compute I_tot and R_tower

Print status only: OK / ATTENTION (no fault open label).
No final score.

Author: Alexandre Giacomelli Leal
"""

import numpy as np
import os


# =====================================================
# 0) PARÂMETROS
# =====================================================

MODEL_FILE = "grounding_model_Zonly.npz"

TH_OK = 0.35
I_CP_MIN = 1e-3

TOL_ALPHA = 0.05
TOLZ_REL = 0.05

TOL_RAT_REL = 0.05  # 5% relativo para health_Rat (ajustável)

# pesos para saúde do contrapeso (pode ajustar)
W_I   = 0.35   # HI_I
W_RAT = 0.55   # HI_Rat
W_ZCP = 0.10   # healthZ_CP


# =====================================================
# 1) CARREGAR MODELO OFFLINE
# =====================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Arquivo não encontrado: {MODEL_FILE}")

model = np.load(MODEL_FILE, allow_pickle=True)

Z_off = model["Z"]
Rat_pred_ref = model["Rat_pred_ref"]
alpha_ref_cp = model["alpha_ref_cp"]
cp_label_normal = str(np.array(model["cp_label_normal"]).reshape(-1)[0])
electrodes = list(model["electrodes"])

print(f"Modelo carregado: {MODEL_FILE}")
print(f"cp_label_normal (npz) = {cp_label_normal}")


# =====================================================
# 2) FUNÇÕES AUXILIARES
# =====================================================

def robust_scale_k(Rat_meas, Rat_pred_ref):
    """
    k = median( Rat_meas / Rat_pred_ref ) usando somente valores finitos.
    """
    Rat_meas = np.asarray(Rat_meas, dtype=float).reshape(4)
    Rat_pred_ref = np.asarray(Rat_pred_ref, dtype=float).reshape(4)

    mask = np.isfinite(Rat_meas) & np.isfinite(Rat_pred_ref) & (Rat_pred_ref > 1e-12)

    if np.sum(mask) == 0:
        # sem Rat válido -> k=1 (fallback)
        return 1.0

    ratios = Rat_meas[mask] / Rat_pred_ref[mask]
    return float(np.median(ratios))


def health_Rat(Rat_meas, Rat_pred, tol_rel=TOL_RAT_REL):
    """
    health_Rat = 1 / (1 + |Rat_meas - Rat_pred| / (tol_rel*|Rat_pred|) )
    """
    Rat_meas = np.asarray(Rat_meas, dtype=float).reshape(4)
    Rat_pred = np.asarray(Rat_pred, dtype=float).reshape(4)

    out = np.zeros(4, dtype=float)
    for k in range(4):
        if not np.isfinite(Rat_meas[k]) or not np.isfinite(Rat_pred[k]) or abs(Rat_pred[k]) < 1e-12:
            out[k] = 0.0
        else:
            denom = tol_rel * abs(Rat_pred[k]) + 1e-12
            out[k] = 1.0 / (1.0 + abs(Rat_meas[k] - Rat_pred[k]) / denom)
    return out


def health_I_CP(I_CP, I_tot_est, alpha_ref, tol=TOL_ALPHA):
    """
    HI_I baseado na fração de corrente alpha = I_CP / I_tot_est
    """
    I_CP = np.asarray(I_CP, dtype=float).reshape(4)
    I_tot_est = max(float(I_tot_est), 1e-12)

    alpha = np.abs(I_CP) / I_tot_est
    return 1.0 / (1.0 + np.abs(alpha - alpha_ref) / tol)


def solve_equipotential_Zonly(Z_on, I_CP_meas, connected_cp=None):
    """
    Estima I_F (4) e V_tower usando apenas:
    - Z_on (8x8)
    - I_CP_meas (4)

    Monta para cada eletrodo conectado:
        V_tower = Z_iF I_F + Z_iCP I_CP_meas
    Reorganiza:
        Z_iF I_F - V_tower = - Z_iCP I_CP_meas

    Unknowns: [I_FA, I_FB, I_FC, I_FD, V_tower] (5)
    """

    I_CP_meas = np.asarray(I_CP_meas, dtype=float).reshape(4)

    # flags de conexão dos CP (se None -> assume todos conectados)
    if connected_cp is None:
        connected_cp = np.array([True, True, True, True], dtype=bool)
    else:
        connected_cp = np.asarray(connected_cp, dtype=bool).reshape(4)

    # eletrodos conectados: fundações sempre conectadas + CP conectados
    connected_electrodes = [0, 1, 2, 3]  # FA..FD
    for k in range(4):
        if connected_cp[k]:
            connected_electrodes.append(4 + k)

    # Monta A x = b
    # x = [I_F(4), Vt(1)]
    A_rows, b_rows = [], []

    for i in connected_electrodes:
        Z_iF  = Z_on[i, 0:4]  # 1x4
        Z_iCP = Z_on[i, 4:8]  # 1x4

        # Z_iF * I_F  - 1*Vt = -Z_iCP * I_CP_meas
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
    healthZ por eletrodo com base em:
        rZ_i = (Z_on @ I_hat)_i - Vt_est
    """
    Vtol = tol_rel * abs(float(Vt_est)) + 1e-12
    V_model = Z_on @ I_hat
    rZ = V_model - float(Vt_est)
    healthZ = 1.0 / (1.0 + np.abs(rZ) / Vtol)
    return V_model, rZ, healthZ


def ok_attention(h):
    return "OK" if float(h) > TH_OK else "ATTENTION"


# =====================================================
# 3) >>> MEDIÇÃO ONLINE (USUÁRIO COLA AQUI) <<<
# =====================================================

# Correntes nos contrapesos (A)
I_CP_meas = np.array([0.0946, 0.0505, 0.0686, 0.1400], dtype=float)

# Resistências de loop medidas (ohm) - RatA..RatD
# Cole suas medições aqui (se não tiver, use np.nan)
Rat_meas = np.array([4.54, 6.54, 5.9, 4.06], dtype=float)


# =====================================================
# 4) PROCESSAMENTO ONLINE (Z-ONLY)
# =====================================================

# 4.1) Estima escala global k a partir do Rat
k = robust_scale_k(Rat_meas, Rat_pred_ref)

Z_on = k * Z_off
Rat_pred_on = k * Rat_pred_ref

# 4.2) Define CP conectado se:
# - corrente não é ~0 E
# - Rat é finito (se Rat é NaN, não força desconexão)
connected_cp = np.ones(4, dtype=bool)
for i in range(4):
    if abs(I_CP_meas[i]) < I_CP_MIN:
        connected_cp[i] = False
    if np.isfinite(Rat_meas[i]) and Rat_meas[i] > 1e6:
        connected_cp[i] = False

# 4.3) Resolve I_F e V_tower via equipotencialidade
I_F_est, Vt_est = solve_equipotential_Zonly(Z_on, I_CP_meas, connected_cp=connected_cp)

# 4.4) Corrente total estimada
I_hat = np.hstack([I_F_est, I_CP_meas])
I_tot_est = float(np.sum(np.abs(I_hat))) + 1e-12

# 4.5) Resistência equivalente estimada
R_tower_est = Vt_est / I_tot_est

# 4.6) healthZ em todos eletrodos
V_model, rZ, healthZ = healthZ_vector(Z_on, I_hat, Vt_est, tol_rel=TOLZ_REL)

# 4.7) Saúde dos CPs
HI_Rat = health_Rat(Rat_meas, Rat_pred_on, tol_rel=TOL_RAT_REL)
HI_I   = health_I_CP(I_CP_meas, I_tot_est, alpha_ref_cp, tol=TOL_ALPHA)
healthZ_cp = healthZ[4:8]

HI_total = np.zeros(4, dtype=float)
status_cp = [""] * 4

for kcp in range(4):
    # se CP desconectado -> ATTENTION (sem rótulo de falha)
    if not connected_cp[kcp]:
        HI_total[kcp] = 0.0
        status_cp[kcp] = "ATTENTION"
    else:
        HI_total[kcp] = (W_I * HI_I[kcp]) + (W_RAT * HI_Rat[kcp]) + (W_ZCP * healthZ_cp[kcp])
        status_cp[kcp] = ok_attention(HI_total[kcp])


# =====================================================
# 5) PRINTS (mesmo padrão: OK / ATTENTION)
# =====================================================

print("\n=== ONLINE ESTIMATION (Z-ONLY: I_CP + Rat) ===")
print(f"cp_label_normal (npz) = {cp_label_normal}")
print(f"Scale factor k (Rat calibration) = {k:.4f}")
print(f"V_tower_est = {Vt_est:.6f} V")
print(f"I_tot_est   = {I_tot_est:.6f} A")
print(f"R_tower_est = {R_tower_est:.6f} ohm\n")

print("Inferred foundation currents (I_F):")
for name, val in zip(["FA","FB","FC","FD"], I_F_est):
    print(f"  {name}: {val:.6f} A")

print("\nMeasured counterpoise currents (I_CP):")
for name, val in zip(["CA","CB","CC","CD"], I_CP_meas):
    print(f"  {name}: {val:.6f} A")

print("\nRat check (meas vs pred):")
for i, name in enumerate(["CA","CB","CC","CD"]):
    meas = Rat_meas[i]
    pred = Rat_pred_on[i]
    if np.isfinite(meas) and np.isfinite(pred) and abs(pred) > 1e-12:
        err = 100.0 * (meas - pred) / pred
        print(f"  {name}: Rat_meas={meas:.6f} | Rat_pred={pred:.6f} | err={err:+.2f}%")
    else:
        print(f"  {name}: Rat_meas={meas} | Rat_pred={pred} | err=NA")

print("\nElectrode health via Z (healthZ):")
for n, h in zip(electrodes, healthZ):
    print(f"  {n}: healthZ={h:.3f} -> {ok_attention(h)}")

print("\nCounterpoise health (HI_I / HI_Rat / healthZ_CP / HI_TOTAL):")
for i, name in enumerate(["CA","CB","CC","CD"]):
    print(
        f"  {name}: HI_I={HI_I[i]:.3f} | HI_Rat={HI_Rat[i]:.3f} | "
        f"healthZ_CP={healthZ_cp[i]:.3f} | HI_TOTAL={HI_total[i]:.3f} -> {status_cp[i]}"
    )
