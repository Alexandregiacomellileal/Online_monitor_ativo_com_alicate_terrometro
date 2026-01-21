# 📌 Grounding Monitoring — Z-only (Counterpoise Currents + Loop Resistances)

This repository contains the Python implementation of a **Z-only online monitoring method** for estimating the **tower grounding resistance** using only:

✅ **Counterpoise currents:** `I_CA`, `I_CB`, `I_CC`, `I_CD`  
✅ **Loop resistances:** `R_atA`, `R_atB`, `R_atC`, `R_atD`

🚫 No surface potential sensors (`V0..V4`)  
🚫 No total tower current measurement (`I_tot`)

The method is based on an **offline-identified multi-terminal impedance matrix** `Z` and an **online calibration factor** `k` derived from loop resistances.
