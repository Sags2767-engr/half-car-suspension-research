"""run_washout.py  -  Counts how many of 100 perturbed plants (+-20 %) give an unstable closed loop with the
Kalman-filter chain: without washout, with a 0.2 Hz and with a 0.5 Hz washout on the estimated body velocity.
Also stores the smallest stability margin (1 - spectral radius) at 0.5 Hz. Writes results3.json."""
import json
from real import *
R = json.load(open('results.json')); KB = np.array(R['K_B']); P = NOM
keys = ['mb','Ib','mwf','mwr','kbf','kbr','kwf','kwr','cbf','cbr']
rng = np.random.default_rng(7); plants = []
for i in range(100):
    p = dict(P)
    for k in keys: p[k] = P[k]*rng.uniform(0.8, 1.2)
    plants.append(design(p))
Dn = design(P); dt = Dn['dt']; K10 = np.hstack([KB, np.zeros((2, 2))])
def rho(Dp, fc):
    Phi, Gam, C, Lk = Dn['Phi'], Dn['Gam'], Dn['C'], Dn['L']; Pp, Gp, Cp = Dp['Phi'], Dp['Gam'], Dp['C']
    Gu = Gp[:, :2]; Gun = Gam[:, :2]
    if fc is None:
        A = np.block([[Pp, -Gu@K10], [Lk@Cp, Phi - Gun@K10 - Lk@C]])
    else:
        a = np.exp(-2*np.pi*fc*dt); Sel = np.zeros((2, 10)); Sel[0, 4] = 1; Sel[1, 5] = 1; Kz = -(K10@Sel.T)
        A = np.zeros((22, 22)); A[:10, :10] = Pp; A[:10, 10:20] = -Gu@K10; A[:10, 20:] = -Gu@Kz
        A[10:20, :10] = Lk@Cp; A[10:20, 10:20] = Phi - Gun@K10 - Lk@C; A[10:20, 20:] = -Gun@Kz
        A[20:, 10:20] = (1-a)*Sel; A[20:, 20:] = a*np.eye(2)
    return np.abs(np.linalg.eigvals(A)).max()
out = {str(fc): int(sum(rho(Dp, fc) > 1+1e-9 for Dp in plants)) for fc in (None, 0.2, 0.5)}
out['margin_05'] = float(1 - max(rho(Dp, 0.5) for Dp in plants))
print(out); json.dump(out, open('results3.json', 'w'))