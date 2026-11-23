INDEPENDENT ROAD EVALUATION PATCH

Purpose
-------
Separate controller tuning roads from performance-evaluation roads.

Seed policy
-----------
Development/tuning set: 1000-1007 (unchanged in run_base.py's RB list).
Held-out evaluation set: 2000-2019 (used for random-road comparisons,
Wk-weighted comparisons, realism ladder random-road tests, and actuator
sensitivity road tests).
Plant-uncertainty Monte Carlo: its existing independent RNG policy is retained.

Run order (from this directory, in the user's Python environment)
------------------------------------------------------------------
1. python run_base.py
2. python run_mc.py
3. python run_real.py

Important
---------
This patch changes random-road evaluation seeds only. It does not change
controller gains, passive tuning, plant parameters, actuator parameters,
observer design, Wk implementation, road PSD levels, or metric definitions.
The new outputs must replace the corresponding old random-road results in
the manuscript only after they have been generated and checked.

Class B/Class C comparison
--------------------------
The existing code uses Gd=64e-6 for Class B and Gd=256e-6 for Class C with
matched seeds. Since the PSD parameter is four times larger, the generated
road amplitude scales by two. This is a paired amplitude-scaling comparison,
not an independent road-class validation. The manuscript must describe it
that way unless a separate independent Class C protocol is designed.

Reproducibility
---------------
Record Python, NumPy and SciPy versions and retain the newly generated
results.json, results2.json, and figs/base.npz with the manuscript archive.