# Backend v0.1.6 — conditional height/geometry risk
Status: DRAFT/HOLD. Optional offline Python-only helper; existing CLI unchanged.
Given declared maximum vertical contact offset h and measured camera K/R/C,
the maximum XY bias is h*norm(ray_xy)/abs(ray_z). Speed bias is bounded by
3.6*(endpoint_bias_1+endpoint_bias_2)/elapsed_seconds in km/h.
This is not a 3D point-height estimator, physical accuracy measurement or
probabilistic confidence interval. Lens/pose/terrain systematic errors are
excluded. The height limit is user-declared, not observed.
Further independent physical geometry stress, resource limits and licensing
remain open. No frontend and no real data testing at this stage.
