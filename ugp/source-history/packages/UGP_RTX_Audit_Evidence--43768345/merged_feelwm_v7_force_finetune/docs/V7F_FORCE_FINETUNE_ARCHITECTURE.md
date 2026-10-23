# Five-stream V7F force-specialist continuation

The frozen upstream network is the completed per-leg Feel-WM/V7 model. It keeps five aligned query streams: one global/body stream plus FL, FR, HL, and HR streams. FL and FR use their action-conditioned terrain at `t-2`; HL and HR use their action-conditioned terrain at `t-3`. All branches use the current rollout action. The leg order is fixed as FL, FR, HL, HR through cache preparation, model input, prediction, and evaluation.

The trainable continuation is only `V7ForceSpecialist`. For every horizon and leg it receives the existing 1,285-dimensional force context, performs causal attention across the five prediction horizons, predicts contact phase, mixes four force experts, and fuses direct vector correction with magnitude/direction reconstruction. Its refined force is fed back into the next autoregressive rollout step.

The continuation uses all 28,103 eligible training anchors across 28 trajectories. The separate 5,632-anchor development split selects `best_force.pt` using normalized force MAE. Five K=5 horizons are weighted uniformly. Scheduled sampling decreases from 50 percent teacher forcing in epoch one to fully autoregressive training in epoch seven. The four strict test trajectories are never used for optimization or checkpoint selection.
