
# V7 — FULL FOUR-BRANCH TERRAIN-CONDITIONED PHYSICAL RESPONSE TRANSFORMER

## Goal

Build one final model family that keeps the full original four-branch vision design and combines the strongest parts of V4/V5/V6:

- preserve V4/V5 residual/direct multi-horizon prediction;
- preserve V5 contact-regime-conditioned force/torque head;
- preserve V5 soft physics features/losses;
- preserve V4-style binary contact output;
- use V6 continuous force magnitude/direction only as an auxiliary regularizer, NOT as the final force reconstruction;
- do NOT use V6 hard physics + torque residual as the final torque construction;
- hard expert selection: SHORT at +0.1 s; LONG at +0.2 ... +1.0 s.

## Vision: EXACTLY FOUR PARALLEL BRANCHES

### Branch A — semantic dense appearance
Backbone: DINOv3 ViT-L/16, frozen initially.
Input: calibrated 448x448 ground-relevant RGB crop.
Output: dense 28x28 patch tokens, 1024D -> adapter -> 256D.

### Branch B — local texture
Backbone: DINOv3 ConvNeXt-Large, frozen initially.
Input: same calibrated 448x448 RGB crop.
Output: multi-scale convolutional feature map -> FPN -> 28x28x256.

### Branch C — metric RGB-D geometry
Backbone: Utonia-pretrained Point Transformer V3, frozen initially.
Input: 4096 metric points from aligned depth backprojection.
Per-point features:
[x,y,z,r,g,b,nx,ny,nz,height,local_roughness]
Coordinates must be transformed into the verified ANYmal base frame.
Output: point tokens -> adapter -> 256D.

### Branch D — temporal motion
Backbone: WAFT optical flow, frozen initially.
Input: consecutive RGB frames.
For each flow pair produce:
[u,v,flow_magnitude,divergence,curl,confidence/validity]
Stack recent flow pairs and encode with a temporal CNN.
Output: spatial 28x28x256 motion tokens.

## No heuristic image ROI

The current V4/V5 center/lower crop must be replaced.

For TartanGround pinhole camera:
width = 640
height = 640
fx = fy = 320
cx = cy = 320

Backproject depth:
X = (u-cx) Z / fx
Y = (v-cy) Z / fy
Z = depth

Then use verified camera->base extrinsic to select physically relevant ground points.
Project the selected 3D region back to the RGB image.
The SAME geometric mask/crop must be used by DINOv3, ConvNeXt and WAFT.

No camera-to-base transform may be guessed.

## Vision token compression

Each branch can contain hundreds/thousands of tokens, so use a Perceiver-style resampler:

DINOv3 tokens        -> 16 learned latent tokens x 256
ConvNeXt tokens      -> 16 learned latent tokens x 256
Utonia/PTv3 tokens   -> 16 learned latent tokens x 256
WAFT/TCN tokens      -> 16 learned latent tokens x 256

Total = 64 visual tokens x 256.

Add a learned modality embedding to each branch.

## Foot-conditioned cross-modal fusion

Create 5 query tokens:

1 GLOBAL terrain query
4 FOOT terrain queries: FL, FR, RL, RR

Each foot query is conditioned on:
- learned leg identity;
- current joint state;
- verified forward-kinematic foot position;
- planned short displacement from commanded body motion.

The 5 queries cross-attend to all 64 vision tokens using a 4-layer Transformer:
d_model = 256
heads = 8
FFN = 1024

Output:
T_global: 256
T_FL: 256
T_FR: 256
T_RL: 256
T_RR: 256

## Proprioception branch — keep the proven causal design

Per leg at each 100 Hz timestep:
P12 + U6 + L3 + E3 = 24 channels

Shared leg causal Transformer:
input 24
d_model 128
2 layers
4 heads
FFN 512
GELU
causal mask

This yields one 128D token per leg.

Body:
B12 + U6 = 18 channels
same 128D, 2-layer, 4-head causal Transformer.

## Per-leg visual/proprio fusion

For each leg i:

leg_state_i: 128
foot_vision_i: 256

shared MLP:
384 -> 256 -> 256

Output:
Z_FL, Z_FR, Z_RL, Z_RR each 256D.

No graph-attention block is required here.
Inter-leg information is composed later with body context and soft physics.

## Global dynamics context

Concatenate:
4 x per-leg fused tokens = 1024
body token = 128
global vision token = 256

total = 1408

Context MLP:
1408 -> 768 -> 384

For every prediction horizon add:
future/planned controls -> 128
horizon embedding -> 128
soft physics features -> 128

Final shared horizon context:
384 + 128 + 128 + 128 = 768

## Output heads — strongest parts of V4/V5/V6

### 1. Binary contact head
Primary contact output.
4 sigmoid logits.

Use this for contact probability because V4-style binary contact was more reliable than V5's four-class accuracy.

### 2. Four-regime load head
Auxiliary + conditioning:
0 = no contact
1 = light
2 = medium
3 = heavy

4 classes x 4 feet.

Do NOT use this as the sole contact output.
Embed the regime probabilities and use them to condition the force/torque head.

### 3. State/body residual head
Predict:
q residual 12
dq residual 12
body residual 12

Total 36.

Baseline:
q_base(t+h) = q_t + h*dq_t
dq_base = dq_t
body_base = body_t

final = baseline + learned residual.

### 4. Specialized per-leg force/torque head
This is the most important head.

For each foot independently but with SHARED weights, concatenate:
- final horizon context 768
- its per-leg fused token 256
- its foot vision token 256
- its 4 regime probabilities
- binary contact probability

Input approximately 1285D.

MLP:
1285 -> 512 -> 256 -> 6

Outputs:
tau_x/tau_HAA, tau_HFE, tau_KFE
Fx, Fy, Fz

The output is a residual around persistence:
tau_base = measured tau_t
F_base = measured F_t

final = base + residual.

This keeps the V5 behavior that produced the best force/torque.

### 5. Continuous force auxiliary head
Use the useful part of V6 only as an auxiliary target:

predict:
log(1 + |F|)
unit force direction

Do NOT reconstruct the final force from this head.
It regularizes the direct V5-style force vector head.

### 6. Optional uncertainty head
For tau and F output log sigma.
Train with a robust heteroscedastic likelihood in addition to point loss.
This gives confidence/risk estimates without changing the direct mean prediction.

## Physics

Use V5-style physics as a SOFT constraint and input feature, because V5 improved tau/F and V6's hard torque construction regressed.

Current causal identified proxy:
tau ~= Phi(q,dq,ddq,F,contact-regime) W

plus:
sum(F_i) -> body acceleration proxy

Soft losses:
kinematic consistency
inverse-dynamics consistency
Newton/body-force consistency

Do not hard-add tau_physics to the network output.

If and only if the exact TartanGround ANYmal robot model + frames are verified, replace the proxy with Pinocchio:
M(q)ddq + C(q,dq)dq + g(q) - J(q)^T F
as an additional soft loss / expert.

## Temporal experts

### SHORT expert
history = 0.5 s = about 50 proprio samples
vision = current RGB-D + recent flow history
prediction = ONLY +0.1 s

### LONG expert
history = 4.0 s = about 400 proprio samples
vision = current four-branch visual state + longer motion summary
prediction = DIRECT:
+0.2,+0.3,...,+1.0 s

No autoregressive sensor feedback.

### Deployment rule
+0.1 s -> SHORT only
+0.2 ... +1.0 s -> LONG only

No Gaussian short-proxy interpolation.

## Training

### Stage 0 — visual cache
Run every heavy visual backbone separately and cache outputs:
DINOv3
ConvNeXt
Utonia/PTv3
WAFT

Unload each backbone before running the next one.
This allows strong frozen backbones even on limited GPU memory.

### Stage 1 — global prior
Train adapters + vision fusion + proprio Transformers + output heads across all available causal training runs.

Do NOT try to fine-tune all four foundation backbones on ~65 local windows.

### Stage 2 — local adaptation
For each ttest:
copy global prior
take previous 20 s only
oldest -> newest
one optimizer.step per window
recency weighted
no shuffle

### Stage 3 — optional adapter fine-tuning
Only after the global model works:
LoRA / small adapters in final DINOv3 blocks,
ConvNeXt final stage,
Utonia final stage,
WAFT remains frozen initially.

## Loss

Recommended starting loss:

L =
  1.0 Lq
+ 1.0 Ldq
+ 1.5 Ltau
+ 2.0 Lforce_vector
+ 0.5 Lbody
+ 0.25 Lbinary_contact
+ 0.40 L4regime
+ 0.25 Lforce_logmag_aux
+ 0.10 Lforce_direction_aux
+ 0.20 Lkinematic
+ 0.20 Linverse_dynamics
+ 0.10 Lnewton
+ 0.10 Ltrend

Use normalized Smooth-L1/Huber for q,dq,tau,F,body rather than letting force scale dominate.

These are initial weights, not experimentally validated constants.

## Current acceptance targets

The new model is NOT "done" merely because it is larger.
It must beat the best existing native experts.

At +0.1 s:
V5 short:
q ~= 0.0725 rad
dq ~= 0.5004 rad/s
tau ~= 2.949
force ~= 6.208 N

Desired V7:
q <= about 0.075
dq <= about 0.50
tau < 2.95
force < 6.21 N

At +1.0 s:
V5 long:
q ~= 0.643
dq ~= 1.250
tau ~= 6.695
force ~= 13.694 N

V4 long gives better kinematics:
q ~= 0.626
dq ~= 0.971
tau ~= 6.597
but worse force ~= 21.077 N.

Desired V7:
q <= 0.64
dq preferably <= 1.1
tau < 6.6
force < 13.7 N

## Important data constraint

TartanGround supplies RGB, depth, pose/meta, LiDAR and proprioception.
Optical flow can be generated from pose + intrinsics + depth.
The visual cache used by V4/V5 is insufficient for V7 because it only contains DINOv2/ConvNeXt/depth features.

V7 therefore REQUIRES a new visual-preprocessing/cache stage.
Do not pretend this can be added as a tiny post-processing cell to the existing cache.
