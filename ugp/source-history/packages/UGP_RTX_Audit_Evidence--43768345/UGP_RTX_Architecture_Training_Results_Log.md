# UGP RTX experiment log: architectures, mathematics, training and results

**Audit date:** 5 October 2026. **Scope:** UGP robotics experiments associated with V7, V7-F, FeelWM and TartanGround, including the RTX V8 A–F extensions. This document follows the executed code and retained result files. Every result belongs to the experiment and evaluation protocol stated next to it.


## Contents

- [Notation and optimizer updates](#mathematical-notation-and-what-a-training-update-means)
- [V7 and V7-F architecture, losses and results](#ugp-v7-and-v7-f-implemented-architecture-and-experiments)
- [V8 A–F architecture and RTX results](#v8-af-architecture-and-experiment-evidence)
- [FeelWM architecture, dimensions and losses](#feelwm--tartanground-architecture-audit)
- [FeelWM / TartanGround experiment records](#feelwm--tartanground-experiment-records-and-results)
- [Metric glossary](#metric-interpretation-and-comparisons)

## How this log was built

This is a read-only audit of the project files, server logs, saved configurations, checkpoint tensor metadata and exported metrics. Memory was used to locate projects; numerical results and current job status were checked against their files. The audit did not train, evaluate, resume, pause or modify any model.

The server identified itself as `UGP_HOST`. A live hardware query on 5 October 2026 returned **two NVIDIA GeForce RTX 4080 GPUs**, each reporting **16,376 MiB**, driver **580.178.04**. The FeelWM environment reported **PyTorch 2.5.1+cu121**, CUDA build **12.1**. A CUDA build version is the version used to build PyTorch; it is not a statement about every notebook environment or the installed driver runtime.

The main evidence roots are:

- `/path/to/ugp/V7F_4080_Run`: deployed notebooks, chronological execution logs, V7/V7-F results and V8 A–F artifacts.
- `/path/to/ugp/merged_feelwm_v7`: current merged world-model source and launch scripts.
- `/path/to/ugp/merged_feelwm_v7_runs`: P2000 pilot, matched baseline/per-leg pilot, ten-epoch continuation and queued force-specialist continuation.
- `/path/to/ugp/merged_feelwm_v7_force_finetune`: the isolated proposed specialist continuation source.

The code, configurations and metric tables cited in this document are retained in the accompanying evidence package. A filename containing “RTX,” a configuration requesting training, or an unexecuted notebook cell was not by itself treated as execution evidence. Colab/T4-only V8/V9/FORCE-5 results are outside this RTX log. Prior V4/V5/V6 outputs sometimes appear as evaluation references inside V7 artifacts; their presence does not establish that those older models were trained on these RTX GPUs.

### Reading order

1. Read the experiment ledger to see what actually ran and what remains unfinished.
2. Read the V7/V7-F architecture section and its immediately following experiment results.
3. Read V8 A–F: each module has its inputs, model, objective, training protocol and recorded results.
4. Read the FeelWM architecture, followed by the individual pilot and ten-epoch run records.
5. Use the metric glossary and detailed CSV appendices to interpret the numbers.

### Evidence labels

| Label | Meaning |
|---|---|
| Recorded | A value appears in a retained log, checkpoint or result artifact. |
| Derived | Calculated from inspected code, tensor shapes or recorded values; the derivation is stated. |
| Configured | Requested by the code/configuration; execution evidence is stated separately. |
| Unavailable | The required measurement or provenance was not retained or not yet produced. |

The latest notebook run completed its recorded cell sequence on 28 September. Several cells restored already-trained models and skipped their global training loops. Earlier RTX logs contain the actual V7 and V7-F training. These are recorded as separate events below.

The ten-epoch comparison has **three completed components and one paused component**. The paused per-leg K5 process was observed as PID `469841`, process group `284634`, state `Tl`. Its last logged optimizer step is **33,477**, and its latest durable checkpoint is **33,475**. The final forty-case strict evaluation and test-only ZIP had not been produced in the audited ten-epoch run directory.

## Mathematical notation and what a training update means

Let $B$ be batch size, $T$ the number of history samples, $L=4$ the legs, $K$ the prediction horizons, $N$ the number of visual tokens, and $d$ a feature width. A shape $B\times K\times4\times3$ means a three-component vector for each leg at each future horizon for every batch item. The leg order used in the merged project is **FL, FR, HL, HR**; logged robot topic names `lf/rf/lh/rh` map to those legs.

Let $\theta$ denote the trainable parameters, $\phi$ frozen parameters, $x$ an input, $y$ a target and $f_{\theta,\phi}(x)$ the model. Training minimizes a scalar objective $\mathcal L$ assembled from the terms documented for that experiment. An optimizer step computes gradients with respect to the trainable graph and applies an update. Frozen feature builders and frozen priors receive no optimizer updates.

### Epoch, microbatch and optimizer step

An **epoch** is one traversal of the training loader. A **microbatch** is one loader batch. With gradient accumulation $A$, approximately $A$ microbatches contribute to one optimizer step:

$$g_s=\frac1A\sum_{a=1}^{A}\nabla_\theta\mathcal L_a(\theta_s).$$

For $n$ examples, per-device batch $b$ and $D$ distributed workers, nominal steps per epoch are approximately

$$S_{\mathrm{epoch}}=\left\lceil\frac{\lceil n/(bD)\rceil}{A}\right\rceil,$$

with exact handling determined by the sampler and final partial accumulation. In the single-GPU matched comparison, $n=28{,}103$, $b=1$, $A=8$, so **3,513 optimizer steps per nominal epoch**, or **35,130 for ten epochs**. The final accumulation has seven microbatches. The current implementation divides each microbatch loss by eight even in that partial accumulation, so its contribution is scaled by $7/8$ relative to a full group.

One optimizer step concerns the model's trainable parameter tensors together. It does **not** mean “update one of 110 million parameters.” A particular tensor can have no gradient because its branch was not used, and a particular component can have a zero gradient. Thus “all trainable parameters are updated once per epoch” is not the correct description: an epoch normally contains thousands of updates; optimizer membership and the computation graph determine which parameters participate in each one.

### The AdamW update used by the merged trainer

The retained merged checkpoints confirm $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$. For gradient $g_s$, AdamW forms

$$m_s=\beta_1m_{s-1}+(1-\beta_1)g_s,\qquad v_s=\beta_2v_{s-1}+(1-\beta_2)g_s^2,$$

$$\widehat m_s=\frac{m_s}{1-\beta_1^s},\qquad \widehat v_s=\frac{v_s}{1-\beta_2^s},$$

$$\theta_{s+1}=(1-\eta_s\lambda_{\mathrm{wd}})\theta_s-\eta_s\frac{\widehat m_s}{\sqrt{\widehat v_s}+\epsilon}.$$

The implementation clips the global gradient norm to one and uses weight decay **0.05** for eligible matrices, with **zero** decay for the separate non-decay group. Learning rate scheduling and mixed precision are specified in the experiment sections. These AdamW settings should not be substituted for the separate V7/V7-F notebook optimizers.

**“200 weight updates” means 200 calls at optimizer-step boundaries.** With accumulation eight, that normally corresponds to about 1,600 example presentations in a single-GPU run. It was an early pilot computation limit. **Pruning** instead changes connectivity or masks/removes weights, for example $\theta'_i=m_i\theta_i$ with $m_i\in\{0,1\}$. No pruning procedure is present in the inspected pilot trainer. AdamW weight decay shrinks eligible weights continuously; it is also distinct from pruning.

### Attention and residual prediction

For query/key/value matrices $Q,K,V$, one attention head computes

$$\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.$$

The mask $M$ is zero for permitted pairs and negative infinity for forbidden pairs. A time-causal mask means the query cannot attend to later physical times; it does not automatically prevent leakage in preprocessing, target-based sample selection or later replay updates. Those issues are audited separately below.

Many robot outputs use a physical baseline plus a learned standardized residual:

$$\widehat y=y_{\mathrm{baseline}}+\sigma_y\,r_\theta(x).$$

Consequently, a standardized training residual loss has no direct Newton, Newton-meter or radian interpretation. Physical metrics are computed after reconstructing outputs in physical units.

For SmoothL1/Huber-style terms, the PyTorch SmoothL1 form with parameter $\beta$ is

$$\operatorname{SL1}_\beta(e)=\begin{cases}e^2/(2\beta),&|e|<\beta,\\|e|-\beta/2,&|e|\ge\beta.\end{cases}$$

All reductions, masks and coefficients must be included when comparing objectives. A small composite loss or high contact accuracy alone does not establish accurate force vectors or generalization to unseen trajectories.



---

<a id="ugp-v7-and-v7-f-implemented-architecture-and-experiments"></a>

## UGP V7 and V7-F: implemented architecture and experiments

The notebook implements two V7 residual dynamics experts and two additional V7-F force specialists. V7 predicts joint angles, joint velocities, measured torques, contact forces, body signals, contact probabilities, and force-load regimes. V7-F freezes the V7 models and learns corrections to their contact-force vectors. The original Colab experiment, the expanded RTX diagnostic experiment, the temporal-gap experiment, and the strict unseen-trajectory experiment use different data or adaptation protocols; their results are reported separately below.

This section is based on the collected server notebook, source extracts, saved configurations and result CSVs. It is an evidence audit: no training, inference, tests or model-source changes were performed while preparing it. The snapshot contains the runner's final state from 2026-09-28, and the evidence collection is dated 2026-10-05. It does not establish the current live process state.

Primary sources: [collected server notebook][v7-nb], [completed RTX runner state][v7-state], [RTX training log from 2026-09-28 13:52:35 UTC][v7-trainlog], [latest RTX execution log][v7-latest].

### Inputs, histories, baselines and targets

| Input | Shape excluding batch | Meaning |
| --- | --- | --- |
| P | T × 4 × 12 | Per leg: q3, dq3, measured torque3, measured force Fx/Fy/Fz |
| B | T × 12 | IMU acceleration3, IMU gyro3, estimated linear velocity3, estimated angular velocity3 |
| U | T × 6 | Recorded high-level linear/angular twist command |
| L | T × 4 × 3 | Recorded low-level commanded joint torque |
| E | T × 4 × 3 | At n>0: measured torque[n] − commanded torque[n−1]; first sample uses same-index command |
| Leg history input | 4 × T × 24 | P12 + U6 + L3 + E3; a shared encoder is applied separately to each leg |
| Body history input | T × 18 | B12 + U6 |
| Future-control input | H × 36 | Last U6/L12 and mean U6/L12 over the recorded anchor-to-horizon control interval |
| Physics input | H × 47 | Normalized constant-velocity q12, proxy torque12, proxy acceleration3, force magnitude4, regime one-hot16 |
| Foot-query kinematics | 4 × 6 | Official ANYmal-D URDF foot FK position3 + current linear command × 0.30 s displacement3 |

SHORT uses 0.5 s context and predicts only +0.1 s. LONG uses 4.0 s context and jointly produces +0.2, +0.3, …, +1.0 s. History length is `round(context_sec / median(diff(time)))`; the original saved positional tensors have 50 and 400 history positions, consistent with 100 Hz samples. Horizon offsets are rounded on the robot time grid. Each inference anchor selects the latest camera frame at or before the anchor and rejects a future visual timestamp. [active configuration][v7-config], [saved tensor metadata from original ZIP checkpoints][v7-ckpt].

The raw baseline is `q_baseline(h) = q_now + h*dq_now`; dq, torque and force remain at their current values, and all body channels remain at the current body value. V7 learns `(target − baseline) / channel_std`, then reconstructs `prediction = baseline + predicted_residual*channel_std`. Binary contact labels use measured force norm **>10 N**; regime labels use **≥10 N** and learned positive-force tertiles. The difference at exactly 10 N is present in the implementation.

Per-run normalization uses channelwise mean/std, with P pooled across time and legs. L uses measured-torque mean/std; E uses measured-torque std without mean subtraction. V7 local adaptation recomputes statistics and soft physics on the latest 20 s through the test anchor. The V7-F specialist preserves the original training-run statistics and physics that its frozen base saw. [signal extraction and alignment][v7-prep], [global and local training][v7-fit], [force-head training and causal validation][v7-ffit].

Future recorded U and L are supplied to the horizon heads. Thus the benchmark is conditional on recorded future controller commands. A deployment would need planned commands or another forecasting assumption for these inputs. These are not autonomous forecasts from past sensors alone.

### Frozen visual feature extraction and trainable visual fusion

| Branch | Foundation encoder actually requested | Cached input to V7 | Processing |
| --- | --- | --- | --- |
| DINOv3 | timm/vit_large_patch16_dinov3.lvd1689m | 64 × 1025 | 1024 visual channels + ground-mask validity; last feature map |
| ConvNeXt | timm/convnext_large.dinov3_lvd1689m | 64 × 2881 | All four stage features pooled and concatenated (192+384+768+1536=2880), plus mean validity |
| Geometry | Pointcept/Utonia pretrained PTv3 | 64 × 1234 | 4096 input points; hierarchical recovered feature1224 + xyz3 + normal3 + RGB3 + roughness1 |
| Optical flow | WAFT a1, tar-c-t checkpoint | 4 × 64 × 6 | Four flows ending at or before the camera frame; channels u,v,magnitude,divergence,curl,confidence |

The RGB branches use a geometry-defined ground support crop, resized to 448×448 with ImageNet normalization, and mask-weighted 8×8 pooling. Metric depth is accepted only in the expected encoded/float formats, with usable depth 0.25–8.0 m. Support bounds in the camera-heading, gravity-aligned frame are x=0.15–4.0 m, |y|≤2.5 m, z=−2.0–0.35 m, and normal z≥0.35; the crop has 0.08 margin and requires 500 support pixels. The geometry cloud is sampled to 4096 points; 64 representative points are retained after pretrained encoding. Roughness is `1 − abs(normal_z)`, clipped to the valid range. It is a geometric proxy.

All four foundation models are put in eval mode, frozen, used under inference mode, and unloaded after their float16 caches are written. They are external to the V7 task checkpoints. Their weights do not receive gradients during V7 or V7-F training. The cached flow stack is causal by construction. Its trainable temporal CNN uses two kernel-3, padding-1 convolutions (6→64→128) across the four already observed flows, selects the last output and projects 128→256. Padding does not create access to future images. [frozen visual cache builders][v7-cache].

Each branch has a trainable Perceiver resampler: LayerNorm(input), Linear(input→256), GELU, LayerNorm(256), 16 learned latent tokens, and **2** cross-attention/FFN layers. Cross-attention has **8** heads; each FFN is 256→1024→256 with GELU and dropout 0.1. Flow first passes through its temporal CNN, then a resampler with input width 256. The four sets of 16 latent tokens receive modality embeddings and are concatenated into **64×256** visual tokens.

Five terrain queries (one learned global query and four foot queries) pass through **4** fusion blocks. Each block applies 8-head cross-attention to all 64 visual tokens, 8-head self-attention among the five queries, and a 256→1024→256 FFN with dropout 0.1. A foot query is a leg embedding plus an MLP of current normalized proprioception12, command6 and raw FK/displacement6: LayerNorm24→Linear24→256→GELU→Linear256→256. Outputs are a global terrain feature256 and four foot features256. These visual cross-attention and query self-attention operations are unmasked; all tokens are derived from the current or past inputs. [active V7 layers][v7-model].

The active V7 classes have **no FiLM layer**. They use query conditioning, cross-attention and concatenation followed by MLPs. They also do not apply explicit FL/FR t−2 versus rear-leg t−3 visual delays; all four foot queries read the same causal visual frame. Such FiLM/delay choices belong to another model implementation in the UGP workspace and must not be attributed to this notebook.

### Exact V7 task layers and output dimensions

| Module | Layers / dimensions | Result |
| --- | --- | --- |
| Shared leg history encoder | Linear24→128; learned positions; 2 Transformer encoder layers; 4 heads; FFN512; GELU; dropout0.1; pre-norm; final LayerNorm | Last history token128 + leg embedding128 |
| Body history encoder | Same architecture with input18→128 | Last history token128 |
| Leg/visual fusion | LayerNorm384; Linear384→256→GELU→Linear256→256→GELU→LayerNorm256 | Four fused leg features256 |
| Context MLP | LayerNorm1408; Linear1408→768; GELU; dropout0.1; Linear768→384; GELU; LayerNorm384 | Context384; input=4×256 + body128 + global visual256 |
| Control projection | 36→128; GELU; LayerNorm128 | 128 per horizon |
| Horizon projection | 1→64→GELU→128→GELU→LayerNorm128 | 128 per horizon |
| Physics projection | 47→128→GELU→128→GELU→LayerNorm128 | 128 per horizon |
| Horizon context | Concatenate context384 + control128 + horizon128 + physics128 | H × 768 |
| Binary contact head | 768→256→GELU→4 | H × 4 logits |
| Load-regime head | 768→256→GELU→16 | H × 4 × 4 logits |
| State/body head | 768→384→GELU→dropout0.1→36 | 24 joint q/dq residuals + body12 residuals per horizon |
| Torque/force head | LayerNorm1285; 1285→512→GELU→dropout0.1→256→GELU→6 | Per horizon/leg: torque3 + force3 residuals |
| Auxiliary force head | LayerNorm1285; 1285→256→GELU→4 | Log magnitude1 + normalized direction3 |

The 1285-dimensional per-leg head input is horizon context768 + fused leg256 + foot visual256 + contact probability1 + regime probabilities4. Joint q/dq and torque/force residuals are assembled as `(batch,H,4,12)`; body residuals are `(batch,H,12)`. The history encoders use a strictly upper-triangular Boolean attention mask and retain the final token. They share weights across legs but have separate leg identity embeddings. No time mask is used in the horizon-independent V7 output heads. [causal history encoder][v7-history], [active V7 layers][v7-model].

The freshly inspected server state tensors contain exactly **16,314,094** parameters for SHORT and **16,403,694** for LONG, and **2,448,554** for each specialist. The original ZIP checkpoint counts agree. The 89,600-parameter SHORT/LONG difference equals two positional tables ×350 additional positions×128. These counts cover the task models, not the frozen foundation encoders. The module breakdown below was derived from original ZIP tensors; the fresh server tensor totals and dimensions confirm the same task architecture. [fresh server saved tensor metadata][v7-fresh-ckpt], [saved tensor metadata from original ZIP checkpoints][v7-ckpt].

| Saved task module | SHORT parameter count | LONG parameter count |
| --- | --- | --- |
| leg_encoder | 406,400 | 451,200 |
| leg_identity | 512 | 512 |
| body_encoder | 405,632 | 450,432 |
| terrain_encoder | 12,083,608 | 12,083,608 |
| leg_visual_fuse | 165,632 | 165,632 |
| context_net | 1,380,992 | 1,380,992 |
| control_proj | 4,992 | 4,992 |
| horizon_proj | 8,704 | 8,704 |
| physics_proj | 22,912 | 22,912 |
| binary_contact_head | 197,892 | 197,892 |
| regime_head | 200,976 | 200,976 |
| state_body_head | 309,156 | 309,156 |
| tau_force_head | 793,872 | 793,872 |
| force_aux_head | 332,814 | 332,814 |

### V7-F specialist layers and force reconstruction

| Module | Exact structure |
| --- | --- |
| Input projection | LayerNorm1285; 1285→256; GELU; dropout0.10; 256→256; GELU; LayerNorm256 |
| Joint-horizon transformer | Two pre-norm Transformer encoder layers; width256; 8 heads; FFN1024; GELU; dropout0.10; applied separately per leg |
| Phase classifier | LayerNorm256; 256→128→GELU→4 |
| Four force experts | Each: LayerNorm256; 256→192→GELU→dropout0.08→96→GELU→3 |
| Expert gate | LayerNorm265; 265→128→GELU→4; softmax |
| Direct correction | LayerNorm256; 256→192→GELU→3 |
| Magnitude/direction correction | LayerNorm256; 256→192→GELU→4 |
| Fusion gate | LayerNorm265; 265→96→GELU→1; sigmoid |

V7-F extracts detached V7 context `x(batch,H,4,1285)`, raw base force `F_base`, contact probability and regime probabilities. The horizon transformer has **no attention mask**. Its tokens represent the planned conditional predictions at all queried horizons; they contain horizon/control/physics conditioning, not future measured state or force labels. SHORT has one token per leg, LONG has nine.

The phase targets are swing, touchdown, stance and liftoff. They are formed from contact transitions between the input anchor and target horizon, with the label agreeing with the final contact state. The gate input has 256 latent +4 phase probabilities +4 load-regime probabilities +1 contact probability =265 channels. The phase classifier guides a soft expert gate; the implementation does not hard-assign one named phase to one particular expert.

With expert weights `w_e`, per-axis force standard deviation `σ_F`, and MLP corrections `c_e,c_d,m,d`, the two paths and final output are:

`c = Σ_e w_e*c_e + c_d`

`F_direct = F_base + σ_F*c`

`m_base = max(||F_base||, 1e−5)`

`m_new = max(expm1(log1p(m_base) + 0.75*tanh(m)), 0)`

`d_new = normalize(normalize(F_base) + 0.20*tanh(d))`

`F_magdir = m_new*d_new`

`α = sigmoid(fusion_gate); F_final = α*F_direct + (1−α)*F_magdir`.

The correction-output layers and final fusion-gate layer are zero-initialized. Consequently both paths start at the base prediction up to numerical effects and the 1e−5 magnitude floor. Forces stay in the recorded measurement coordinate system: geometry normals are not used to invent an unverified camera-to-base force rotation. [active specialist layers and fusion][v7-specialist].

For the diagnostic V7-F CSV, only force rows are replaced in the existing adapted-V7 sensor table. Its joint, torque, body and contact rows are copied from V7. The force specialist itself runs on the **frozen global V7 prior**, not on each full-model locally adapted V7 copy used to construct the earlier diagnostic table. Thus the force comparison measures the complete V7-F pipeline versus the adapted V7 pipeline; it is not simply adding a head to the same locally adapted prediction.

### Physics proxies and complete training losses

The inverse-dynamics proxy for each leg is the identified regression

`τ_proxy = [ddq, dq, sin(q), cos(q), F, regime_onehot_or_probabilities, 1] W_leg`, with `W_leg` shape20×3.

The body proxy is `acc_proxy = [Σ_legs F, 1] W_body`, with `W_body` shape4×3. Ridge fitting uses λ=0.001 after RMS feature scaling, including its constant column. This is an affine calibrated soft physical consistency proxy, rather than a full rigid-body dynamics model with known mass, inertia and calibrated force frame. Official URDF FK supplies foot-query positions; it is not the inverse-dynamics solver.

The fitting function selects rows in the requested time interval only **after** computing `np.gradient(dq,time)` and per-joint 99.5th-percentile derivative clipping limits over the **entire trajectory**. Central differences at a cutoff can use a later sample, and the clipping scale includes later trajectory data. Actuator-to-joint assignment is also inferred once using the first min(5000,N) aligned samples. These procedures mean that the notebook's word “causal” is stronger than the implemented preprocessing isolation. The gap tests and diagnostics do not establish strictly past-only calibration. [physics identification and derivative preprocessing][v7-physics], [signal extraction and alignment][v7-prep].

Let `Sβ(a,b)` denote Smooth L1/Huber loss with PyTorch beta β, and let normalized residuals be `r̂_P,r_P,r̂_B,r_B`. The q,dq,torque,force and body data losses use beta0.5, average their feature/leg axes, then average horizons with `w_h=linspace(1,1.5,H)` (or1 for SHORT); the weights are not normalized back to mean1. Other terms below use their direct implementation averages. [V7 loss implementation][v7-loss].

| V7 loss term | Coefficient | Exact definition |
| --- | --- | --- |
| q | 1.00 | Horizon-weighted S0.5 of normalized q residuals |
| dq | 1.00 | Horizon-weighted S0.5 of normalized dq residuals |
| torque | 1.50 | Horizon-weighted S0.5 of normalized torque residuals |
| force vector | 2.00 | Horizon-weighted S0.5 of normalized force residuals |
| body | 0.50 | Horizon-weighted S0.5 of all12 normalized body residuals |
| binary contact | 0.25 | BCE with logits against measured contact labels |
| force-load regime | 0.40 | Weighted cross entropy over4 classes |
| auxiliary log magnitude | 0.25 | S0.25(aux_logmag, log1p(||F_truth||)) |
| auxiliary direction | 0.10 | Contact-weighted mean of1−cos(aux_dir,F_truth_dir); denominator sum(contact)+1e−6 |
| kinematic consistency | 0.20 | S0.5((q_pred−q_now−0.5*h*(dq_now+dq_pred))/σ_q,0) |
| inverse-dynamics consistency | 0.20 | S0.5((τ_pred−τ_proxy(q_pred,dq_pred,(dq_pred−dq_now)/(h+1e−6),F_pred,regime_probs))/σ_τ,0) |
| Newton/body consistency | 0.10 | S0.5((acc_pred−acc_proxy(F_pred))/σ_acc,0) |
| zero-force in noncontact | 0.08 | Mean((1−p_contact)*||F_pred||/(largest regime center+1e−6)) |
| long-horizon trend | 0.10 | S0.5(Δ_hP_pred/σ_P, Δ_hP_truth/σ_P), zero for SHORT |

The total V7 objective is the sum of the 14 terms above. Standard-deviation denominators use the code’s 1e−6 safeguards. Regime class weights are inverse square-root class frequency, normalized to mean1 and clipped to[0.5,3.0]. For the expanded RTX run, thresholds are[10,222.11116,273.2027] N, centers[0,195.37396,249.0397,295.5505] N and class counts[618440,221511,221510,221511].

The V7-F total objective is

`L_F = 2.00L_vector + 1.60L_logmag + 0.10L_direction + 0.25L_phase + 0.45L_temporal + 0.35L_integral + 0.015L_anchor + 0.005L_gate`.

Its vector term uses normalized component error and beta0.35, multiplied per horizon/leg by `a = 1 + 1.25*clip(||F_truth||/max(last_regime_threshold,1),0,2) + 0.50*I(phase in touchdown,liftoff)`. The maximum such weight is4.0. [V7-F loss implementation][v7-floss].

| V7-F term | Exact definition |
| --- | --- |
| Vector | Mean(a * mean_components(S0.35((F_pred−F_truth)/(σ_F+1e−6),0))) |
| Log magnitude | S0.20(log1p(||F_pred||),log1p(||F_truth||)) |
| Direction | Mean1−cos on ||F_truth||≥10 N; denominator valid count+1e−6 |
| Phase | Unweighted4-class cross entropy |
| Temporal | S0.35(Δ_hF_pred/(σ_F+1e−6),Δ_hF_truth/(σ_F+1e−6)); zero for SHORT |
| Integral | S0.35((Σ_h0.1F_pred−Σ_h0.1F_truth)/(σ_F+1e−6),0); zero for SHORT |
| Anchor | Mean square of direct normalized correction c; does not penalize the entire magnitude/direction path |
| Gate | Mean squared difference between batch-mean expert probabilities and uniform1/4 |

The “impulse” term is a discrete sum over uniformly spaced horizon predictions. It is not a independently measured impulse target. The variable named gate entropy implements mean-gate squared balancing, rather than entropy. Every coefficient above comes from active code, including the internal0.75/0.20 force reconstruction bounds,1.25/0.50 example weighting and0.1 s integral step.

### Training protocol and what the RTX logs actually executed

| Stage | Configuration | Parameters updated |
| --- | --- | --- |
| V7 global SHORT | Seed791; 3 epochs; 0.3 s hop; AdamW lr0.0002; weight decay0.0001; clip1.0; one sample per update | All task-model layers; frozen foundation caches |
| V7 global LONG | Seed801; 4 epochs; 1.0 s hop; same optimizer | All task-model layers |
| V7 diagnostic local adaptation | 4 epochs each expert; lr0.00008; latest20 s; half-life5 s SHORT/6 s LONG; chronological windows; test-dependent seed | A fresh full V7 copy per test; local stats/physics refit |
| V7-F global SHORT | Seed7882; 5 epochs; 0.15 s hop; batch16; lr0.0003; AdamW wd0.0001; clip1.0 | Force head only; two-GPU DataParallel |
| V7-F global LONG | Seed7883; 6 epochs; 0.25 s hop; batch8; same optimizer | Force head only; two-GPU DataParallel |
| V7-F diagnostic local adaptation | 5 epochs; lr0.00006; hops0.3/1.0 s; latest20 s; half-life5/6 s; newest20% windows validation; best validation epoch restored | Force-head copy only; frozen global V7 base; original source-run calibration |
| Same-run gap adaptation | Gaps4,7,9,10 s from latest gradient input anchor; same5 epochs/lr0.00006; no validation split or best-epoch selection | Force-head copy only, independently per gap/test; two GPU workers |
| Strict unseen | No local V7/V7-F updates; no heldout normalization/physics fit; four tests per trajectory | No task parameters updated; cache extraction and frozen inference |

The expanded training set has seven trajectories in each of four terrain groups, **28 unique trajectories total**. The four diagnostic trajectories remain part of training. Their global prior cutoff is the first diagnostic time; the other24 training trajectories use cutoff `run_end−1.05 s`. The four unseen trajectories are excluded from both task-model training stages. [exact 28-trajectory training manifest][v7-split], [strict holdout configuration][v7-zero-cfg].

Training keys by environment are ForestEnv P2000/P2003/P2007/P2011/P2015/P2019/P2022; GreatMarsh P2000/P2005/P2010/P2015/P2020/P2023/P2029; SeasonalForestWinter P2000/P2003/P2009/P2012/P2015/P2018/P2021; OldTownFall P2000/P2002/P2003/P2004/P2005/P2006/P2007.

The16 diagnostic anchors are ForestEnv/P2000 at28,45,75,90 s; GreatMarsh/P2023 at47,61,73,87 s; SeasonalForestWinter/P2000 at29,46,64,92 s; OldTownFall/P2000 at32,84,100,115 s. Independent V7 local seeds are `781 + terrain_index*1000 + test_id*100 +1` (SHORT) or+2 (LONG), with zero-based terrain index and one-based test id. V7-F starts its seed family at781+7100=7881 and uses+1/+2 for global experts. [active configuration][v7-config], [global and local training][v7-fit], [force-head training and causal validation][v7-ffit].

V7-F global examples are cached detached half-precision contexts and full-precision targets/calibration. Mini-batches draw across trajectories. The sampler balances **force bins**, with duplication of each rare bin capped at2.5× its population; it does not guarantee equal terrain counts within a mini-batch. These densely overlapping windows are optimization samples, not independent trajectories.

The earlier RTX log `training_20260928T135235Z.log` records real global V7 and V7-F optimizer updates and local diagnostic adaptations. V7 prior raw losses per epoch were SHORT1.180507371→0.843577968→0.745650106 and LONG2.446516626→1.702973012→1.472296120→1.322666874. V7-F had1969 batches per SHORT epoch and1540 per LONG epoch, with the executed losses below. The log contains all16 local V7 and16 local V7-F tests. [RTX training log from 2026-09-28 13:52:35 UTC][v7-trainlog].

| Epoch | V7-F SHORT executed loss (1969 batches) | V7-F LONG executed loss (1540 batches) |
| --- | --- | --- |
| 1 | 1.5687 | 2.0814 |
| 2 | 1.3267 | 1.7704 |
| 3 | 1.2359 | 1.6543 |
| 4 | 1.1872 | 1.5783 |
| 5 | 1.1313 | 1.5248 |
| 6 | — | 1.4819 |

The **latest** runner starts at2026-09-28 15:56:44 UTC and finishes at16:45:46 UTC with status`complete` and code cells1–23 completed. It logs two NVIDIA GeForce RTX4080 GPUs. It does **not** rerun the global training epochs or the16 original local diagnostics: these come from persisted checkpoints/results. The log prints an RNG restoration error (“RNG state must be a torch.ByteTensor”) after loading each model, optimizer and completed epoch. In the active code, these assignments precede RNG restoration inside a broad try block, and the exception does not undo them. The subsequent epoch loops therefore execute zero epochs; the completed diagnostic signatures match and all16 V7 and16 V7-F tests are skipped. This code trace and the absence of global epoch updates explain the very short V7/V7-F cell times; the warning is not evidence of training from scratch. It also means exact RNG restoration was unsuccessful in this rerun. [latest RTX execution log][v7-latest], [global and local training][v7-fit], [force-head training and causal validation][v7-ffit], [completed RTX runner state][v7-state].

The latest run actually executes all64 same-run gap jobs (16 tests×4 gaps), with a SHORT and LONG5-epoch head adaptation in each job; it then extracts features for the four strict holdouts and performs16 zero-shot tests. Gap cell22 takes130.8 s; unseen cell23 takes2703.9 s including download/preprocessing. The unseen timing CSV reports0.218205 s for all zero-shot inference with caches prepared, and heldout preprocessing about657–687 s per trajectory. This cached inference timing does not include the foundation encoders or establish real-time end-to-end deployment speed. [saved unseen timing][v7-timing].

Optional V5 checkpoint paths are absent in the RTX log. V5 numerical predictions are not training targets in these V7/V7-F stages. The eight stored “best existing V5” values below are hard-coded diagnostic evaluation references.

### Metric definitions and aggregation

Force **component MAE** is the mean absolute error across Fx/Fy/Fz, then averaged over vectors. Force **vector error** is `||F_pred−F_truth||₂` in N; **magnitude error** is `abs(||F_pred||₂−||F_truth||₂)` in N. Direction cosine is reported only where ground-truth force meets the contact cutoff (and, in the gap/unseen functions, predicted magnitude exceeds1e−9); noncontact directions are N/A. “Vector error percent of mean force” is100×mean vector error/mean true force magnitude, including all vectors. It is a ratio of means and differs from averaging per-vector or per-test percentages.

Each diagnostic/unseen horizon has16 test anchors×4 legs=64 force vectors; each unseen terrain/leg mean has4 anchors. The ten horizons are conditional predictions at the same16 anchors, rather than160 independent trajectories. Diagnostic sensor-group RMSE is computed inside each test/horizon/group and then averaged across16 tests. NRMSE divides that within-test RMSE by the group’s within-test ground-truth std plus1e−6, then averages. It is not a global pooled RMSE or a force-relative percentage.

Contact prediction is thresholded at0.5; accuracy/precision/recall/F1 use the64 leg labels per horizon, Brier is mean squared probability error and AUROC uses probability ranking when both classes occur. Trend correlation is Pearson correlation over the nine LONG horizons for each sensor/test, averaged by group; first-difference sign accuracy is the fraction of adjacent horizon changes whose signs agree. [metric definitions][v7-metrics].

### Original executed Colab diagnostic results (historical, distinct from RTX)

The executed original notebook retains the following force comparison over its familiar four trajectories with local adaptation. These numbers are historical Colab outputs and are **not** the expanded RTX CSVs. [executed original Colab outputs][v7-old].

| Horizon s | V7 component MAE N | V7-F component MAE N | Reduction % |
| --- | --- | --- | --- |
| 0.1 | 6.941602 | 6.077722 | 12.444966 |
| 0.2 | 18.441826 | 10.539195 | 42.851674 |
| 0.3 | 20.194385 | 13.798430 | 31.671947 |
| 0.4 | 15.052799 | 9.208241 | 38.827047 |
| 0.5 | 18.171151 | 11.492774 | 36.752636 |
| 0.6 | 17.251262 | 11.916269 | 30.925237 |
| 0.7 | 19.692355 | 13.657293 | 30.646725 |
| 0.8 | 16.426138 | 9.799497 | 40.342055 |
| 0.9 | 22.386615 | 14.078678 | 37.111181 |
| 1.0 | 17.682948 | 10.967694 | 37.975875 |

Original V7 vector errors were14.502486 N at+0.1 s and38.374624 N at+1.0 s. Comparing those numbers directly to the expanded unseen experiment changes both the training set and adaptation protocol, so it does not constitute a controlled improvement estimate.

### Expanded RTX diagnostics: V7 versus V7-F

| Horizon s | V7 component MAE N | V7-F component MAE N | Reduction % | V7 vector error N | V7-F vector error N | V7-F magnitude error N | V7-F cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 5.2203 | 3.8688 | 25.89 | 11.4895 | 8.2053 | 6.9463 | 0.999428 |
| 0.2 | 8.7047 | 4.6349 | 46.75 | 19.2049 | 9.9554 | 8.0810 | 0.997802 |
| 0.3 | 10.9375 | 6.4646 | 40.89 | 24.5584 | 14.7517 | 12.5548 | 0.980607 |
| 0.4 | 8.3898 | 4.3694 | 47.92 | 18.1128 | 9.1365 | 7.6911 | 0.999278 |
| 0.5 | 11.3371 | 6.8327 | 39.73 | 24.8213 | 14.0625 | 12.5834 | 0.997128 |
| 0.6 | 11.9728 | 9.1761 | 23.36 | 25.7818 | 20.1453 | 17.4419 | 0.994592 |
| 0.7 | 11.7257 | 8.1351 | 30.62 | 27.7032 | 18.6732 | 16.8540 | 0.999055 |
| 0.8 | 7.5457 | 4.0518 | 46.30 | 16.4601 | 8.5768 | 7.0328 | 0.999200 |
| 0.9 | 13.6586 | 7.6655 | 43.88 | 30.3020 | 16.0536 | 14.1187 | 0.995235 |
| 1.0 | 9.7211 | 5.8876 | 39.44 | 20.9454 | 12.4737 | 10.7056 | 0.998053 |

Sources: [RTX force comparison][v7-forcecomp], [RTX V7 vector metrics][v7-v7vec], [RTX V7-F vector metrics][v7-fvec].

The following V7 sensor MAEs are also present in the V7-F diagnostic sensor table because the specialist replaces only force rows. Units: q rad, dq rad/s, torque N·m, acceleration m/s², gyro and body angular velocity rad/s, body linear velocity m/s. “Contact” is probability absolute error.

| Horizon s | q | dq | Torque | Acc | Gyro | Body linear v | Body angular v | Contact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.0342 | 0.3499 | 2.0791 | 0.5854 | 0.0365 | 0.0235 | 0.0392 | 0.0000 |
| 0.2 | 0.0955 | 0.5400 | 2.8960 | 0.4976 | 0.0373 | 0.0431 | 0.0418 | 0.0302 |
| 0.3 | 0.1284 | 0.5738 | 3.3706 | 0.4478 | 0.0296 | 0.0311 | 0.0521 | 0.0343 |
| 0.4 | 0.1387 | 0.5714 | 2.6595 | 0.4338 | 0.0364 | 0.0284 | 0.0437 | 0.0029 |
| 0.5 | 0.1455 | 0.5795 | 2.8844 | 0.4946 | 0.0491 | 0.0369 | 0.0403 | 0.0184 |
| 0.6 | 0.1695 | 0.6254 | 2.7406 | 0.4842 | 0.0323 | 0.0287 | 0.0370 | 0.0271 |
| 0.7 | 0.1831 | 0.4832 | 2.6728 | 0.4581 | 0.0346 | 0.0351 | 0.0455 | 0.0202 |
| 0.8 | 0.2127 | 0.5568 | 2.9654 | 0.4889 | 0.0430 | 0.0366 | 0.0378 | 0.0057 |
| 0.9 | 0.2540 | 0.5999 | 2.8520 | 0.6022 | 0.0346 | 0.0393 | 0.0405 | 0.0440 |
| 1.0 | 0.2996 | 0.6212 | 2.7736 | 0.4575 | 0.0370 | 0.0362 | 0.0486 | 0.0267 |

[full RTX sensor MAE/RMSE/NRMSE and baseline metrics][v7-sens] retains every group/horizon and baseline value. Endpoint comparisons to the recorded V5 reference are:

| Horizon s | Group | Recorded V5 reference | RTX V7 MAE | RTX V7-F MAE |
| --- | --- | --- | --- | --- |
| 0.1 | q | 0.0725 | 0.0342 | 0.0342 |
| 0.1 | dq | 0.5004 | 0.3499 | 0.3499 |
| 0.1 | tau | 2.9492 | 2.0791 | 2.0791 |
| 0.1 | force | 6.2079 | 5.2203 | 3.8688 |
| 1.0 | q | 0.6432 | 0.2996 | 0.2996 |
| 1.0 | dq | 1.2505 | 0.6212 | 0.6212 |
| 1.0 | tau | 6.6952 | 2.7736 | 2.7736 |
| 1.0 | force | 13.6940 | 9.7211 | 5.8876 |

All eight stored reference comparisons pass the code’s strict lower-MAE check. This evaluates fixed familiar diagnostics and does not establish terrain generalization. [RTX V7-F acceptance reference table][v7-accept].

| Horizon s | Accuracy | Precision | Recall | F1 | Brier | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.2 | 0.968750 | 0.970588 | 0.970588 | 0.970588 | 0.017656 | 0.998039 |
| 0.3 | 0.968750 | 0.971429 | 0.971429 | 0.971429 | 0.019977 | 0.998030 |
| 0.4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000135 | 1.000000 |
| 0.5 | 0.984375 | 1.000000 | 0.969697 | 0.984615 | 0.014805 | 0.998045 |
| 0.6 | 0.984375 | 1.000000 | 0.969697 | 0.984615 | 0.016458 | 0.998045 |
| 0.7 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.007824 | 1.000000 |
| 0.8 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.001080 | 1.000000 |
| 0.9 | 0.968750 | 1.000000 | 0.939394 | 0.968750 | 0.024364 | 0.996090 |
| 1.0 | 0.968750 | 0.941176 | 1.000000 | 0.969697 | 0.018752 | 1.000000 |

[RTX contact metrics][v7-contact]. The pooled4-regime confusion matrix below contains640 predictions (16 tests×10 horizons×4 legs):

| True regime | Pred no | Pred light | Pred medium | Pred heavy |
| --- | --- | --- | --- | --- |
| true_0 | 304 | 6 | 1 | 0 |
| true_1 | 3 | 97 | 9 | 0 |
| true_2 | 1 | 6 | 76 | 12 |
| true_3 | 2 | 0 | 14 | 109 |

Correct predictions586/640=91.5625%. [RTX regime confusion matrix][v7-regime].

| LONG signal group | Mean Pearson correlation | First-difference sign accuracy |
| --- | --- | --- |
| acc | 0.723903 | 0.786458 |
| body_angular_velocity | 0.707978 | 0.770833 |
| body_linear_velocity | 0.835670 | 0.893229 |
| contact | 0.974203 | 0.318359 |
| dq | 0.950783 | 0.908203 |
| force | 0.863286 | 0.602865 |
| gyro | 0.635241 | 0.742188 |
| q | 0.700023 | 0.766276 |
| tau | 0.980701 | 0.898438 |

[RTX long-horizon trend metrics][v7-trend]. High contact classification accuracy and lower contact trend-sign accuracy measure different properties; exact sign agreement is sensitive to nearly constant probabilities and ties.

### Same-run temporal-gap force experiment

For each original test anchor and requested gap g, the latest **local gradient input anchor** is t_test−g. The latest target label can be g−0.1 s before the test for SHORT and g−1.0 s before it for LONG. For example, gap4 s at test28 s uses input24.0 s, with latest labels24.1 s (SHORT) or25.0 s (LONG). All local labels remain before the test, and the audit CSV records actual anchor/target gaps. The specialist trains on all permitted local windows for five epochs; the final head is evaluated. [actual gap adaptation implementation][v7-gap], [actual anchor and target-gap audit][v7-gap-audit].

The saved configuration says no training or validation/model-selection data follows the cutoff, but the source shows a narrower restriction: the **local input anchors** obey the cutoff; their target labels can follow it, and the frozen global priors/source normalization/physics are not retrained to that cutoff. At the first diagnostic test, prior calibration reaches the test anchor itself. Thus this experiment is a gap in additional local adaptation, not a fully isolated gap in all training data or preprocessing.

| Requested input gap s | Horizon s | Component MAE N | Vector error N | Magnitude error N | Direction cosine | Vector error / mean true force % |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.1 | 4.2193 | 9.0373 | 7.6983 | 0.999494 | 6.93 |
| 4 | 0.2 | 5.3372 | 11.3159 | 8.7259 | 0.997217 | 8.87 |
| 4 | 0.3 | 7.2895 | 16.4617 | 14.6193 | 0.978969 | 13.22 |
| 4 | 0.4 | 4.4326 | 9.1660 | 7.6437 | 0.999360 | 7.06 |
| 4 | 0.5 | 6.7724 | 14.1840 | 12.8645 | 0.997480 | 11.01 |
| 4 | 0.6 | 10.2287 | 21.9948 | 19.5752 | 0.993868 | 17.64 |
| 4 | 0.7 | 8.6478 | 19.8762 | 18.0172 | 0.998887 | 14.20 |
| 4 | 0.8 | 4.8354 | 10.5880 | 8.8275 | 0.999063 | 8.03 |
| 4 | 0.9 | 8.2949 | 17.6857 | 15.7989 | 0.994573 | 13.31 |
| 4 | 1.0 | 6.6322 | 13.9211 | 12.1329 | 0.997775 | 11.12 |
| 7 | 0.1 | 3.4092 | 7.5559 | 6.4016 | 0.999666 | 5.79 |
| 7 | 0.2 | 4.9783 | 10.8257 | 8.4349 | 0.997427 | 8.49 |
| 7 | 0.3 | 7.1716 | 16.1547 | 14.2686 | 0.978777 | 12.98 |
| 7 | 0.4 | 4.7188 | 9.7914 | 8.2910 | 0.999243 | 7.55 |
| 7 | 0.5 | 6.8978 | 14.3175 | 12.6452 | 0.998689 | 11.11 |
| 7 | 0.6 | 10.2271 | 22.8156 | 20.2976 | 0.994018 | 18.30 |
| 7 | 0.7 | 9.0173 | 20.4058 | 18.5082 | 0.998687 | 14.57 |
| 7 | 0.8 | 4.5369 | 9.7033 | 7.9120 | 0.999152 | 7.36 |
| 7 | 0.9 | 7.7232 | 16.4612 | 14.3289 | 0.991415 | 12.39 |
| 7 | 1.0 | 6.1866 | 13.1387 | 11.0824 | 0.997862 | 10.49 |
| 9 | 0.1 | 4.2509 | 9.1471 | 7.8193 | 0.999421 | 7.01 |
| 9 | 0.2 | 4.8760 | 10.2426 | 7.6673 | 0.997143 | 8.03 |
| 9 | 0.3 | 7.2669 | 16.4098 | 14.1568 | 0.979047 | 13.18 |
| 9 | 0.4 | 4.6411 | 9.8768 | 8.1608 | 0.999217 | 7.61 |
| 9 | 0.5 | 6.7625 | 14.1003 | 12.8461 | 0.995533 | 10.94 |
| 9 | 0.6 | 8.7792 | 19.0134 | 16.2341 | 0.994899 | 15.25 |
| 9 | 0.7 | 8.8059 | 20.1350 | 18.2945 | 0.998722 | 14.38 |
| 9 | 0.8 | 4.7695 | 10.5306 | 8.5167 | 0.999012 | 7.99 |
| 9 | 0.9 | 8.2977 | 17.6812 | 15.9129 | 0.995424 | 13.31 |
| 9 | 1.0 | 6.0625 | 12.5854 | 10.7246 | 0.997525 | 10.05 |
| 10 | 0.1 | 4.0068 | 8.8593 | 7.6305 | 0.999558 | 6.79 |
| 10 | 0.2 | 4.7335 | 10.1596 | 7.3123 | 0.997049 | 7.96 |
| 10 | 0.3 | 6.5252 | 14.4811 | 12.6307 | 0.980036 | 11.63 |
| 10 | 0.4 | 4.5054 | 9.5142 | 7.9233 | 0.999223 | 7.33 |
| 10 | 0.5 | 6.9946 | 14.4999 | 12.7668 | 0.996296 | 11.25 |
| 10 | 0.6 | 8.7865 | 18.8338 | 16.4418 | 0.994774 | 15.10 |
| 10 | 0.7 | 8.6413 | 19.6354 | 17.9399 | 0.998838 | 14.02 |
| 10 | 0.8 | 3.9859 | 8.3887 | 6.6448 | 0.999170 | 6.36 |
| 10 | 0.9 | 7.9493 | 16.7268 | 14.5307 | 0.973512 | 12.59 |
| 10 | 1.0 | 6.3672 | 13.3939 | 11.9042 | 0.997985 | 10.70 |

Each row averages64 vectors. [RTX gap overall results][v7-gap-result]; [complete gap results by terrain and leg][v7-gap-leg]. Errors do not increase monotonically with input gap. This small, familiar-trajectory diagnostic does not identify a universally preferred gap.

### Strict unseen-trajectory zero-shot force experiment

The excluded holdouts are ForestEnv/P2001, GreatMarsh/P2027, SeasonalForestWinter/P2006 and OldTownFall/P2001. Each is tested at20,27,36,46 s, producing spacing7,9,10 s. Training/holdout key overlap is empty. Both V7 experts and V7-F global heads are frozen: there are no heldout gradient updates, no local adaptations, and no fitting of normalization or physics on heldout runs. Each terrain borrows the corresponding original training-run calibration. [strict holdout configuration][v7-zero-cfg], [zero-shot source-calibration implementation][v7-unseen].

The run holds out trajectories within the same four environments used for training. It is evidence of unseen-trajectory transfer, not unseen-terrain-category transfer. Heldout sensor histories and current measured force remain valid observed inputs, while future recorded controls remain supplied as described above. The upstream actuator assignment and source physics preprocessing limitations also remain relevant.

| Horizon s | Vectors | Component MAE N | Vector error N | Mean true force N | Magnitude error N | Direction cosine | Vector error / mean true force % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 64 | 8.2270 | 18.7853 | 126.4182 | 16.0853 | 0.945258 | 14.86 |
| 0.2 | 64 | 11.7447 | 25.7238 | 131.6640 | 22.2658 | 0.988003 | 19.54 |
| 0.3 | 64 | 5.9861 | 13.0257 | 126.0564 | 10.5172 | 0.992867 | 10.33 |
| 0.4 | 64 | 6.6131 | 14.6187 | 125.5982 | 13.3578 | 0.999235 | 11.64 |
| 0.5 | 64 | 6.1754 | 13.4856 | 118.7507 | 11.7868 | 0.998339 | 11.36 |
| 0.6 | 64 | 13.6965 | 32.6429 | 139.2593 | 30.3331 | 0.997178 | 23.44 |
| 0.7 | 64 | 11.3634 | 25.4710 | 135.0439 | 22.4646 | 0.941958 | 18.86 |
| 0.8 | 64 | 9.2419 | 21.2623 | 130.5248 | 18.0023 | 0.996525 | 16.29 |
| 0.9 | 64 | 6.0112 | 12.8173 | 122.1884 | 9.9297 | 0.997969 | 10.49 |
| 1.0 | 64 | 6.2817 | 13.4419 | 129.0070 | 10.3886 | 0.943842 | 10.42 |

[RTX strict unseen overall results][v7-zero-result]. At+0.1 s the force-vector error is18.7853 N; at+1.0 s it is13.4419 N. The nonmonotonic curve should be reported as measured rather than smoothed into a worsening-with-horizon claim.

Endpoint detail by heldout terrain and leg follows. Every cell averages four tests; direction cosine excludes undefined noncontact directions.

| Terrain | Horizon s | Leg | Component MAE N | Vector error N | Mean true force N | Magnitude error N | Cosine | Relative vector error % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forest | 0.1 | FL | 9.6854 | 19.4680 | 154.9615 | 16.7871 | 0.998120 | 12.56 |
| forest | 0.1 | FR | 2.8447 | 6.4989 | 80.5305 | 6.3271 | 0.999863 | 8.07 |
| forest | 0.1 | RL | 5.9355 | 11.6309 | 53.4798 | 4.0764 | 0.144731 | 21.75 |
| forest | 0.1 | RR | 8.0949 | 16.8371 | 220.0799 | 14.2492 | 0.999325 | 7.65 |
| forest | 1.0 | FL | 10.2192 | 23.7662 | 203.6185 | 15.9524 | 0.552351 | 11.67 |
| forest | 1.0 | FR | 4.6147 | 9.0010 | 52.2029 | 7.3194 | 0.996753 | 17.24 |
| forest | 1.0 | RL | 3.6019 | 6.5982 | 63.2314 | 5.1729 | 0.998267 | 10.43 |
| forest | 1.0 | RR | 6.2279 | 13.8607 | 211.2654 | 11.2554 | 0.982105 | 6.56 |
| rough_uneven | 0.1 | FL | 10.2674 | 25.3422 | 115.6565 | 20.4239 | 0.995012 | 21.91 |
| rough_uneven | 0.1 | FR | 11.0523 | 22.5838 | 111.5578 | 15.1574 | 0.981730 | 20.24 |
| rough_uneven | 0.1 | RL | 33.3735 | 82.9066 | 147.7745 | 81.4116 | 0.999052 | 56.10 |
| rough_uneven | 0.1 | RR | 11.6697 | 28.0458 | 94.0973 | 18.1159 | 0.978087 | 29.81 |
| rough_uneven | 1.0 | FL | 5.8427 | 12.7827 | 133.6761 | 8.4150 | 0.998510 | 9.56 |
| rough_uneven | 1.0 | FR | 11.4543 | 22.8484 | 130.7756 | 19.5697 | 0.996002 | 17.47 |
| rough_uneven | 1.0 | RL | 2.9583 | 7.6093 | 134.5134 | 3.3401 | 0.999379 | 5.66 |
| rough_uneven | 1.0 | RR | 8.8911 | 17.0493 | 120.9945 | 8.6644 | 0.992870 | 14.09 |
| swamp_marsh | 0.1 | FL | 1.9554 | 3.7647 | 141.6681 | 2.6498 | 0.999866 | 2.66 |
| swamp_marsh | 0.1 | FR | 3.3666 | 8.1370 | 120.9682 | 7.3701 | 0.999781 | 6.73 |
| swamp_marsh | 0.1 | RL | 1.5837 | 3.2085 | 136.5276 | 2.6352 | 0.999915 | 2.35 |
| swamp_marsh | 0.1 | RR | 3.1047 | 6.4902 | 114.4802 | 5.9415 | 0.999696 | 5.67 |
| swamp_marsh | 1.0 | FL | 1.5032 | 3.5336 | 0.0000 | 3.5336 |  |  |
| swamp_marsh | 1.0 | FR | 7.5651 | 15.3236 | 281.7371 | 7.2024 | 0.998995 | 5.44 |
| swamp_marsh | 1.0 | RL | 3.8033 | 8.0483 | 226.6884 | 6.1945 | 0.999773 | 3.55 |
| swamp_marsh | 1.0 | RR | 1.0059 | 2.4137 | 0.0000 | 2.4137 |  |  |
| wet_uneven_proxy | 0.1 | FL | 7.5862 | 18.3430 | 90.9294 | 18.1442 | 0.994649 | 20.17 |
| wet_uneven_proxy | 0.1 | FR | 9.0381 | 21.4086 | 201.0410 | 20.7265 | 0.999780 | 10.65 |
| wet_uneven_proxy | 0.1 | RL | 7.7543 | 16.7315 | 177.3313 | 15.9336 | 0.999672 | 9.44 |
| wet_uneven_proxy | 0.1 | RR | 4.3204 | 9.1679 | 61.6070 | 7.4153 | 0.954213 | 14.88 |
| wet_uneven_proxy | 1.0 | FL | 7.1778 | 13.4042 | 105.1359 | 11.5850 | 0.998581 | 12.75 |
| wet_uneven_proxy | 1.0 | FR | 7.7261 | 17.8085 | 130.7952 | 17.1688 | 0.999577 | 13.62 |
| wet_uneven_proxy | 1.0 | RL | 6.3300 | 16.5371 | 127.5608 | 16.4543 | 0.999949 | 12.96 |
| wet_uneven_proxy | 1.0 | RR | 11.5854 | 24.4856 | 141.9173 | 21.9758 | 0.996494 | 17.25 |

[complete unseen terrain/leg results at all horizons][v7-zero-leg], [complete unseen results by test][v7-zero-test], [unseen test manifest][v7-zero-manifest]. A given leg can have excellent direction and sizable magnitude error; direction alone is insufficient to characterize force quality. Four observations per terrain/leg also limit how strongly to interpret an individual percentage.

### Foundation-model dimensions and serialized weight totals

Fresh safetensors headers and CPU checkpoint metadata establish the following **serialized tensor-element totals**. Foundation totals can include buffers and unused heads. They are not counts of active UGP trainable parameters, and WAFT's checkpoint already contains its Depth-Anything component, so adding the separate Depth-Anything checkpoint would double-count that component. [fresh server saved tensor metadata][v7-fresh-ckpt].

| Serialized foundation checkpoint | Tensor count | Tensor-element total | Verified shape / depth evidence |
| --- | --- | --- | --- |
| DINOv3 ViT-L/16 | 318 | 303,079,424 | 24 Transformer blocks; width1024; QKV3072×1024; FFN4096; task uses last feature map |
| DINOv3 ConvNeXt-L | 342 | 196,230,336 | Stage depths3/3/27/3; widths192/384/768/1536; depthwise7×7 kernels; MLP expansion4× |
| WAFT a1 tar-c-t | 604 | 64,154,559 | Entire serialized flow model, including pretrained feature components |
| Depth-Anything-V2 ViT-S checkpoint | 239 | 24,785,089 | Separate initializer/component checkpoint; not an additional V7 visual branch |
| Utonia pretrained PTv3 | 477 | 137,253,816 | Encoder depths3/3/3/12/3; widths54/108/216/432/576; embedding input9; no decoder tensors |

The DINOv3 HF config identifies `vit_large_patch16_dinov3`, `num_features=1024`, `num_classes=0`, and flexible input size. The task explicitly supplies448×448 crops. Its checkpoint shape index verifies24 blocks and FFN4096. The stored config/header does not encode the attention-head split, so the exact head count is not independently verified here. ConvNeXt stage depths and widths above come directly from its tensor keys and shapes.

The collected WAFT `tar-c-t.json` selects `dav2_backbone=vits`, `network_backbone=vits`, `algorithm=waft-a1`, five iterative updates, uncertainty range0–10 and `use_var=true`. The collected constructor shows the Depth-Anything-V2 Small feature model with features64 and output widths[48,96,192,384], a ResNet18 deconvolution feature pyramid with widths64/128/256/512, a ViT refinement branch with patch8 and intermediate indices[2,5,8,11], and six flow-head outputs: flow2 plus mixture weights2 and log-scale2. These are cache-extractor details; its dataset/batch/lr/200000-step fields describe the supplied pretrained WAFT training configuration, not UGP task training. [collected WAFT pretrained config][v7-waft-config], [collected WAFT a1 constructor][v7-waft-model], [collected WAFT refinement transformer][v7-waft-vit].

Utonia's worker explicitly overrides encoder patch sizes to[1024,1024,1024,1024,1024] and disables Flash Attention. It uses the official default transform with scale0.5, no enforced positive z and no coordinate normalization. Fresh state shapes verify encoder widths[54,108,216,432,576] and depths[3,3,3,12,3]. Each stage's saved RoPE inverse-frequency vector has length3; the collected RoPE formula and compatible channel divisibility imply head dimension18, hence heads[3,6,12,24,32]. This is a shape/source derivation rather than a quoted checkpoint-config field. MLP matrices expand widths4×. There are no saved decoder tensors. The geometry worker concatenates recovered stage features216+432+576=1224, then appends10 geometric/color channels, yielding the actual1234-wide cache. The source constructor's different default widths[48,96,192,384,512] must not replace these loaded-weight dimensions. [fresh server saved tensor metadata][v7-fresh-ckpt], [collected Utonia class and checkpoint loader][v7-utonia-model].

These frozen weights are loaded sequentially for cache extraction and are not resident trainable branches inside the saved task models. A sum of serialized checkpoint elements would neither establish simultaneous model memory use nor identify the exact number of active parameters in an end-to-end deployment.

### Interpretation supported by the artifacts

The expanded RTX diagnostic CSVs show lower force component MAE for V7-F at every horizon compared with the adapted V7 pipeline, while the copied V7 non-force rows retain their reported errors. The strict unseen CSVs establish a separate zero-shot result on four excluded trajectories, with18.7853 N vector error at+0.1 s and13.4419 N at+1.0 s. These results do not establish generalization to unseen terrain categories, a fully causal preprocessing protocol, real-time deployment performance, or a matched causal ablation of every training/adaptation choice. The final runner’s completed state confirms its23 cells finished; earlier logs provide the actual global training evidence and the latest log provides the gap/unseen execution evidence.

[v7-model]: <C:/path/to/ugp/notebook_02_live.txt:2062>

[v7-history]: <C:/path/to/ugp/notebook_02_live.txt:938>

[v7-config]: <C:/path/to/ugp/notebook_02_live.txt:516>

[v7-specialist]: <C:/path/to/ugp/notebook_02_live.txt:3185>

[v7-loss]: <C:/path/to/ugp/notebook_02_live.txt:2353>

[v7-floss]: <C:/path/to/ugp/notebook_02_live.txt:3412>

[v7-prep]: <C:/path/to/ugp/notebook_02_live.txt:709>

[v7-physics]: <C:/path/to/ugp/notebook_02_live.txt:1975>

[v7-cache]: <C:/path/to/ugp/notebook_02_live.txt:1371>

[v7-fit]: <C:/path/to/ugp/notebook_02_live.txt:2412>

[v7-ffit]: <C:/path/to/ugp/notebook_02_live.txt:3480>

[v7-gap]: <C:/path/to/ugp/notebook_02_live.txt:5356>

[v7-unseen]: <C:/path/to/ugp/notebook_02_live.txt:6034>

[v7-metrics]: <C:/path/to/ugp/notebook_02_live.txt:2577>

[v7-ckpt]: <C:/path/to/ugp/v7_checkpoint_metadata.json>

[v7-fresh-ckpt]: <C:/path/to/ugp/remote_model_metadata.json>

[v7-old]: <C:/path/to/ugp/notebook_04_V7_FULL_4_BRANCH_COLAB_PY313_FIXED (1).txt:4334>

[v7-latest]: <C:/path/to/ugp/V7F_4080_Run/storage/training_latest.log:523>

[v7-trainlog]: <C:/path/to/ugp/V7F_4080_Run/storage/training_20260928T135235Z.log:1282>

[v7-state]: <C:/path/to/ugp/V7F_4080_Run/storage/training_latest.state.json>

[v7-split]: <C:/path/to/ugp/V7F_4080_Run/storage/results/V7_EXPANDED_TRAIN_TRAJECTORIES.csv>

[v7-zero-cfg]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_unseen_trajectory_zero_shot_2GPU/CONFIG.json>

[v7-nb]: <C:/path/to/ugp/V7F_4080_Run/V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL.ipynb>

[v7-timing]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_unseen_trajectory_zero_shot_2GPU/TIMING.csv>

[v7-forcecomp]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_force_specialist/V7F_VS_V7_FORCE_BY_HORIZON.csv>

[v7-v7vec]: <C:/path/to/ugp/V7F_4080_Run/storage/results/V7_FORCE_VECTOR_METRICS_BY_HORIZON.csv>

[v7-fvec]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_force_specialist/V7F_FORCE_VECTOR_METRICS_BY_HORIZON.csv>

[v7-sens]: <C:/path/to/ugp/V7F_4080_Run/storage/results/V7_MEAN_METRICS_ACROSS_16_TESTS.csv>

[v7-accept]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_force_specialist/V7F_ACCEPTANCE_VS_BEST_EXISTING.csv>

[v7-contact]: <C:/path/to/ugp/V7F_4080_Run/storage/results/V7_CONTACT_METRICS_BY_HORIZON.csv>

[v7-regime]: <C:/path/to/ugp/V7F_4080_Run/storage/results/V7_4REGIME_CONFUSION_MATRIX.csv>

[v7-trend]: <C:/path/to/ugp/V7F_4080_Run/storage/results/V7_LONG_HORIZON_TREND_METRICS_MEAN.csv>

[v7-gap-audit]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_same_run_gap_sweep_2GPU/SAME_RUN_GAP_AUDIT.csv>

[v7-gap-result]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_same_run_gap_sweep_2GPU/SAME_RUN_GAP_SWEEP_OVERALL.csv>

[v7-gap-leg]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_same_run_gap_sweep_2GPU/SAME_RUN_GAP_SWEEP_BY_TERRAIN_LEG.csv>

[v7-zero-result]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_OVERALL.csv>

[v7-zero-leg]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_BY_TERRAIN_LEG.csv>

[v7-zero-test]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_BY_TEST.csv>

[v7-zero-manifest]: <C:/path/to/ugp/V7F_4080_Run/storage/V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TEST_MANIFEST.csv>

[v7-waft-config]: <C:/path/to/ugp/V7F_4080_Run/storage/.work/v7_external/WAFT/config/a1/tar-c-t.json>

[v7-waft-model]: <C:/path/to/ugp/V7F_4080_Run/storage/.work/v7_external/WAFT/model/waft_a1.py>

[v7-waft-vit]: <C:/path/to/ugp/V7F_4080_Run/storage/.work/v7_external/WAFT/model/backbone/vit.py>

[v7-utonia-model]: <C:/path/to/ugp/V7F_4080_Run/storage/.work/v7_external/Utonia/utonia/model.py:646>


---

<a id="v8-af-architecture-and-experiment-evidence"></a>

# V8 A–F: architecture and experiment evidence

This section is a read-only source/artifact audit. It did not execute training, inference, or tests. Notebook cell indices below are **zero based**; `L` numbers are one based within the indicated cell source. Code-derived dimensions and limitations are identified separately from persisted measurements. Source paths are absolute.

## Sources and execution scope

- **R**: `/path/to/ugp/local`. V8 cells 17–22 contain source but have null execution counts and no saved outputs. This file specifies the implementation; its filename alone does not prove execution.
- **R extract**: `/path/to/ugp/local`. Source references such as `C18:L105` are explicit searchable markers in this extraction.
- **S (deployed RTX source)**: `/path/to/ugp/local`. Whole-file SHA256 `4643a108aa89264fe4f569c879ab4b954b1a0f93fb56fbc0c70611d5fcfc1f31`. Its V8 code is in cells **16–21**; cells A–E have byte-identical joined source to R cells17–21. S cell21 fixes one curiosity-row ordering issue described below. S also has null notebook execution counts and no saved outputs, so execution is established by the separate persisted log/state and result files.
- **S extract**: `/path/to/ugp/local`. Markers use the actual deployed indices, e.g. `C21:L45`. The exact source-hash mapping appears in the RTX measurements section.
- **L (RTX execution log)**: `/path/to/ugp/local`. **State** is the adjacent `training_latest.state.json`.
- **I**: `/path/to/ugp/local`, snapshot UTC `2026-10-05T09:58:35.721819+00:00`. It records V8 model, recovery, and A–F result files on the RTX project disk at `/path/to/ugp/V7F_4080_Run/storage/V8_contactmap`, with September 28 timestamps. A fresh local copy of those metric files provides the RTX measurement section below.
- **T**: `/path/to/ugp/local(1) (3).ipynb`. Cell 2 output explicitly identifies `DEVICE: cuda Tesla T4`; cell 30 also explicitly identifies Tesla T4. Its saved V8 and V9 results are **Colab/T4 evidence**. They must not be substituted for the RTX measurements. The notebook restores the frozen V7 short prior, then executes V8 cells 15/18/21/24/26/28; cell 30 snapshots the resulting live objects.
- **T ZIPs**: `/path/to/ugp/local` and `/path/to/ugp/local (1).zip` contain `V8_contactmap_standalone` results. Both include A–E metrics/models but omit `F_active_adaptation` result files despite the A–F README. They are not the RTX archive inventoried in I.

## Dataset, targets, and evaluation protocol

The V8 loops use `run_records`, which R cell 13 L147–155 deliberately defines as the **four original diagnostic trajectories**, not all 28 V7 training trajectories and not the four strict trajectory holdouts. R cell 5 L237–240 specifies:

| Terrain label | Environment/run | Diagnostic times (s) |
|---|---|---|
| forest | ForestEnv/P2000 | 28, 45, 75, 90 |
| swamp_marsh | GreatMarsh/P2023 | 47, 61, 73, 87 |
| rough_uneven | SeasonalForestWinter/P2000 | 29, 46, 64, 92 |
| wet_uneven_proxy | OldTownFall/P2000 | 32, 84, 100, 115 |

Thus “16 tests” denotes four later-time diagnostics on each of four known trajectories. B/C training uses only early windows before each trajectory’s first diagnostic, with future-target guards. Their validation takes the last 15% of those early source times within each run. V8 does not inherit the separate strict-holdout zero-shot experiment merely by being appended to that notebook.

R cell 6 L83–87/L124–129/L162–166/L219 gives target provenance: joint position/velocity/effort come from `/state_estimator/joint_states`; force vectors come from `/state_estimator/contact_force_{lf,rf,lh,rh}_foot`. Per-leg `P12 = [q3,dq3,tau3,F3]`. Body `B12 = [acc3,gyro3,body_twist6]`; command `U6` comes from `/path_planning_and_following/twist`. **Contact is a derived pseudo label**, `1[||F||₂ > 10 N]` (threshold R cell 5 L169), not independently annotated physical touchdown. Future torque/force targets are logged estimator channels, not newly measured intervention outcomes.

Foot positions are another offline proxy: ANYmal-D forward kinematics of logged joint angles, a y-axis sign flip, and the dataset camera pose. Camera optical center is used as the robot origin, with `V8_BASE_IN_CAM_HEADING=[0,0,0]`; no verified camera-to-base translation is applied. R cell 17 L12–17/L49–87 explicitly labels this limitation. A/B/C/D/E/F spatial claims inherit it.

## A — persistent spatial memory (no new learned model)

**Representation.** A reuses the trained short-prior V7 DINOv3, ConvNeXt, Utonia, and WAFT branch adapters. The 64 metric Utonia anchors per frame are projected to the matching 8×8 image/motion grid. The point-specific geometry feature and matched image/motion features are averaged, then L2 normalized:

`z_cell = normalize((z_DINO[g] + z_CONV[g] + z_GEOM[k] + z_FLOW[g])/4)`, `z_cell ∈ R^256`.

Adapters include the learned V7 modality embeddings. This A operation is analytic spatial association and storage of frozen learned representations; there is no A optimizer or A-specific training loss. Source: R cell 17 L107–119/L144–173.

Each event stores XY, height, timestamp, float16 256-D token, world normal3, roughness1, confidence1, frame/grid identity, and integer XY cell. Cell resolution is 0.05 m; extraction batch24; confidence threshold0.02. Confidence averages the DINO/Conv validity channels and a finite-flow proxy. Source: R cell 17 L42–47/L162–190.

**Retrieval.** A first fetches at most 48 spatial nearest neighbors within0.18 m, then keeps events with `event_time ≤ query_time+1e−8`. It selects the smallest `age + 1.5·distance`. This returns one event, not a learned Bayesian terrain posterior. Because spatial KNN occurs before the time filter, a causal event outside the nearest48 may be missed. Source: R cell 17 L196–218.

**Metric.** At 10 Hz, only legs labeled in contact are queried at their contemporaneous FK/camera foot proxy. Memory hit is the fraction with a returned event; median distance and age refer to successful hits. `seen_0p5/1p0/2p0s_earlier` asks whether the **selected** event is at least that old, counting misses as false. It does not establish that every footprint was observed at those leads, nor search existentially over all older observations. Source: R cell 17 L232–256.

## B — learned future per-leg contact/position distribution

**Input/architecture.** `x ∈ R^(batch×4×T×30)` uses0.5 s causal history per leg: P12+U6+B12; at100 Hz `T=50`. Each leg is independently processed with shared weights: input30→128; learned position embedding capacity128×128; three causal Transformer layers, model128, four attention heads, FF512, GELU, dropout0.1, pre-norm; final time-token LayerNorm; learned leg embedding4×128 added after temporal encoding. There is no cross-leg attention in this B transformer. Horizon embedding MLP1→64→128 plus LayerNorm. Concatenate leg128+horizon128; fusion LayerNorm256→Linear256→256→GELU→dropout0.1→Linear256→128→GELU. Three heads128→1 contact logit,128→3 position mean,128→3 log standard deviation clamped[−4,1.5]. Source: R cell 18 L44–51/L105–124.

Horizons are0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0 s. Output shapes are contact `(batch,8,4)`, mean/sigma `(batch,8,4,3)`:224 scalars per example if all logits, means, and log standard deviations are counted. XYZ is expressed in the **current** camera-heading frame. Future logged FK+camera-pose feet supply position targets; thresholded future force supplies contact targets. B predicts occupancy at sampled horizons, not a continuous-time independent touchdown-event hazard. Source: R cell 18 L30/L54–66/L121–124.

**Loss.** With contact label `c`, position `y`, mean `μ`, logstd `ℓ`, variance `exp(2ℓ)`:

`L_B = BCEWithLogits(logit,c; positive_weight=1.6) + 0.75·[Σ c·Σ_xyz 0.5((y−μ)^2/exp(2ℓ)+2ℓ)/(Σc+1e−6)] + 0.20·SmoothL1(μ,y;β=0.08)`.

The contact-masked Gaussian position term omits the constant log(2π). Weak position tracking also supervises swing. Negative composite values can occur and are not evidence of zero prediction error. Source: R cell 18 L127–137.

**Training.** Window hop0.10 s; start0.5 s; source cutoff`first_test−2.0−0.05 s`; standardize all 30 input channels with mean/std from the full early-window set (floor std1e−4); temporal85/15 split; train batch32/validation64;18 epochs; AdamW lr3e−4, weight decay1e−4; gradient norm clip1; select lowest validation loss; seed`SEED+8200` (9537 if SEED1337). Checkpoint after each epoch includes optimizer/RNG/best state and compatibility signature. Normalization statistics are computed **before** the validation split. Source: R cell 18 L30–41/L69–90/L139–178.

**Metrics.** Contact precision/recall/F1 threshold probability≥0.5. `contact_position_MAE_m` is **mean Euclidean XYZ error on true-contact rows**, not per-coordinate MAE; Recall@5/10/20cm are fractions within those Euclidean radii; sigma is mean predicted coordinate sigma on true-contact rows. Timing is the difference between the first sampled horizon whose contact label/probability crosses0.5; unavailable true/predicted contacts are NaN and excluded from aggregate median. Thus a zero timing median alone does not show accurate continuous touchdown timing. Source: R cell 18 L190–220.

## C — learned local contact response, conditional on oracle location

**Input297.** Terrain token256; geometry7=`height1+world_normal3+roughness1+confidence1+capped_age/5`; current state34=`q/dq/tau9+command6+body12+FK_position3+Jacobian_velocity3+horizon1`. Current measured force is excluded from C input. Numerical Jacobian uses forward differences eps1e−4. Source: R cell 19 L41–76.

**MoE.** Shared trunk LayerNorm297→Linear297→512→GELU→dropout0.1→Linear512→256→GELU→LayerNorm256. Gate256→128→GELU→4 logits→softmax. Four experts each256→256→GELU→12 outputs, split six `[tau3,F3]` means and six logstd clamped[−4,2]. The returned mean is the mixture-weighted expert mean; returned variance follows total variance, `Σπ(exp(2ℓ)+μ²)−(Σπμ)²`, floored1e−6. Source: R cell 19 L118–131.

**Loss.** In standardized six-target units, use true diagonal-Gaussian mixture NLL, including log(2π), plus mean-response SmoothL1 penalties and average-gate balance:

`L_C = −mean logΣ_e π_e N(y;μ_e,diag(exp(2ℓ_e))) + 0.25·SmoothL1(μmix_tau,y_tau;β=.4) + 0.65·SmoothL1(μmix_F,y_F;β=.4) + 0.10·mean_e(πbar_e−1/4)^2`.

Source: R cell 19 L134–145.

**Protocol.** Horizons0.1/0.3/0.5/0.7/1.0 s; source-hop0.15 s; start0.6 s; cutoff`first_test−1.0−0.05 s`. For each source time/leg/horizon, the **actual future** FK+camera-pose foot XY selects a memory event while retrieval time remains the source time. Missing-memory samples are skipped. Targets are future logged `[tau3,F3]`. Therefore C’s metrics are an **oracle-location conditional law test**: terrain observations are causal, but query location is retrospectively known. They do not measure a fully causal end-to-end predicted-foot placement system. Source: R cell 19 L79–103/L199–217; T cell20 explicitly names this oracle-location interpretation.

Inputs/targets are standardized from all early C samples before validation; floors input1e−4/target1e−3. Temporal last15% validation; train batch128/validation256;22 epochs; AdamW lr3e−4, wd1e−4; clip1; lowest validation loss; seed`SEED+8300` (9637 if1337); epoch recovery. Tau/force MAE is mean absolute component error over three components, then mean over query rows; force-vector error is mean Euclidean vector error. This metric includes each memory-supported leg query, not only contact-positive targets. Source: R cell 19 L27–38/L101–114/L147–183/L213–221.

## D — learned matched-pair terrain-response difference

**Pair construction.** Same leg and same horizon; nearest cross-terrain match on standardized C **non-token41 features**, which include geometry7 as well as robot/command/horizon34. At most350 source samples per terrain/leg/horizon group; retain distances≤55th percentile (the comment says “worst half,” while the code retains55%); max12000 pairs. `DY=observed_response_b−observed_response_a`. These are differences between separate observed states/terrains, not intervention ground truth at an exactly identical state. Source: R cell 20 L36–90.

**Descriptor1321.** Concatenate `[z_a,z_b,z_b−z_a,abs(z_b−z_a),z_a*z_b]` =1280 dimensions and average standardized non-token state41. MLP LayerNorm1321→768→GELU→dropout0.1→384→GELU→dropout0.1→192→GELU; output192→12, split six delta means and six logstd clamped[−4,2]. Source: R cell 20 L83–109.

**Loss/training.** Normalize DY using all pairs, floor std1e−3. `L_D=mean[0.5((DYnorm−μ)/expℓ)^2+ℓ]+0.2 SmoothL1(delta_tau;β=.4)+0.6 SmoothL1(delta_F;β=.4)`. Random80/20 **pair** split, not disjoint-source/run split; paired source examples can therefore be shared across train/validation.20 epochs, train128/validation256, AdamW3e−4/wd1e−4, clip1, best validation checkpoint; seed`SEED+8400` (9737 if1337). Source: R cell 20 L91–145.

D evaluation reports delta torque/force component MAE, Euclidean delta-force error, and a sign-ranking accuracy comparing whether `||F_a+predicted_delta||−||F_a||` has the same sign as `||F_b||−||F_a||`. It supplies logged F_a to evaluate ranking. The resulting statistic is neither executed alternative-terrain success nor a strict unseen-trajectory result. Source: R cell 20 L156–170.

## E — analytic micro-foothold shadow planner

No learned E weights. Pick B’s first horizon in[0.4,1.5]s with contact probability≥0.5, else the largest probability in that interval. Generate0.05 m offsets inside nominal0.22 m radius; retrieve causal A events, deduplicate cells, filter slope≤32° and roughness≤0.60, and test approximate Jacobian reachability: pseudoinverse rcond1e−3, max joint displacement≤0.48 rad and norm≤0.75. Candidate position is the retrieved anchor XYZ, not necessarily the originally sampled grid point. Source: R cell 21 L28–52/L65–100.

C predicts candidate response. D refines nominal response to a candidate terrain, blended75% direct C+25% nominal-plus-D; sigma combines direct variance and0.25² delta variance. Engineering risk score is:

`||F||/300 + .45||tau||/30 + .40||sigma_F||/80 + .30 slope_deg/35 + .25 rough + .20 offset/.22 + .10 min(age,5)/5`.

Keep top4 candidates per leg; jointly enumerate top3 for legs with any candidate. Add0.35 crowding penalty for pair distances<0.14 m and0.35 support-area penalty below0.035 m² (only with≥3 active legs). Report nominal independent score minus candidate joint score. Source: R cell 21 L35–36/L55–62/L100–118/L138–168.

This is a counterfactual **predicted risk** comparison. It does not command a robot, contain alternative-foothold ground truth, or establish navigation success. Partial active-leg plans can be reported. The reachability check is a local linear displacement approximation, not full IK, collision checking, verified controller feasibility, or dynamic stability proof. Source: R cell 21 L11–13/L138–156.

## F — bounded residual memory and curiosity overlay

No backbone/contact-field weights are trained online. A new sparse residual memory is initialized for each diagnostic trajectory. Future0.1s contact-positive targets are replayed at0.10s source hop; select causal memory via actual future foot XY, predict C, add prior local residual correction, score, then update residual`logged_target−base_prediction`. This retains C's oracle future-location condition. Source: **S cell21 L25–33/L79–103**.

Residual lookup radius0.30m; kernel scale0.12m; token cosine≥0.45; weight`exp(−.5(d/.12)^2)·max(cos,0)^2·exp(−age/12)`; discard weights≤1e−5. Weighted mean correction is norm-capped8Nm torque/40N force. Each cell keeps running mean, count and M2; token EMA0.9/0.1; capacity800; evict least-recently updated cell. M2 is stored but is not used to form a posterior variance in this implementation. Source: **S cell21 L41–74**.

Curve buckets count prior contacts in compatible neighboring cells (`0`,`1`,`2–4`,`5+`). Component MAEs are compared before/after correction; improvement percent is100·(baseMAE−adaptMAE)/baseMAE. The printed “negative-transfer rate” uses `not improved_force`, so equal/no-change predictions count in that rate. Replay spans the whole four diagnostic trajectories, including early training times; it is not a separately held-out online deployment test. Source: **S cell21 L108–118/L144**.

**Causality limit verified in the deployed RTX source:** within one source time, each leg’s residual is updated at future target time `t+0.1` immediately before the next leg is scored at source time `t`. Residual lookup does not filter `last_t≤t`; its age simply clamps negative time differences to0. Consequently earlier legs’ future residuals may be available to later legs if spatial/token gates match. “Strict causal replay” is the intended semantics in the comments, but this exact loop lacks an explicit pending-update queue or time eligibility check. This is a code-derived limitation, not a claim that every row actually leaked. Source: **S cell21 L45–54/L83–103**.

For shadow curiosity, info gain proxy is`.5 log(1+(unc_norm/15)^2)·(1+min(age,5)/5)`. Candidate safety gate`score≤1.12·safest_score+.02`; minimize`score−.10·info_gain`. It operates on E candidates and represents hypothetical exploratory touches. Confidence is mentioned in the comment but is not directly present in this formula. **S cell21 L124–127 selects the `safest` row after assigning `info_gain`; R cell22 selected it beforehand.** This is the only F source difference, allowing the deployed saved summary to read `safest.info_gain`. The replay causality limit is unchanged. Source: **S cell21 L120–135**; exact diff: `/path/to/ugp/local`.

## RTX persisted measurements

All values below come from the freshly collected **RTX project artifacts**. Define **Q** as the absolute local result directory:

`/path/to/ugp/local`.

CSV paths below resolve under Q. Rounded display values preserve the full-precision originals in those files. Descriptive aggregates computed from the persisted CSVs are saved in `/path/to/ugp/local`; no model code was executed to create them.

### Source identity and recorded execution

These hashes are SHA256 of `''.join(cell['source']).encode('utf-8')`, including stored source newlines, without normalization:

| Stage | S deployed cell | Matching R cell | Exact deployed source SHA256 |
|---|---:|---:|---|
| A | 16 | 17 | `fc1ac7659f632ee8ee4ca1752183c495dd9cf7fc437c1bc4f16afaabc5f33980` |
| B | 17 | 18 | `a821e99cff82622a8903022a2849062d98da421a4d381603b6bbdeae59caa840` |
| C | 18 | 19 | `b7433fa50c69ac59e6f0b82086086b67f4709bb2a3449a3f548403722f954903` |
| D | 19 | 20 | `9cb79e9bfdc2ebeae88e6e5da4d489d8883b1e0d3fe2e2ba87274acc53af519c` |
| E | 20 | 21 | `0c4710e0c018ee7419791deb6daf4033f4c8a7f4964f14be0b878f07df158d61` |
| F | 21 | R22 differs only in curiosity-row ordering | `c39ee9775eedbff3eee6838d02287fea8502da80064dbcbae0ad796fa30abfa8` |

L lines11–12/73–77 explicitly identify **two NVIDIA GeForce RTX4080 devices** and `DEVICE: cuda:0`. Its A–F block, lines795–898, records deployed cells16–21 complete on September28; F completed before cell22 started at **2026-09-28T15:58:31.595432Z**. The state records notebook PID61345, start **15:56:44.249742Z**, `status="complete"`, completed cells1–23, and completion **16:45:46.280268Z**. These are historical execution records; the audit did not check current PID liveness. Cells22–23 belong to later V7/V7F work and do not change the V8 diagnostic sample protocol.

The latest log reuses all four A event stores and reports B **1260** training windows, C **16897** early memory-supported samples, and D **9294** constructed pairs. It prints `Ignoring unreadable V8-{B,C,D} recovery: RNG state must be a torch.ByteTensor` and contains no B/C/D epoch-loss lines in this execution. In S, model/optimizer weights, best state, and completed-epoch counters are assigned **before** RNG restoration; the exception does not reset those assignments, and each cell subsequently loads its nonempty best state. Thus this log documents reuse and result generation from existing recovery artifacts, not fresh B/C/D training curves. The18/22/20 epochs above are the configured training lengths; this collected log alone does not establish the earlier convergence history. Source: S C17:L156–178, C18:L162–183, C19:L125–145; L815–861.

Q's `V8_SHA256_MANIFEST.csv` lists27 members. All **19 copied non-self members**—metric/raw CSVs, four A metadata files, and README—match its stored SHA256 and size. The seven large event/model files are inventoried but were not copied into this snapshot. The manifest's **own** row differs: S C21:L154–158 hashes the existing manifest before overwriting it, so the row describes the previous manifest. That self-entry is not a current-file integrity check. None of the measured CSVs has a hash mismatch.

### Learned-module sizes

The learned counts below were calculated from the exact deployed source, including Linear biases, attention biases, LayerNorm affine parameters, and learned embeddings, and **confirmed against static metadata from the RTX exported checkpoints**. Metadata is saved at `/path/to/ugp/local`; its recorded method is an inert static pickle metadata parser with no saved globals imported or executed. B stores an additional8-element `horizons` buffer (S C17:L108), so its total state element count is724751 while its learned parameter count is724743. C and D have no additional buffers in the extracted state.

| Stage | New learned parameters | Breakdown | Source |
|---|---:|---|---|
| A | 0 | Frozen V7 learned adapters reused; event storage adds no optimizer parameters | S C16:L107–119/L144–173 |
| B | **724743** | Input3968; leg512; position16384; three Transformer layers594816; final norm256; horizon8704; fusion99200; heads903. Checkpoint61 tensors/724751 state elements, including horizon buffer8 | S C17:L105–124; Q/B_touchdown/V8B_TOUCHDOWN_MODEL.pt metadata |
| C | **593926** | Trunk285010; gate33412; four experts275504. Checkpoint28 tensors/593926 state elements | S C18:L118–131; Q/C_contact_field/V8C_CONTACT_FIELD.pt metadata |
| D | **1389470** | Input norm2642; three MLP layers1384512; output2316. Checkpoint10 tensors/1389470 state elements | S C19:L103–109; Q/D_counterfactual/V8D_SIAMESE_DELTA.pt metadata |
| E | 0 | Analytic enumeration/scoring uses B/C/D | S C20:L44–168 |
| F | 0 | Sparse residual statistics and analytic curiosity; no online neural weight update | S C21:L41–74/L120–135 |

B+C+D add **2708139** learned parameters, excluding the reused V7 prior and frozen feature backbones.

### A — recorded contact-memory coverage

Source: Q/`A_memory/V8A_CAUSAL_MEMORY_CONTACT_COVERAGE_SUMMARY.csv`; event counts come from the four adjacent `*_V8A_MEMORY_META.json` files. Lead percentages count all contact queries, including misses; age/distance medians include successful retrievals only.

| Terrain | Stored events | Contact queries | Hit rate | Median distance (m) | Median age (s) | Selected event ≥0.5s old | ≥1s old | ≥2s old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| forest | 80064 | 2501 | 97.761% | 0.1174 | 0.4600 | 48.061% | 5.718% | 0.160% |
| rough_uneven | 82560 | 2670 | 99.663% | 0.1166 | 0.6200 | 51.873% | 17.079% | 1.199% |
| swamp_marsh | 82688 | 2634 | 98.823% | 0.1162 | 0.4300 | 49.279% | 7.175% | 1.822% |
| wet_uneven_proxy | 80064 | 2617 | 99.083% | 0.1438 | 0.5400 | 49.675% | 49.217% | 1.070% |
| **All, raw-row aggregate** | **325376** | **10422** | **98.849%** | **0.1263** | **0.5200** | **49.751%** | **19.919%** | **1.075%** |

The aggregate is **10302 hits/10422 queries**, with ≥0.2s selected-event age on63.280% of all queries. These are whole-trajectory contact-coverage rows, not just the16 diagnostic instants. They show frequent nearby causal retrieval under the camera-origin proxy; they do not validate hardware alignment or long-lead footprint coverage.

### B — future contact and foot-position diagnostics

Source: Q/`B_touchdown/V8B_TOUCHDOWN_METRICS_BY_HORIZON.csv`. Each horizon contains64 rows =16 diagnostic instants×4 legs. Position error and radius recalls use only true-contact rows; probabilities use the0.5 contact threshold. All recall/precision entries below are percentages except F1.

| Horizon (s) | Precision | Recall | F1 | Contact position Euclidean error (m) | Recall≤5cm | Recall≤10cm | Recall≤20cm | Mean coordinate sigma (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 53.125% | 100% | 0.6939 | 0.0770 | 41.176% | 70.588% | 100.000% | 0.0534 |
| 0.4 | 50.000% | 100% | 0.6667 | 0.0796 | 50.000% | 75.000% | 87.500% | 0.0529 |
| 0.6 | 51.563% | 100% | 0.6804 | 0.1184 | 6.061% | 36.364% | 90.909% | 0.0548 |
| 0.8 | 51.563% | 100% | 0.6804 | 0.1137 | 30.303% | 60.606% | 84.848% | 0.0607 |
| 1.0 | 50.000% | 100% | 0.6667 | 0.1398 | 15.625% | 46.875% | 87.500% | 0.0677 |
| 1.2 | 51.563% | 100% | 0.6804 | 0.1077 | 24.242% | 60.606% | 90.909% | 0.0774 |
| 1.5 | 50.000% | 100% | 0.6667 | 0.1607 | 15.625% | 28.125% | 71.875% | 0.1003 |
| 2.0 | 50.000% | 100% | 0.6667 | 0.1900 | 12.500% | 37.500% | 68.750% | 0.1382 |

Raw verification from `V8B_16TEST_TOUCHDOWN_RAW.csv`: **all512/512 predictions are contact-positive**, with probabilities0.583327–0.657658, against261 true-contact rows. Thus100% recall comes from predicting contact at every sampled horizon, and precision equals the true-contact prevalence. The contact classifier does not separate swing from contact on these diagnostics. The table still measures its predicted positions on the true-contact subset; no empirical calibration/coverage result is supplied for sigma.

Source: Q/`B_touchdown/V8B_CONTACT_TIMING.csv`. All64 leg/test cases predict their first contact at0.2s. True first sampled contact is0.2s in34 cases,0.4s in23, and0.6s in7. Consequently median absolute timing error is **0.000s**, mean **0.115625s**, maximum **0.400s**; errors are0s/0.2s/0.4s in34/23/7 cases. The median reflects this discrete sample distribution and the all-positive predictor, not demonstrated continuous touchdown-event accuracy.

### C — oracle-location conditional contact-response diagnostics

Source: Q/`C_contact_field/V8C_CONTACT_FIELD_METRICS_BY_HORIZON.csv`. All320 possible diagnostic leg/horizon queries had a memory-supported row (64 per horizon,80 per terrain). Tau and force MAEs average absolute component errors; vector error averages Euclidean force error. Units below are Nm/N. Mean predicted force sigma is an output spread, with no calibration check in this artifact.

| Horizon (s) | Query rows | Torque component MAE (Nm) | Force component MAE (N) | Force vector error (N) | Mean force sigma (N) |
|---|---:|---:|---:|---:|---:|
| 0.1 | 64 | 14.3571 | 47.6417 | 127.8810 | 51.0246 |
| 0.3 | 64 | 14.8881 | 44.6182 | 116.4237 | 50.6599 |
| 0.5 | 64 | 15.6128 | 48.8508 | 130.0533 | 50.8259 |
| 0.7 | 64 | 14.9428 | 52.5853 | 140.1000 | 51.2141 |
| 1.0 | 64 | 14.7977 | 47.0160 | 123.7878 | 51.4398 |
| **All320 raw rows** | **320** | **14.9197** | **48.1424** | **127.6492** | **51.0328** |

These results condition on **actual future logged foot XY** to choose terrain memory, while restricting terrain event time to the source instant. They establish neither a causal predicted-location pipeline nor an unseen-trajectory result. Actual deployed protocol: S C18:L79–103/L199–217.

### D — held-out matched-pair diagnostics

Source: Q/`D_counterfactual/V8D_MATCHED_PAIR_SUMMARY.csv`, verified against1859 raw rows in `V8D_MATCHED_PAIR_RAW.csv`.

| Constructed pairs | Random pair train split | Validation pairs | Median validation state distance | Delta torque component MAE (Nm) | Delta force component MAE (N) | Delta force vector error (N) | Force-magnitude sign ranking accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9294 | 7435 | 1859 | 2.7232 | 6.5855 | 20.5046 | 46.9867 | **36.686%** |

The9294-pair construction median distance was2.7157 (L853); the table's2.7232 belongs to validation only. Distances are in standardized41-feature units, not metres. Random pair splitting can share originating C samples, trajectories, or times across train/validation. Ranking supplies logged baseline force and checks the sign of a predicted magnitude change; the measured36.686% does not demonstrate reliable alternative-terrain ranking. Exact deployed matching/split/evaluation source: S C19:L36–101/L156–170.

### E — shadow-plan coverage and predicted score changes

Source: Q/`E_microplanner/V8E_16TEST_SHADOW_PLANS.csv`, with candidate rows in `V8E_ALL_CANDIDATES_TOPK.csv`. Score reductions are unitless engineering-score differences, `nominal_independent_score−joint_score`; positive means a lower predicted candidate score. Joint penalties are included in the candidate score, so this comparison is not a measured physical risk reduction.

| Terrain | Plans/diagnostic instants | Four-leg plans | Positive predicted reductions | Mean reduction | Median reduction | Mean support area (m²) |
|---|---:|---:|---:|---:|---:|---:|
| forest | 4/4 | 1 | 2 | −0.0523 | 0.0103 | 0.1243 |
| swamp_marsh | 3/4 | 0 | 3 | 0.5168 | 0.6540 | 0.0518 |
| rough_uneven | 4/4 | 2 | 4 | 0.4185 | 0.3779 | 0.1255 |
| wet_uneven_proxy | 4/4 | 4 | 2 | −0.0502 | −0.0179 | 0.2574 |
| **All** | **15/16** | **7** | **11** | **0.1876** | **0.1375** | **0.1456** |

The omitted diagnostic is swamp_marsh test4 at87s. Plan active-leg counts are1/2/3/4 in1/3/4/7 plans; support area is zero for fewer than3 active legs. All114 stored top candidates, covering47 leg/test groups, use0.4s horizons because B exceeds the contact threshold at the first eligible horizon. Mean nominal independent score is5.4343; mean joint score is5.2467.

Although the search-grid offsets are sampled within0.22m, the retrieved anchor becomes the actual candidate position. **19/114 saved candidates have actual offsets>0.22m**, with maximum **0.32897m**; source S C20:L89–100 applies no final offset-radius rejection after anchor retrieval. The code's0.22m setting therefore bounds the query grid, not every returned anchor. E remains an offline shadow planner: no executed alternative footholds, controller success, or navigation improvement was measured.

### F — residual replay and curiosity

Source: Q/`F_active_adaptation/V8F_ADAPTATION_CURVE.csv`. Rates are strict per-row error reductions, while improvement percentages compare mean errors within each bucket.

| Prior compatible contacts | Rows | Base force MAE (N) | Adapted force MAE (N) | Rows with lower force error | Mean force improvement |
|---|---:|---:|---:|---:|---:|
| 0 | 941 | 48.9420 | 48.9420 | 0.000% | 0.000% |
| 1 | 1108 | 47.2035 | 38.7547 | 89.711% | 17.899% |
| 2–4 | 3588 | 53.8204 | 43.2307 | 91.667% | 19.676% |
| 5+ | 4661 | 47.8845 | 37.2553 | 91.783% | 22.198% |

| Prior compatible contacts | Rows | Base torque MAE (Nm) | Adapted torque MAE (Nm) | Rows with lower torque error | Mean torque improvement |
|---|---:|---:|---:|---:|---:|
| 0 | 941 | 15.7248 | 15.7248 | 0.000% | 0.000% |
| 1 | 1108 | 15.4044 | 13.4881 | 84.657% | 12.440% |
| 2–4 | 3588 | 15.1879 | 14.5930 | 65.301% | 3.917% |
| 5+ | 4661 | 15.6999 | 16.7833 | 31.882% | **−6.900%** |

Raw-row aggregates from Q/`F_active_adaptation/V8F_CAUSAL_CONTACT_REPLAY_RAW.csv`:

| Terrain | Replay rows | Base→adapted force MAE (N) | Force improvement | Base→adapted torque MAE (Nm) | Torque improvement |
|---|---:|---:|---:|---:|---:|
| forest | 2442 | 54.8806→46.8672 | 14.601% | 16.5365→16.6709 | −0.813% |
| swamp_marsh | 2603 | 51.8579→42.8531 | 17.364% | 16.2074→16.2981 | −0.560% |
| rough_uneven | 2660 | 47.2848→38.1884 | 19.237% | 15.4251→15.2751 | 0.973% |
| wet_uneven_proxy | 2593 | 46.2287→34.7765 | 24.773% | 13.8587→14.1004 | −1.744% |
| **All** | **10298** | **49.9760→40.5664** | **18.828%** | **15.4920→15.5689** | **−0.496%** |

Force error strictly improves on83.133% of all rows. The printed16.867% “negative-transfer rate” combines **796 actually worsened rows** and **941 equal/no-correction rows**; the actual worsening fraction is7.730%. Torque strictly improves on46.291%, worsens on4590 rows, and is equal on941. The5+ bucket's torque error increases despite its lower mean force error. These are whole-trajectory, oracle-location replay measurements with the **deployed per-leg future-update/time-gate limitation described above**. They must not be presented as verified strict-causal adaptation gains or independent online deployment evidence.

Source: Q/`F_active_adaptation/V8F_SAFE_CURIOSITY_SHADOW.csv`. There are **47** candidate groups; the overlay differs from the safest candidate in **3/47 =6.383%**. Mean extra information proxy across all groups is **0.011922** and mean extra engineering score is0.000614. Changes occur at swamp_marsh test2 leg1, wet_uneven_proxy test1 leg2, and wet_uneven_proxy test4 leg2. All are hypothetical selections over E's stored candidates; there are no executed exploratory touches or measured information gains.

## Scope boundary

This section reports the collected RTX V8 A–F artifacts. T4-only V9/FORCE-5 results are excluded.


---

<a id="feelwm--tartanground-architecture-audit"></a>

# FeelWM / TartanGround architecture and training evidence

This section describes the actual `merged_feelwm_v7` source collected from the server on **2026-10-05 at 10:18:27 UTC**, its packaged pilot predecessors, and the separate force finetuning copy. It covers the model, preparation, recurrent training, and checkpoint selection. Run completion and numerical test results belong in the accompanying run ledger. This architecture section was assembled from the saved source and metadata using static parameter arithmetic; preparing it did not require training, model evaluation, model changes, or tests.

The central model consumes cached V7 visual features resampled into four blocks of `16 × 256` tokens per frame. It expands four frames into nine FeelWM groups, processes **576 tokens of width 800** with an **eight-layer, ten-head** world transformer, and decodes visual state, joint/body state, torque, contact, load, and refined force. The per-leg variant adds four independent transformers before the shared world transformer. Its total is **141,380,144 parameters**, compared with **110,526,384** for the baseline. Both contain the **2,448,554-parameter V7-F force specialist**.

## 1. Source provenance and what can be attributed to each version

The [collection manifest](C:/path/to/ugp/evidence_manifest.json:1) records matching remote and downloaded hashes. The fresh [checkpoint metadata](C:/path/to/ugp/feelwm_checkpoint_metadata.json:1) records model tensor shapes, counts, configuration, optimizer groups, and counters without a training run.

| Source/version | Established source behavior | Attribution limit |
|---|---|---|
| `work/review-baseline/merged_feelwm_v7` | Original shared world transformer; no leg pretransformers; later horizons consume the prepared per-horizon query, physics, and residual baselines; base V7 force enters recurrent history; V7-F runs once after the rollout. | A precursor implementation, not evidence that the later pilot or ten-epoch runs retained its future-state inputs. |
| Packaged `tartanground_p2000_k1_results_step200/source` | Same original model design, with an added `forward()` wrapper; the loss also supervises the base magnitude/direction auxiliary head. | A K=1 experiment; its model's possible K>1 behavior was not exercised by K=1. |
| Packaged `tartanground_k1_k5_leg_compare/source` | Leg pretransformers, predicted recurrent query/physics/baselines, teacher forcing, causal specialist prefix feedback, and the current horizon-weighted loss. | Captures packaged source. It does not independently record an immutable source hash at every optimizer update. |
| Fresh `/path/to/ugp/merged_feelwm_v7` | Its model, loss, trainer, base configuration, indexed dataset, rollout schedule, and preparation script match the local known source byte for byte. Those seven files also match the pilot package in text after normalizing CRLF line endings. | Current deployed source is confirmed; historical source-at-every-update provenance still requires contemporaneous source records. |
| Fresh `/path/to/ugp/merged_feelwm_v7_force_finetune` | Model, loss, indexed dataset, and preparation are identical to the main deployed copy. Trainer/configuration/teacher-forcing schedule add specialist-only updates, a force selection metric, full-development validation, and forcing annealing. | The collected phase file records waiting for the comparison launcher; this is configured continuation work, not a completed force finetuning result. |

The deployed model SHA-256 is `05cffa195c77873619c48e20d2d0a3e9a39fbbfb75999aff0d81bf4f1234e7cc`. The pilot archive's raw-byte hash differs because its file uses CRLF; its normalized-text hash is the same. The deployed loss SHA-256 is `66591a6a9c0ed07d2206c8c47173d2c20121f60358ee680a0fe323360a9399a5`; it also matches the local known source and the archive's normalized text. The earlier P2000 and review-baseline models differ from this model.

The old recurrent behavior is directly visible in [review-baseline model.py:496](C:/path/to/ugp/merged_feelwm_v7/model.py:496): every horizon reads prepared query/physics/baselines, feedback uses `base_force`, and V7-F runs after K steps. The current behavior is directly visible in [deployed model.py:618](C:/path/to/ugp/merged_feelwm_v7/model.py:618) and [model.py:660](C:/path/to/ugp/merged_feelwm_v7/model.py:660). The P2000 wrapper is in [archived model.py:484](C:/path/to/ugp/model.py:484), and its auxiliary supervision begins at [archived losses.py:44](C:/path/to/ugp/losses.py:44).

## 2. Cached feature preparation and input tensors

FeelWM training starts from precomputed V7 DINOv3, ConvNeXt, Utonia geometry, WAFT flow, and physics caches. Its indexed preparation script does not rerun these pretrained backbones from raw RGB-D or optimize their weights. It loads only the V7 visual resampler submodules from `results/V7_PRIOR_SHORT.pt`, checks missing/unexpected keys, and sets them to evaluation mode with all gradients disabled. It does not initialize the entire merged FeelWM model from that checkpoint. K=1 comparison configurations have no `init_weights_only` source; their merged-model parameters start from the seeded initialization.

| Preparation branch | Cached input to the frozen preparation modules | Operations | Per-frame output |
|---|---|---|---|
| DINOv3 | `64 × 1025` | LayerNorm → linear `1025→256` → GELU → LayerNorm; learned 16 latent queries; two cross-attention/FFN stages, eight heads, FFN `256→1024→256` | `16 × 256` |
| ConvNeXt | `64 × 2881` | Same resampler, input adapter `2881→256` | `16 × 256` |
| Utonia geometry | `64 × 1234` | Same resampler, input adapter `1234→256` | `16 × 256` |
| WAFT flow | `4 × 64 × 6` | For each spatial slot, temporal Conv1d `6→64`, kernel 3/padding 1 → GELU → Conv1d `64→128`, kernel 3/padding 1 → GELU; select final temporal position; linear `128→256`; same two-stage Perceiver resampler | `16 × 256` |

The upstream notebook declares an `8×8` raw spatial grid, 16 latents per branch, width 256, and four flow frames in [notebook_02_live.txt:535](C:/path/to/ugp/notebook_02_live.txt:535). Its flow channels are `(u, v, magnitude, divergence, curl, confidence)`, with causal stacks ending at the current frame, in [notebook_02_live.txt:1812](C:/path/to/ugp/notebook_02_live.txt:1812). The input widths and actual frozen modules are defined in [pilot.py:11](C:/path/to/ugp/merged_feelwm_v7/tartanground/pilot.py:11), [pilot.py:49](C:/path/to/ugp/merged_feelwm_v7/tartanground/pilot.py:49), [pilot.py:70](C:/path/to/ugp/merged_feelwm_v7/tartanground/pilot.py:70), and [pilot.py:94](C:/path/to/ugp/merged_feelwm_v7/tartanground/pilot.py:94).

The four outputs concatenate in fixed order **DINO, Conv, Geometry, Flow** into `64 × 256`. Selected camera frames are written to a float16 latent cache; targets use the same frozen resamplers. The recorded resampler checkpoint hash is `a7170559a55cf4cfa43c25ea2778be1da834df66fca17d61462c220d67fdcd4e`. Preparation computes strict URDF forward-kinematic foot positions and a finite-difference `3×3` local Jacobian per leg, with joint perturbation `1e-4`. Sources: [prepare_tartanground_indexed.py:131](C:/path/to/ugp/merged_feelwm_v7/scripts/prepare_tartanground_indexed.py:131), [preparation script:166](C:/path/to/ugp/merged_feelwm_v7/scripts/prepare_tartanground_indexed.py:166), and [preparation metadata](C:/path/to/ugp/preparation_metadata.json:1).

### Histories, commands, standardization, and targets

Leg order is always **FL, FR, HL, HR**. Cached proprioception `P` has 12 physical channels per leg: `q3, dq3, measured torque3, measured force3`. Body `B` contains 12 channels: accelerometer3, gyroscope3, and body twist6. Cached `U` has six control channels; `L_LEG` and error `E` have three channels per leg. The upstream assignment is in [notebook_02_live.txt:832](C:/path/to/ugp/notebook_02_live.txt:832).

Statistics are fitted only from training anchors and the history/target indices needed by those anchors. Per-channel standard deviations use population standard deviation plus `1e-6`. Proprioception, command, and error statistics pool across all four legs; the 36-channel joint/body output scale retains each output channel. Load thresholds are the 25th, 50th, and 75th percentiles of training force magnitudes greater than 10. The same saved statistics are loaded for train, development, and test samples. Sources: [preparation script:223](C:/path/to/ugp/merged_feelwm_v7/scripts/prepare_tartanground_indexed.py:223) and [preparation script:257](C:/path/to/ugp/merged_feelwm_v7/scripts/prepare_tartanground_indexed.py:257).

The initial dataset histories use

\[
P_n=(P-\mu_P)/\sigma_P,\quad B_n=(B-\mu_B)/\sigma_B,\quad
U_n=(U-\mu_U)/\sigma_U,\quad L_n=(L-\mu_L)/\sigma_L,\quad E_n=E/\sigma_E.
\]

Error is scaled without mean subtraction. The leg history concatenates `(P_n12, U_n6, L_n3, E_n3)` to 24 channels. The body history concatenates `(B_n12,U_n6)` to 18 channels. Each history has **50 samples**. The three leg histories end at the anchor physics index, anchor minus 10, and anchor minus 20; the body history ends at the anchor. The macro rollout is labelled at `0.1 k` seconds and shifts ten high-rate history samples per model step in the comparison configurations. Timestamp alignment chooses the most recent physics sample at or before each camera timestamp, rather than interpolating. Sources: [indexed.py:74](C:/path/to/ugp/merged_feelwm_v7/tartanground/indexed.py:74), [indexed.py:175](C:/path/to/ugp/merged_feelwm_v7/tartanground/indexed.py:175), and [preparation script:84](C:/path/to/ugp/merged_feelwm_v7/scripts/prepare_tartanground_indexed.py:84).

| Batched model input | Shape | Content |
|---|---|---|
| `z_window` | `[B,4,64,256]` | Visual latents at `t−3,t−2,t−1,t` |
| `leg_hist_t`, `leg_hist_tm1`, `leg_hist_tm2` | `[B,4,50,24]` each | Three standardized leg histories |
| `body_hist_t` | `[B,50,18]` | Standardized body/control history |
| `body_action_seq` | `[B,K,4]` | Physical `[U0·0.1,U1·0.1,U5·0.1,0.1]` per rollout step |
| `leg_action_seq` | `[B,K,4,4]` | Standardized cached command3 plus standardized norm of the change to the next camera-step command |
| `leg_query_features_seq` | `[B,K,4,24]` | Standardized proprio12, standardized control6, physical foot position3, planned translation3 |
| `future_action_seq` | `[B,K,36]` | End-of-interval standardized `U6/L12`, followed by interval-mean standardized `U6/L12` |
| `physics_seq` | `[B,K,47]` | Normalized ballistic q12, normalized torque12, normalized body acceleration3, normalized force magnitude4, contact-weighted load one-hot16 |
| `horizon_seq` | `[B,K,1]` | `0.1,0.2,…,0.1K` seconds |
| `jointbody_baseline_seq` | `[B,K,36]` | Physical per-leg `(q+0.1dq,dq)` and current body12 |
| `tau_baseline_seq`, `force_baseline_seq` | `[B,K,4,3]` each | Current physical torque and force |
| `jointbody_std`, `tau_std`, `force_std` | `[B,36]`, `[B,3]`, `[B,3]` | Training output scales |
| `target_z_seq` | `[B,K,64,256]` | Frozen future visual latents |
| `target_jointbody_seq` | `[B,K,36]` | Physical future joint/body state |
| `target_tau_seq`, `target_force_seq` | `[B,K,4,3]` each | Physical future torque and force |
| Contact, load, phase targets | `[B,K,4]` each | Binary contact, four load classes, four transition phases |
| `foot_jacobian` | `[B,4,3,3]` | Local URDF Jacobian at the initial measured pose |

The indexed path actually uses `encode_cache_native_action`: `(command3, ||command_next−command_now||2)`, standardized by training command and scalar delta statistics. The older `build_leg_action` helper encodes joint-command deltas and torque-command components; that helper is not the action construction used by the indexed comparison. Planned translation is `0.30·U[:3]` repeated across legs. Phase classes are `0=swing→swing`, `1=touchdown`, `2=stance→stance`, `3=liftoff`. Sources: [windows.py:70](C:/path/to/ugp/merged_feelwm_v7/tartanground/windows.py:70), [indexed.py:113](C:/path/to/ugp/merged_feelwm_v7/tartanground/indexed.py:113), [indexed.py:159](C:/path/to/ugp/merged_feelwm_v7/tartanground/indexed.py:159), and [pilot.py:113](C:/path/to/ugp/merged_feelwm_v7/tartanground/pilot.py:113).

These are forecasts conditioned on supplied future recorded controller commands. Deployment would need an available planned command sequence. Later query inputs retain the supplied control and planned displacement channels, while replacing measured future proprioception and foot position with recurrent predictions.

Preparation uses two 700-camera-frame windows centred near `0.25(F−1)` and `0.75(F−1)`, with starts clamped to valid ranges. Overlapping frames are cached once with both memberships. Twenty-eight training trajectories supply chronological 80% training anchors and development anchors after a 50-camera-frame purge. Four entire trajectories are strict test holdouts: `ForestEnv/P2001`, `GreatMarsh/P2027`, `OldTownFall/P2001`, `SeasonalForestWinter/P2006`. Saved counts are **28,103 train**, **5,632 development**, and **5,037 strict test** anchors; K=1 and K=5 share compatible anchors. Sources: [windows.py:24](C:/path/to/ugp/merged_feelwm_v7/tartanground/windows.py:24), [windows.py:51](C:/path/to/ugp/merged_feelwm_v7/tartanground/windows.py:51), and [preparation script:335](C:/path/to/ugp/merged_feelwm_v7/scripts/prepare_tartanground_indexed.py:335).

## 3. Exact model layers and tensor flow

### History encoders and embeddings

There are two history encoders: a **shared leg encoder** applied to all four legs, and a separate body encoder. Each projects its input to width 128, adds learned `[1,512,128]` positional embeddings, applies two causal attention blocks with four heads (32 channels/head) and FFN `128→512→128`, applies LayerNorm, and selects the final history token. The leg summaries receive a learned `4×128` leg identity embedding. The three delayed leg histories reuse the same encoder weights; they are not three independent encoders. Source: [model.py:60](C:/path/to/ugp/merged_feelwm_v7/model.py:60), [model.py:263](C:/path/to/ugp/merged_feelwm_v7/model.py:263), and [model.py:347](C:/path/to/ugp/merged_feelwm_v7/model.py:347).

Each standard attention block uses pre-LayerNorm, residual multihead attention, and a second pre-LayerNorm residual GELU FFN. General dropout is **0.10**. Attention is `nn.MultiheadAttention(..., batch_first=True, need_weights=False)`; these declarations do not establish a particular fused attention kernel. Source: [model.py:27](C:/path/to/ugp/merged_feelwm_v7/model.py:27) and [model.py:43](C:/path/to/ugp/merged_feelwm_v7/model.py:43).

Before visual projection, a learned modality identity tensor `[1,4,1,256]` is added to the four branch blocks. A linear `256→800` projects every visual token. After group construction, learned group `[1,9,1,800]`, slot `[1,1,64,800]`, and time `[1,4,800]` embeddings are added. Time IDs are `[0,0,0,1,1,1,2,3,3]`. Sources: [model.py:280](C:/path/to/ugp/merged_feelwm_v7/model.py:280) and [model.py:388](C:/path/to/ugp/merged_feelwm_v7/model.py:388).

### FiLM timing and the nine world groups

The shared leg action encoder and separate body action encoder are MLPs `4→128→128`. Linear history projections are `128→128`. Each of five independent FiLM generators is `128→512→1600`, split into gamma and beta of width 800. Its final layer is zero-initialized, so FiLM starts at identity:

\[
\operatorname{FiLM}(Z,c)=Z\odot(1+\gamma(c))+\beta(c),\qquad
c_\ell=W_hh_{\ell,\mathrm{delay}}+\operatorname{Enc}_a(a_{\ell,t}).
\]

Every leg uses the **current rollout-step action**. Delayed terrain and delayed histories are explicit and differ by one step:

| Group | Terrain latents | FiLM history | Action | Tokens |
|---|---|---|---|---|
| G0 | `Z(t−3)` | None | None | `64×800` |
| G1 | `Z(t−3)` | HL history ending `t−2` | HL action at `t` | `64×800` |
| G2 | `Z(t−3)` | HR history ending `t−2` | HR action at `t` | `64×800` |
| G3 | `Z(t−2)` | None | None | `64×800` |
| G4 | `Z(t−2)` | FL history ending `t−1` | FL action at `t` | `64×800` |
| G5 | `Z(t−2)` | FR history ending `t−1` | FR action at `t` | `64×800` |
| G6 | `Z(t−1)` | None | None | `64×800` |
| G7 | `Z(t)` | None | None | `64×800` |
| G8 | `Z(t)` | Body history ending `t` | Body action at `t` | `64×800` |

Sources: [model.py:128](C:/path/to/ugp/merged_feelwm_v7/model.py:128), [model.py:354](C:/path/to/ugp/merged_feelwm_v7/model.py:354), and [model.py:376](C:/path/to/ugp/merged_feelwm_v7/model.py:376).

### Shared world transformer and per-leg addition

The baseline flattens `[B,9,64,800]` to `[B,576,800]`. Eight attention blocks use ten heads (80 channels/head), FFN ratio seven (`800→5600→800`), and final LayerNorm. The physical-time mask forbids keys from a later group time, while allowing all tokens/groups at the same time to attend each other. It is not a strictly triangular mask over the 576 token positions. Sources: [config.py:20](C:/path/to/ugp/merged_feelwm_v7/config.py:20), [model.py:120](C:/path/to/ugp/merged_feelwm_v7/model.py:120), and [model.py:393](C:/path/to/ugp/merged_feelwm_v7/model.py:393).

The per-leg variant inserts **four separately parameterized** transformers on the 64-token FiLM groups G4/FL, G5/FR, G1/HL, and G2/HR. Each applies `800→384`, four attention blocks with eight heads (48 channels/head), FFN `384→1536→384`, and `384→800`, then returns

\[
X_\ell'=X_\ell+0.1\,W_{\mathrm{out}}T_\ell(W_{\mathrm{in}}X_\ell).
\]

These leg transformers have no temporal mask because each processes a single 64-token bank. They operate after FiLM and the group/slot/time embeddings, immediately before shared attention. Other groups are passed through directly. They are constructed after all baseline modules so common baseline parameters share the same seeded initial values. The residual scale is small, but the leg addition is **not zero-initialized**, so per-leg outputs do not initially equal baseline outputs. Sources: [model.py:142](C:/path/to/ugp/merged_feelwm_v7/model.py:142), [model.py:330](C:/path/to/ugp/merged_feelwm_v7/model.py:330), and [model.py:402](C:/path/to/ugp/merged_feelwm_v7/model.py:402).

### Visual heads, terrain queries, and V7 decoding

Four independent MLP heads `800→512→256` operate on the corresponding 16 modality slots of G8. Their concatenation predicts the next `64×256` visual state.

The five query vectors are global, FL, FR, HL, HR. The global vector adds a learned `[1,1,256]` query to a `128→256` body-history projection. Each leg combines a learned `4×256` query embedding with a state adapter `LayerNorm24→Linear24→256→GELU→Linear256→256`. Its visual bank is projected by `LayerNorm800→Linear800→256`; aligned banks are **G8, G4, G5, G1, G2** in that order. Four query-fusion blocks each perform separate one-query cross-attention to the corresponding 64-token bank, self-attention among all five queries, and FFN `256→1024→256`, with eight heads (32 channels/head). Final query LayerNorm gives `[B,256]` global and `[B,4,256]` leg vectors. Sources: [model.py:293](C:/path/to/ugp/merged_feelwm_v7/model.py:293), [model.py:84](C:/path/to/ugp/merged_feelwm_v7/model.py:84), and [model.py:425](C:/path/to/ugp/merged_feelwm_v7/model.py:425).

| Decoder stage | Exact width/shape |
|---|---|
| Per-leg history/visual fusion | Concatenate current history128 and query256; `LayerNorm384→Linear384→256→GELU→Linear256→256→GELU→LayerNorm256`; output `[B,4,256]` |
| Global context | Flatten four fused legs1024 + body history128 + global query256 = 1408; `LayerNorm1408→Linear1408→768→GELU→Dropout→Linear768→384→GELU→LayerNorm384` |
| Control projection | `36→128→GELU→LayerNorm128` |
| Horizon projection | `1→64→GELU→128→GELU→LayerNorm128` |
| Physics projection | `47→128→GELU→128→GELU→LayerNorm128` |
| Dynamics vector | Context384 + control128 + horizon128 + physics128 = `[B,768]` |
| Contact head | `768→256→GELU→4`; sigmoid → `[B,4]` |
| Load regime head | `768→256→GELU→16`; reshape `[B,4,4]`; softmax |
| Joint/body residual head | `768→384→GELU→Dropout→36` |
| Per-leg force context | Dynamics768 + fused leg256 + terrain query256 + contact1 + load probabilities4 = `[B,4,1285]` |
| Torque/base force head | `LayerNorm1285→1285→512→GELU→Dropout→256→GELU→6`; first3 torque, last3 force |
| Base magnitude/direction auxiliary head | `LayerNorm1285→1285→256→GELU→4`; one log magnitude and three normalized direction channels |

The model predicts **physical outputs by scaled residuals**, not standardized target values:

\[
\hat s=b_s+r_s\odot\sigma_s,\qquad
\hat\tau=b_\tau+r_\tau\odot\sigma_\tau,\qquad
F_b=b_F+r_F\odot\sigma_F.
\]

Sources: [model.py:308](C:/path/to/ugp/merged_feelwm_v7/model.py:308) and [model.py:442](C:/path/to/ugp/merged_feelwm_v7/model.py:442).

### V7-F force specialist

The specialist takes `[B,K,4,1285]`. Input processing is `LayerNorm1285→Linear1285→256→GELU→Dropout0.10→Linear256→256→GELU→LayerNorm256`. It reshapes to `[B·4,K,256]` so each leg attends across available horizons independently, using two causal attention blocks, eight heads, and FFN `256→1024→256`. There is no separate learned horizon positional embedding inside this specialist; horizon information already enters its dynamics context.

Its phase head is `LayerNorm256→256→128→GELU→4`. The gate input is **265 dimensions**: latent256 + phase probabilities4 + load probabilities4 + contact1. Four independent experts each use `LayerNorm256→256→192→GELU→Dropout0.08→96→GELU→3`. Their softmax gate uses `LayerNorm265→265→128→GELU→4`. A direct correction head uses `LayerNorm256→256→192→GELU→3`; a magnitude/direction head uses `LayerNorm256→256→192→GELU→4`; fusion uses `LayerNorm265→265→96→GELU→1` and sigmoid.

For expert weights \(w_e\), expert corrections \(E_e\), direct correction \(D\), and magnitude/direction outputs \(m,d\), the exact force construction is

\[
c=\sum_{e=1}^{4}w_eE_e+D,\qquad F_{\mathrm{direct}}=F_b+c\odot\sigma_F,
\]
\[
M'=\max\{0,\operatorname{expm1}[\log(1+\|F_b\|)+0.75\tanh(m)]\},
\]
\[
u'=\operatorname{normalize}(\operatorname{normalize}(F_b)+0.20\tanh(d)),\qquad
F_{\mathrm{md}}=M'u',
\]
\[
F_{\mathrm{V7F}}=\alpha F_{\mathrm{direct}}+(1-\alpha)F_{\mathrm{md}}.
\]

The base magnitude is clamped to `1e-5`; normalizations use epsilon `1e-6`. Expert, direct correction, magnitude/direction, and fusion final layers start at zero, giving zero correction and fusion `α=0.5`, so the specialist starts at the base force up to the magnitude/normalization clamps. Source: [model.py:169](C:/path/to/ugp/merged_feelwm_v7/model.py:169).

## 4. K=1 and K=5 recurrent behavior

K=1 predicts `t+0.1`. K=5 repeats the same model five times, labelling predictions `t+0.1,…,t+0.5`. It does not use five separately trained horizon models. The first step uses observed initial histories/visuals plus the first prepared query, physics, and residual baselines. For later steps:

1. Rebuild proprioceptive query channels from predicted physical joint state, torque, and **refined V7-F force**, then standardize them with training statistics. Retain only the supplied control6 and planned displacement3 from the prepared per-horizon query.
2. Update foot position using the initial cached local Jacobian and changes in predicted joint angles. It never copies a later measured foot position into the recurrent query.
3. Rebuild physics from predicted `q+0.1dq`, torque, body acceleration, force magnitude, and contact-weighted predicted regime probabilities. Set residual baselines to the preceding predicted joint/body state, torque, and refined force.
4. Run the world model and evaluate the force specialist on the entire currently available causal prefix. Feed the last refined force into the next recurrent step.
5. Shift the visual window and append predicted latents. Shift leg/body histories by ten samples, repeating the next macro state ten times. Move delayed leg-history banks backward: `tm2←tm1`, `tm1←t`.

There is no detach between recurrent steps. The history appender repeats macro outputs; it does not integrate ten independently predicted high-rate transitions. The error channels use the initial `err_seq[:,0]` at every step, whereas known command channels vary with the supplied sequence. Sources: [model.py:503](C:/path/to/ugp/merged_feelwm_v7/model.py:503), [model.py:537](C:/path/to/ugp/merged_feelwm_v7/model.py:537), [model.py:556](C:/path/to/ugp/merged_feelwm_v7/model.py:556), and [model.py:670](C:/path/to/ugp/merged_feelwm_v7/model.py:670).

K=5 comparison training deterministically alternates examples between fully recurrent feedback and teacher-forced feedback. The mode is selected by parity of `sample_offset + example_index + seed`; the initial observation is always shared. A forced example replaces previous-step predicted visual state, joint/body state, torque, force, contact, and regime with previous-step targets before the next step. It is one mode per example, not independent random forcing per horizon. Validation omits the mask and is fully recurrent. Sources: [rollout.py:6](C:/path/to/ugp/merged_feelwm_v7/tartanground/rollout.py:6), [train.py:237](C:/path/to/ugp/merged_feelwm_v7/train.py:237), and [model.py:677](C:/path/to/ugp/merged_feelwm_v7/model.py:677).

### Confirmed source issues affecting the current indexed K=5 path

**Mixed units in recurrent histories.** Initial histories contain standardized `P_n` and `B_n`, but `_make_leg_history_step` appends physical predicted `q,dq,torque,force` and `_make_body_history_step` appends physical predicted body12. They concatenate those with standardized control/command/error fields and append directly, without restandardizing the predicted proprioception/body channels. Thus recurrent histories mix standardized observed values and physical predicted values, even on teacher-forced examples. By the fifth prediction, the current 50-sample history has incorporated four ten-sample appends. This is a confirmed code mismatch, not a measured estimate of its error impact. Sources: [indexed.py:79](C:/path/to/ugp/merged_feelwm_v7/tartanground/indexed.py:79), [model.py:513](C:/path/to/ugp/merged_feelwm_v7/model.py:513), and [model.py:702](C:/path/to/ugp/merged_feelwm_v7/model.py:702).

**Wrong units in the first foot-position increment.** `previous_q` is initialized from the last three q channels of the standardized initial leg history. The next q is in physical radians. The first Jacobian increment therefore computes `J·(q_physical_prediction−q_standardized_initial)`. Subsequent `previous_q` values are physical, so later differences have consistent units, but the first offset remains in the accumulated foot positions. Sources: [model.py:612](C:/path/to/ugp/merged_feelwm_v7/model.py:612) and [model.py:696](C:/path/to/ugp/merged_feelwm_v7/model.py:696). The Jacobian is held at the initial pose throughout the rollout.

**Different later-step q baselines.** The initial additive state baseline is ballistic `q+0.1dq`; later additive state baselines are the preceding predicted state without that q increment, although the rebuilt physics vector still includes predicted `q+0.1dq`. This is an exact description of the current implementation, rather than a claim that its accuracy consequence has been established.

**Specialist prefixes are recomputed during training.** Force actually used for earlier feedback is saved in `refined_force_seq`. The loss reads `out['v7f']`, which is the final prefix evaluation and recomputes earlier prefix outputs. Active dropout can make these training realizations differ. In evaluation, dropout is disabled and the causal mask prevents later horizons from contributing to earlier specialist outputs. Sources: [model.py:662](C:/path/to/ugp/merged_feelwm_v7/model.py:662), [model.py:716](C:/path/to/ugp/merged_feelwm_v7/model.py:716), and [losses.py:43](C:/path/to/ugp/merged_feelwm_v7/losses.py:43).

These issues appear in both the fresh deployed source and the packaged K1/K5 pilot source. The older review-baseline future-state inputs should therefore not be attributed to the current code. Conversely, the corrected query/physics feedback does not resolve the history and initial-FK unit mismatches. This read-only audit leaves the measured run results unchanged and does not quantify the improvements a fix would produce.

## 5. Exact objective, including all coefficients

Let `mean` average all remaining sample/leg/token/channel dimensions. Define Smooth-L1

\[
H_\beta(e)=\begin{cases}e^2/(2\beta),&|e|<\beta,\\|e|-\beta/2,&\text{otherwise},\end{cases}
\qquad
\mathcal H_\beta(x,y;\sigma)=\operatorname{mean}H_\beta((x-y)/(\sigma+10^{-6})).
\]

The normalized cosine loss is `1−mean(normalize(x)·normalize(y))` with epsilon `1e-8`. General force/state normalization uses epsilon `1e-6`. Each horizon's four visual losses are

\[
L_D=\operatorname{MSE}(\hat Z_D,Z_D)+0.1L_{\cos}(\hat Z_D,Z_D),\quad
L_C=\operatorname{MSE}(\hat Z_C,Z_C)+0.1L_{\cos}(\hat Z_C,Z_C),
\]
\[
L_G=\operatorname{mean}H_1(\hat Z_G-Z_G)+0.1L_{\cos}(\hat Z_G,Z_G),\quad
L_W=\operatorname{mean}H_1(\hat Z_W-Z_W)+0.1L_{\cos}(\hat Z_W,Z_W).
\]

Contact uses BCE with logits; load regime uses four-class cross-entropy. Joint/body36, torque3, and base force3 each use normalized Smooth-L1 with **β=0.35**. The base force auxiliary loss is

\[
A_b=0.10\operatorname{mean}H_{0.25}(\hat\ell_b-\log(1+\|F\|))+
0.02\frac{\sum(1-\hat u_b\cdot u_F)\,\mathbf1[\|F\|\ge20]}{\sum\mathbf1[\|F\|\ge20]+10^{-6}}.
\]

For V7-F, let \(M=\|F\|\), \(\hat M=\|\hat F\|\), and \(s=\max(1,\operatorname{median}(M))\), where the detached median is across the whole batch/horizon/leg tensor. The detached force weight is

\[
\omega=1+1.25\operatorname{clip}(M/s,0,2)+0.50\mathbf1[\text{phase}\in\{1,3\}].
\]

For each horizon k,

\[
V_k=\operatorname{mean}_{B,L}\left[\omega_k\operatorname{mean}_{xyz}H_{0.35}\left(\frac{\hat F_k-F_k}{\sigma_F+10^{-6}}\right)\right],
\]
\[
M_k=\operatorname{mean}H_{0.20}(\log(1+\hat M_k)-\log(1+M_k)),
\]
\[
D_k=\frac{\sum(1-\hat u_k\cdot u_k)\mathbf1[M_k^{\mathrm{true}}\ge20]}{\sum\mathbf1[M_k^{\mathrm{true}}\ge20]+10^{-6}},\quad
P_k=\operatorname{CE}(\text{phase logits}_k,\text{phase target}_k),
\]
\[
R_k=\operatorname{mean}(c_k^2),\qquad
G_k=\operatorname{mean}_e\left[(\operatorname{mean}_{B,L}w_{k,e}-1/4)^2\right].
\]

Here `M_k` in the magnitude-loss symbol denotes a loss; the direction mask uses the **true physical force magnitude**. The specialist horizon loss is exactly

\[
S_k=2.00V_k+1.60M_k+0.10D_k+0.25P_k+0.015R_k+0.005G_k.
\]

With the comparison weights, the complete supervised horizon loss is

\[
L_k=L_D+L_C+L_G+L_W+0.50L_{\mathrm{contact}}+0.25L_{\mathrm{load}}
+\mathcal H_{0.35}(\hat s,s;\sigma_s)+\mathcal H_{0.35}(\hat\tau,\tau;\sigma_\tau)
+\mathcal H_{0.35}(F_b,F;\sigma_F)+A_b+S_k.
\]

For K>1 the cross-horizon terms are

\[
T=\operatorname{mean}H_{0.35}\left(\frac{(\hat F_{k+1}-\hat F_k)-(F_{k+1}-F_k)}{\sigma_F+10^{-6}}\right),
\]
\[
I=\operatorname{mean}H_{0.35}\left(\frac{\sum_{k=1}^K0.1\hat F_k-\sum_{k=1}^K0.1F_k}{\sigma_F+10^{-6}}\right),
\qquad
L=\sum_{k=1}^K\lambda_kL_k+0.45T+0.35I.
\]

The temporal term matches target force changes; it is not a penalty driving all force changes toward zero. Temporal and impulse regularizers sit **outside** the lambda-weighted horizon sum and are zero for K=1. The ten primary weights are `w_dino=w_conv=w_geom=w_flow=w_jointbody=w_tau=w_base_force=w_v7f_force=1`, `w_contact=0.50`, and `w_load=0.25`. The specialist-only finetuning configuration sets all primary weights except `w_v7f_force` to zero, while retaining the listed specialist coefficients. Sources: [config.py:120](C:/path/to/ugp/merged_feelwm_v7/config.py:120), [losses.py:11](C:/path/to/ugp/merged_feelwm_v7/losses.py:11), [losses.py:47](C:/path/to/ugp/merged_feelwm_v7/losses.py:47), [losses.py:62](C:/path/to/ugp/merged_feelwm_v7/losses.py:62), [losses.py:94](C:/path/to/ugp/merged_feelwm_v7/losses.py:94), and [losses.py:137](C:/path/to/ugp/merged_feelwm_v7/losses.py:137).

Logged component values are unweighted means across horizons; `total` uses lambdas. A weighted total should not be reconstructed by simply summing the printed unweighted component means.

## 6. Training schedules, validation, and resumability

| Setting | Matched capped pilot | Current ten-epoch comparison | Configured V7-F continuation |
|---|---|---|---|
| K=1 stage | 200 optimizer updates; YAML epoch ceiling 100 | 10 nominal epochs; `max_steps=null` | None |
| K=5 initialization | Model weights from matching variant's K=1 `best.pt`; fresh optimizer/scheduler | Same, using matching ten-epoch K=1 `best.pt` | Selected per-leg K=5 checkpoint, by full-development normalized force error |
| K=5 stage | 100 optimizer updates; YAML epoch ceiling 100 | 10 nominal epochs; `max_steps=null` | 7 nominal epochs; `max_steps=null` |
| Trainable parameters | Entire baseline or per-leg merged model, including specialist | Same | Only names under `v7f.` |
| K=1 lambdas | `[1.0]` | `[1.0]` | N/A |
| K=5 lambdas | `[0.2,0.2,0.2,0.2,0.2]` in actual resolved configs | `[0.516,0.258,0.129,0.065,0.032]` | `[0.2,0.2,0.2,0.2,0.2]` |
| Teacher forcing | K=1 off; K=5 deterministic 50/50 examples | Same | Probability annealed from 0.50 to 0 across seven epochs |
| Per-GPU batch/accumulation | `1 / 8` | `1 / 8` | `1 / 8` |
| AdamW learning rate | `2e−4` | `2e−4` | `5e−5` |
| AdamW weight decay | `0.05` for decay group; `0` for excluded tensors | Same | `0.01` for decay group; `0` for excluded tensors |
| Warmup | 10 optimizer steps | 10 optimizer steps | 100 optimizer steps |
| Minimum LR ratio | 0.05 | 0.05 | 0.05 |
| Gradient clipping | Norm 1.0 | Norm 1.0 | Norm 1.0 |
| Precision | bf16 autocast, TF32 enabled, compile off | Same | Same |
| Validation frequency | Every 50 K=1 / 25 K=5 optimizer steps | Same | Every 3,513 optimizer steps |
| Validation subset | First 50 development batches | First 50 development batches | All development batches |
| Best checkpoint metric/name | Minimum `total` / `best.pt` | Minimum `total` / `best.pt` | Minimum normalized force MAE / `best_force.pt` |

Pilot resolved configurations are preserved in [baseline K1](C:/path/to/ugp/resolved_config.json:1), [baseline K5](C:/path/to/ugp/resolved_config.json:1), and [per-leg K5](C:/path/to/ugp/resolved_config.json:1). Current settings are in [baseline K1 YAML](C:/path/to/ugp/merged_feelwm_v7/configs/tartanground_baseline_k1.yaml:1), [baseline K5 YAML](C:/path/to/ugp/merged_feelwm_v7/configs/tartanground_baseline_k5.yaml:1), [per-leg K5 YAML](C:/path/to/ugp/merged_feelwm_v7/configs/tartanground_leg_k5.yaml:1), and [force finetuning YAML](C:/path/to/ugp/merged_feelwm_v7_force_finetune/configs/tartanground_leg_k5_v7f_finetune.yaml:1). The earlier P2000-only 200-update K=1 stage used accumulation four, not eight; its exact [resolved configuration](C:/path/to/ugp/resolved_config.json:1) should be kept distinct from the matched four-variant pilot.

AdamW uses betas `(0.9,0.95)` and observed epsilon `1e−8`. Biases, tensors with fewer than two dimensions, and parameter names containing `norm` or `embedding` are excluded from weight decay. The scheduler multiplier at optimizer step j is

\[
f(j)=\begin{cases}\max(10^{-8},j/W),&j<W,\\
0.05+0.95\,\frac{1+\cos(\pi\operatorname{clip}((j-W)/(S-W),0,1))}{2},&j\ge W,
\end{cases}
\]

with protected denominators `max(1,W)` and `max(1,S−W)`. Nominal `S=epochs·ceil(len(train_loader)/accumulation)`, limited by `max_steps` when a cap exists. There are `ceil(28,103/8)=3,513` nominal updates per complete comparison epoch. The final seven-sample accumulation still divides each loss by eight, so that last update has 7/8 of the full-batch weighting. BF16 does not enable GradScaler; the scaler is enabled only for fp16. Sources: [train.py:57](C:/path/to/ugp/merged_feelwm_v7/train.py:57), [train.py:72](C:/path/to/ugp/merged_feelwm_v7/train.py:72), [train.py:187](C:/path/to/ugp/merged_feelwm_v7/train.py:187), and [train.py:231](C:/path/to/ugp/merged_feelwm_v7/train.py:231).

The indexed sampler shuffles samples in short trajectory blocks with seed `781+epoch`. Its implementation visits each anchor once and shuffles the blocks; despite its `BalancedTrajectorySampler` name, it does not duplicate smaller trajectories to equalize their counts. In DDP, a distributed sampler replaces it. Development rows are interleaved by trajectory, so the first-50-batch comparison validation draws roughly one or two anchors per training trajectory, rather than all 5,632 development anchors. Best checkpoint selection uses `total`, although the trainer also logs a physical score. Sources: [indexed.py:212](C:/path/to/ugp/merged_feelwm_v7/tartanground/indexed.py:212), [preparation script:290](C:/path/to/ugp/merged_feelwm_v7/scripts/prepare_tartanground_indexed.py:290), [train.py:97](C:/path/to/ugp/merged_feelwm_v7/train.py:97), and [train.py:286](C:/path/to/ugp/merged_feelwm_v7/train.py:286).

The ten-epoch K=5 horizon weighting was selected by three 20-update development calibration continuations: uniform `(0.20×5)`, mild decay `(0.30,0.25,0.20,0.15,0.10)`, and strong decay `(0.516,0.258,0.129,0.065,0.032)`. Selection minimizes

\[
\text{physical score}=\tfrac14(\operatorname{MAE}(q/\sigma_q)+\operatorname{MAE}(\tau/\sigma_\tau)+\operatorname{MAE}(F_{\mathrm{V7F}}/\sigma_F)+\operatorname{MAE}(p_{\mathrm{contact}})),
\]

where each MAE means prediction-minus-target error and all horizons are equally averaged. Calibration inherits the normal 50-batch validation limit. The selected strong-decay profile is confirmed by [lambda_calibration.json](C:/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare_10ep_test10/lambda_calibration/lambda_calibration.json:1). Sources: [calibrate_lambdas.py:14](C:/path/to/ugp/merged_feelwm_v7/scripts/calibrate_lambdas.py:14), [calibration script:50](C:/path/to/ugp/merged_feelwm_v7/scripts/calibrate_lambdas.py:50), and [evaluate.py:37](C:/path/to/ugp/merged_feelwm_v7/tartanground/evaluate.py:37).

### GPU use and engineering evidence

The architecture comparison launches baseline on physical GPU0 and per-leg on physical GPU1 as **independent single-GPU jobs**. It is not one two-GPU DDP comparison model. The earlier `full_probe` uses dummy data with the full baseline dimensions; the collected log records a two-GPU DDP engineering update. That probe establishes a runnable update with the configured tensor sizes, not learning or convergence on TartanGround. Sources: [run_tartanground_compare.sh:35](C:/path/to/ugp/merged_feelwm_v7/scripts/run_tartanground_compare.sh:35), [full_probe_remote.yaml:20](C:/path/to/ugp/merged_feelwm_v7/configs/full_probe_remote.yaml:20), and [full_probe train.log](C:/path/to/ugp/merged_feelwm_v7_runs/full_probe/train.log:1).

### Checkpoint resume repeats the unfinished epoch

Checkpoints persist model, optimizer, scheduler, scaler, RNG state, epoch, global step, best validation score, and configuration. They do **not** persist a batch cursor or accumulated gradients. Mid-epoch saves record the current epoch; resume starts `for epoch in range(saved_epoch,epochs)` and enumerates the loader from its beginning. A mid-epoch restart therefore revisits already processed batches in that epoch while retaining the optimizer/scheduler global step. End-of-epoch checkpoints save `epoch+1`. Thus ten nominal epoch counters do not establish exactly ten nonrepeated passes, and resumed K=1 runs can contain more than `10×3,513` optimizer updates. Sources: [checkpointing.py:43](C:/path/to/ugp/merged_feelwm_v7/checkpointing.py:43), [train.py:198](C:/path/to/ugp/merged_feelwm_v7/train.py:198), [train.py:224](C:/path/to/ugp/merged_feelwm_v7/train.py:224), and [train.py:311](C:/path/to/ugp/merged_feelwm_v7/train.py:311).

### Configured force finetuning

Only `v7f.` parameter names have `requires_grad=True`; **2,448,554 of 141,380,144** parameters can update. The trainer still sets the whole model to training mode, so frozen modules retain training dropout. Freezing their weights does not remove recurrent forward computation or the gradient path through their inputs. The same current model source retains the history/FK unit issues described above. Source: [force train.py:72](C:/path/to/ugp/merged_feelwm_v7_force_finetune/train.py:72).

For zero-based epoch e in seven epochs, forcing probability is `0.5−0.5e/6`, giving **0.50, 0.4167, 0.3333, 0.25, 0.1667, 0.0833, 0**. For p=0.5 it preserves parity selection; otherwise a deterministic multiplicative hash of sample index chooses fully forced examples. Source: [force rollout.py:6](C:/path/to/ugp/merged_feelwm_v7_force_finetune/tartanground/rollout.py:6).

Checkpoint selection minimizes `mean(abs(F_V7F−F_target)/clamp(σ_F,1e−6))` over all development samples, horizons, legs, and axes. The initial checkpoint selector explicitly visits the full development loader; training validation also has `validation_max_batches=null`. The launcher waits for completion of both comparison K=5 jobs and the comparison phase before selecting a source and starting seven epochs. The collected [phase.txt](C:/path/to/ugp/merged_feelwm_v7_runs/tartanground_k1_k5_leg_compare_10ep_test10/v7f_force_finetune/phase.txt:1) contains `waiting_for_comparison_pid_284634`, not training completion. Sources: [force evaluate.py:17](C:/path/to/ugp/merged_feelwm_v7_force_finetune/tartanground/evaluate.py:17), [select_force_checkpoint.py:25](C:/path/to/ugp/merged_feelwm_v7_force_finetune/scripts/select_force_checkpoint.py:25), [force train.py:128](C:/path/to/ugp/merged_feelwm_v7_force_finetune/train.py:128), and [force launcher:18](C:/path/to/ugp/merged_feelwm_v7_force_finetune/scripts/run_v7f_force_finetune_after_compare.sh:18).

## 7. Parameter counts and their scope

The totals below include all trainable merged-model weights, biases, learned embeddings, and V7-F. They exclude the frozen cached-feature backbones and the frozen preparation resamplers. They are supported by exact checkpoint state totals (**459 tensors baseline, 667 tensors per-leg**) and by independently summing the source layer formulas. A Linear has `in·out+out` parameters, LayerNorm has `2d`, and a same-width multihead attention layer has `4d²+4d`. One standard attention block with FFN ratio r has `(4+2r)d²+(r+9)d` parameters. No model execution was needed for these arithmetic checks.

| Component | Parameters |
|---|---:|
| Shared leg history encoder | 465,536 |
| Body history encoder | 464,768 |
| Leg identity embeddings | 512 |
| Leg/body action encoders | 34,304 |
| Leg/body conditioning projections | 33,024 |
| Five FiLM generators | 4,434,240 |
| Visual projection and modality/group/slot/time embeddings | 268,224 |
| Shared eight-layer world transformer and final norm | 92,264,000 |
| Four next-visual heads | 2,165,760 |
| Query-bank projection and initial query construction | 313,200 |
| Four query-fusion blocks and final query norm | 4,216,320 |
| Per-leg history/visual fusion | 165,632 |
| Global context | 1,380,992 |
| Control, horizon, and physics projections | 36,608 |
| Contact, regime, and joint/body heads | 708,024 |
| Base torque/force and magnitude/direction heads | 1,126,686 |
| V7-F specialist | 2,448,554 |
| **Baseline merged model** | **110,526,384** |
| Each independent leg pretransformer | 7,713,440 |
| Four leg pretransformers added | 30,853,760 |
| **Per-leg merged model** | **141,380,144** |

The specialist's 2,448,554 parameters decompose into input projection 398,090; two horizon attention blocks 1,579,520; phase head 33,924; four experts 274,700; expert gate 35,094; direct correction 50,435; magnitude/direction 50,628; and fusion gate 26,163. Its count is already included in both merged totals.

Count evidence: [baseline checkpoint metadata:18](C:/path/to/ugp/feelwm_checkpoint_metadata.json:18) and [per-leg checkpoint metadata:6442](C:/path/to/ugp/feelwm_checkpoint_metadata.json:6442). Source construction starts at [model.py:256](C:/path/to/ugp/merged_feelwm_v7/model.py:256).


---

<a id="feelwm--tartanground-experiment-records-and-results"></a>

# FeelWM / TartanGround experiment records and results

These records follow the architecture section above. `K=1` predicts one 0.1 s transition; `K=5` predicts five successive transitions through 0.5 s. A pilot cap is an optimizer-step budget. The ten-epoch study is a separate run using the same prepared data.

## Engineering probes

`full_probe` completed one optimizer step with the full 110,526,384-parameter baseline under **DDP, world size two, BF16**, according to its log. Its recorded training total was **24.4411125183** and elapsed launcher timing **4.97 seconds**. Inputs were synthetic integration data, so this is proof of execution/backpropagation plumbing, not a robotics generalization result.

`tiny_integration` retained one training step (total **24.5023097992**) and one validation event (total **25.9814872742**). Its saved configuration uses a reduced model. A corresponding GPU-identifying log was not retained in this collected record; it is listed as an engineering artifact rather than asserted as a separate scientific RTX result. Neither probe supplies physical test metrics.

## Single-trajectory P2000 K1 pilot

**Architecture:** the earlier baseline merged model, 110.53 million trainable parameters, with the earlier loss implementation preserved in the P2000 archive. The later per-leg/K5 implementation is audited separately.

**Data/training:** ForestEnv/P2000, 1,200 camera frames over 119.9 s; 834 training samples and 238 later validation samples, separated by a 5 s purge. Two RTX 4080 GPUs with DDP; per-GPU batch one, accumulation four, approximately eight examples per optimizer update; BF16 and TF32. The retained result summary records **200 optimizer steps**, approximately **1.91 passes** through the training examples and approximately **592 s** for the successful pipeline. The metrics stream has an earlier repeated step-one attempt; the final checkpoint/result is the step-200 lineage.

**Evaluation:** validation events at steps 50/100/150/200, capped to 50 batches per GPU (100 samples per event). This is later-time validation within P2000. It is not an unseen-trajectory test. No physical-unit RMSE, accuracy, precision, recall or F1 was exported.

| Step | Total loss | DINO MSE | Conv MSE | Geometry MSE | Flow MSE |
| --- | --- | --- | --- | --- | --- |
| 50 | 22977.511 | 10079.147 | 12844.022 | 23.242483 | 22.913247 |
| 100 | 19440.593 | 8393.5964 | 10999.865 | 20.755019 | 20.566069 |
| 150 | 17146.58 | 7328.14 | 9774.7829 | 19.654464 | 18.936581 |
| 200 | 16473.875 | 7017.4925 | 9414.1099 | 19.253818 | 18.391376 |

Final recorded validation components:

| Loss component | Value |
| --- | --- |
| total | 16473.875 |
| dino | 7017.4925 |
| conv | 9414.1099 |
| geom | 19.253818 |
| flow | 18.391376 |
| contact | 0.049390045 |
| load | 0.43248138 |
| jointbody | 0.32419428 |
| tau | 0.56029044 |
| base_force | 0.40332079 |
| base_force_aux | 0.036607956 |
| v7f | 3.1710548 |
| v7f_vec | 0.96297888 |
| v7f_mag | 0.73879521 |
| v7f_dir | 0.34140777 |
| v7f_phase | 0.11495621 |
| v7f_temp | 0 |
| v7f_impulse | 0 |

The two image-latent losses dominate this composite objective. A value such as `v7f_vec=0.96297888` is a standardized training term; it is not 0.96 N force error. The checkpoint SHA-256 recorded by its export is `c463db53de0c1fedef9803bd150dc6d56b2d2c298856e37314b964afc71f671e`. The frozen visual-resampler checkpoint SHA-256 is `a7170559a55cf4cfa43c25ea2778be1da834df66fca17d61462c220d67fdcd4e`. Those are recorded export provenance, not newly recomputed multi-gigabyte hashes in this audit.

## Matched baseline / per-leg K1 / K5 pilot

**Data:** 28 training trajectories (seven in each environment) and four strict held-out trajectories. Each trajectory contributes two 700-camera-frame windows centered at one-quarter and three-quarters of its available sequence, with boundary clamping and overlap cached once. The retained preparation metadata and freshly counted manifests agree on **28,103 training anchors, 5,632 development anchors and 5,037 strict-test anchors** for each K. The preparation purge is 50 camera frames. A trajectory can occur in both training and development; the strict-test trajectory identities must occur in neither.

| Strict held-out trajectory |
| --- |
| ForestEnv/P2001 |
| GreatMarsh/P2027 |
| OldTownFall/P2001 |
| SeasonalForestWinter/P2006 |

The pilot evaluation used all 5,037 strict-test anchors, not the later forty-case selection and not the V7 notebook's sixteen diagnostic tests. K5 evaluates five horizons at each anchor.

**Execution:** baseline was assigned physical GPU0 and per-leg physical GPU1 as separate **single-GPU jobs**, not DDP between the two architectures. Each uses batch one, accumulation eight, BF16, TF32, AdamW lr2e-4, warmup10 steps, cosine decay to5% of peak, clip1, weight decay0.05 (separate non-decay group0), seed781. K1 ended at200 optimizer steps; K5 ended at100 optimizer steps. YAML `epochs=100` did not cause 100 epochs: the explicit caps terminated the jobs. K5 initialized model weights from its architecture's K1 `best.pt` and began a new optimizer/scheduler.

**Checkpoint/evaluation:** validation was capped at50 loader batches; `best.pt` minimized composite validation `total`. The strict-test evaluator uses each architecture/K's best checkpoint and free rollout. All four jobs reached their recorded cap and the pilot launcher reached `complete` on30 September. The server archive size was12,149,956,904 bytes. This audit read its metadata and selected small members; it did not claim that an older incomplete local12GB transfer was complete.

| Architecture | K | q MAE deg | Torque MAE Nm | Force component MAE N | Force component RMSE N | Contact force cosine | Contact F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 3.8356683 | 11.280144 | 17.926031 | 38.594505 | 0.98872972 | 0.97091206 |
| baseline | 5 | 8.4852743 | 15.528279 | 16.304628 | 34.632328 | 0.98866051 | 0.97328862 |
| per_leg | 1 | 3.9817481 | 11.858758 | 25.769657 | 61.922443 | 0.72110474 | 0.97476083 |
| per_leg | 5 | 8.4323854 | 16.043526 | 32.28215 | 82.353546 | 0.2750448 | 0.97769424 |

At this capped budget, the added leg transformers increased force error, especially at K5. They slightly increased contact F1. This is the outcome of these trained pilot checkpoints, not proof about their eventual converged ranking. K1 and K5 values have different prediction tasks/horizon aggregation, so comparing their global MAEs alone is not a controlled same-horizon comparison.

### Pilot baseline K1

**Model/training:** 110,526,384 saved parameter scalars; 200 optimizer steps; final logged epoch index0. Best checkpoint step200; validation total16189.7167578. K5 horizon weights are **uniform [0.2,0.2,0.2,0.2,0.2]** in its recorded config. K1 weight is1.

**Results — every global exported metric:**

| Metric | Value |
| --- | --- |
| q_mae_rad | 0.066945046 |
| q_rmse_rad | 0.11615387 |
| q_mae_deg | 3.8356683 |
| q_rmse_deg | 6.6551261 |
| tau_mae_nm | 11.280144 |
| tau_rmse_nm | 17.503363 |
| force_vector_mae | 17.926031 |
| force_vector_rmse | 38.594505 |
| force_magnitude_mae | 36.156319 |
| force_direction_cosine_contact | 0.98872972 |
| contact_accuracy | 0.96997221 |
| contact_precision | 0.98430493 |
| contact_recall | 0.95787876 |
| contact_f1 | 0.97091206 |
| load_accuracy | 0.80747469 |

**By trajectory:**

| trajectory | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ForestEnv/P2001 | 3.6403344 | 11.173263 | 16.4491 | 32.815929 | 0.99103624 | 0.97720247 | 0.84027778 |
| GreatMarsh/P2027 | 3.3784008 | 10.400093 | 15.818004 | 31.979895 | 0.99014986 | 0.96845727 | 0.83002383 |
| OldTownFall/P2001 | 3.6708157 | 11.168476 | 17.189432 | 34.451435 | 0.98972207 | 0.97216853 | 0.83346582 |
| SeasonalForestWinter/P2006 | 4.6524982 | 12.377868 | 22.244755 | 45.371998 | 0.98399258 | 0.96588731 | 0.72619048 |

**By horizon:**

| horizon | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0.1s | 3.8356683 | 11.280144 | 17.926031 | 36.156319 | 0.98872972 | 0.97091206 | 0.80747469 |

**By leg:**

| leg | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL | 3.6456897 | 10.826285 | 17.410646 | 34.555862 | 0.98769379 | 0.97015213 | 0.81496923 |
| FR | 4.1438327 | 11.871914 | 19.380377 | 39.266678 | 0.98654515 | 0.96830518 | 0.76990272 |
| HL | 3.8439674 | 11.413018 | 17.946709 | 37.152596 | 0.98923004 | 0.97384997 | 0.82191781 |
| HR | 3.7091844 | 11.009357 | 16.966404 | 33.650146 | 0.99154967 | 0.97136132 | 0.82310899 |

### Pilot baseline K5

**Model/training:** 110,526,384 saved parameter scalars; 100 optimizer steps; final logged epoch index0. Best checkpoint step100; validation total9059.37808594. K5 horizon weights are **uniform [0.2,0.2,0.2,0.2,0.2]** in its recorded config. K1 weight is1.

**Results — every global exported metric:**

| Metric | Value |
| --- | --- |
| q_mae_rad | 0.14809598 |
| q_rmse_rad | 0.20905425 |
| q_mae_deg | 8.4852743 |
| q_rmse_deg | 11.977925 |
| tau_mae_nm | 15.528279 |
| tau_rmse_nm | 23.361937 |
| force_vector_mae | 16.304628 |
| force_vector_rmse | 34.632328 |
| force_magnitude_mae | 32.802383 |
| force_direction_cosine_contact | 0.98866051 |
| contact_accuracy | 0.97248362 |
| contact_precision | 0.98868442 |
| contact_recall | 0.95836496 |
| contact_f1 | 0.97328862 |
| load_accuracy | 0.82034941 |

**By trajectory:**

| trajectory | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ForestEnv/P2001 | 8.1144295 | 15.085316 | 14.914661 | 29.950621 | 0.99179333 | 0.97889478 | 0.8531746 |
| GreatMarsh/P2027 | 7.3349595 | 15.341397 | 14.211843 | 28.365295 | 0.99138647 | 0.97133221 | 0.82728356 |
| OldTownFall/P2001 | 8.1909952 | 15.239007 | 15.637725 | 31.43759 | 0.98997051 | 0.9770044 | 0.84395866 |
| SeasonalForestWinter/P2006 | 10.299335 | 16.446785 | 20.451567 | 41.450336 | 0.98145717 | 0.96596384 | 0.75702381 |

**By horizon:**

| horizon | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0.1s | 6.3142576 | 10.894423 | 15.123409 | 28.278334 | 0.9883948 | 0.97461294 | 0.82891602 |
| +0.2s | 7.6094871 | 15.234648 | 17.603096 | 36.108173 | 0.98912609 | 0.97163223 | 0.81313282 |
| +0.3s | 9.391346 | 17.558739 | 18.462515 | 39.738899 | 0.98718023 | 0.96987226 | 0.81402621 |
| +0.4s | 8.8696671 | 17.599403 | 15.426446 | 30.765963 | 0.98994327 | 0.97516037 | 0.82171928 |
| +0.5s | 10.241616 | 16.354181 | 14.907677 | 29.120537 | 0.98865825 | 0.97516516 | 0.82395275 |

**By leg:**

| leg | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL | 7.5019889 | 15.223992 | 16.284725 | 32.987747 | 0.99010766 | 0.96872936 | 0.80647211 |
| FR | 7.9503741 | 17.490023 | 17.88485 | 37.320038 | 0.98884493 | 0.9699017 | 0.79221759 |
| HL | 9.1882143 | 13.396887 | 15.468389 | 31.494892 | 0.98823583 | 0.97954052 | 0.83351201 |
| HR | 9.3005219 | 16.002213 | 15.580557 | 29.406849 | 0.98742253 | 0.97495971 | 0.84919595 |

### Pilot per_leg K1

**Model/training:** 141,380,144 saved parameter scalars; 200 optimizer steps; final logged epoch index0. Best checkpoint step200; validation total16184.2716602. K5 horizon weights are **uniform [0.2,0.2,0.2,0.2,0.2]** in its recorded config. K1 weight is1.

**Results — every global exported metric:**

| Metric | Value |
| --- | --- |
| q_mae_rad | 0.06949462 |
| q_rmse_rad | 0.12240673 |
| q_mae_deg | 3.9817481 |
| q_rmse_deg | 7.0133886 |
| tau_mae_nm | 11.858758 |
| tau_rmse_nm | 18.208319 |
| force_vector_mae | 25.769657 |
| force_vector_rmse | 61.922443 |
| force_magnitude_mae | 53.376221 |
| force_direction_cosine_contact | 0.72110474 |
| contact_accuracy | 0.97394282 |
| contact_precision | 0.98810916 |
| contact_recall | 0.96176833 |
| contact_f1 | 0.97476083 |
| load_accuracy | 0.81561445 |

**By trajectory:**

| trajectory | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ForestEnv/P2001 | 3.7828953 | 11.765242 | 24.598249 | 50.541122 | 0.72321087 | 0.98162831 | 0.84583333 |
| GreatMarsh/P2027 | 3.4230578 | 10.972683 | 23.843594 | 49.574211 | 0.73290998 | 0.97365407 | 0.84034948 |
| OldTownFall/P2001 | 3.7697277 | 11.742702 | 25.005505 | 51.592155 | 0.72273338 | 0.97880539 | 0.83644674 |
| SeasonalForestWinter/P2006 | 4.9505324 | 12.953522 | 29.628538 | 61.791542 | 0.70538723 | 0.96501346 | 0.73988095 |

**By horizon:**

| horizon | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0.1s | 3.9817481 | 11.858758 | 25.769657 | 53.376221 | 0.72110474 | 0.97476083 | 0.81561445 |

**By leg:**

| leg | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL | 3.8995624 | 11.059015 | 28.200727 | 54.373425 | 0.45488232 | 0.97427035 | 0.81357951 |
| FR | 4.2458439 | 11.583862 | 22.228142 | 47.748623 | 0.98692483 | 0.97261316 | 0.77764542 |
| HL | 3.8319647 | 12.744309 | 21.90267 | 49.363152 | 0.98889244 | 0.97509579 | 0.84256502 |
| HR | 3.9496236 | 12.047846 | 30.747087 | 62.01968 | 0.44005561 | 0.9771261 | 0.82866786 |

### Pilot per_leg K5

**Model/training:** 141,380,144 saved parameter scalars; 100 optimizer steps; final logged epoch index0. Best checkpoint step100; validation total9055.1884375. K5 horizon weights are **uniform [0.2,0.2,0.2,0.2,0.2]** in its recorded config. K1 weight is1.

**Results — every global exported metric:**

| Metric | Value |
| --- | --- |
| q_mae_rad | 0.1471729 |
| q_rmse_rad | 0.2133092 |
| q_mae_deg | 8.4323854 |
| q_rmse_deg | 12.221716 |
| tau_mae_nm | 16.043526 |
| tau_rmse_nm | 23.453165 |
| force_vector_mae | 32.28215 |
| force_vector_rmse | 82.353546 |
| force_magnitude_mae | 61.250084 |
| force_direction_cosine_contact | 0.2750448 |
| contact_accuracy | 0.97702998 |
| contact_precision | 0.9935154 |
| contact_recall | 0.96236906 |
| contact_f1 | 0.97769424 |
| load_accuracy | 0.8247568 |

**By trajectory:**

| trajectory | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ForestEnv/P2001 | 8.0642319 | 15.793301 | 31.186823 | 58.270813 | 0.27721766 | 0.98225494 | 0.85293651 |
| GreatMarsh/P2027 | 7.0995388 | 15.656142 | 30.85 | 57.338566 | 0.28420478 | 0.97403095 | 0.8327641 |
| OldTownFall/P2001 | 8.1209736 | 15.833729 | 31.810535 | 60.087158 | 0.27759653 | 0.98241555 | 0.84046105 |
| SeasonalForestWinter/P2006 | 10.443248 | 16.890289 | 35.279373 | 69.298859 | 0.26102445 | 0.97211278 | 0.77289683 |

**By horizon:**

| horizon | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +0.1s | 5.7830706 | 10.738683 | 21.995089 | 39.155579 | 0.72236228 | 0.97710878 | 0.82415128 |
| +0.2s | 7.3603487 | 15.584012 | 31.000418 | 58.349354 | 0.4513222 | 0.97511334 | 0.82152075 |
| +0.3s | 9.2786007 | 18.755161 | 36.520771 | 70.897949 | 0.17404766 | 0.97791281 | 0.82067699 |
| +0.4s | 9.1623507 | 18.870871 | 35.895668 | 67.591545 | 0.013362351 | 0.97872546 | 0.82698035 |
| +0.5s | 10.577559 | 16.268898 | 35.998821 | 70.256004 | 0.014019379 | 0.97961151 | 0.83045464 |

**By leg:**

| leg | q_mae_deg | tau_mae_nm | force_vector_mae | force_magnitude_mae | force_direction_cosine_contact | contact_f1 | load_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL | 7.8220468 | 15.383176 | 47.467045 | 86.674446 | -0.44878989 | 0.97293734 | 0.80444709 |
| FR | 7.6608191 | 16.714161 | 18.045828 | 37.604214 | 0.98927796 | 0.97550399 | 0.80162795 |
| HL | 9.0290537 | 16.240095 | 15.074945 | 30.340151 | 0.98996329 | 0.980806 | 0.84534445 |
| HR | 9.2176228 | 15.836665 | 48.540791 | 90.381538 | -0.46551907 | 0.98157152 | 0.8476077 |

### Development-only horizon-weight calibration

Three K5 candidates each trained20 optimizer steps on the same calibration setup and were scored on development data. Candidate selection minimized the normalized physical score described below; strict-test data did not select the weights. Each candidate starts from the corresponding K1 initialization rather than continuing another candidate.

**tartanground_k1_k5_leg_compare:** selected `uniform`.

| Candidate | Horizon weights | Development physical score |
| --- | --- | --- |
| uniform | [0.2, 0.2, 0.2, 0.2, 0.2] | 0.62846229 |
| mild_decay | [0.3, 0.25, 0.2, 0.15, 0.1] | 0.63163024 |
| strong_decay | [0.516, 0.258, 0.129, 0.065, 0.032] | 0.64762971 |

**tartanground_k1_k5_leg_compare_10ep_test10:** selected `strong_decay`.

| Candidate | Horizon weights | Development physical score |
| --- | --- | --- |
| uniform | [0.2, 0.2, 0.2, 0.2, 0.2] | 0.41552295 |
| mild_decay | [0.3, 0.25, 0.2, 0.15, 0.1] | 0.39589581 |
| strong_decay | [0.516, 0.258, 0.129, 0.065, 0.032] | 0.38466316 |

Thus the earlier pilot used uniform weights; the ten-epoch study used strong decay `[0.516,0.258,0.129,0.065,0.032]`. Source equivalence does not imply identical configuration or identical checkpoints.

## Ten-epoch comparison with ten selected cases per unseen trajectory

**Recipe:** same prepared training/development manifests, same parameterized architectures and separate GPU assignments, `epochs=10`, `max_steps=null`, batch1/accumulation8, lr2e-4, AdamW settings as above. K1 validates/checkpoints every50 optimizer steps; K5 every25. Ten uninterrupted passes imply35,130 optimizer steps. K5 initializes only the model weights from its architecture's ten-epoch K1 best checkpoint. Teacher forcing is configured on, with about half of training rollout steps forced by the mask; evaluation is free rollout.

**Resume semantics:** the trainer saves epoch and global optimizer step but no batch cursor. A mid-epoch resume begins that epoch's loader again. The K1 log contains a failed RNG restoration attempt followed by a successful resume at baseline epoch4/step15,600 and per-leg epoch3/step11,950. Consequently the completed K1 runs have extra updates: **1,548 baseline**, **1,411 per-leg** beyond35,130. This is why their final steps exceed the uninterrupted ten-epoch total. Epoch index9 is the tenth epoch; a completed latest checkpoint stores epoch10.

| Architecture | K | Status | Last logged step | Latest durable step | Best step | Best dev total | Latest dev total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | Completed training | 36678 | 36678 | 30850 | 9.6086339 | 10.284466 |
| baseline | 5 | Completed training | 35130 | 35130 | 9125 | 8.5908252 | 12.343047 |
| per_leg | 1 | Completed training | 36541 | 36541 | 24650 | 9.5020302 | 10.362296 |
| per_leg | 5 | Paused in tenth epoch | 33477 | 33475 | 19300 | 8.9718875 | 5713.1246 |

**Per-leg K5:** paused at the audited process snapshot; last logged step33,477, durable checkpoint33,475, leaving1,653 nominal optimizer steps from the logged counter. This is roughly4.7% of nominal K5 updates. That fraction is not an ETA while the process is suspended. Its latest validation total5713.12464844 is far above its best8.97188745618 at step19,300. The late-run objective degraded; the best checkpoint remains the earlier one. The source normalization/FK issues and exposure change identified in the architecture audit are relevant limits, but the retained logs do not establish a unique cause of this degradation.

**Strict test selection:** exactly40 shared anchors, ten evenly spaced common valid camera anchors for each of the four held-out trajectories. The selected set is the same for K1 and K5 and both architectures. Forty cases means40 input anchors for each model/K evaluation. With four legs, this yields160 leg rows for K1 and800 for K5, or1,920 rows across two architectures and both K values. These row counts are derived packaging dimensions, not1,920 independent trajectory cases.

**Results status:** the launcher remains in `k5_baseline_and_per_leg`. It has not reached its recorded strict-evaluation/package phases. No ten-epoch strict-test metrics or completed `tartanground_10epoch_unseen_test_only.zip` were found in that run's collected inventory. The pilot metrics above must not be relabeled as ten-epoch results. The test-only packager is implemented to require the forty-case metrics/prediction/selection/CSV artifacts for all four components and excludes training checkpoints, logs, source and raw caches. The existing `tartanground_10ep_update.zip` is a source/configuration update bundle, not a result archive.

### Ten-epoch baseline K1 recorded development losses

Best and latest below are retained development events, each capped to50 loader batches. They have not been converted to physical strict-test metrics. The best event is selected by `total`, not force MAE alone.

| Loss/score | Best-total event | Latest event |
| --- | --- | --- |
| step | 30850 | 36650 |
| total | 9.6086339 | 10.284466 |
| dino | 0.14932942 | 0.13043566 |
| conv | 8.6614987 | 9.3541272 |
| geom | 0.06969636 | 0.067465678 |
| flow | 0.021019404 | 0.028334955 |
| contact | 0.021045194 | 0.021004142 |
| load | 0.10748423 | 0.10077815 |
| jointbody | 0.067622235 | 0.066849704 |
| tau | 0.02779273 | 0.027280756 |
| base_force | 0.049163553 | 0.049442593 |
| base_force_aux | 0.0035614177 | 0.0034957358 |
| v7f_vec | 0.15612335 | 0.16163242 |
| v7f_mag | 0.12941629 | 0.12242034 |
| v7f_dir | 0.00079261743 | 0.00078712229 |
| v7f_phase | 0.0059589589 | 0.0058612621 |
| v7f_anchor | 0.044886114 | 0.043616762 |
| v7f_gate | 0.00022175597 | 0.0002282394 |
| v7f | 0.52155617 | 0.5213368 |
| horizon_1 | 9.6086339 | 10.284466 |
| lambda_1 | 1 | 1 |
| temporal_regularizer | 0 | 0 |
| impulse_regularizer | 0 | 0 |
| v7f_temp | 0 | 0 |
| v7f_impulse | 0 | 0 |
| physical_score | 0.070565318 | 0.071103195 |

### Ten-epoch baseline K5 recorded development losses

Best and latest below are retained development events, each capped to50 loader batches. They have not been converted to physical strict-test metrics. The best event is selected by `total`, not force MAE alone.

| Loss/score | Best-total event | Latest event |
| --- | --- | --- |
| step | 9125 | 35125 |
| total | 8.5908252 | 12.343047 |
| dino | 0.39752029 | 0.1923274 |
| conv | 5.2354397 | 13.326248 |
| geom | 0.16075608 | 0.096304324 |
| flow | 0.065282233 | 0.035812925 |
| contact | 0.019764713 | 0.01780149 |
| load | 0.15675107 | 0.12435249 |
| jointbody | 0.1353755 | 0.10570615 |
| tau | 0.051105831 | 0.032943712 |
| base_force | 0.090558269 | 0.063336866 |
| base_force_aux | 0.0043248701 | 0.0039947842 |
| v7f_vec | 0.25544744 | 0.1862358 |
| v7f_mag | 0.31873298 | 0.12761061 |
| v7f_dir | 0.0048997795 | 0.0027136983 |
| v7f_phase | 0.041556545 | 0.029503854 |
| v7f_anchor | 0.14160988 | 0.085946047 |
| v7f_gate | 0.00092770261 | 0.001307887 |
| v7f | 1.0338755 | 0.58559161 |
| horizon_1 | 8.5969996 | 10.096703 |
| lambda_1 | 0.51599997 | 0.51599997 |
| horizon_2 | 9.6307763 | 14.246978 |
| lambda_2 | 0.25799999 | 0.25799999 |
| horizon_3 | 9.1983533 | 13.837554 |
| lambda_3 | 0.12899999 | 0.12899999 |
| horizon_4 | 4.2168639 | 15.600874 |
| lambda_4 | 0.064999998 | 0.064999998 |
| horizon_5 | 4.4735491 | 18.629168 |
| lambda_5 | 0.032000002 | 0.032000002 |
| temporal_regularizer | 0.061999133 | 0.057726768 |
| impulse_regularizer | 0.0041971845 | 0.0044662473 |
| v7f_temp | 0.13777585 | 0.12828171 |
| v7f_impulse | 0.011991956 | 0.012760707 |
| physical_score | 0.12448559 | 0.093842207 |

### Ten-epoch per_leg K1 recorded development losses

Best and latest below are retained development events, each capped to50 loader batches. They have not been converted to physical strict-test metrics. The best event is selected by `total`, not force MAE alone.

| Loss/score | Best-total event | Latest event |
| --- | --- | --- |
| step | 24650 | 36500 |
| total | 9.5020302 | 10.362296 |
| dino | 0.16753497 | 0.13586084 |
| conv | 8.3509417 | 9.3405602 |
| geom | 0.099352732 | 0.07581355 |
| flow | 0.014224999 | 0.019378633 |
| contact | 0.021686199 | 0.018940171 |
| load | 0.11575122 | 0.11448678 |
| jointbody | 0.07385923 | 0.067597653 |
| tau | 0.030183628 | 0.026728532 |
| base_force | 0.059237767 | 0.051015816 |
| base_force_aux | 0.0042470233 | 0.0037349686 |
| v7f_vec | 0.19351082 | 0.17289375 |
| v7f_mag | 0.16983125 | 0.15854533 |
| v7f_dir | 0.0011074182 | 0.00088530342 |
| v7f_phase | 0.010769276 | 0.01141965 |
| v7f_anchor | 0.065607254 | 0.065533471 |
| v7f_gate | 0.025616815 | 0.025455005 |
| v7f | 0.66266689 | 0.60351377 |
| horizon_1 | 9.5020302 | 10.362296 |
| lambda_1 | 1 | 1 |
| temporal_regularizer | 0 | 0 |
| impulse_regularizer | 0 | 0 |
| v7f_temp | 0 | 0 |
| v7f_impulse | 0 | 0 |
| physical_score | 0.077505976 | 0.072079109 |

### Ten-epoch per_leg K5 recorded development losses

Best and latest below are retained development events, each capped to50 loader batches. They have not been converted to physical strict-test metrics. The best event is selected by `total`, not force MAE alone.

| Loss/score | Best-total event | Latest event |
| --- | --- | --- |
| step | 19300 | 33475 |
| total | 8.9718875 | 5713.1246 |
| dino | 0.52522367 | 4366.3317 |
| conv | 5.9633578 | 4505.8295 |
| geom | 0.11709564 | 8.0367388 |
| flow | 0.04494598 | 21.533543 |
| contact | 0.012093198 | 0.021963671 |
| load | 0.14205209 | 0.25278022 |
| jointbody | 0.1210108 | 0.76188235 |
| tau | 0.044416777 | 0.27301148 |
| base_force | 0.083887429 | 0.34943278 |
| base_force_aux | 0.0040105717 | 0.0085921466 |
| v7f_vec | 0.25101426 | 0.93195845 |
| v7f_mag | 0.2074246 | 1.5454476 |
| v7f_dir | 0.0029150287 | 0.010441721 |
| v7f_phase | 0.034816452 | 1.0875185 |
| v7f_anchor | 0.11701422 | 0.083875856 |
| v7f_gate | 0.026375759 | 0.10992006 |
| v7f | 0.84479059 | 4.6113647 |
| horizon_1 | 9.4480981 | 599.07394 |
| lambda_1 | 0.51599997 | 0.51599997 |
| horizon_2 | 9.3404978 | 10550.752 |
| lambda_2 | 0.25799999 | 0.25799999 |
| horizon_3 | 7.9435869 | 13702.757 |
| lambda_3 | 0.12899999 | 0.12899999 |
| horizon_4 | 5.9838821 | 8609.2 |
| lambda_4 | 0.064999998 | 0.064999998 |
| horizon_5 | 6.2354303 | 11077.266 |
| lambda_5 | 0.032000002 | 0.032000002 |
| temporal_regularizer | 0.068444557 | 0.16115293 |
| impulse_regularizer | 0.0051671972 | 0.021546526 |
| v7f_temp | 0.15209902 | 0.35811764 |
| v7f_impulse | 0.014763421 | 0.061561503 |
| physical_score | 0.11146819 | 0.39252007 |

## Isolated V7-F-only continuation: implemented, queued

The isolated continuation source and YAML are present. It freezes the merged world/dynamics trunk and trains only the2,448,554-parameter V7-F specialist, with full-development normalized force MAE selecting checkpoints, scheduled teacher forcing0.5→0 over seven epochs, and full strict-manifest evaluation. The retained phase file reads **`waiting_for_comparison_pid_284634`** at2 October05:45:40IST. No completed continuation training/result record was collected. This is a configured queued experiment; it has no measured result to place next to the pilot. Its exact proposed recipe and objective are in the architecture section/configuration appendix.


---

<a id="metric-interpretation-and-comparisons"></a>

# Metric interpretation and comparisons

## Physical regression metrics

For $n$ scalar errors $e_i=\widehat y_i-y_i$, MAE is $\frac1n\sum_i|e_i|$ and RMSE is $\sqrt{\frac1n\sum_i e_i^2}$. RMSE penalizes large errors more strongly. Joint-angle errors are in radians; degrees are radians multiplied by $180/\pi$. Torque errors are in N·m and force errors in N after the model's residual reconstruction. Joint velocities, acceleration, gyro, translation and body twist each have their own channel units; averaging their physical MAEs into one undifferentiated number would not be meaningful.

**TartanGround's `force_vector_mae` name means mean absolute component error:**

$$\mathrm{ForceComponentMAE}=\frac{1}{3n}\sum_{i=1}^n\sum_{c=x,y,z}|\widehat F_{ic}-F_{ic}|.$$

Its `force_vector_rmse` is the corresponding component RMSE. This differs from the Euclidean vector error used in several notebook tables:

$$\mathrm{MeanVectorError}=\frac1n\sum_i\|\widehat{\mathbf F}_i-\mathbf F_i\|_2.$$

Neither should be interchanged with **force-magnitude MAE**, $\frac1n\sum_i|\|\widehat{\mathbf F}_i\|_2-\|\mathbf F_i\|_2|$. A prediction can have nearly correct magnitude and completely wrong direction. V7/V7-F's explicitly named vector metrics and their reductions are specified in their own section; V8-C/F's component and vector metrics are also distinguished there.

**Direction cosine:**

$$c_i=\frac{\widehat{\mathbf F}_i\cdot\mathbf F_i}{\max(\|\widehat{\mathbf F}_i\|_2\|\mathbf F_i\|_2,\epsilon)}.$$

Cosine one means aligned, zero perpendicular, negative one opposite. The TartanGround evaluator averages it only where true force magnitude is at least20N; denominator floor is1e-8. Notebook contact thresholds differ (often10N). A reported direction score therefore depends on both its geometric formula and the force/contact mask.

**Correlation and explained variance:** where exported, Pearson correlation measures centered linear co-variation, not absolute calibration. $R^2=1-\sum_i(\widehat y_i-y_i)^2/\sum_i(y_i-\bar y)^2$ compares squared error against the constant-mean predictor. It may be negative. Constant or nearly constant targets can make correlation or $R^2$ undefined or unstable. Follow the actual evaluator's undefined-value handling documented in the V7 section.

**Percentage/normalized MAE:** the exact denominator is experiment-specific. Dividing a component MAE by a mean absolute target is different from averaging pointwise percentage errors, and both are different from dividing by training standard deviation. Near-zero targets make percentages unstable or undefined; an undefined percentage is not zero error. Cross-family percentage columns must not be compared without matching their formula.

## Classification metrics

Given true positives TP, false positives FP, false negatives FN and true negatives TN:

$$\mathrm{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN},\quad\mathrm{Precision}=\frac{TP}{TP+FP},\quad\mathrm{Recall}=\frac{TP}{TP+FN},$$

$$F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.$$

The TartanGround evaluator thresholds predicted contact probability at0.5, uses true contact labels≥0.5, and protects empty precision/recall denominators with `max(1,count)`. It evaluates load class by argmax over four predicted probabilities. Its reported contact accuracy/F1 and load accuracy are fractions (0.97 means97%). V8 tables also contain fractions and percentage recall columns; column names and the accompanying text specify which is which.

Contact labels in these datasets are derived from estimator force thresholds. These metrics measure agreement with those pseudo labels. They do not independently verify physical ground contact with a separate annotation system.

V8-B demonstrates why recall alone is insufficient: predicting contact everywhere produces recall one while false positives remain. Its contact-position recall at a spatial radius is a separate localization quantity, not the classification recall above.

## Normalized physical score used for TartanGround calibration

The development selection score is the equal average of four families, after averaging each family across horizons:

$$S=\frac14\left[\left\langle\frac{|\widehat q-q|}{\sigma_q}\right\rangle+\left\langle\frac{|\widehat\tau-\tau|}{\sigma_\tau}\right\rangle+\left\langle\frac{|\widehat F-F|}{\sigma_F}\right\rangle+\langle|\widehat p_c-c|\rangle\right].$$

The standard deviations are training/prepared-data scales clamped to1e-6. The force prediction is the refined V7-F output. This score is dimensionless. Contact contributes a probability absolute error, not classification error after thresholding. The score selected horizon weights; the ordinary comparison trainer's `best.pt` still minimizes composite validation total. The queued specialist continuation changes the checkpoint criterion to normalized force MAE alone.

## Spatial, uncertainty and planning metrics

V8-A memory hit rate measures retrieval availability at logged foot proxies; memory age and distance are conditional on retrieval. V8-B contact-position “MAE” is mean Euclidean position error on true-contact samples; recall at5/10/20cm means the fraction within those radii. Reported sigma is predicted standard deviation, not calibrated coverage. A Gaussian NLL can be negative when a narrow density places enough mass around its target; negative NLL does not imply negative physical error.

V8-D's rank/sign agreement checks the predicted sign of paired-response change, not demonstrated causal intervention improvement. V8-E's predicted risk reduction is a model score difference between candidates; it is not measured robot benefit. V8-F's improvement is an offline replay comparison. Its source timing caveat means the result should not be promoted to a strictly causal deployed controller evaluation.

## Aggregation and fair comparison

An average of per-test RMSE values is generally not the same as a pooled RMSE over all errors. Macro averages weight tests/legs equally; sample-weighted aggregates weight longer or denser groups more. The notebook and TartanGround exports use their own grouping/reduction rules, detailed next to each experiment. Reported global metrics therefore retain their source identity instead of being merged into one cross-family leaderboard.

The strongest comparisons here share the same cases, labels and metric code within an experiment. The sixteen locally adapted diagnostic tests, same-run gap sweep, four-trajectory zero-shot evaluation, capped5037-anchor FeelWM pilot and unfinished forty-case ten-epoch study answer different questions. Performance differences across these protocols do not isolate architecture alone.

The final missing measurements remain explicitly missing. Producing a detailed historical log does not complete a paused training job, create its final predictions or repair the causal/provenance limitations identified in its code.



# Accompanying files and audit record

The ZIP is the **documentation evidence package**, not the unfinished ten-epoch test-results ZIP. It includes this log, the run ledger, all exported strict-test metric breakdowns, checkpoint tensor specifications, source snapshots/configurations, and retained small result tables/logs.

- [UGP_RTX_run_ledger.csv](<C:/path/to/ugp/UGP_RTX_run_ledger.csv>)
- [UGP_TartanGround_all_strict_metrics.csv](<C:/path/to/ugp/UGP_TartanGround_all_strict_metrics.csv>)
- [UGP_Model_Tensor_Specs.csv](<C:/path/to/ugp/UGP_Model_Tensor_Specs.csv>)

Unknown or unproduced measurements are identified in the relevant experiment sections. The snapshot reports paused training as paused and queued training as queued.
